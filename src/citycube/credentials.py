"""Live checks that every configured data-provider credential still works.

Credentials expire silently (CDSE OAuth clients, the Earthdata bearer
token) and a run only finds out after queueing, sometimes hours in. These
checks authenticate against each provider with a minimal request so an
expired or revoked credential is reported up front.

Only the outcome (HTTP status, expiry date) is ever reported: response
bodies can contain tokens and are never read into the result.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
import requests

from .config import ClientConfig

TIMEOUT_S = 15
# A credential expiring sooner than this is reported as a warning.
EXPIRY_WARNING_DAYS = 7
HYP3_USER_URL = "https://hyp3-api.asf.alaska.edu/user"

OK, WARNING, ERROR, NOT_CONFIGURED = "ok", "warning", "error", "not_configured"


@dataclass(frozen=True)
class CredentialCheck:
    service: str
    status: str
    detail: str
    expires_at: str | None = None

    @property
    def failed(self) -> bool:
        return self.status == ERROR

    def to_dict(self) -> dict[str, str | None]:
        return {"status": self.status, "detail": self.detail, "expires_at": self.expires_at}


def check_credentials(env_file: str | Path | None = ".env", *, timeout: float = TIMEOUT_S) -> dict[str, CredentialCheck]:
    """Check every provider's credentials concurrently; never raises."""

    if env_file is not None:
        load_dotenv(Path(env_file))
    checks: dict[str, Callable[[float], CredentialCheck]] = {
        "cdse_oauth_client": _check_cdse_client,
        "cdse_account": _check_cdse_account,
        "earthdata": _check_earthdata,
        "cds": _check_cds,
        "ads": _check_ads,
        "openaq": _check_openaq,
    }

    def run(item: tuple[str, Callable[[float], CredentialCheck]]) -> CredentialCheck:
        service, check = item
        try:
            return check(timeout)
        except requests.RequestException as exc:
            return CredentialCheck(service, ERROR, f"unreachable: {type(exc).__name__}")
        except Exception as exc:  # noqa: BLE001 - one broken check must not hide the others
            return CredentialCheck(service, ERROR, f"check failed: {type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=len(checks)) as executor:
        results = list(executor.map(run, checks.items()))
    return {result.service: result for result in results}


def _status(service: str, response: requests.Response, *, expires_at: datetime | None = None) -> CredentialCheck:
    if response.status_code != 200:
        return CredentialCheck(service, ERROR, f"rejected: HTTP {response.status_code}", _iso(expires_at))
    return _expiry(service, expires_at)


def _expiry(service: str, expires_at: datetime | None) -> CredentialCheck:
    if expires_at is None:
        return CredentialCheck(service, OK, "authenticated")
    days_left = (expires_at - datetime.now(timezone.utc)).total_seconds() / 86400
    if days_left <= 0:
        return CredentialCheck(service, ERROR, "expired", _iso(expires_at))
    if days_left < EXPIRY_WARNING_DAYS:
        return CredentialCheck(service, WARNING, f"authenticated; expires in {days_left:.1f} days", _iso(expires_at))
    return CredentialCheck(service, OK, f"authenticated; expires in {days_left:.0f} days", _iso(expires_at))


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _token_url() -> str:
    return os.getenv("CDSE_TOKEN_URL") or ClientConfig.token_url


def _check_cdse_client(timeout: float) -> CredentialCheck:
    client_id = os.getenv("CDSE_CLIENT_ID") or os.getenv("SH_CLIENT_ID")
    client_secret = os.getenv("CDSE_CLIENT_SECRET") or os.getenv("SH_CLIENT_SECRET")
    if not client_id or not client_secret:
        return CredentialCheck("cdse_oauth_client", NOT_CONFIGURED, "CDSE_CLIENT_ID/CDSE_CLIENT_SECRET not set")
    response = requests.post(
        _token_url(),
        data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret},
        timeout=timeout,
    )
    return _status("cdse_oauth_client", response)


def _check_cdse_account(timeout: float) -> CredentialCheck:
    username, password = os.getenv("CDSE_USERNAME"), os.getenv("CDSE_PASSWORD")
    if not username or not password:
        return CredentialCheck("cdse_account", NOT_CONFIGURED, "CDSE_USERNAME/CDSE_PASSWORD not set (downloads then use the OAuth client)")
    response = requests.post(
        _token_url(),
        data={"grant_type": "password", "client_id": "cdse-public", "username": username, "password": password},
        timeout=timeout,
    )
    return _status("cdse_account", response)


def _jwt_expiry(token: str) -> datetime | None:
    """Expiry claim of a JWT, read without verifying it (only the date is used)."""

    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return datetime.fromtimestamp(int(claims["exp"]), tz=timezone.utc)
    except (IndexError, KeyError, ValueError, TypeError):
        return None


def _check_earthdata(timeout: float) -> CredentialCheck:
    token = os.getenv("EARTHDATA_BEARER_TOKEN")
    if not token:
        return CredentialCheck("earthdata", NOT_CONFIGURED, "EARTHDATA_BEARER_TOKEN not set (needed for ECOSTRESS and hyp3_rtc)")
    response = requests.get(HYP3_USER_URL, headers={"Authorization": f"Bearer {token}"}, timeout=timeout)
    return _status("earthdata", response, expires_at=_jwt_expiry(token))


def _cdsapirc() -> tuple[str | None, str | None]:
    path = Path(os.getenv("CDSAPI_RC", Path.home() / ".cdsapirc"))
    if not path.is_file():
        return None, None
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        values[key.strip()] = value.strip()
    return values.get("url"), values.get("key")


def _check_cds_api(service: str, url: str | None, key: str | None, timeout: float) -> CredentialCheck:
    if not url or not key:
        return CredentialCheck(service, NOT_CONFIGURED, "API URL/key not set")
    response = requests.get(f"{url.rstrip('/')}/retrieve/v1/jobs", params={"limit": 1}, headers={"PRIVATE-TOKEN": key}, timeout=timeout)
    return _status(service, response)


def _check_cds(timeout: float) -> CredentialCheck:
    url = os.getenv("CDS_API_URL") or os.getenv("CDSAPI_URL")
    key = os.getenv("CDS_API_KEY") or os.getenv("CDSAPI_KEY")
    if not (url and key):
        url, key = _cdsapirc()
    return _check_cds_api("cds", url, key, timeout)


def _check_ads(timeout: float) -> CredentialCheck:
    return _check_cds_api("ads", os.getenv("CAMS_API_URL"), os.getenv("CAMS_API_KEY"), timeout)


def _check_openaq(timeout: float) -> CredentialCheck:
    key = os.getenv("OPENAQ_API_KEY")
    if not key:
        return CredentialCheck("openaq", NOT_CONFIGURED, "OPENAQ_API_KEY not set")
    base_url = os.getenv("OPENAQ_BASE_URL", "https://api.openaq.org/v3").rstrip("/")
    response = requests.get(f"{base_url}/locations", params={"limit": 1}, headers={"X-API-Key": key}, timeout=timeout)
    return _status("openaq", response)
