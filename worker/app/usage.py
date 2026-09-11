"""Best-effort quota/credentials snapshot for the auxiliary providers this
worker calls on behalf of sentinel_analysis.

Unlike ASF's HyP3 (a numeric processing-credit balance), CDSE, CDS/ADS and
OpenAQ are not credit-balance systems:
- CDSE (Sentinel-1/2/3/5P catalog + downloads): OAuth client-credentials
  quota, no exposed numeric balance.
- CDS/ADS (ERA5, CAMS): request-queue based, no exposed numeric balance.
- OpenAQ v3: a real per-key rate limit, reported via response headers
  (X-RateLimit-Limit / -Remaining / -Reset) - this is the one provider here
  that can actually answer "how much do I have left, and when does it reset".

Each check is independent and never raises: a network hiccup on one provider
must not blank out the others.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

_TIMEOUT_S = 8


def _check_openaq() -> dict[str, Any]:
    from sentinel_analysis.providers import OpenAQConfig

    config = OpenAQConfig.from_env()
    if not config.api_key:
        return {"configured": False, "note": "OPENAQ_API_KEY not set"}
    try:
        response = requests.get(
            f"{config.base_url}/parameters",
            params={"limit": 1},
            headers={"X-API-Key": config.api_key},
            timeout=_TIMEOUT_S,
        )
        headers = response.headers
        return {
            "configured": True,
            "reachable": True,
            "limit": _as_int(headers.get("x-ratelimit-limit")),
            "remaining": _as_int(headers.get("x-ratelimit-remaining")),
            "reset_seconds": _as_int(headers.get("x-ratelimit-reset")),
        }
    except requests.RequestException as exc:
        logger.warning("OpenAQ usage check failed: %s", exc)
        return {"configured": True, "reachable": False, "error": str(exc)}


def _check_cdse() -> dict[str, Any]:
    from sentinel_analysis.config import ClientConfig

    try:
        ClientConfig.from_env()
        configured = True
    except Exception:
        configured = False
    return {
        "configured": configured,
        "note": "OAuth client-credentials quota; CDSE exposes no numeric remaining-balance API.",
    }


def _check_cds_ads() -> dict[str, Any]:
    from sentinel_analysis.providers import CAMSConfig, ERA5Config

    era5 = ERA5Config.from_env()
    cams = CAMSConfig.from_env()
    return {
        "era5": {"configured": bool(era5.api_key)},
        "cams": {"configured": bool(cams.api_key)},
        "note": "Request-queue based (CDS/ADS); no numeric remaining-balance API.",
    }


def _as_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def check_all() -> dict[str, Any]:
    return {
        "openaq": _check_openaq(),
        "cdse": _check_cdse(),
        "cds_ads": _check_cds_ads(),
    }
