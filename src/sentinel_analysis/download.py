"""Authenticated, provider-neutral CDSE product downloads."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from threading import Lock
import time
import zipfile
import re

import requests

from .catalog import ProductRef
from .config import ClientConfig
from .http import http_session
from .cache import AssetCache


class CDSEDownloader:
    """Authenticated, resumable-by-cache downloader for any CDSE product."""

    def __init__(self, config: ClientConfig, *, session: requests.Session | None = None, cache: AssetCache | None = None):
        self.config = config
        self.session = session or http_session()
        self._token: str | None = None
        self._token_lock = Lock()
        self.cache = cache

    def _access_token(self) -> str:
        with self._token_lock:
            if self._token:
                return self._token
            response = self.session.post(
            self.config.token_url,
            data=(
                {
                    "grant_type": "password",
                    "client_id": "cdse-public",
                    "username": self.config.username,
                    "password": self.config.password,
                }
                if self.config.username and self.config.password
                else {
                    "grant_type": "client_credentials",
                    "client_id": self.config.client_id,
                    "client_secret": self.config.client_secret,
                }
            ),
            timeout=(self.config.cdse_connect_timeout_s, self.config.cdse_read_timeout_s),
            )
            response.raise_for_status()
            token = response.json().get("access_token")
            if not token:
                raise RuntimeError("CDSE token response did not contain access_token")
            self._token = str(token)
            return self._token

    def download(self, product: ProductRef, output: str | Path) -> Path:
        """Download one product atomically and return its local path."""

        destination = Path(output)
        if destination.suffix.lower() not in {".zip", ".sen3", ".nc", ".h5", ".hdf"}:
            destination.mkdir(parents=True, exist_ok=True)
            product_suffix = Path(product.name).suffix.lower()
            filename = product.name if product_suffix in {".nc", ".h5", ".hdf"} else f"{product.name}.zip"
            destination = destination / filename
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
        return self._fetch(
            product.download_url, destination, product.product_id,
            metadata={"name": product.name, "product_type": product.product_type},
        )

    def download_files(self, product: ProductRef, names: Iterable[str], output_dir: str | Path) -> Path:
        """Download only some files of a product, laid out as its SAFE directory.

        Uses CDSE's OData ``Nodes`` endpoint
        (``Products(<id>)/Nodes(<product>)/Nodes(<file>)/$value``), so e.g.
        the ~16 MB an SLSTR LST analysis reads are fetched instead of the
        ~70 MB archive. Returns ``output_dir/<product name>``, which every
        SAFE reader here accepts like an extracted archive.
        """

        root = Path(output_dir) / product.name
        root.mkdir(parents=True, exist_ok=True)
        base = product.download_url.rsplit("/$value", 1)[0]
        for name in names:
            url = f"{base}/Nodes({product.name})/Nodes({name})/$value"
            self._fetch(url, root / name, f"{product.product_id}/{name}", metadata={"name": product.name, "file": name})
        return root

    def _fetch(self, url: str, destination: Path, cache_key: str, *, metadata: dict) -> Path:
        """Resumable, retried, token-refreshing transfer of one URL."""

        if destination.exists() and destination.stat().st_size > 0 and (self.cache is None or self.cache.valid(cache_key, destination)):
            return destination

        partial = destination.with_suffix(destination.suffix + ".part")
        retries = max(1, self.config.cdse_download_retries)
        timeout = (self.config.cdse_connect_timeout_s, self.config.cdse_read_timeout_s)
        for attempt in range(retries):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Authorization": f"Bearer {self._access_token()}"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with self.session.get(url, headers=headers, stream=True, timeout=timeout) as response:
                    if response.status_code == 401:
                        self._token = None
                        headers["Authorization"] = f"Bearer {self._access_token()}"
                        with self.session.get(url, headers=headers, stream=True, timeout=timeout) as refreshed:
                            response = refreshed
                            response.raise_for_status()
                            mode = "ab" if offset and response.status_code == 206 else "wb"
                            self._append_response(partial, response, mode, offset)
                        break
                    if response.status_code == 416:
                        partial.unlink(missing_ok=True)
                        continue
                    response.raise_for_status()
                    mode = "ab" if offset and response.status_code == 206 else "wb"
                    self._append_response(partial, response, mode, offset)
                break
            except (requests.RequestException, OSError):
                if attempt == retries - 1:
                    raise
                time.sleep(min(60, 2**attempt))
        partial.replace(destination)
        if self.cache:
            self.cache.record(cache_key, destination, metadata=metadata)
        return destination

    @staticmethod
    def _append_response(partial: Path, response: requests.Response, mode: str, offset: int) -> None:
        """Write a response and reject a silently truncated transfer."""

        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        expected_total: int | None = None
        headers = getattr(response, "headers", {})
        content_range = headers.get("Content-Range", "")
        match = re.search(r"bytes \d+-\d+/(\d+)", content_range)
        if match:
            expected_total = int(match.group(1))
        elif headers.get("Content-Length"):
            expected_total = offset + int(headers["Content-Length"]) if mode == "ab" else int(headers["Content-Length"])
        if expected_total is not None and partial.stat().st_size < expected_total:
            raise requests.ConnectionError(
                f"CDSE transfer ended at {partial.stat().st_size} of {expected_total} bytes"
            )

    @staticmethod
    def extract(archive: str | Path, output_dir: str | Path) -> Path:
        """Safely extract a CDSE archive and return its product root when unique."""

        archive = Path(archive)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        root = output_dir.resolve()
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                target = (output_dir / member.filename).resolve()
                if target != root and root not in target.parents:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            zipped.extractall(output_dir)
        products = [*output_dir.glob("*.SAFE"), *output_dir.glob("*.SEN3")]
        return products[0] if len(products) == 1 else output_dir
