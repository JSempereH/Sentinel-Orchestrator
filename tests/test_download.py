from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import requests

from sentinel_analysis.catalog import ProductRef
from sentinel_analysis.config import ClientConfig
from sentinel_analysis.download import CDSEDownloader


class _Response:
    status_code = 200

    def __init__(self, payload=None, chunks=()):
        self.payload = payload
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload

    def iter_content(self, chunk_size):
        yield from self.chunks


class _Session:
    def post(self, *args, **kwargs):
        return _Response({"access_token": "test-token"})

    def get(self, *args, **kwargs):
        return _Response(chunks=(b"netcdf",))


def test_cdse_downloader_preserves_netcdf_product_extension(tmp_path: Path):
    config = ClientConfig(client_id="id", client_secret="secret")
    product = ProductRef(
        product_id="product-id",
        name="S5P_OFFL_L2__NO2____20250619.nc",
        product_type="L2__NO2___",
        start_datetime=None,
        end_datetime=None,
        timeliness=None,
        online=True,
        download_url="https://example.test/product",
        metadata={},
    )

    path = CDSEDownloader(config, session=cast(Any, _Session())).download(product, tmp_path)

    assert path.name == product.name
    assert path.read_bytes() == b"netcdf"


class _ResumeResponse(_Response):
    def __init__(self, *, fail: bool):
        super().__init__()
        self.fail = fail
        self.status_code = 200 if fail else 206

    def iter_content(self, chunk_size):
        if self.fail:
            yield b"net"
            raise requests.ConnectionError("simulated interrupted download")
        yield b"cdf"


class _ResumeSession(_Session):
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(kwargs)
        return _ResumeResponse(fail=len(self.calls) == 1)


def test_cdse_downloader_resumes_partial_download(tmp_path: Path):
    session = _ResumeSession()
    config = ClientConfig(client_id="id", client_secret="secret")
    product = ProductRef(
        product_id="product-id",
        name="product.nc",
        product_type="test",
        start_datetime=None,
        end_datetime=None,
        timeliness=None,
        online=True,
        download_url="https://example.test/product",
        metadata={},
    )

    path = CDSEDownloader(config, session=cast(Any, session)).download(product, tmp_path)

    assert path.read_bytes() == b"netcdf"
    assert session.calls[1]["headers"]["Range"] == "bytes=3-"
