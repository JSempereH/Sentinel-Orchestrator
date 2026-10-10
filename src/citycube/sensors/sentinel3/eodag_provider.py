"""Optional EODAG adapter for CDSE product discovery and downloads."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ...config import AOI


class EODAGProvider:
    """Thin adapter around EODAG's ``S3_SLSTR_L2LST`` collection.

    EODAG is optional so users who only need openEO or direct OData access do
    not pay its dependency cost. Install the package with ``[cdse]``.
    """

    collection = "S3_SLSTR_L2LST"

    def __init__(self, *, provider: str = "cop_dataspace", config_path: str | None = None):
        try:
            EODataAccessGateway = import_module("eodag").EODataAccessGateway
        except ImportError as exc:
            raise RuntimeError("Install citycube[cdse] to use EODAGProvider") from exc
        kwargs: dict[str, Any] = {}
        if config_path:
            kwargs["config_path"] = config_path
        self.gateway = EODataAccessGateway(**kwargs)
        self.gateway.set_preferred_provider(provider)

    def search(
        self,
        aoi: AOI,
        start: str,
        end: str,
        *,
        timeliness: str | None = "NTC",
    ):
        """Search the EODAG provider using the exact L2 LST collection."""

        query: dict[str, Any] = {
            "collection": self.collection,
            "start": start,
            "end": end,
            "geom": aoi.as_geojson(),
        }
        if timeliness:
            query["product:timeliness"] = "NT" if timeliness.upper() == "NTC" else "NR"
        return self.gateway.search(**query)

    def download(self, results, output_dir: str):
        """Download EODAG search results into a local directory."""

        return self.gateway.download_all(results, outputs_prefix=output_dir)
