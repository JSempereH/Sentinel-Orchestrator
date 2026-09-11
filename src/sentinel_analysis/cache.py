"""Checksum-backed local asset cache."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass
class AssetCache:
    """Record downloaded assets and verify them before reuse."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self._lock = Lock()

    def _load(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _save(self, manifest: dict[str, Any]) -> None:
        temporary = self.manifest_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.manifest_path)

    @staticmethod
    def checksum(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def record(self, key: str, path: Path, *, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            manifest = self._load()
            manifest[key] = {"path": str(path), "sha256": self.checksum(path), "metadata": metadata or {}}
            self._save(manifest)

    def valid(self, key: str, path: Path) -> bool:
        with self._lock:
            entry = self._load().get(key)
        return bool(entry and path.exists() and entry.get("sha256") == self.checksum(path))
