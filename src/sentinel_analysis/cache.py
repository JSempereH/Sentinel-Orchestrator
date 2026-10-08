"""Checksum-backed local asset cache."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from threading import Lock
from typing import Any


_MANIFEST_LOCKS: dict[Path, Lock] = {}
_LOCKS_GUARD = Lock()


@dataclass
class AssetCache:
    """Record downloaded assets and verify them before reuse.

    Each entry stores the file's SHA-256 together with its size and
    modification time. ``valid()`` trusts an unchanged size+mtime and only
    re-hashes when either differs (or when ``verify=True``) - hashing a
    multi-GB Sentinel-1/2 archive on every cache lookup is far more expensive
    than the corruption it guards against, which always changes the stat.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        # Shared by every AssetCache on the same manifest: parallel product
        # workers each create their own instance, and per-instance locks let
        # their read-modify-write cycles interleave and drop entries.
        with _LOCKS_GUARD:
            self._lock = _MANIFEST_LOCKS.setdefault(self.manifest_path.resolve(), Lock())

    def _load(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _save(self, manifest: dict[str, Any]) -> None:
        # A unique temporary name per write, so concurrent writers (other
        # processes included) never interleave inside one shared file.
        handle, name = tempfile.mkstemp(dir=self.root, prefix="manifest.", suffix=".part")
        with os.fdopen(handle, "w", encoding="utf-8") as temporary:
            temporary.write(json.dumps(manifest, indent=2, sort_keys=True))
        Path(name).replace(self.manifest_path)

    @staticmethod
    def checksum(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _stat(path: Path) -> dict[str, int]:
        stat = path.stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    @staticmethod
    def _unchanged(entry: dict[str, Any], path: Path, stat: dict[str, int]) -> bool:
        return entry.get("path") == str(path) and entry.get("size") == stat["size"] and entry.get("mtime_ns") == stat["mtime_ns"]

    def record(self, key: str, path: Path, *, metadata: dict[str, Any] | None = None) -> str:
        """Record ``path`` under ``key`` and return its SHA-256."""

        path = Path(path)
        stat = self._stat(path)
        with self._lock:
            entry = self._load().get(key)
        digest = entry["sha256"] if entry and self._unchanged(entry, path, stat) else self.checksum(path)
        with self._lock:
            manifest = self._load()
            manifest[key] = {"path": str(path), "sha256": digest, **stat, "metadata": metadata or {}}
            self._save(manifest)
        return digest

    def valid(self, key: str, path: Path, *, verify: bool = False) -> bool:
        """Return whether ``path`` still matches the asset recorded under ``key``."""

        path = Path(path)
        with self._lock:
            entry = self._load().get(key)
        if not entry or not path.exists():
            return False
        stat = self._stat(path)
        if not verify and self._unchanged(entry, path, stat):
            return True
        if entry.get("sha256") != self.checksum(path):
            return False
        # Upgrade entries written before size/mtime were stored, so the next
        # lookup takes the fast path.
        with self._lock:
            manifest = self._load()
            if key in manifest:
                manifest[key].update(stat)
                self._save(manifest)
        return True
