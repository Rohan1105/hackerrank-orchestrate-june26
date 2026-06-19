"""Disk cache keyed by a hash of the claim's actual content (images + text + strategy).

Lets us rerun the pipeline during development -- or resume after a partial failure --
without re-spending vision tokens on rows we've already scored.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class DiskCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(image_paths: list[Path], user_claim: str, claim_object: str, strategy: str) -> str:
        h = hashlib.sha256()
        h.update(strategy.encode())
        h.update(claim_object.encode())
        h.update(user_claim.encode())
        for p in image_paths:
            h.update(p.read_bytes())
        return h.hexdigest()

    @staticmethod
    def image_key(path: Path) -> str:
        """Content hash of a single image's bytes.

        Used for the image-level cache (e.g. EXIF/authenticity checks): if the same
        image file is reused across multiple claim rows, we inspect it once instead of
        once per claim that references it.
        """
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def get(self, key: str) -> dict | None:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def set(self, key: str, value: dict) -> None:
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
