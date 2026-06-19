"""Loads claims/user_history/evidence_requirements CSVs and resolves image paths
relative to the dataset root, so the pipeline never hardcodes a single CSV's layout.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Claim:
    user_id: str
    image_paths: list[str]          # raw paths as given in the CSV, semicolon-split
    user_claim: str
    claim_object: str
    expected: dict = field(default_factory=dict)  # present only for sample_claims.csv

    @property
    def image_ids(self) -> list[str]:
        return [Path(p).stem for p in self.image_paths]


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_claims(csv_path: Path) -> list[Claim]:
    rows = _read_csv(csv_path)
    claims = []
    input_cols = {"user_id", "image_paths", "user_claim", "claim_object"}
    for row in rows:
        expected = {k: v for k, v in row.items() if k not in input_cols}
        claims.append(
            Claim(
                user_id=row["user_id"],
                image_paths=[p.strip() for p in row["image_paths"].split(";") if p.strip()],
                user_claim=row["user_claim"],
                claim_object=row["claim_object"],
                expected=expected,
            )
        )
    return claims


def load_user_history(csv_path: Path) -> dict[str, dict]:
    return {row["user_id"]: row for row in _read_csv(csv_path)}


def load_evidence_requirements(csv_path: Path) -> list[dict]:
    return _read_csv(csv_path)


def requirements_for(claim_object: str, requirements: list[dict]) -> list[dict]:
    return [r for r in requirements if r["claim_object"] in (claim_object, "all")]


def resolve_image_paths(claim: Claim, dataset_root: Path) -> list[Path]:
    """Resolve image paths, skipping (not raising on) ones that don't exist.

    A missing file is itself evidence the image set is unusable -- the caller (see
    pipeline/runner.py) treats an empty result as grounds to short-circuit straight to
    "not_enough_information" instead of crashing the whole run on one bad row.
    """
    resolved = []
    for rel in claim.image_paths:
        candidate = dataset_root / rel
        if candidate.exists():
            resolved.append(candidate)
    return resolved
