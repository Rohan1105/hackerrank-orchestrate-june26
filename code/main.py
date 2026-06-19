#!/usr/bin/env python3
"""Entry point: run the claim-review pipeline over an input CSV and write output.csv.

Usage:
    python main.py --input ../dataset/claims.csv --dataset-root .. --output ../output.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pipeline import data_loader
from pipeline.runner import run_pipeline
from pipeline.schema import OUTPUT_COLUMNS

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-modal evidence review pipeline")
    parser.add_argument("--input", default=str(REPO_ROOT / "dataset" / "claims.csv"))
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "dataset"))
    parser.add_argument("--output", default=str(REPO_ROOT / "output.csv"))
    parser.add_argument(
        "--user-history", default=str(REPO_ROOT / "dataset" / "user_history.csv")
    )
    parser.add_argument(
        "--evidence-requirements",
        default=str(REPO_ROOT / "dataset" / "evidence_requirements.csv"),
    )
    parser.add_argument("--strategy", choices=["single", "two_stage"], default="single")
    parser.add_argument("--cache-dir", default=str(THIS_DIR / ".cache"))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    claims = data_loader.load_claims(Path(args.input))
    user_history = data_loader.load_user_history(Path(args.user_history))
    requirements = data_loader.load_evidence_requirements(Path(args.evidence_requirements))

    rows, stats = run_pipeline(
        claims,
        dataset_root=Path(args.dataset_root),
        user_history=user_history,
        requirements=requirements,
        strategy=args.strategy,
        cache_dir=None if args.no_cache else Path(args.cache_dir),
    )

    out_path = Path(args.output)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {len(rows)}/{len(claims)} rows to {out_path}\n"
        f"calls={stats.calls} cache_hits={stats.cache_hits} "
        f"images={stats.images_processed} input_tokens={stats.input_tokens} "
        f"output_tokens={stats.output_tokens} "
        f"total_latency={stats.total_latency_seconds:.1f}s"
    )
    if stats.stopped_early:
        print(
            f"\nSTOPPED EARLY after a model API error: {stats.stop_reason}\n"
            f"{len(stats.unprocessed_user_ids)} claim(s) not yet processed: "
            f"{', '.join(stats.unprocessed_user_ids)}\n"
            "Rerun this exact command with a fresh GEMINI_API_KEY (or once quota resets) "
            "to resume -- already-completed claims are cached and won't re-call the API."
        )


if __name__ == "__main__":
    main()
