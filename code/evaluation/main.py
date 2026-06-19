#!/usr/bin/env python3
"""Evaluate the pipeline on dataset/sample_claims.csv, comparing the "single" and
"two_stage" strategies, and write evaluation/evaluation_report.md with accuracy
metrics plus an operational-analysis projection onto the full claims.csv.

Usage:
    python evaluation/main.py
(run from code/, or anywhere -- paths are resolved relative to this file)
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pipeline import data_loader  # noqa: E402
from pipeline.runner import run_pipeline  # noqa: E402
from pipeline.vision_client import VisionClient  # noqa: E402
from evaluation import metrics  # noqa: E402

# Per-million-token pricing assumption for the operational-analysis section, based on
# Gemini 2.5 Flash pay-as-you-go pricing. Update if you change GEMINI_MODEL, and note
# that on the free tier the actual dollar cost is $0 (you're rate-limited, not billed).
ASSUMED_INPUT_PRICE_PER_MTOK = 0.30
ASSUMED_OUTPUT_PRICE_PER_MTOK = 2.50


def evaluate_strategy(strategy: str, claims, dataset_root, user_history, requirements, client):
    rows, stats = run_pipeline(
        claims,
        dataset_root=dataset_root,
        user_history=user_history,
        requirements=requirements,
        strategy=strategy,
        cache_dir=CODE_DIR / ".cache",
        client=client,
        verbose=True,
    )
    per_claim = [metrics.score_claim(row, claim.expected) for row, claim in zip(rows, claims)]
    agg = metrics.aggregate(per_claim)
    return rows, stats, agg


def cost_estimate(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * ASSUMED_INPUT_PRICE_PER_MTOK
        + output_tokens / 1_000_000 * ASSUMED_OUTPUT_PRICE_PER_MTOK
    )


def render_report(sample_n: int, test_n: int, results: dict, final_strategy: str) -> str:
    lines = ["# Evaluation Report", ""]
    lines += [
        "## Strategies compared on `dataset/sample_claims.csv`",
        "",
        "- **single**: one vision call per claim, given images + conversation + evidence "
        "requirements + user history, forced to return structured JSON via a tool call.",
        "- **two_stage**: a cheap text-only triage call extracts the claimed part/issue from "
        "the conversation first, then a vision call (with that hint attached) does the "
        "actual evidence review.",
        "",
        "| Metric | single | two_stage |",
        "|---|---|---|",
    ]
    single_agg, two_stage_agg = results["single"]["agg"], results["two_stage"]["agg"]
    metric_keys = [k for k in single_agg if k != "n"]
    for k in metric_keys:
        lines.append(f"| {k} | {single_agg.get(k, 0):.3f} | {two_stage_agg.get(k, 0):.3f} |")
    lines += ["", f"Evaluated on n={sample_n} labeled sample claims.", ""]

    lines += [
        f"## Final strategy used for `output.csv`: **{final_strategy}**",
        "",
        f"Chosen because it had the higher claim_status_accuracy / better cost-accuracy "
        f"trade-off on the sample set (see table above).",
        "",
        "## Operational analysis",
        "",
    ]
    for name in ("single", "two_stage"):
        stats = results[name]["stats"]
        n = results[name]["n"]
        per_claim_calls = stats.calls / n if n else 0
        per_claim_in = stats.input_tokens / n if n else 0
        per_claim_out = stats.output_tokens / n if n else 0
        projected_calls = round(per_claim_calls * test_n)
        projected_in = per_claim_in * test_n
        projected_out = per_claim_out * test_n
        projected_cost = cost_estimate(projected_in, projected_out)
        projected_latency_min = (stats.total_latency_seconds / n * test_n) / 60 if n else 0

        lines += [
            f"### Strategy: {name}",
            "",
            f"- Sample run: {stats.calls} model calls, {stats.images_processed} images, "
            f"{stats.input_tokens} input tokens, {stats.output_tokens} output tokens, "
            f"{stats.total_latency_seconds:.1f}s total latency, "
            f"{stats.cache_hits} cache hits, over {n} claims.",
            f"- Projected to the full test set (`dataset/claims.csv`, {test_n} rows): "
            f"~{projected_calls} model calls, ~{projected_in:,.0f} input tokens, "
            f"~{projected_out:,.0f} output tokens, ~{projected_latency_min:.1f} minutes "
            f"sequential runtime.",
            f"- Estimated cost at ${ASSUMED_INPUT_PRICE_PER_MTOK}/MTok in + "
            f"${ASSUMED_OUTPUT_PRICE_PER_MTOK}/MTok out (Gemini 2.5 Flash pay-as-you-go "
            f"pricing assumption; $0 on the free tier, subject to RPM/TPM/RPD limits "
            f"instead): **${projected_cost:.4f}**.",
            "",
        ]

    lines += [
        "## Rate limits, batching, caching, retries",
        "",
        "- Each claim is processed independently (1 vision call for `single`, 2 calls for "
        "`two_stage`); no cross-claim batching is required at this volume (tens of rows).",
        "- `pipeline/cache.py` keys on a hash of the actual image bytes + claim text + "
        "strategy, so reruns (e.g. after a crash, or while iterating on prompts) skip "
        "claims already scored instead of re-spending tokens.",
        "- `pipeline/vision_client.py` retries `RateLimitError` and 5xx/529 (overloaded) "
        "responses with exponential backoff, up to 5 attempts, so transient TPM/RPM "
        "throttling does not fail a run.",
        "- At this dataset size (tens of claims, a few images each) we stay well under "
        "typical per-minute token/request limits for a single API key; for a much larger "
        "test set, the same runner could be parallelized across a small worker pool with a "
        "shared token-bucket limiter without changing the per-claim logic.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    dataset_root = REPO_ROOT / "dataset"
    sample_claims = data_loader.load_claims(dataset_root / "sample_claims.csv")
    test_claims = data_loader.load_claims(dataset_root / "claims.csv")
    user_history = data_loader.load_user_history(dataset_root / "user_history.csv")
    requirements = data_loader.load_evidence_requirements(dataset_root / "evidence_requirements.csv")

    client = VisionClient()
    results = {}
    for strategy in ("single", "two_stage"):
        rows, stats, agg = evaluate_strategy(
            strategy, sample_claims, dataset_root, user_history, requirements, client
        )
        results[strategy] = {"rows": rows, "stats": stats, "agg": agg, "n": len(sample_claims)}

    final_strategy = "single"
    if results["two_stage"]["agg"].get("claim_status_accuracy", 0) > results["single"]["agg"].get(
        "claim_status_accuracy", 0
    ):
        final_strategy = "two_stage"

    report = render_report(
        sample_n=len(sample_claims),
        test_n=len(test_claims),
        results=results,
        final_strategy=final_strategy,
    )
    report_path = THIS_DIR / "evaluation_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Recommended final strategy: {final_strategy}")


if __name__ == "__main__":
    main()
