"""Shared pipeline runner used by both code/main.py (production run on claims.csv) and
evaluation/main.py (scoring run on sample_claims.csv), so the two never drift apart.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from google.genai import errors as genai_errors

from . import data_loader, image_checks, postprocess
from .cache import DiskCache
from .prompt import PROMPT_VERSION
from .vision_client import DEFAULT_MODEL, VisionClient

STRATEGIES = ("single", "two_stage")

DEFAULT_NO_IMAGE_OUTPUT = {
    "evidence_standard_met": False,
    "evidence_standard_met_reason": "No usable images were available for this claim.",
    "risk_flags": ["damage_not_visible"],
    "issue_type": "unknown",
    "object_part": "unknown",
    "claim_status": "not_enough_information",
    "claim_status_justification": "No loadable images were submitted, so the claim "
    "could not be evaluated against any visual evidence.",
    "supporting_image_ids": [],
    "valid_image": False,
    "severity": "unknown",
}


@dataclass
class RunStats:
    calls: int = 0
    cache_hits: int = 0
    images_processed: int = 0
    short_circuited: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_seconds: float = 0.0
    rows: list[dict] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str = ""
    unprocessed_user_ids: list[str] = field(default_factory=list)


def _check_images_locally(
    image_paths: list[Path], image_cache: DiskCache | None
) -> tuple[list[Path], list[str]]:
    """Run the deterministic EXIF/authenticity check on each image (cached per image
    hash). Returns (loadable_paths, aggregated_risk_flags) -- unloadable images are
    dropped from what we send to the vision model but still contribute risk flags.
    """
    loadable_paths = []
    flags: set[str] = set()
    for path in image_paths:
        cached = image_cache.get(DiskCache.image_key(path)) if image_cache else None
        if cached is not None:
            inspection = image_checks.inspection_from_cache_dict(cached)
        else:
            inspection = image_checks.inspect_image(path)
            if image_cache:
                image_cache.set(
                    DiskCache.image_key(path), image_checks.inspection_to_cache_dict(inspection)
                )
        flags.update(inspection.risk_flags)
        if inspection.loadable:
            loadable_paths.append(path)
    return loadable_paths, sorted(flags)


def run_pipeline(
    claims: list[data_loader.Claim],
    dataset_root: Path,
    user_history: dict[str, dict],
    requirements: list[dict],
    strategy: str = "single",
    cache_dir: Path | None = None,
    client: VisionClient | None = None,
    verbose: bool = True,
) -> tuple[list[dict], RunStats]:
    assert strategy in STRATEGIES, f"unknown strategy: {strategy}"
    cache = DiskCache(cache_dir / "claims") if cache_dir else None
    image_cache = DiskCache(cache_dir / "images") if cache_dir else None
    stats = RunStats()
    rows = []
    client_holder = [client]  # lazily create only if a real call is needed

    def get_client() -> VisionClient:
        if client_holder[0] is None:
            client_holder[0] = VisionClient()
        return client_holder[0]

    for i, claim in enumerate(claims, start=1):
        all_image_paths = data_loader.resolve_image_paths(claim, dataset_root)
        loadable_paths, exif_risk_flags = _check_images_locally(all_image_paths, image_cache)
        stats.images_processed += len(all_image_paths)

        if not loadable_paths:
            # Obviously-bad case: no usable images at all. Skip the vision (and, for
            # two_stage, the triage) call entirely -- there is nothing for either to
            # look at, so spending tokens on it would be pure waste.
            stats.short_circuited += 1
            row = postprocess.build_output_row(
                claim, DEFAULT_NO_IMAGE_OUTPUT, user_history.get(claim.user_id), exif_risk_flags
            )
            rows.append(row)
            stats.rows.append(row)
            if verbose:
                print(
                    f"[{i}/{len(claims)}] {claim.user_id} -> short-circuited (no usable images)",
                    file=sys.stderr,
                )
            continue

        req_rows = data_loader.requirements_for(claim.claim_object, requirements)
        user_row = user_history.get(claim.user_id)

        cache_key = None
        cached = None
        if cache:
            model_name = client_holder[0].model if client_holder[0] else DEFAULT_MODEL
            cache_key = cache.make_key(
                loadable_paths,
                claim.user_claim,
                claim.claim_object,
                f"{strategy}|{model_name}|{PROMPT_VERSION}",
            )
            cached = cache.get(cache_key)

        if cached is not None:
            stats.cache_hits += 1
            model_output = cached
        else:
            try:
                client = get_client()
                triage_hint = None
                if strategy == "two_stage":
                    triage_result = client.triage_claim(claim)
                    stats.calls += 1
                    stats.input_tokens += triage_result.input_tokens
                    stats.output_tokens += triage_result.output_tokens
                    stats.total_latency_seconds += triage_result.latency_seconds
                    triage_hint = triage_result.data["hint"]

                result = client.review_claim(
                    claim, loadable_paths, user_row, req_rows, triage_hint=triage_hint
                )
                stats.calls += 1
                stats.input_tokens += result.input_tokens
                stats.output_tokens += result.output_tokens
                stats.total_latency_seconds += result.latency_seconds
                model_output = result.data
                if cache:
                    cache.set(cache_key, model_output)
            except genai_errors.APIError as e:
                # Most likely a quota wall (429) that retries won't clear today. Stop
                # here rather than crash -- everything processed so far (and cached)
                # is preserved, and main.py can resume later with a fresh key/quota:
                # already-cached claims are skipped automatically next run.
                stats.stopped_early = True
                stats.stop_reason = str(e)
                stats.unprocessed_user_ids = [c.user_id for c in claims[i - 1 :]]
                if verbose:
                    print(
                        f"[{i}/{len(claims)}] STOPPED EARLY: {e}\n"
                        f"{len(stats.unprocessed_user_ids)} claim(s) remain unprocessed; "
                        "rerun with a fresh GEMINI_API_KEY to resume (cache preserves progress).",
                        file=sys.stderr,
                    )
                break

        row = postprocess.build_output_row(claim, model_output, user_row, exif_risk_flags)
        rows.append(row)
        stats.rows.append(row)

        if verbose:
            print(f"[{i}/{len(claims)}] {claim.user_id} -> {row['claim_status']}", file=sys.stderr)

    return rows, stats
