"""Turns a raw model tool-call payload into a final output row: clamps every field to
the allowed vocabulary, folds in rule-based user-history risk, and formats the
semicolon-separated list fields the output CSV expects.
"""
from __future__ import annotations

from . import schema

# Users with this many recent claims or any rejected claim get a history risk flag.
# These are simple, auditable thresholds rather than a learned risk score, since the
# task explicitly says history should add context, not drive the decision.
RECENT_CLAIM_RISK_THRESHOLD = 3
HISTORY_FLAGS_NONE = {"", "none", "None", None}


def _clamp(value: str, allowed: set[str], default: str) -> str:
    return value if value in allowed else default


def history_risk_flags(user_history_row: dict | None) -> list[str]:
    if not user_history_row:
        return []
    flags = []
    try:
        rejected = int(user_history_row.get("rejected_claim") or 0)
        recent = int(user_history_row.get("last_90_days_claim_count") or 0)
    except ValueError:
        rejected, recent = 0, 0
    if rejected > 0 or recent >= RECENT_CLAIM_RISK_THRESHOLD:
        flags.append("user_history_risk")
    if user_history_row.get("history_flags") not in HISTORY_FLAGS_NONE:
        flags.append("user_history_risk")
    return flags


def build_output_row(
    claim,
    model_output: dict,
    user_history_row: dict | None,
    extra_risk_flags: list[str] | None = None,
) -> dict:
    claim_object = claim.claim_object if claim.claim_object in schema.CLAIM_OBJECTS else "car"
    object_part_allowed = schema.OBJECT_PART[claim_object]

    issue_type = _clamp(model_output.get("issue_type", "unknown"), schema.ISSUE_TYPE, "unknown")
    object_part = _clamp(model_output.get("object_part", "unknown"), object_part_allowed, "unknown")
    claim_status = _clamp(
        model_output.get("claim_status", "not_enough_information"),
        schema.CLAIM_STATUS,
        "not_enough_information",
    )
    severity = _clamp(model_output.get("severity", "unknown"), schema.SEVERITY, "unknown")

    raw_flags = model_output.get("risk_flags") or []
    flags = {f for f in raw_flags if f in schema.RISK_FLAGS and f != "none"}
    flags.update(history_risk_flags(user_history_row))
    flags.update(f for f in (extra_risk_flags or []) if f in schema.RISK_FLAGS and f != "none")
    risk_flags = ";".join(sorted(flags)) if flags else "none"

    supporting_ids = model_output.get("supporting_image_ids") or []
    valid_ids = set(claim.image_ids)
    supporting_ids = [i for i in supporting_ids if i in valid_ids]
    supporting_image_ids = ";".join(supporting_ids) if supporting_ids else "none"

    return {
        "user_id": claim.user_id,
        "image_paths": ";".join(claim.image_paths),
        "user_claim": claim.user_claim,
        "claim_object": claim.claim_object,
        "evidence_standard_met": str(bool(model_output.get("evidence_standard_met", False))).lower(),
        "evidence_standard_met_reason": model_output.get("evidence_standard_met_reason", ""),
        "risk_flags": risk_flags,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": model_output.get("claim_status_justification", ""),
        "supporting_image_ids": supporting_image_ids,
        "valid_image": str(bool(model_output.get("valid_image", False))).lower(),
        "severity": severity,
    }
