"""Scores predicted rows against the labels embedded in sample_claims.csv."""
from __future__ import annotations

EXACT_MATCH_FIELDS = [
    "evidence_standard_met",
    "issue_type",
    "object_part",
    "claim_status",
    "valid_image",
    "severity",
]


def _as_set(semicolon_list: str) -> set[str]:
    if not semicolon_list or semicolon_list == "none":
        return set()
    return {x.strip() for x in semicolon_list.split(";") if x.strip()}


def score_claim(predicted: dict, expected: dict) -> dict:
    field_correct = {
        field: predicted.get(field) == expected.get(field) for field in EXACT_MATCH_FIELDS
    }

    pred_support = _as_set(predicted.get("supporting_image_ids", ""))
    exp_support = _as_set(expected.get("supporting_image_ids", ""))
    union = pred_support | exp_support
    support_iou = len(pred_support & exp_support) / len(union) if union else 1.0

    pred_risk = _as_set(predicted.get("risk_flags", ""))
    exp_risk = _as_set(expected.get("risk_flags", ""))
    risk_union = pred_risk | exp_risk
    risk_iou = len(pred_risk & exp_risk) / len(risk_union) if risk_union else 1.0

    return {**field_correct, "support_iou": support_iou, "risk_iou": risk_iou}


def aggregate(per_claim_scores: list[dict]) -> dict:
    if not per_claim_scores:
        return {}
    n = len(per_claim_scores)
    agg = {}
    for field in EXACT_MATCH_FIELDS:
        agg[f"{field}_accuracy"] = sum(s[field] for s in per_claim_scores) / n
    agg["supporting_image_ids_mean_iou"] = sum(s["support_iou"] for s in per_claim_scores) / n
    agg["risk_flags_mean_iou"] = sum(s["risk_iou"] for s in per_claim_scores) / n
    agg["n"] = n
    return agg
