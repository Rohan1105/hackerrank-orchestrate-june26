"""Prompt construction and the forced-tool-call JSON schema for the vision review call.

We force the model to call a single tool with a strict JSON schema (rather than asking
for free-form JSON) so parsing is deterministic and never needs a JSON-repair step.
"""
from __future__ import annotations

from . import schema

# Bump whenever SYSTEM_PROMPT/TRIAGE_SYSTEM_PROMPT/build_tool_schema materially change --
# included in the cache key (see pipeline/runner.py) so a prompt edit can't silently
# serve stale cached model outputs.
#
# v2-fewshot (text-only calibration examples, no images) was tried and measured against
# v1 on a 5-claim held-out probe from sample_claims.csv: it never beat v1 on issue_type
# or severity accuracy, and regressed one previously-correct case (a dent misread as
# broken_part) -- likely the high-severity example anchored the model toward "high" more
# than the accompanying "use high sparingly" instruction pulled it back. Reverted to v1.
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a claims evidence reviewer for an insurance-style damage claim \
system covering three object types: car, laptop, package.

The submitted images are the primary source of truth. The conversation tells you what to \
check. User history adds risk context only -- it must never override clear visual evidence \
by itself.

For the single claim you are given:
1. Read the conversation and extract the actual damage claim being made (which part, what \
   kind of issue).
2. Inspect every submitted image.
3. Decide if the image evidence is sufficient to evaluate the claim, per the evidence \
   requirements provided.
4. Identify the visible issue type and the relevant object part.
5. Decide claim_status: "supported" if the images confirm the claimed issue, "contradicted" \
   if the images show the claimed area is fine / inconsistent with the claim, or \
   "not_enough_information" if the images cannot confirm or deny it.
6. Select which image IDs actually support your decision (use "none" if none do).
7. Flag any image quality, mismatch, authenticity, or user-history risk concerns.
8. Estimate severity.
9. Justify your decision concisely, grounded in what is visible in the images, citing image \
   IDs where helpful.

Use only the allowed values given to you for each field. If you are unsure, prefer "unknown" \
over guessing for issue_type/object_part/severity, and use "not_enough_information" for \
claim_status when the images genuinely do not resolve the question."""


def build_tool_schema(claim_object: str) -> dict:
    return {
        "name": "submit_claim_review",
        "description": "Submit the structured review decision for this claim.",
        "input_schema": {
            "type": "object",
            "properties": {
                "evidence_standard_met": {"type": "boolean"},
                "evidence_standard_met_reason": {"type": "string"},
                "risk_flags": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(schema.RISK_FLAGS)},
                },
                "issue_type": {"type": "string", "enum": sorted(schema.ISSUE_TYPE)},
                "object_part": {
                    "type": "string",
                    "enum": sorted(schema.OBJECT_PART[claim_object]),
                },
                "claim_status": {"type": "string", "enum": sorted(schema.CLAIM_STATUS)},
                "claim_status_justification": {"type": "string"},
                "supporting_image_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Image IDs (filename without extension), or [] if none.",
                },
                "valid_image": {"type": "boolean"},
                "severity": {"type": "string", "enum": sorted(schema.SEVERITY)},
            },
            "required": [
                "evidence_standard_met",
                "evidence_standard_met_reason",
                "risk_flags",
                "issue_type",
                "object_part",
                "claim_status",
                "claim_status_justification",
                "supporting_image_ids",
                "valid_image",
                "severity",
            ],
        },
    }


def build_user_text(
    claim,
    user_history_row: dict | None,
    requirements: list[dict],
    triage_hint: str | None = None,
) -> str:
    req_lines = "\n".join(
        f"- [{r['applies_to']}] {r['minimum_image_evidence']}" for r in requirements
    )
    history_lines = (
        "\n".join(f"- {k}: {v}" for k, v in user_history_row.items() if k != "user_id")
        if user_history_row
        else "- no history on file for this user"
    )
    image_id_list = ", ".join(claim.image_ids)

    parts = [
        f"claim_object: {claim.claim_object}",
        f"image IDs in submission order: {image_id_list}",
        "",
        "Conversation:",
        claim.user_claim,
        "",
        "Minimum evidence requirements for this object type:",
        req_lines,
        "",
        "User history (risk context only -- do not let this override visual evidence):",
        history_lines,
    ]
    if triage_hint:
        parts += ["", f"Triage hint from a prior text-only pass: {triage_hint}"]
    parts += [
        "",
        "Call submit_claim_review with your decision. object_part must be one of the "
        f"values valid for claim_object={claim.claim_object}.",
    ]
    return "\n".join(parts)


TRIAGE_SYSTEM_PROMPT = """You read only the conversation transcript of a damage claim (no \
images yet). Extract a short hint of what to check: which object part is claimed, what kind \
of issue is claimed, and anything ambiguous worth flagging to a reviewer who will look at the \
photos next. One or two sentences, no more."""


def build_triage_user_text(claim) -> str:
    return f"claim_object: {claim.claim_object}\n\nConversation:\n{claim.user_claim}"
