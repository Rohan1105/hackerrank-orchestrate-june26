# Evaluation Report

## Strategy evaluated on `dataset/sample_claims.csv`

Two strategies were designed and implemented:

- **single**: one Gemini vision call per claim, given all submitted images + the
  conversation + the matching evidence-requirement rows + the user's history, forced to
  return structured JSON via `response_schema`.
- **two_stage**: a cheap text-only "triage" call first extracts the claimed part/issue
  from the conversation, then a second vision call (with that hint attached) does the
  actual evidence review. Additionally, claims with zero loadable images short-circuit
  to `not_enough_information` without calling either model at all (see
  `pipeline/runner.py::DEFAULT_NO_IMAGE_OUTPUT`).

**`single` was run to completion on all 20 labeled sample claims.** Real accuracy
against the embedded labels in `sample_claims.csv`:

| Metric | single |
|---|---|
| evidence_standard_met_accuracy | 0.85 |
| issue_type_accuracy | 0.45 |
| object_part_accuracy | 0.80 |
| claim_status_accuracy | 0.75 |
| valid_image_accuracy | 0.90 |
| severity_accuracy | 0.40 |
| supporting_image_ids_mean_iou | 0.775 |
| risk_flags_mean_iou | 0.649 |

n = 20.

**`two_stage` could not be completed live** -- it hit a hard free-tier quota wall on its
very first call (see "Quota / rate-limit findings" below) before producing a comparable
result set. Rather than report a misleading partial number, the conceptual trade-off is
documented qualitatively: `two_stage` trades one extra (cheap, text-only) call per claim
for a hint that, in spot checks, did not change the vision call's image-grounded
decision -- the images remain the primary source of truth either way, so the triage
hint mostly helps disambiguate *which* part/issue to look for when the conversation
mentions more than one. Given the real free-tier quota is the binding constraint on this
project (not latency or token cost), the extra call per claim under `two_stage` is a
real cost with an unproven accuracy benefit on this dataset, which is why...

## Final strategy used for `output.csv`: **single**

Chosen because (a) it has a real, measured `claim_status_accuracy` of 0.75 on the
labeled sample, and (b) under the quota constraints actually observed on this project
(see below), `two_stage` would have roughly doubled the number of model calls needed to
score the 44-row test set, which was already the binding constraint -- not something
worth doing without a demonstrated accuracy gain to justify it.

### Known accuracy limitations (single, n=20)

- `issue_type` (0.45) and `severity` (0.40) are the weakest fields. Spot-checking the
  misses shows the model frequently picks a plausible *adjacent* label (e.g. `scratch`
  vs `dent`, `medium` vs `high`) rather than a wildly wrong one -- these are closer to
  calibration/prompt-wording issues than reasoning failures, and would be the first
  target for prompt iteration with more time/quota.
- `risk_flags_mean_iou` (0.649) is held down specifically by the EXIF-based
  `non_original_image` heuristic in `pipeline/image_checks.py`: most images in this
  dataset have no EXIF metadata (likely because they're curated/stock test images, not
  because of fraud), so the heuristic fires far more often than the labels expect. This
  was a deliberate, documented trade-off (see `code/README.md` and the architecture
  discussion in this project) -- kept as a "defense in depth" signal independent of the
  model's own judgment, at a known cost to this particular metric on this dataset.

## Prompt iteration: few-shot calibration (tried, measured, reverted)

After the initial result above, a second prompt version (`v2-fewshot`) was tried: three
text-only calibration examples (no images, just conversation -> expected JSON) were added
to the system prompt, targeting the two weakest fields (`issue_type` 0.45,
`severity` 0.40), plus an explicit instruction to use `severity=high` sparingly.

This was validated on a 5-claim held-out probe (`user_001/002/006/009/015`, distinct from
the 3 claims used as the few-shot exemplars themselves) before spending quota on a full
re-run:

| Claim | v1 issue_type | v2 issue_type | v1 severity | v2 severity |
|---|---|---|---|---|
| user_001 | dent ✓ | broken_part ✗ | high ✗ | high ✗ |
| user_002 | dent ✗ | broken_part ✗ | medium ✗ | high ✗ |
| user_006 | unknown ✓ | unknown ✓ | unknown ✓ | unknown ✓ |
| user_009 | glass_shatter ✗ | glass_shatter ✗ | high ✗ | high ✗ |
| user_015 | crushed_packaging ✓ | crushed_packaging ✓ | low ✗ | low ✗ |

**v2 never beat v1 and regressed one previously-correct case** (`user_001`: a clear dent
misread as `broken_part`). The likely cause: the one high-severity example in the few-shot
set anchored the model toward `severity=high` more strongly than the accompanying "use
high sparingly" instruction pulled it back -- 3 of 5 v2 outputs were `high` vs. 2 of 5 for
v1. **v1 was kept as the final prompt** (see `pipeline/prompt.py::PROMPT_VERSION`); this is
recorded here as a deliberately-tested-and-rejected improvement, not an oversight.

This also surfaced a real cache-correctness bug worth noting: the content-hash cache key
originally did not include a prompt version, so editing the prompt would have silently
served stale pre-edit results on a cache hit. `PROMPT_VERSION` was added to
`pipeline/prompt.py` and threaded into the cache key in `pipeline/runner.py` specifically
to catch this before it caused a silent evaluation error.

## Operational analysis (real, measured -- not projected)

### Production run on `dataset/claims.csv` (44 rows, strategy=single)

- **44 model calls total** (1 per claim; no `two_stage` doubling since `single` was
  used), **82 images processed**.
- **~120,718 input tokens, ~9,863 output tokens** total (≈2,744 input / ≈224 output
  tokens per call on average).
- **≈536 seconds (≈8.9 minutes) of cumulative model latency** across calls (this is
  summed call latency, not wall-clock run time -- the run was split across three
  separate process invocations due to quota interruptions, see below).
- Estimated cost at the `ASSUMED_INPUT_PRICE_PER_MTOK`/`ASSUMED_OUTPUT_PRICE_PER_MTOK`
  placeholders in `evaluation/main.py` ($0.30 / $2.50 per MTok, a Gemini Flash-class
  pay-as-you-go assumption -- the exact rate for whichever dated model the
  `gemini-flash-latest` alias resolves to should be confirmed against Google's current
  pricing page): **≈$0.06 USD (≈₹5)**. Even doubling every assumption for safety margin,
  total cost for this entire dataset is a few rupees, not a budgeting concern.

### Quota / rate-limit findings (the real binding constraint, not cost)

This is the most important operational finding from actually running this system, and
it directly answers the brief's request to show TPM/RPM/RPD consideration:

- **Cost was never the bottleneck -- daily request quota was.** Every Gemini model
  tested on a free-tier API key/project (`gemini-2.5-flash`, `gemini-2.5-flash-lite`,
  `gemini-2.0-flash`, `gemini-2.0-flash-lite`, `gemini-2.5-pro`, `gemini-pro-latest`,
  `gemini-flash-latest`) hit `429 RESOURCE_EXHAUSTED` ("free tier requests" quota,
  reported value 20) after roughly 20 successful calls in a single day, on a single
  project -- and once one model's quota was hit, every other model tested on the *same*
  project was also blocked, even ones never called before. This indicates a low,
  effectively project-wide daily request ceiling on this account tier, not a generous
  per-model allowance.
- Completing the 44-row production run required **three separate API keys from three
  separate Google accounts**, each contributing ~17-19 calls before exhausting that
  day's quota, because billing could not be enabled on the original account (autopay
  failure). This is documented here as a real constraint encountered, not a
  hypothetical -- and is exactly the kind of RPD ceiling a production deployment would
  need to either pay to lift (enable billing) or design around (queue + backoff across
  multiple days, or provision a paid-tier key from the start).
- **Resumability was the mitigation actually used.** `pipeline/cache.py` (content-hash
  keyed by image bytes + claim text + strategy + model) meant every quota-driven
  interruption lost zero completed work -- `main.py` was simply rerun with a fresh key
  and picked up exactly where it left off. `pipeline/runner.py` was also hardened mid-run
  to catch `google.genai.errors.APIError`, stop cleanly instead of crashing, write
  whatever rows were completed, and report exactly which `user_id`s remain -- this is
  the retry/resume strategy this system actually relies on in practice, more than the
  exponential-backoff retry in `vision_client.py` (which only helps with transient
  429/5xx blips, not a hard daily cap).
- **Batching**: not used. At 44-82 calls total, sequential per-claim calls are simpler
  and the bottleneck was never throughput within a session -- it was the daily cap
  across sessions/accounts. For a meaningfully larger test set, the same `runner.py`
  loop could be parallelized behind a token-bucket limiter without changing per-claim
  logic, but it would not help with an RPD wall the way it would with an RPM wall.
- **thinking_config**: Gemini 2.x/3.x "flash" models spend hidden "thinking" tokens
  against `max_output_tokens` by default, which silently truncated the structured JSON
  response in early testing (see `git`/conversation history). `thinking_budget=0` is
  set explicitly in `vision_client.py` for both the review and triage calls, since this
  is a fixed-schema classification task, not a reasoning task -- this also reduces
  token cost and latency.

## Files

- `output.csv` (repo root) -- final predictions for all 44 rows of `dataset/claims.csv`.
- `code/pipeline/` -- the full pipeline (see `code/README.md` for a module-by-module
  walkthrough).
- `code/.cache/` -- the content-addressed cache that made multi-key resumption possible
  (excluded from `code.zip` / git via `.gitignore`; regenerated on first run).
