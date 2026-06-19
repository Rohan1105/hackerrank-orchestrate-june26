# Multi-Modal Evidence Review — Solution

## Setup

```bash
cd code
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env && edit .env to add your GEMINI_API_KEY   # never hardcode this
# or: export GEMINI_API_KEY=...
```

Get a free Gemini API key at https://aistudio.google.com/apikey.

**Free-tier note (encountered in practice, not hypothetical):** every Gemini model
tested on a single free-tier project hit a `429 RESOURCE_EXHAUSTED` daily-quota wall
after ~20 successful calls, and once one model's quota was spent, every other model on
that *same* project was blocked too — the practical ceiling appears to be project-wide,
not per-model. If you hit this, either enable billing on the project (cost for this
dataset is a few cents/rupees — see `evaluation/evaluation_report.md`), or use a fresh
key from a different account: the pipeline is fully resumable (see below), so switching
keys mid-run loses no completed work.

## Run

```bash
# Evaluate on the labeled sample set first (compares two strategies, writes a report)
python evaluation/main.py

# Produce the final predictions for the unlabeled test set
python main.py --input ../dataset/claims.csv --dataset-root .. --output ../output.csv
```

`main.py` defaults already point at `../dataset/claims.csv` and `../output.csv`, so
`python main.py` with no flags works from the `code/` directory.

## How it works

- `pipeline/schema.py` — the allowed-value vocabulary from `problem_statement.md`, shared
  by the prompt (what we ask for) and post-processing (what we accept).
- `pipeline/data_loader.py` — reads the three input CSVs and resolves image paths.
- `pipeline/prompt.py` — system prompt + a forced tool-call JSON schema, so the model's
  response is always valid structured output (no free-form JSON parsing/repair).
- `pipeline/vision_client.py` — Gemini SDK wrapper: builds the multi-image content list,
  forces structured output via `response_schema`, retries on rate-limit/5xx with backoff.
- `pipeline/postprocess.py` — clamps any out-of-vocabulary model output to `unknown`/`none`,
  and adds a rule-based `user_history_risk` flag from `user_history.csv` (recent claim
  count / rejection history) so user history adds context without overriding the model's
  visual read of the images.
- `pipeline/image_checks.py` — deterministic, code-only EXIF/authenticity check run
  before any model call: flags a known photo-editor signature in EXIF as
  `possible_manipulation`, and flags missing EXIF as a soft `non_original_image` signal.
  Independent of, and a complement to, the model's own authenticity judgment.
- `pipeline/cache.py` — disk cache keyed by a hash of the actual image bytes + claim text
  + strategy + model, so re-running during development (or resuming after a quota/
  network failure) doesn't re-spend tokens or calls on claims already scored. There's
  also a separate image-level cache (keyed by image hash alone) for the EXIF check, so
  a reused image file is only inspected once across the whole dataset.
- `pipeline/runner.py` — shared orchestration used by both `main.py` and
  `evaluation/main.py`. Claims with zero loadable images short-circuit straight to
  `not_enough_information` without any model call. If a model call raises an API error
  (most realistically a quota wall), the run stops cleanly, writes everything completed
  so far, and reports exactly which `user_id`s remain — rerunning with a fresh key/quota
  resumes from there via the cache.

## Strategies

- **single** — one vision call per claim: images + conversation + evidence requirements +
  user history in, structured JSON out. **This is the strategy used for `output.csv`.**
  Real accuracy on the 20 labeled sample claims: `claim_status_accuracy` 0.75 (see
  `evaluation/evaluation_report.md` for the full breakdown).
- **two_stage** — a cheap text-only triage call extracts the claimed issue/part from the
  conversation first; that hint is then passed into the vision call. Doubles the call
  count per claim. Implemented and wired into the same evaluation harness, but could not
  be run to completion against the live API in this submission window because of the
  free-tier daily quota constraint described above — see
  `evaluation/evaluation_report.md` for the reasoning behind sticking with `single`.

`evaluation/main.py` runs both strategies against `dataset/sample_claims.csv` and scores
them against the embedded labels; `main.py` defaults to `single` for the production run
given the results above, but `--strategy two_stage` works identically if rerun with more
quota available.

## Determinism / reproducibility notes

- Every output field is clamped against the fixed allowed-value sets, so the model can
  never silently change the output vocabulary.
- The disk cache is content-addressed, so re-running on the same inputs reproduces the
  same output without new API calls.
- No test labels from `sample_claims.csv` are read anywhere in `main.py`'s production
  path — they're only used by `evaluation/main.py` for scoring.
