"""Thin wrapper around the Gemini SDK: builds multi-image + text content, forces a
structured JSON response via response_schema, retries on rate-limit/overload errors,
and reports token usage so the caller can accumulate operational-analysis numbers.
"""
from __future__ import annotations

import json
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from . import prompt as prompt_mod

load_dotenv()  # picks up code/.env if present; never overrides an already-exported var

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 2.0
RETRIABLE_STATUS_CODES = {429, 500, 502, 503}


@dataclass
class CallResult:
    data: dict
    input_tokens: int
    output_tokens: int
    latency_seconds: float


def _image_part(path: Path) -> types.Part:
    media_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=media_type)


class VisionClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Put it in code/.env or export it as an "
                "environment variable (never hardcode it)."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _call_with_retry(self, **kwargs):
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                return self.client.models.generate_content(**kwargs)
            except genai_errors.APIError as e:
                last_err = e
                retriable = getattr(e, "code", None) in RETRIABLE_STATUS_CODES
                if not retriable or attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(BASE_BACKOFF_SECONDS * (2**attempt))
        raise last_err

    def review_claim(
        self,
        claim,
        image_paths: list[Path],
        user_history_row: dict | None,
        requirements: list[dict],
        triage_hint: str | None = None,
        max_tokens: int = 1024,
    ) -> CallResult:
        response_schema = prompt_mod.build_tool_schema(claim.claim_object)["input_schema"]
        user_text = prompt_mod.build_user_text(
            claim, user_history_row, requirements, triage_hint=triage_hint
        )
        contents = [_image_part(p) for p in image_paths] + [user_text]

        start = time.monotonic()
        response = self._call_with_retry(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=prompt_mod.SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=response_schema,
                max_output_tokens=max_tokens,
                # This is a fixed-schema classification task, not a reasoning task --
                # disable Gemini's extended thinking so its (hidden) "thought" tokens
                # don't eat the max_output_tokens budget and truncate the JSON reply.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        latency = time.monotonic() - start

        usage = response.usage_metadata
        return CallResult(
            data=json.loads(response.text),
            input_tokens=usage.prompt_token_count or 0,
            output_tokens=usage.candidates_token_count or 0,
            latency_seconds=latency,
        )

    def triage_claim(self, claim, max_tokens: int = 200) -> CallResult:
        start = time.monotonic()
        response = self._call_with_retry(
            model=self.model,
            contents=[prompt_mod.build_triage_user_text(claim)],
            config=types.GenerateContentConfig(
                system_instruction=prompt_mod.TRIAGE_SYSTEM_PROMPT,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        latency = time.monotonic() - start
        usage = response.usage_metadata
        return CallResult(
            data={"hint": response.text or ""},
            input_tokens=usage.prompt_token_count or 0,
            output_tokens=usage.candidates_token_count or 0,
            latency_seconds=latency,
        )
