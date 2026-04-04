"""--------------------------
Groq LPU inference client for LLM extraction and validation.

Groq's Language Processing Unit architecture delivers 500-800 tokens/sec,
making it the primary provider for low-latency extraction and validation
calls. The free tier provides sufficient daily tokens for the full
20-document-pair evaluation dataset.

Primary model  : llama-3.3-70b-versatile
Fallback model : mixtral-8x7b-32768 (extended 32K context window)
"""
from __future__ import annotations

import json
import time
from typing import Any

from groq import Groq


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
EXTENDED_CONTEXT_MODEL = "mixtral-8x7b-32768"


class GroqLLMClient:
    """
    Wraps the Groq API for fast LLM inference.

    Note: Groq does not support multimodal (image) inputs. This client
    is used for text-only validation calls (Layer 7). For visual extraction
    (Layer 5), NvidiaLLMClient is used instead.

    Parameters
    ----------
    api_key     : Groq API key
    model       : Groq model name
    temperature : sampling temperature (0.0 = deterministic)
    max_tokens  : maximum tokens in the response
    max_retries : number of retry attempts on failure
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GROQ_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = Groq(api_key=api_key)

    def complete(self, prompt: str) -> str:
        """
        Send a text prompt to Groq and return the response content.

        Parameters
        ----------
        prompt : the full prompt string

        Returns
        -------
        str : raw LLM response content

        Raises
        ------
        RuntimeError : if all retry attempts fail
        """
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Groq LLM failed after {self.max_retries} attempts: {last_error}"
        )

    def complete_json(self, prompt: str) -> dict:
        """
        Like complete(), but strips markdown fences and parses JSON.

        Parameters
        ----------
        prompt : the full prompt string

        Returns
        -------
        dict : parsed JSON response

        Raises
        ------
        RuntimeError : if LLM call fails
        ValueError   : if JSON parsing fails after retries
        """
        last_parse_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            raw = self.complete(prompt)
            try:
                return self._parse_json(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                last_parse_error = exc
                if attempt < self.max_retries:
                    prompt = (
                        prompt
                        + "\n\nIMPORTANT: Return ONLY valid JSON. "
                        "Do not include markdown fences or explanation text."
                    )

        raise ValueError(
            f"JSON parse failed after {self.max_retries} attempts. "
            f"Last error: {last_parse_error}"
        )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Strip markdown code fences and parse JSON."""
        clean = raw.strip()
        if clean.startswith("```"):
            parts = clean.split("```")
            if len(parts) >= 2:
                body = parts[1]
                if body.startswith("json"):
                    body = body[4:]
                clean = body.strip()
        clean = clean.rstrip("`").strip()
        return json.loads(clean)
