"""
----------------------------
NVIDIA NIM API client for both embedding and LLM inference.

Provides two capabilities:
  1. embed()   — Dense vector embeddings via llama-3.2-nemoretriever-300m-embed-v1
  2. complete() — Multimodal LLM completions via meta/llama-4-maverick-17b-128e-instruct

Both use the OpenAI-compatible NVIDIA NIM endpoint at
https://integrate.api.nvidia.com/v1

Cost: $0.00 — free tier throughout.
"""
from __future__ import annotations

import json
import time
from typing import Any

import requests
from openai import OpenAI


# ── Constants ─────────────────────────────────────────────────────────────────

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_EMBED_MODEL = "nvidia/llama-3.2-nemoretriever-300m-embed-v1"
DEFAULT_LLM_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
NVIDIA_CHAT_URL = f"{NVIDIA_BASE_URL}/chat/completions"


class NvidiaEmbeddingClient:
    """
    Wraps the NVIDIA NIM embedding endpoint.

    Uses llama-3.2-nemoretriever-300m-embed-v1, a retrieval-optimised
    300M-parameter model that distinguishes passage (document) and
    query embeddings via the input_type parameter.

    Parameters
    ----------
    api_key    : NVIDIA NIM API key
    model      : embedding model name (default: NemoRetriever 300M)
    base_url   : NVIDIA NIM base URL
    truncate   : how to handle inputs exceeding token limit
                 'END' | 'START' | 'NONE'
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_EMBED_MODEL,
        base_url: str = NVIDIA_BASE_URL,
        truncate: str = "END",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.truncate = truncate
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """
        Embed document passages for indexing.

        Uses input_type='passage' — optimised for document content
        that will be searched against query embeddings.

        Parameters
        ----------
        texts : list of text strings to embed

        Returns
        -------
        list of float vectors, one per input text
        """
        return self._embed(texts, input_type="passage")

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single retrieval query.

        Uses input_type='query' — optimised for search queries that
        will be compared against passage embeddings.

        Parameters
        ----------
        query : retrieval query string

        Returns
        -------
        single float embedding vector
        """
        return self._embed([query], input_type="query")[0]

    # ── private ───────────────────────────────────────────────────────────────

    def _embed(
        self,
        texts: list[str],
        input_type: str,
    ) -> list[list[float]]:
        """
        Call the NVIDIA NIM embedding endpoint.

        Parameters
        ----------
        texts      : texts to embed
        input_type : 'passage' for documents, 'query' for search queries

        Returns
        -------
        list of embedding vectors

        Raises
        ------
        RuntimeError : on API error
        """
        response = self._client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
            extra_body={
                "input_type": input_type,
                "truncate": self.truncate,
            },
        )
        return [item.embedding for item in response.data]


class NvidiaLLMClient:
    """
    Wraps the NVIDIA NIM multimodal LLM endpoint for visual extraction
    and validation.

    Uses meta/llama-4-maverick-17b-128e-instruct which accepts interleaved
    text and base64-encoded image content blocks.

    Parameters
    ----------
    api_key     : NVIDIA NIM API key
    model       : LLM model name
    base_url    : NVIDIA NIM API URL
    temperature : sampling temperature (0.0 = deterministic)
    max_tokens  : maximum tokens in the response
    max_retries : number of retry attempts on failure
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_LLM_MODEL,
        base_url: str = NVIDIA_CHAT_URL,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def complete(
        self,
        prompt: str,
        images_b64: list[str] | None = None,
    ) -> str:
        """
        Send a prompt (optionally with page images) to the NVIDIA LLM.

        Parameters
        ----------
        prompt      : text prompt (system + user instructions)
        images_b64  : list of base64-encoded PNG page images to attach

        Returns
        -------
        str : raw LLM response content

        Raises
        ------
        RuntimeError : if all retry attempts fail
        """
        messages = self._build_messages(prompt, images_b64 or [])
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 1.0,
            "stream": False,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"NVIDIA API HTTP {response.status_code}: "
                        f"{response.text[:300]}"
                    )
                return response.json()["choices"][0]["message"]["content"]
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)   # exponential back-off

        raise RuntimeError(
            f"NVIDIA LLM failed after {self.max_retries} attempts: {last_error}"
        )

    def complete_json(
        self,
        prompt: str,
        images_b64: list[str] | None = None,
    ) -> dict:
        """
        Like complete(), but strips markdown fences and parses JSON.

        Parameters
        ----------
        prompt     : text prompt
        images_b64 : optional list of base64 page images

        Returns
        -------
        dict : parsed JSON response

        Raises
        ------
        RuntimeError  : if LLM call fails after retries
        ValueError    : if JSON parsing fails after retries
        """
        last_parse_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            raw = self.complete(prompt, images_b64)
            try:
                return self._parse_json(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                last_parse_error = exc
                if attempt < self.max_retries:
                    # Re-inject schema reminder on parse failures
                    prompt = (
                        prompt
                        + "\n\nIMPORTANT: Return ONLY valid JSON. "
                        "Do not include markdown fences or explanation text."
                    )

        raise ValueError(
            f"JSON parse failed after {self.max_retries} attempts. "
            f"Last error: {last_parse_error}"
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _build_messages(
        self,
        prompt: str,
        images_b64: list[str],
    ) -> list[dict]:
        """
        Construct OpenAI-compatible multimodal message list.

        The prompt is the text content block; each image is appended
        as an image_url content block with an inline base64 data URI.
        """
        content: list[dict] = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        return [{"role": "user", "content": content}]

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Strip markdown code fences and parse JSON."""
        clean = raw.strip()
        # Remove opening fence + optional language tag
        if clean.startswith("```"):
            parts = clean.split("```")
            if len(parts) >= 2:
                body = parts[1]
                if body.startswith("json"):
                    body = body[4:]
                clean = body.strip()
        # Remove trailing fence
        clean = clean.rstrip("`").strip()
        return json.loads(clean)
