"""The one model call discovery needs: documents and a prompt in, a validated Pydantic object out.

``JsonModel`` is the seam: ``GeminiJsonModel`` calls Gemini on Vertex AI with
schema-constrained output at temperature 0 and a fixed seed, and tests pass a
scripted fake. ``CachedModel`` keeps every answer on disk keyed by the whole
request, so a rerun over the same documents makes no new calls and gives the
same answers.

Discovery is a batch job with no tools to call, so it uses ``google-genai``
directly rather than an ADK agent.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel


@dataclass(frozen=True)
class Blob:
    data: bytes
    mime_type: str


class JsonModel(Protocol):
    name: str

    def generate(self, prompt: str, blobs: list[Blob], schema: type[BaseModel]) -> dict[str, Any]:
        """Raw JSON for ``schema``; the caller validates it."""
        ...


class GeminiJsonModel:
    """Gemini via google-genai on Vertex AI (``GOOGLE_CLOUD_PROJECT`` / ``GOOGLE_CLOUD_LOCATION``)."""

    def __init__(
        self, model: str = "gemini-2.5-pro", project: str | None = None, location: str | None = None, seed: int = 7
    ):
        from google import genai

        self.name = model
        self.seed = seed
        self._client = genai.Client(
            vertexai=True,
            project=project or os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )

    def generate(self, prompt: str, blobs: list[Blob], schema: type[BaseModel]) -> dict[str, Any]:
        from google.genai import types

        parts = [types.Part.from_bytes(data=b.data, mime_type=b.mime_type) for b in blobs]
        resp = self._client.models.generate_content(
            model=self.name,
            contents=[*parts, prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                seed=self.seed,
                response_mime_type="application/json",
                response_json_schema=schema.model_json_schema(),
            ),
        )
        return json.loads(resp.text)


class CachedModel:
    def __init__(self, inner: JsonModel, cache_dir: str | Path):
        self.inner = inner
        self.name = inner.name
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0
        self._lock = threading.Lock()

    def _key(self, prompt: str, blobs: list[Blob], schema: type[BaseModel]) -> str:
        h = hashlib.sha256()
        for part in (self.name, prompt, json.dumps(schema.model_json_schema(), sort_keys=True)):
            h.update(part.encode())
        for b in blobs:
            h.update(b.mime_type.encode())
            h.update(hashlib.sha256(b.data).digest())
        return h.hexdigest()

    def generate(self, prompt: str, blobs: list[Blob], schema: type[BaseModel]) -> dict[str, Any]:
        path = self.dir / f"{self._key(prompt, blobs, schema)}.json"
        if path.exists():
            with self._lock:
                self.hits += 1
            return json.loads(path.read_text())
        out = self.inner.generate(prompt, blobs, schema)
        path.write_text(json.dumps(out, ensure_ascii=False))
        with self._lock:
            self.misses += 1
        return out
