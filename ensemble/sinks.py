"""Where results go. ``MongoSink`` stores the result and its trace in MongoDB Atlas."""

from __future__ import annotations

import asyncio
from typing import Any

from .types import ExtractionResult


class MongoSink:
    """Upserts one document per (doc_id, schema_version) and one per trace."""

    def __init__(self, uri: str, database: str = "ensemble", results: str = "results", traces: str = "traces") -> None:
        from pymongo import MongoClient  # optional dependency: pip install ensemble[mongo]

        self.db = MongoClient(uri)[database]
        self.results = results
        self.traces = traces

    async def save(self, result: ExtractionResult, spans: list[dict[str, Any]]) -> None:
        doc = result.model_dump(mode="json")
        key = {"doc_id": result.doc_id, "schema_version": result.schema_version}

        def write() -> None:
            self.db[self.results].replace_one(key, doc, upsert=True)
            self.db[self.traces].insert_one({"trace_id": result.trace_id, "doc_id": result.doc_id, "spans": spans})

        await asyncio.to_thread(write)


class MemorySink:
    """Keeps results in a list (tests, notebooks)."""

    def __init__(self) -> None:
        self.saved: list[tuple[ExtractionResult, list[dict[str, Any]]]] = []

    async def save(self, result: ExtractionResult, spans: list[dict[str, Any]]) -> None:
        self.saved.append((result, spans))
