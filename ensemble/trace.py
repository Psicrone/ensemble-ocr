"""Minimal span recorder. Spans are plain dicts so they store directly in MongoDB.

Exporting to Cloud Trace via OpenTelemetry can wrap ``Tracer.span`` later
without changing any call site.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class Tracer:
    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex
        self.spans: list[dict[str, Any]] = []

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        record: dict[str, Any] = {"name": name, **attrs}
        start = time.perf_counter()
        try:
            yield record
        finally:
            record["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            self.spans.append(record)

    def event(self, name: str, **attrs: Any) -> None:
        self.spans.append({"name": name, **attrs})

    def count(self, name: str) -> int:
        return sum(s["name"] == name for s in self.spans)
