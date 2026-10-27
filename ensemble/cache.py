"""Tool-result cache keyed by (tool name, tool version, doc_id, input hash)."""

from __future__ import annotations

from typing import Any

Key = tuple[str, str, str, str]


class Cache:
    """In-process cache. Swap for a Redis/Mongo-backed one with the same two methods."""

    def __init__(self) -> None:
        self._data: dict[Key, Any] = {}

    def get(self, key: Key) -> Any | None:
        return self._data.get(key)

    def set(self, key: Key, value: Any) -> None:
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)
