"""Dotted-path helpers: ``{"a": {"b": [ {"c": 1} ]}}`` <-> ``{"a.b[0].c": 1}``."""

from __future__ import annotations

import re
from typing import Any

_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Map every leaf (scalar or None) to its dotted path."""
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            out.update(flatten(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        if not value and prefix:
            out[prefix] = []
        for i, child in enumerate(value):
            out.update(flatten(child, f"{prefix}[{i}]"))
    else:
        out[prefix] = value
    return out


def split(path: str) -> list[str | int]:
    return [name if name else int(index) for name, index in _TOKEN.findall(path)]


def unflatten(flat: dict[str, Any]) -> Any:
    root: Any = None
    for path, leaf in flat.items():
        keys = split(path)
        if root is None:
            root = [] if isinstance(keys[0], int) else {}
        node = root
        for key, nxt in zip(keys, keys[1:], strict=False):
            node = _child(node, key, [] if isinstance(nxt, int) else {})
        _assign(node, keys[-1], leaf)
    return root if root is not None else {}


def get(value: Any, path: str, default: Any = None) -> Any:
    node = value
    for key in split(path):
        try:
            node = node[key]
        except (KeyError, IndexError, TypeError):
            return default
    return node


def generic(path: str) -> str:
    """``line_items[3].amount`` -> ``line_items[].amount`` (for per-field config)."""
    return re.sub(r"\[\d+\]", "[]", path)


def _child(node: Any, key: str | int, empty: Any) -> Any:
    if isinstance(key, int):
        while len(node) <= key:
            node.append(None)
        if node[key] is None:
            node[key] = empty
        return node[key]
    return node.setdefault(key, empty)


def _assign(node: Any, key: str | int, leaf: Any) -> None:
    if isinstance(key, int):
        while len(node) <= key:
            node.append(None)
    node[key] = leaf
