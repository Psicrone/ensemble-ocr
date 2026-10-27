"""Build the extraction schema (a Pydantic model) from a document-type spec.

The spec (see ``ensemble.evaluation.specs``) is the single source for labels, scoring
and extraction: a flat list of fields per document type::

    doc_types:
      invoice:
        version: 1
        fields:
          - {path: invoice_number, type: id, required: true, critical: true}
          - {path: vendor.entity_number, type: id, critical: true, pattern: '^\\d{4}-\\d{2}$'}
          - {path: currency, type: enum, values: [ZAR, USD, EUR]}
          - {path: "line_items[].amount", type: amount}

This module turns one doc type into a Pydantic model whose fields carry the
``x-*`` keys the validators read (``x-critical``, ``x-format``, ``x-enum``).
Every leaf is optional: whether a required field is present is for the
validators to flag, not for the decoder to invent.
"""

from __future__ import annotations

from datetime import date
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from .evaluation.specs import load_specs

TYPES: dict[str, Any] = {
    "amount": float,
    "number": float,
    "date": date,
    "id": str,
    "text": str,
    "exact": str,
    "enum": str,
    "bool": bool,
}


def load_spec(path: str | Path, doc_type: str) -> dict[str, Any]:
    """One doc type from a spec file or directory, validated by ``ensemble.evaluation.specs``."""
    specs = load_specs(path)
    if doc_type not in specs:
        raise KeyError(f"{path}: no doc type {doc_type!r}; have {sorted(specs)}")
    spec = specs[doc_type]
    return {"doc_type": doc_type, "fields": spec.fields, "version": spec.version, "description": spec.description}


def _extra(f: dict[str, Any]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if f.get("critical"):
        extra["x-critical"] = True
    if f.get("required"):
        extra["x-required"] = True
    if "pattern" in f:
        extra["x-format"] = f["pattern"]
    if "values" in f:
        extra["x-enum"] = list(f["values"])
    return extra


def _description(f: dict[str, Any]) -> str | None:
    parts = [f.get("description")]
    if "example" in f:
        parts.append(f"e.g. {f['example']}")
    if f.get("type") == "date":
        parts.append("YYYY-MM-DD")
    return ". ".join(str(p) for p in parts if p) or None


def _title(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace("[]", "").split("_")) or "Item"


def _build(name: str, tree: dict[str, Any], config: ConfigDict | None = None) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for key, node in tree.items():
        is_list = key.endswith("[]")
        attr = key[:-2] if is_list else key
        if "__field__" in node:  # leaf
            f = node["__field__"]
            ann: Any = TYPES[f.get("type", "text")]
            kwargs = {"description": _description(f), "json_schema_extra": _extra(f) or None}
            if is_list:  # list of scalars
                fields[attr] = (list[ann], Field(default_factory=list, **kwargs))
            else:
                fields[attr] = (ann | None, Field(None, **kwargs))
        else:
            sub = _build(f"{name}{_title(attr)}", node)
            if is_list:
                fields[attr] = (list[sub], Field(default_factory=list))
            else:
                fields[attr] = (sub | None, Field(None))
    return create_model(name, __config__=config, **fields)


@cache
def _model_from_file(path: str, doc_type: str) -> type[BaseModel]:
    return model_from_spec(load_spec(path, doc_type))


def model_from_spec(spec: dict[str, Any] | str | Path, doc_type: str | None = None) -> type[BaseModel]:
    """Pydantic model for one doc type (``spec`` is a parsed spec dict or a YAML path)."""
    if not isinstance(spec, dict):
        if doc_type is None:
            raise ValueError("doc_type is required when loading a spec file")
        return _model_from_file(str(Path(spec).resolve()), doc_type)
    name = spec["doc_type"]
    tree: dict[str, Any] = {}
    for f in spec.get("fields", []):
        f = {"path": f} if isinstance(f, str) else f
        node = tree
        parts = f["path"].split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault(parts[-1], {})["__field__"] = f
    config = ConfigDict(json_schema_extra={"x-version": f"{name}.v{spec.get('version', 1)}"})
    model = _build(_title(name), tree, config)
    model.__doc__ = spec.get("description") or None
    return model
