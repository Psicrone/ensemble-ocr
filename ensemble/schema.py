"""Schema helpers: per-field declarations and the grounded output wrapper.

A doc type's schema is a Pydantic model. Fields declare their checks with
``json_schema_extra``, which also lands in the exported JSON schema::

    entity_number: str = Field(json_schema_extra={
        "x-validators": ["rules.format", "docai.ocr_match", "api.entity_registry"],
        "x-format": r"^\\d{4}-\\d{2}$",
        "x-critical": True,
    })

For extraction every leaf ``T`` is wrapped as ``{value: T, source_tokens:
[str], page: int}`` so the model cites the OCR tokens a value came from.
"""

from __future__ import annotations

import fnmatch
import types
import typing
from functools import cache
from typing import Any

from pydantic import BaseModel, Field, create_model

from .paths import generic

GROUNDED_KEYS = frozenset({"value", "source_tokens", "page"})


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _list_item(annotation: Any) -> Any | None:
    if typing.get_origin(annotation) in (list, tuple, set):
        return typing.get_args(annotation)[0]
    return None


@cache
def field_specs(schema: type[BaseModel]) -> dict[str, dict[str, Any]]:
    """Generic path (``lines[].amount``) -> declared ``x-*`` keys for every leaf."""
    out: dict[str, dict[str, Any]] = {}

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, info in model.model_fields.items():
            path = f"{prefix}.{name}" if prefix else name
            extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
            inner, _ = _unwrap_optional(info.annotation)
            item = _list_item(inner)
            if _is_model(inner):
                walk(inner, path)
            elif item is not None and _is_model(_unwrap_optional(item)[0]):
                out[path] = dict(extra)  # list-level spec (e.g. layout table checks)
                walk(_unwrap_optional(item)[0], f"{path}[]")
            else:
                out[path] = dict(extra)

    walk(schema, "")
    return out


def spec_for(schema: type[BaseModel] | None, path: str) -> dict[str, Any]:
    return field_specs(schema).get(generic(path), {}) if schema else {}


def covers(tool: Any, path: str, schema: type[BaseModel] | None) -> bool:
    """Does ``tool`` check ``path``? Field-level ``x-validators`` win over the tool's globs."""
    declared = spec_for(schema, path).get("x-validators")
    if declared is not None:
        return tool.name in declared
    globs = getattr(tool, "fields", None) or ["*"]
    gpath = generic(path)
    return any(fnmatch.fnmatchcase(gpath, g) or fnmatch.fnmatchcase(path, g) for g in globs)


def is_critical(schema: type[BaseModel] | None, path: str) -> bool:
    return bool(spec_for(schema, path).get("x-critical"))


def is_required(schema: type[BaseModel] | None, path: str) -> bool:
    spec = spec_for(schema, path)
    return bool(spec.get("x-required") or spec.get("x-critical"))


def schema_version(schema: type[BaseModel]) -> str:
    extra = schema.model_config.get("json_schema_extra")
    if isinstance(extra, dict) and "x-version" in extra:
        return str(extra["x-version"])
    return getattr(schema, "__version__", "1")


# --- grounded wrapper -----------------------------------------------------------


@cache
def _grounded_leaf(annotation: Any) -> type[BaseModel]:
    inner, _ = _unwrap_optional(annotation)
    label = getattr(inner, "__name__", str(inner)).replace("[", "_").replace("]", "")
    return create_model(
        f"Grounded_{label}",
        value=(inner | None, None),
        source_tokens=(list[str], Field(default_factory=list, description="OCR token ids, e.g. ['t12','t13']")),
        page=(int | None, Field(None, description="0-based page index")),
    )


def _grounded_annotation(annotation: Any) -> Any:
    inner, optional = _unwrap_optional(annotation)
    item = _list_item(inner)
    if _is_model(inner):
        result: Any = grounded_model(inner)
    elif item is not None:
        result = list[_grounded_annotation(item)]
    else:
        return _grounded_leaf(inner)
    return result | None if optional else result


@cache
def grounded_model(schema: type[BaseModel]) -> type[BaseModel]:
    """``schema`` with every leaf wrapped as ``{value, source_tokens, page}``."""
    fields: dict[str, Any] = {}
    for name, info in schema.model_fields.items():
        ann = _grounded_annotation(info.annotation)
        default = None if not info.is_required() else ...
        fields[name] = (ann, Field(default, description=info.description))
    return create_model(f"Grounded{schema.__name__}", __doc__=schema.__doc__, **fields)


def ungrounded(data: Any, prefix: str = "") -> list[tuple[str, Any, list[str], int | None]]:
    """Flatten a grounded dump to ``(path, value, source_tokens, page)`` per leaf."""
    out: list[tuple[str, Any, list[str], int | None]] = []
    if isinstance(data, dict) and GROUNDED_KEYS >= set(data) and "value" in data:
        out.append((prefix, data.get("value"), list(data.get("source_tokens") or []), data.get("page")))
    elif isinstance(data, dict):
        for key, child in data.items():
            out.extend(ungrounded(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(data, list):
        for i, child in enumerate(data):
            out.extend(ungrounded(child, f"{prefix}[{i}]"))
    elif prefix:
        out.append((prefix, data, [], None))
    return out


@cache
def leaf_types(schema: type[BaseModel]) -> dict[str, Any]:
    """Generic path -> the leaf's type annotation (for format/type checks)."""
    out: dict[str, Any] = {}

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, info in model.model_fields.items():
            path = f"{prefix}.{name}" if prefix else name
            inner, _ = _unwrap_optional(info.annotation)
            item = _list_item(inner)
            if _is_model(inner):
                walk(inner, path)
            elif item is not None and _is_model(_unwrap_optional(item)[0]):
                walk(_unwrap_optional(item)[0], f"{path}[]")
            elif item is not None:
                out[f"{path}[]"] = item
            else:
                out[path] = info.annotation

    walk(schema, "")
    return out
