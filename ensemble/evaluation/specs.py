"""Document-type specs: one flat list of fields per document type (YAML or JSON).

A spec is the single source for the extractor's output schema (``json_schema``), the
evalkit comparators the evaluation scores with (``eval_schema``), and the labelling
sheets people fill in (``ensemble.evaluation.labels``).

    doc_type: invoice
    version: 1
    description: Supplier tax invoice
    fields:
      - {path: invoice_number, type: id, required: true, critical: true}
      - {path: invoice_date, type: date, required: true}
      - {path: vendor.name, type: text, threshold: 0.9}
      - {path: currency, type: enum, values: [ZAR, USD, EUR]}
      - {path: total, type: amount, critical: true}
      - {path: "line_items[].description", type: text}
      - {path: "line_items[].amount", type: amount}

Paths are dotted; ``[]`` marks a list of objects (table rows). Field keys:
``type`` (amount | number | date | id | text | enum | exact | bool), ``required``,
``critical``, ``description``, ``example``, ``values`` (enum), ``pattern`` (regex a value
must match), and comparator options ``threshold``, ``abs_tol``, ``rel_tol``, ``dayfirst``.
Several specs can share a file under ``doc_types: {name: spec}``.

This module needs only PyYAML at runtime; evalkit is imported only by ``eval_schema``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

TYPES = ["amount", "bool", "date", "enum", "exact", "id", "number", "text"]
_FIELD_KEYS = {"path", "type", "required", "critical", "description", "example", "values",
               "pattern", "threshold", "abs_tol", "rel_tol", "dayfirst"}
_IDX = re.compile(r"\[(\d+)\]")


# =========================================================================== specs
@dataclass
class DocTypeSpec:
    doc_type: str
    fields: list[dict]
    version: str = "1"
    description: str = ""
    dayfirst: bool = True
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, doc_type: str | None = None) -> DocTypeSpec:
        name = d.get("doc_type") or doc_type
        if not name:
            raise ValueError("spec has no doc_type")
        fields = []
        for f in d.get("fields", []):
            f = {"path": f} if isinstance(f, str) else dict(f)
            if "path" not in f:
                raise ValueError(f"{name}: field without a path: {f}")
            f.setdefault("type", "text")
            if f["type"] not in TYPES:
                raise ValueError(f"{name}.{f['path']}: unknown type {f['type']!r}; use one of {TYPES}")
            unknown = set(f) - _FIELD_KEYS
            if unknown:
                raise ValueError(f"{name}.{f['path']}: unknown keys {sorted(unknown)}")
            fields.append(f)
        paths = [f["path"] for f in fields]
        dupes = {p for p in paths if paths.count(p) > 1}
        if dupes:
            raise ValueError(f"{name}: duplicate field paths {sorted(dupes)}")
        return cls(name, fields, str(d.get("version", "1")), d.get("description", ""),
                   d.get("dayfirst", True),
                   {k: v for k, v in d.items() if k not in ("doc_type", "fields", "version",
                                                            "description", "dayfirst")})

    def field(self, path: str) -> dict | None:
        generic = _IDX.sub("[]", path)
        return next((f for f in self.fields if f["path"] == generic), None)

    # ---------------------------------------------------------------- derived configs
    def eval_schema(self):
        """The ``evalkit.structured.Schema`` this document type is scored with."""
        from evalkit.structured import Schema
        opts = ("type", "threshold", "abs_tol", "rel_tol", "dayfirst")
        return Schema({"dayfirst": self.dayfirst,
                       "fields": {f["path"]: {k: f[k] for k in opts if k in f} for f in self.fields},
                       "critical": [f["path"] for f in self.fields if f.get("critical")]})

    def json_schema(self) -> dict:
        """A JSON Schema of the document (all leaves nullable) with x-critical / x-eval."""
        root = {"type": "object", "properties": {}}
        for f in self.fields:
            node, parts = root, f["path"].split(".")
            for part in parts[:-1]:
                node = _child(node, part)
            leaf = _child_name(parts[-1])
            target = node["properties"].setdefault(leaf[0], {})
            if leaf[1]:  # "tags[]": a list of scalars
                target.update({"type": ["array", "null"], "items": _leaf_schema(f)})
            else:
                target.update(_leaf_schema(f))
        return {"$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": self.doc_type, "description": self.description,
                "x-version": self.version, **root}

    def blank(self) -> dict:
        """An empty gold ``fields`` object (lists of rows start empty)."""
        out: dict = {}
        for f in self.fields:
            node, parts = out, f["path"].split(".")
            for part in parts[:-1]:
                name, is_list = _child_name(part)
                if is_list:
                    node.setdefault(name, [])
                    node = None
                    break
                node = node.setdefault(name, {})
            if node is not None:
                name, is_list = _child_name(parts[-1])
                node[name] = [] if is_list else None
        return out

    def required_paths(self) -> list[str]:
        return [f["path"] for f in self.fields if f.get("required")]

    def critical_paths(self) -> list[str]:
        return [f["path"] for f in self.fields if f.get("critical")]

    def columns(self, max_items: int = 5) -> list[str]:
        """Spreadsheet columns: list paths expanded to ``rows[0].x`` .. ``rows[n-1].x``."""
        cols = []
        list_fields = [f["path"] for f in self.fields if "[]." in f["path"]]
        scalar = [f["path"] for f in self.fields if "[]." not in f["path"]]
        cols += scalar
        prefixes = list(dict.fromkeys(p.split("[].")[0] for p in list_fields))
        for pre in prefixes:
            subs = [p for p in list_fields if p.startswith(pre + "[].")]
            for i in range(max_items):
                cols += [p.replace("[]", f"[{i}]", 1) for p in subs]
        return cols


def _child_name(part: str) -> tuple[str, bool]:
    return (part[:-2], True) if part.endswith("[]") else (part, False)


def _child(node: dict, part: str) -> dict:
    name, is_list = _child_name(part)
    if is_list:
        arr = node["properties"].setdefault(name, {"type": ["array", "null"],
                                                   "items": {"type": "object", "properties": {}}})
        return arr["items"]
    return node["properties"].setdefault(name, {"type": ["object", "null"], "properties": {}})


def _leaf_schema(f: dict) -> dict:
    t = f["type"]
    s = ({"type": ["number", "null"]} if t in ("amount", "number") else
         {"type": ["boolean", "null"]} if t == "bool" else
         {"type": ["string", "null"], "format": "date"} if t == "date" else
         {"type": ["string", "null"]})
    if t == "enum" and f.get("values"):
        s["enum"] = list(f["values"]) + [None]
    if f.get("pattern"):
        s["pattern"] = f["pattern"]
    if f.get("description"):
        s["description"] = f["description"]
    if "example" in f:
        s["examples"] = [f["example"]]
    if f.get("critical"):
        s["x-critical"] = True
    s["x-eval"] = {k: f[k] for k in ("type", "threshold", "abs_tol", "rel_tol") if k in f}
    return s


def _read_structured_file(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def load_specs(src) -> dict[str, DocTypeSpec]:
    """Specs from a directory of YAML/JSON files, one file, or a dict.

    A file holds one spec (``doc_type`` + ``fields``) or several under ``doc_types:``.
    """
    if isinstance(src, dict):
        items = [src]
    else:
        p = Path(src)
        files = sorted(x for x in p.iterdir() if x.suffix.lower() in (".yaml", ".yml", ".json")) if p.is_dir() else [p]
        items = [_read_structured_file(f) for f in files]
    specs: dict[str, DocTypeSpec] = {}
    for d in items:
        if not isinstance(d, dict):
            continue
        if "doc_types" in d:
            for name, sub in d["doc_types"].items():
                if isinstance(sub, dict) and "fields" in sub:
                    specs[name] = DocTypeSpec.from_dict(sub, name)
        elif "fields" in d:
            s = DocTypeSpec.from_dict(d)
            specs[s.doc_type] = s
    if not specs:
        raise ValueError(f"no document-type specs found in {src}")
    return specs


def starter_spec(doc_type: str) -> str:
    """A commented YAML spec to copy and edit."""
    return f"""# Fields to extract from a {doc_type}. One entry per field; see evalkit/labels.py.
doc_type: {doc_type}
version: 1
description: ""
dayfirst: true          # how to read ambiguous dates like 08/05/2024
fields:
  - path: document_number
    type: id            # amount | number | date | id | text | enum | exact | bool
    required: true
    critical: true      # a document "passes" only when every critical field is right
    description: Number printed at the top right
  - path: document_date
    type: date
    required: true
  - path: issuer.name
    type: text
    threshold: 0.9      # text similarity needed to count as correct
  - path: issuer.registration_number
    type: id
    pattern: '^\\d{{4}}/\\d{{6}}/\\d{{2}}$'
  - path: currency
    type: enum
    values: [ZAR, USD, EUR]
  - path: total
    type: amount
    abs_tol: 0.01
    critical: true
  - path: line_items[].description
    type: text
  - path: line_items[].quantity
    type: number
  - path: line_items[].amount
    type: amount
"""
