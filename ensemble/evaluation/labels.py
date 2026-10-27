"""Ground-truth labelling for document extraction.

Documents live in ``docs/<doc_type>/<file>``; each gets ``doc_id = "<doc_type>/<file stem>"``.
Gold records are ``{"doc_id", "doc_type", "source", "fields": {...}}`` with ``source`` the
document's path relative to the documents folder.

    template  a folder of documents -> one CSV (or JSONL) per doc type, optionally
              pre-filled with draft values (e.g. a model's first pass) to correct
    import    filled sheets -> canonical gold JSONL (numbers as numbers, ISO dates)
    check     unknown fields, unparseable values, enum/pattern violations, missing required
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from evalkit.structured import Schema, compare, is_empty, parse_date, parse_number, read_records

from .specs import _IDX, DocTypeSpec

DOC_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp", ".webp"}


# =========================================================================== labels
def scan_documents(root) -> list[dict]:
    """Documents under ``root``. A first-level folder name is taken as the doc type."""
    root = Path(root)
    out = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in DOC_SUFFIXES:
            rel = f.relative_to(root)
            out.append({"doc_id": rel.with_suffix("").as_posix(), "source": rel.as_posix(),
                        "doc_type": rel.parts[0] if len(rel.parts) > 1 else ""})
    return out


def documents_from_gold(gold, root) -> dict:
    """``{doc_id: path}`` for ``score_pipeline``, from the gold records' ``source``."""
    root = Path(root)
    return {str(r["doc_id"]): root / r.get("source", r["doc_id"]) for r in read_records(gold)}


def _flat_cells(obj, path="") -> dict:
    """Nested values -> {"vendor.name": v, "line_items[0].amount": v} spreadsheet cells."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_flat_cells(v, f"{path}.{k}" if path else k))
        return out
    if isinstance(obj, list) and any(isinstance(x, dict) for x in obj):
        out = {}
        for i, v in enumerate(obj):
            out.update(_flat_cells(v, f"{path}[{i}]"))
        return out
    return {path: obj}


def write_templates(docs: list[dict], specs: dict[str, DocTypeSpec], out, *, max_items=5,
                    fmt="csv", doc_type: str | None = None, prefill=None) -> list[Path]:
    """One labelling file per document type (CSV for a spreadsheet, or JSONL).

    ``prefill`` (gold-shaped records, e.g. a model's draft values) fills the cells so people
    correct values instead of typing them; its ``doc_type`` overrides the folder name.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pre = {str(r["doc_id"]): r for r in read_records(prefill)} if prefill is not None else {}
    by_type: dict[str, list] = {}
    for d in docs:
        t = doc_type or pre.get(d["doc_id"], {}).get("doc_type") or d["doc_type"]
        if t not in specs:
            t = "_unassigned"
        by_type.setdefault(t, []).append(d)
    written = []
    for t, rows in by_type.items():
        if t == "_unassigned":
            path = out / "_unassigned.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["doc_id", "source", "doc_type"])
                w.writerows([[r["doc_id"], r["source"], r["doc_type"]] for r in rows])
            written.append(path)
            continue
        spec = specs[t]
        if fmt == "jsonl":
            path = out / f"{t}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    fields = pre.get(r["doc_id"], {}).get("fields") or spec.blank()
                    f.write(json.dumps({"doc_id": r["doc_id"], "doc_type": t, "source": r["source"],
                                        "fields": fields}) + "\n")
        else:
            path = out / f"{t}.csv"
            cells = {r["doc_id"]: _flat_cells(pre.get(r["doc_id"], {}).get("fields") or {}) for r in rows}
            n_items = max([max_items] + [int(m) + 1 for c in cells.values() for k in c
                                         for m in _IDX.findall(k)])
            cols = spec.columns(n_items)
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["doc_id", "doc_type", "source"] + cols)
                for r in rows:
                    c = cells[r["doc_id"]]
                    w.writerow([r["doc_id"], t, r["source"]] + ["" if is_empty(c.get(k)) else c.get(k) for k in cols])
        written.append(path)
    return written


def _set_path(obj: dict, path: str, value):
    parts = path.split(".")
    node = obj
    for i, part in enumerate(parts):
        m = re.fullmatch(r"(.+?)\[(\d+)\]", part)
        last = i == len(parts) - 1
        if m:
            name, idx = m.group(1), int(m.group(2))
            lst = node.setdefault(name, [])
            while len(lst) <= idx:
                lst.append({})
            if last:
                lst[idx] = value
            else:
                node = lst[idx]
        elif last:
            node[part] = value
        else:
            node = node.setdefault(part, {})


def canonical(value, f: dict | None, dayfirst=True):
    """A labelled cell in canonical form (numbers as numbers, dates as ISO)."""
    if is_empty(value):
        return None
    if f is None:
        return value
    t = f["type"]
    if t in ("amount", "number"):
        x = parse_number(value)
        return x if x is not None else value
    if t == "date":
        d = parse_date(value, f.get("dayfirst", dayfirst))
        return d.isoformat() if d else value
    if t == "bool" and isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "t"}
    return value.strip() if isinstance(value, str) else value


def _drop_empty_rows(obj):
    if isinstance(obj, dict):
        return {k: _drop_empty_rows(v) for k, v in obj.items()}
    if isinstance(obj, list):
        rows = [_drop_empty_rows(x) for x in obj]
        return [x for x in rows if not (isinstance(x, dict) and all(is_empty(v) for v in _leaves(x)))]
    return obj


def _leaves(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _leaves(v)
    else:
        yield obj


def import_labels(files, specs: dict[str, DocTypeSpec]) -> list[dict]:
    """Labelled CSV or JSONL files -> canonical gold records."""
    records = []
    for file in map(Path, files):
        if file.name.startswith("_"):
            continue
        if file.suffix.lower() == ".csv":
            with open(file, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    t = row.pop("doc_type", "") or file.stem
                    rec = {"doc_id": row.pop("doc_id"), "doc_type": t, "source": row.pop("source", "")}
                    spec = specs.get(t)
                    fields: dict = {}
                    for col, cell in row.items():
                        if col is None:
                            continue
                        fdef = spec.field(col) if spec else None
                        _set_path(fields, col, canonical(cell, fdef, spec.dayfirst if spec else True))
                    rec["fields"] = _drop_empty_rows(fields)
                    records.append(rec)
        else:
            for rec in read_records(file):
                spec = specs.get(rec.get("doc_type"))
                rec["fields"] = _drop_empty_rows(_canon_tree(rec.get("fields") or {}, spec, ""))
                records.append(rec)
    return records


def _canon_tree(obj, spec, path):
    if isinstance(obj, dict):
        return {k: _canon_tree(v, spec, f"{path}.{k}" if path else k) for k, v in obj.items()}
    if isinstance(obj, list) and any(isinstance(x, dict) for x in obj):
        return [_canon_tree(x, spec, path + "[]") for x in obj]
    f = spec.field(path) if spec else None
    return canonical(obj, f, spec.dayfirst if spec else True)


def check_labels(records: list[dict], specs: dict[str, DocTypeSpec]) -> list[dict]:
    """Problems in gold records: unknown types/fields, missing required, bad values, dupes."""
    issues = []
    seen: dict[str, int] = {}

    def add(rec, path, problem):
        issues.append({"doc_id": rec.get("doc_id"), "doc_type": rec.get("doc_type"),
                       "field": path, "problem": problem})

    for rec in records:
        doc_id = str(rec.get("doc_id", ""))
        if not doc_id:
            add(rec, "", "record has no doc_id")
        seen[doc_id] = seen.get(doc_id, 0) + 1
        spec = specs.get(rec.get("doc_type"))
        if spec is None:
            add(rec, "", f"unknown doc_type {rec.get('doc_type')!r}; known: {sorted(specs)}")
            continue
        fields = rec.get("fields") or {}
        # walk every leaf the record has, using compare() against itself for paths
        leaves = [(r["pred_path"] or r["field"], r["field"], r["gold"])
                  for r in compare(fields, fields, Schema()) if not r["field"].endswith("[#]")]
        present = {generic for _, generic, v in leaves if not is_empty(v)}
        for concrete, generic, value in leaves:
            if is_empty(value):
                continue
            f = spec.field(generic)
            if f is None:
                add(rec, concrete, "not in the spec")
                continue
            t = f["type"]
            if t in ("amount", "number") and parse_number(value) is None:
                add(rec, concrete, f"not a number: {value!r}")
            elif t == "date" and parse_date(value, f.get("dayfirst", spec.dayfirst)) is None:
                add(rec, concrete, f"not a date: {value!r}")
            elif t == "enum" and f.get("values") and str(value) not in map(str, f["values"]):
                add(rec, concrete, f"{value!r} is not one of {f['values']}")
            if f.get("pattern") and not re.fullmatch(f["pattern"], str(value)):
                add(rec, concrete, f"{value!r} does not match {f['pattern']}")
        for f in spec.fields:
            if f.get("required") and "[]" not in f["path"] and f["path"] not in present:
                add(rec, f["path"], "required field is empty")
    for doc_id, n in seen.items():
        if n > 1:
            issues.append({"doc_id": doc_id, "doc_type": None, "field": "", "problem": f"doc_id appears {n} times"})
    return issues
