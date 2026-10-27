"""Turn a folder of unlabelled documents into evalkit labelling material.

1. ``classify``: group the documents into ``k`` named classes. A folder already
   sorted as ``docs/<class>/<file>`` uses its folder names and needs no model calls.
   Writes ``classes.json`` for a person to check and edit.
2. ``propose``: per class, show the model a few examples and get a field list back.
   Writes ``specs/<class>.yaml`` in the evalkit spec format (``evalkit/labels.py``).
3. ``prelabel``: per document, fill its class's fields. Writes ``prelabels.jsonl``
   (``{"doc_id", "doc_type", "source", "fields"}``), which
   ``evalkit labels template DOCS --specs specs --prefill prelabels.jsonl`` turns into
   one pre-filled sheet per class for a person to correct.

Each step reads the files the previous one wrote, so a person can fix classes or
specs in between. ``doc_id`` is the file's path under the documents folder without
its extension, the same id ``evalkit labels template`` gives it. A failure on one
document goes to ``errors.jsonl`` and never stops the run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, Field, ValidationError

from ..spec import TYPES, model_from_spec
from .model import Blob, JsonModel

T = TypeVar("T")
R = TypeVar("R")

MIME = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}
FieldType = Literal["amount", "number", "date", "id", "text", "enum", "exact", "bool"]
assert set(FieldType.__args__) == set(TYPES)

# ---------------------------------------------------------------- documents


@dataclass(frozen=True)
class Document:
    root: Path
    source: str  # path under the documents folder, forward slashes
    folder_class: str | None

    @property
    def doc_id(self) -> str:
        return Path(self.source).with_suffix("").as_posix()

    def blob(self) -> Blob:
        return Blob((self.root / self.source).read_bytes(), MIME[Path(self.source).suffix.lower()])


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_") or "unknown"


def scan(root: str | Path) -> list[Document]:
    """Supported files under ``root``, sorted, skipping hidden folders and ``_*`` folders
    (where discovery writes its own output). A file directly inside a first-level folder
    takes that folder's name as its class."""
    root = Path(root)
    docs = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or p.suffix.lower() not in MIME or p.name.startswith("."):
            continue
        if any(part.startswith((".", "_")) for part in rel.parts[:-1]):
            continue
        docs.append(Document(root, rel.as_posix(), slug(rel.parts[0]) if len(rel.parts) == 2 else None))
    return docs


# ---------------------------------------------------------------- response models


class Description(BaseModel):
    kind: str = Field(description="two to four plain words, e.g. 'supplier invoice'")
    issuer_type: str = ""
    summary: str = Field(description="one line on what the document is for")


class ClassDef(BaseModel):
    name: str
    description: str


class Assignment(BaseModel):
    index: int
    class_name: str


class Clustering(BaseModel):
    classes: list[ClassDef]
    assignments: list[Assignment]


class ProposedField(BaseModel):
    path: str
    type: FieldType
    required: bool
    critical: bool
    description: str
    example: str | None = None
    values: list[str] | None = None
    pattern: str | None = None


class ProposedSpec(BaseModel):
    description: str
    dayfirst: bool
    fields: list[ProposedField]


# ---------------------------------------------------------------- prompts

DESCRIBE_PROMPT = """You are cataloguing business documents before anyone has defined document types.
Describe this document so it can be grouped with similar ones: the kind of document in two
to four plain words (for example "supplier invoice", "bank statement", "delivery note"),
the type of issuer if visible, and its purpose in one line.
Do not copy personal details such as names, account numbers or addresses."""

CLUSTER_PROMPT = """Below are short descriptions of {n} documents, one per line as `index: kind | summary`.
Group them into exactly {k} document classes, so that documents in one class would share
the same extraction fields. Give each class a short snake_case name and a one-line
description, then assign every index to exactly one class by its name.

{lines}"""

PROPOSE_PROMPT = """These {n} documents are examples of the class "{name}": {description}

Propose the fields an extraction system should pull from every document of this class.
- Paths are snake_case. Group related fields with dots (vendor.name, vendor.tax_id).
- Repeating table rows use [] (line_items[].description, line_items[].amount).
- type is one of: {types}. amount is money, id is a reference number or code, date is a
  date, enum is a value from a small fixed set (then list the values), exact must match
  character for character, text is everything else.
- required: the field appears on nearly every document of this class.
- critical: an error in this field would make the extraction unusable (ids, totals,
  dates of record).
- pattern: a regex only when the examples show one clear fixed format.
- example: a made-up value in the right format, never one copied from the documents.
- Aim for the 8 to 25 fields a business would use; skip boilerplate and legal text.
- dayfirst: true if dates in these documents are written day before month."""

PRELABEL_PROMPT = """Extract the fields below from this "{doc_type}" document ({description}).

{fields}

- Return null for a field that is not on the document. Never guess.
- Copy ids, codes and exact fields character for character as printed.
- Dates as YYYY-MM-DD.{dayfirst}
- Amounts and numbers as plain numbers, without currency symbols or thousands separators.
- For [] fields return one entry per table row, in printed order."""

# ---------------------------------------------------------------- helpers


class Run:
    """Output folder plus the errors collected while writing it."""

    def __init__(self, out: str | Path):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.errors: list[dict[str, Any]] = []

    def error(self, step: str, source: str | None, error: str) -> None:
        self.errors.append({"step": step, "source": source, "error": error})

    def flush(self) -> None:
        if self.errors:
            with (self.out / "errors.jsonl").open("a", encoding="utf-8") as fh:
                for e in self.errors:
                    fh.write(json.dumps(e, ensure_ascii=False) + "\n")
            self.errors = []


def _pmap(fn: Callable[[T], R], items: Iterable[T], workers: int) -> list[R]:
    items = list(items)
    if workers <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def _spread(docs: list[Document], n: int) -> list[Document]:
    """``n`` documents spread evenly over the sorted list, so examples are not all from one
    batch. Deterministic, so a rerun picks the same examples (and hits the cache)."""
    if len(docs) <= n:
        return docs
    step = len(docs) / n
    return [docs[int(i * step)] for i in range(n)]


def _ask(model: JsonModel, prompt: str, blobs: list[Blob], schema: type[BaseModel]) -> BaseModel:
    return schema.model_validate(model.generate(prompt, blobs, schema))


# ---------------------------------------------------------------- 1. classify


def classify(docs_dir: str | Path, out: str | Path, model: JsonModel, k: int = 5, workers: int = 4) -> dict[str, Any]:
    """Write ``classes.json``: ``{"classes": [{"name", "description"}], "assignments": {source: name}}``."""
    run = Run(out)
    docs = scan(docs_dir)
    if not docs:
        raise ValueError(f"no documents under {docs_dir}")

    if all(d.folder_class for d in docs):
        result: dict[str, Any] = {
            "source": "folders",
            "classes": [{"name": n, "description": ""} for n in sorted({d.folder_class for d in docs})],
            "assignments": {d.source: d.folder_class for d in docs},
        }
    else:

        def describe(d: Document) -> Description | None:
            try:
                return _ask(model, DESCRIBE_PROMPT, [d.blob()], Description)
            except Exception as e:  # one bad file must not stop the run
                run.error("classify", d.source, repr(e))
                return None

        described = [(d, x) for d, x in zip(docs, _pmap(describe, docs, workers), strict=True) if x]
        lines = "\n".join(f"{i}: {x.kind} | {x.summary}" for i, (_, x) in enumerate(described))
        resp = _ask(model, CLUSTER_PROMPT.format(n=len(described), k=k, lines=lines), [], Clustering)

        classes = list(
            {slug(c.name): {"name": slug(c.name), "description": c.description} for c in resp.classes}.values()
        )
        known = {c["name"] for c in classes}
        assignments = {}
        for a in resp.assignments:
            if 0 <= a.index < len(described) and slug(a.class_name) in known:
                assignments[described[a.index][0].source] = slug(a.class_name)
        for d, _ in described:
            if d.source not in assignments:
                run.error("classify", d.source, "model gave no valid class")
        if len(classes) != k:
            run.error("classify", None, f"asked for {k} classes, got {len(classes)}")
        result = {
            "source": model.name,
            "classes": classes,
            "assignments": dict(sorted(assignments.items())),
            "descriptions": {d.source: x.model_dump() for d, x in described},
        }

    (run.out / "classes.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    run.flush()
    return result


def _load_classes(docs_dir: str | Path, out: Path) -> tuple[dict[str, str], dict[str, list[Document]]]:
    path = out / "classes.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run classify first")
    data = json.loads(path.read_text())
    by_source = {d.source: d for d in scan(docs_dir)}
    groups: dict[str, list[Document]] = {c["name"]: [] for c in data["classes"]}
    for source, name in data["assignments"].items():
        if source in by_source and name in groups:
            groups[name].append(by_source[source])
    return {c["name"]: c.get("description", "") for c in data["classes"]}, groups


# ---------------------------------------------------------------- 2. propose


def clean_spec(doc_type: str, proposed: ProposedSpec, description: str = "") -> dict[str, Any]:
    """A proposed spec made valid for evalkit: snake_case paths, no duplicates or
    leaf/parent clashes, enums with values, compilable patterns. The reviewer fixes
    meaning, not syntax."""
    fields: list[dict[str, Any]] = []
    seen: set[str] = set()
    for f in proposed.fields:
        path = ".".join(slug(s[:-2]) + "[]" if s.endswith("[]") else slug(s) for s in f.path.strip().split(".") if s)
        if not path or path in seen:
            continue
        stems = {p.replace("[]", "") for p in seen}
        bare = path.replace("[]", "")
        if any(bare.startswith(s + ".") or s.startswith(bare + ".") for s in stems):
            continue  # "vendor" and "vendor.name" cannot both be leaves
        seen.add(path)
        item: dict[str, Any] = {"path": path, "type": f.type}
        if f.type == "enum" and not f.values:
            item["type"] = "text"
        if f.required:
            item["required"] = True
        if f.critical:
            item["critical"] = True
        if f.description:
            item["description"] = f.description
        if f.example not in (None, ""):
            item["example"] = f.example
        if item["type"] == "enum":
            item["values"] = list(dict.fromkeys(f.values or []))
        if f.pattern:
            try:
                re.compile(f.pattern)
                item["pattern"] = f.pattern
            except re.error:
                pass
        fields.append(item)
    spec = {
        "doc_type": doc_type,
        "version": 1,
        "description": proposed.description or description,
        "dayfirst": proposed.dayfirst,
        "fields": fields,
    }
    model_from_spec(spec)  # raises if the pipeline could not build a schema from it
    return spec


def propose(
    docs_dir: str | Path, out: str | Path, model: JsonModel, examples: int = 5, overwrite: bool = False
) -> dict[str, dict[str, Any]]:
    """Write ``specs/<class>.yaml`` per class. An existing spec is kept unless ``overwrite``,
    since a person may already have edited it."""
    run = Run(out)
    descriptions, groups = _load_classes(docs_dir, run.out)
    spec_dir = run.out / "specs"
    spec_dir.mkdir(exist_ok=True)
    written = {}
    for name, docs in groups.items():
        path = spec_dir / f"{name}.yaml"
        if path.exists() and not overwrite:
            continue
        sample = _spread(docs, examples)
        if not sample:
            run.error("propose", name, "class has no documents")
            continue
        prompt = PROPOSE_PROMPT.format(
            n=len(sample), name=name, description=descriptions.get(name) or name, types=", ".join(FieldType.__args__)
        )
        try:
            spec = clean_spec(
                name, _ask(model, prompt, [d.blob() for d in sample], ProposedSpec), descriptions.get(name, "")
            )
        except Exception as e:
            run.error("propose", name, repr(e))
            continue
        path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
        written[name] = spec
    run.flush()
    return written


# ---------------------------------------------------------------- 3. prelabel


def load_specs(folder: str | Path) -> dict[str, dict[str, Any]]:
    specs = {}
    for p in sorted(Path(folder).glob("*.y*ml")):
        data = yaml.safe_load(p.read_text()) or {}
        if "doc_types" in data:
            specs.update({n: {"doc_type": n, **s} for n, s in data["doc_types"].items()})
        elif "fields" in data:
            specs[data.get("doc_type") or p.stem] = {"doc_type": data.get("doc_type") or p.stem, **data}
    return specs


def _field_lines(spec: dict[str, Any]) -> str:
    rows = []
    for f in spec["fields"]:
        f = {"path": f} if isinstance(f, str) else f
        notes = [f.get("type", "text")]
        if f.get("values"):
            notes.append("one of " + ", ".join(map(str, f["values"])))
        if f.get("pattern"):
            notes.append(f"format {f['pattern']}")
        desc = f": {f['description']}" if f.get("description") else ""
        rows.append(f"- {f['path']} ({'; '.join(notes)}){desc}")
    return "\n".join(rows)


def _prune(value: Any, schema: type[BaseModel]) -> Any:
    """Keep only keys the spec knows, so a chatty model cannot add columns."""
    known = schema.model_json_schema()
    return _prune_json(value, known, known.get("$defs", {}))


def _prune_json(value: Any, node: dict[str, Any], defs: dict[str, Any]) -> Any:
    if "$ref" in node:
        node = defs[node["$ref"].rsplit("/", 1)[-1]]
    for alt in node.get("anyOf", []):
        if alt.get("type") != "null":
            node = defs[alt["$ref"].rsplit("/", 1)[-1]] if "$ref" in alt else alt
            break
    if isinstance(value, dict) and "properties" in node:
        return {k: _prune_json(v, node["properties"][k], defs) for k, v in value.items() if k in node["properties"]}
    if isinstance(value, list) and "items" in node:
        return [_prune_json(v, node["items"], defs) for v in value]
    return value


def prelabel(
    docs_dir: str | Path, out: str | Path, model: JsonModel, workers: int = 4, only: Iterable[str] | None = None
) -> Path:
    """Write ``prelabels.jsonl``: one ``{"doc_id", "doc_type", "source", "fields"}`` draft per
    document. A draft whose values do not fit the spec's types is kept as returned (minus
    unknown keys) and noted in ``errors.jsonl``; the person correcting it sees the raw value."""
    run = Run(out)
    _, groups = _load_classes(docs_dir, run.out)
    specs = load_specs(run.out / "specs")
    wanted = set(only) if only else set(groups)
    for name in sorted(wanted - set(specs)):
        run.error("prelabel", name, f"no spec at specs/{name}.yaml")
    schemas = {n: model_from_spec(specs[n]) for n in wanted & set(specs)}
    jobs = [(n, d) for n in sorted(schemas) for d in groups.get(n, [])]

    def label(job: tuple[str, Document]) -> dict[str, Any] | None:
        name, d = job
        spec, schema = specs[name], schemas[name]
        prompt = PRELABEL_PROMPT.format(
            doc_type=name,
            description=spec.get("description") or name,
            fields=_field_lines(spec),
            dayfirst=" Read ambiguous dates as day/month." if spec.get("dayfirst", True) else "",
        )
        try:
            raw = model.generate(prompt, [d.blob()], schema)
        except Exception as e:
            run.error("prelabel", d.source, repr(e))
            return None
        try:
            fields = schema.model_validate(raw).model_dump(mode="json")
        except ValidationError as e:
            run.error("prelabel", d.source, f"values do not fit the spec: {e.error_count()} errors")
            fields = _prune(raw, schema)
        return {"doc_id": d.doc_id, "doc_type": name, "source": d.source, "fields": fields}

    rows = sorted((r for r in _pmap(label, jobs, workers) if r), key=lambda r: r["doc_id"])
    path = run.out / "prelabels.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    run.flush()
    return path
