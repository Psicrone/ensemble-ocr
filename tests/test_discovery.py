from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ensemble.discovery import CachedModel, classify, clean_spec, prelabel, propose, scan
from ensemble.discovery.discover import Clustering, Description, ProposedSpec


class ScriptedModel:
    """Answers by response type; ``prelabel`` answers depend on the document bytes."""

    name = "scripted"

    def __init__(self, fail_on: bytes | None = None):
        self.calls: list[tuple[str, int]] = []
        self.fail_on = fail_on

    def generate(self, prompt, blobs, schema):
        self.calls.append((schema.__name__, len(blobs)))
        if blobs and blobs[0].data == self.fail_on:
            raise RuntimeError("model unavailable")
        if schema is Description:
            kind = "invoice" if b"INV" in blobs[0].data else "bank statement"
            return {"kind": kind, "issuer_type": "company", "summary": f"a {kind}"}
        if schema is Clustering:
            lines = prompt.split("\n\n")[-1].splitlines()
            return {
                "classes": [
                    {"name": "Invoice", "description": "Supplier invoices"},
                    {"name": "bank statement", "description": "Monthly statements"},
                ],
                "assignments": [
                    {
                        "index": int(line.split(":")[0]),
                        "class_name": "invoice" if "invoice" in line else "Bank Statement",
                    }
                    for line in lines
                ],
            }
        if schema is ProposedSpec:
            return {
                "description": "Supplier invoice",
                "dayfirst": True,
                "fields": [
                    {
                        "path": "Invoice Number",
                        "type": "id",
                        "required": True,
                        "critical": True,
                        "description": "Number at top right",
                        "example": "INV-0001",
                        "pattern": "^INV-\\d{4}$",
                    },
                    {"path": "invoice_date", "type": "date", "required": True, "critical": False, "description": ""},
                    {
                        "path": "currency",
                        "type": "enum",
                        "required": False,
                        "critical": False,
                        "description": "",
                        "values": ["ZAR", "USD"],
                    },
                    {"path": "status", "type": "enum", "required": False, "critical": False, "description": ""},
                    {
                        "path": "total",
                        "type": "amount",
                        "required": True,
                        "critical": True,
                        "description": "",
                        "pattern": "([unclosed",
                    },
                    {"path": "total", "type": "number", "required": False, "critical": False, "description": ""},
                    {
                        "path": "line_items[].amount",
                        "type": "amount",
                        "required": False,
                        "critical": False,
                        "description": "",
                    },
                ],
            }
        # a per-document draft for the class schema built from the spec
        n = blobs[0].data.decode().split()[-1]
        return {
            "invoice_number": f"INV-{n}",
            "invoice_date": "2026-02-03",
            "currency": "ZAR",
            "total": 1100.0,
            "line_items": [{"amount": 500.0}, {"amount": 600.0}],
            "made_up_field": "drop me",
        }


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode())


@pytest.fixture
def unsorted(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    for i in range(4):
        _write(docs, f"batch1/inv_{i}.pdf", f"%PDF INV {i:04d}")
    for i in range(2):
        _write(docs, f"stmt_{i}.png", f"PNG statement {i}")
    _write(docs, "notes.txt", "ignored")
    _write(docs, "_discovery/old.pdf", "%PDF our own output, ignored")
    return docs


def test_scan_reads_folder_classes_and_skips_output(tmp_path: Path):
    _write(tmp_path, "Supplier Invoices/a.pdf", "%PDF")
    _write(tmp_path, "Supplier Invoices/deeper/b.pdf", "%PDF")
    _write(tmp_path, "_discovery/c.pdf", "%PDF")
    docs = scan(tmp_path)
    assert [(d.source, d.folder_class, d.doc_id) for d in docs] == [
        ("Supplier Invoices/a.pdf", "supplier_invoices", "Supplier Invoices/a"),
        ("Supplier Invoices/deeper/b.pdf", None, "Supplier Invoices/deeper/b"),
    ]


def test_sorted_folder_needs_no_model_calls(tmp_path: Path):
    _write(tmp_path, "docs/invoice/a.pdf", "%PDF")
    _write(tmp_path, "docs/receipt/b.jpg", "JPG")
    model = ScriptedModel()
    res = classify(tmp_path / "docs", tmp_path / "out", model)
    assert model.calls == []
    assert res["assignments"] == {"invoice/a.pdf": "invoice", "receipt/b.jpg": "receipt"}


def test_end_to_end_unsorted(unsorted: Path, tmp_path: Path):
    out = tmp_path / "out"
    model = ScriptedModel(fail_on=b"%PDF INV 0003")
    res = classify(unsorted, out, model, k=2, workers=2)
    assert {c["name"] for c in res["classes"]} == {"invoice", "bank_statement"}
    assert res["assignments"]["batch1/inv_0.pdf"] == "invoice"
    assert res["assignments"]["stmt_1.png"] == "bank_statement"
    assert "batch1/inv_3.pdf" not in res["assignments"]  # failed to describe, reported instead

    specs = propose(unsorted, out, model, examples=2)
    assert set(specs) == {"invoice", "bank_statement"}
    on_disk = yaml.safe_load((out / "specs" / "invoice.yaml").read_text())
    by_path = {f["path"]: f for f in on_disk["fields"]}
    assert list(by_path) == ["invoice_number", "invoice_date", "currency", "status", "total", "line_items[].amount"]
    assert by_path["invoice_number"]["pattern"] == "^INV-\\d{4}$" and by_path["invoice_number"]["critical"]
    assert by_path["status"]["type"] == "text"  # enum without values
    assert "pattern" not in by_path["total"]  # regex that does not compile
    assert on_disk["doc_type"] == "invoice" and on_disk["dayfirst"] is True

    # a reviewed spec survives a rerun
    (out / "specs" / "invoice.yaml").write_text(yaml.safe_dump({**on_disk, "description": "edited"}))
    propose(unsorted, out, model)
    assert yaml.safe_load((out / "specs" / "invoice.yaml").read_text())["description"] == "edited"

    path = prelabel(unsorted, out, model, only=["invoice"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["doc_id"] for r in rows] == ["batch1/inv_0", "batch1/inv_1", "batch1/inv_2"]
    first = rows[0]
    assert first["doc_type"] == "invoice" and first["source"] == "batch1/inv_0.pdf"
    assert first["fields"]["invoice_number"] == "INV-0000" and first["fields"]["total"] == 1100.0
    assert first["fields"]["line_items"] == [{"amount": 500.0}, {"amount": 600.0}]
    assert "made_up_field" not in first["fields"]

    errors = [json.loads(line) for line in (out / "errors.jsonl").read_text().splitlines()]
    assert any(e["source"] == "batch1/inv_3.pdf" for e in errors)


def test_cache_makes_reruns_free(unsorted: Path, tmp_path: Path):
    model = ScriptedModel()
    cached = CachedModel(model, tmp_path / "cache")
    classify(unsorted, tmp_path / "a", cached, k=2, workers=1)
    first = len(model.calls)
    classify(unsorted, tmp_path / "b", cached, k=2, workers=1)
    assert len(model.calls) == first and cached.hits == first


def test_clean_spec_drops_leaf_parent_clash():
    spec = clean_spec(
        "x",
        ProposedSpec(
            description="",
            dayfirst=False,
            fields=[
                {"path": "vendor", "type": "text", "required": False, "critical": False, "description": ""},
                {"path": "vendor.name", "type": "text", "required": False, "critical": False, "description": ""},
            ],
        ),
    )
    assert [f["path"] for f in spec["fields"]] == ["vendor"]


def _labels_module():
    """The labelling tools (``labels template`` / ``import``), wherever they live now."""
    for name in ("ensemble.evaluation.labels", "evalkit.labels"):
        try:
            return __import__(name, fromlist=["DocTypeSpec"])
        except ImportError:
            continue
    pytest.skip("labelling tools not installed")


def test_specs_load_in_labelling_tools():
    labels = _labels_module()
    spec = clean_spec("invoice", ProposedSpec.model_validate(ScriptedModel().generate("", [], ProposedSpec)))
    loaded = labels.DocTypeSpec.from_dict(spec)
    assert loaded.field("line_items[3].amount")["type"] == "amount"
