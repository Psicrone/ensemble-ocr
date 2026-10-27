import csv
import json

import pytest
from evalkit.structured import Schema

from ensemble.evaluation.cli import main
from ensemble.evaluation.labels import check_labels, documents_from_gold, import_labels, scan_documents, write_templates
from ensemble.evaluation.score import eval_schemas
from ensemble.evaluation.specs import DocTypeSpec, load_specs

SPEC = {"doc_type": "receipt", "version": 2, "fields": [
    {"path": "number", "type": "id", "required": True, "critical": True},
    {"path": "date", "type": "date", "required": True},
    {"path": "shop.name", "type": "text", "threshold": 0.8},
    {"path": "currency", "type": "enum", "values": ["ZAR", "USD"]},
    {"path": "total", "type": "amount", "critical": True},
    {"path": "items[].name", "type": "text"},
    {"path": "items[].price", "type": "amount"}]}


def test_spec_derives_schema_and_json_schema():
    spec = DocTypeSpec.from_dict(SPEC)
    s = spec.eval_schema()
    assert s.spec("shop.name", "a", "a")["threshold"] == 0.8 and s.is_critical("total")
    js = spec.json_schema()
    item = js["properties"]["items"]["items"]["properties"]["price"]
    assert item["type"] == ["number", "null"] and js["properties"]["number"]["x-critical"]
    assert js["properties"]["currency"]["enum"] == ["ZAR", "USD", None]
    # the JSON Schema round-trips into the same comparators
    back = Schema(js)
    assert back.fields["items[].price"]["type"] == "amount" and back.is_critical("number")
    assert spec.blank() == {"number": None, "date": None, "shop": {"name": None}, "currency": None,
                            "total": None, "items": []}


def test_spec_rejects_typos():
    with pytest.raises(ValueError, match="unknown type"):
        DocTypeSpec.from_dict({"doc_type": "x", "fields": [{"path": "a", "type": "amonut"}]})
    with pytest.raises(ValueError, match="unknown keys"):
        DocTypeSpec.from_dict({"doc_type": "x", "fields": [{"path": "a", "critcal": True}]})


def test_template_prefill_import_check_roundtrip(tmp_path):
    docs = tmp_path / "docs"
    (docs / "receipt").mkdir(parents=True)
    for n in ("r1.pdf", "r2.jpg"):
        (docs / "receipt" / n).write_bytes(b"x")
    (docs / "notes.txt").write_text("ignored")
    found = scan_documents(docs)
    assert [d["doc_id"] for d in found] == ["receipt/r1", "receipt/r2"]

    specs = {"receipt": DocTypeSpec.from_dict(SPEC)}
    pre = [{"doc_id": "receipt/r1", "doc_type": "receipt",
            "fields": {"number": "R-1", "date": "08/05/2024", "total": "R 1 150,00", "currency": "GBP",
                       "items": [{"name": "tea", "price": "12,50"}, {"name": "cake", "price": "30"}]}}]
    out = write_templates(found, specs, tmp_path / "labels", prefill=pre, max_items=1)
    rows = list(csv.DictReader(open(out[0], encoding="utf-8")))
    assert rows[0]["items[1].name"] == "cake"           # columns grow to fit the prefill
    assert rows[1]["number"] == "" and rows[1]["source"] == "receipt/r2.jpg"

    gold = import_labels(out, specs)
    r1 = gold[0]["fields"]
    assert r1["date"] == "2024-05-08" and r1["total"] == 1150.0 and r1["items"][0]["price"] == 12.5
    assert gold[1]["fields"]["items"] == []               # empty table rows are dropped
    problems = {(i["doc_id"], i["field"], i["problem"].split(":")[0]) for i in check_labels(gold, specs)}
    assert ("receipt/r1", "currency", "'GBP' is not one of ['ZAR', 'USD']") in problems
    assert ("receipt/r2", "number", "required field is empty") in problems
    assert documents_from_gold(gold, docs)["receipt/r2"] == docs / "receipt" / "r2.jpg"


def test_specs_drive_per_doc_type_scoring(tmp_path):
    (tmp_path / "s.yaml").write_text(json.dumps({"doc_types": {
        "a": {"fields": [{"path": "x", "type": "amount"}]},
        "b": {"fields": [{"path": "x", "type": "text"}]}}}))
    schemas = eval_schemas(tmp_path / "s.yaml")
    assert schemas["a"].score("x", "1,00", "1")[1] and not schemas["b"].score("x", "1,00", "1")[1]
    assert set(load_specs(tmp_path)) == {"a", "b"}


def test_cli_init_and_json_schema(tmp_path, capsys):
    main(["init", "receipt", "--out", str(tmp_path / "specs")])
    main(["json-schema", "--specs", str(tmp_path / "specs"), "--out", str(tmp_path / "schemas")])
    js = json.loads((tmp_path / "schemas" / "receipt.v1.json").read_text())
    assert js["properties"]["total"]["x-critical"] is True
