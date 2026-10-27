import asyncio
import json
from pathlib import Path

from ensemble.evaluation.adapter import extraction_result_adapter
from ensemble.evaluation.score import score_pipeline, score_results
from ensemble.evaluation.specs import load_specs

SPECS = Path(__file__).parents[2] / "specs"


def _result(doc_id, output, conf):
    return {"doc_id": doc_id, "doc_type": "invoice", "schema_version": "invoice.v1", "rounds": 1,
            "trace_id": "t", "findings": [], "output": output,
            "fields": [{"path": p, "value": None, "confidence": c,
                        "status": "accepted" if c >= .85 else "needs_review" if c >= .6 else "failed"}
                       for p, c in conf.items()]}


def test_adapter_reads_extraction_result():
    rec = _result("h", {"total": 5, "line_items": [{"amount": 5}]}, {"total": .9, "$.line_items[0].amount": .4})
    values, meta = extraction_result_adapter(rec)
    assert values == {"total": 5, "line_items": [{"amount": 5}]}
    assert meta["line_items[0].amount"] == {"confidence": .4, "status": "failed"}


GOLD = [{"doc_id": "invoice/a", "doc_type": "invoice", "source": "invoice/a.pdf",
         "fields": {"invoice_number": "INV-1", "invoice_date": "2024-05-08", "total": 1150.0,
                    "vendor": {"entity_number": "4471-22"},
                    "line_items": [{"description": "Safe", "amount": 1000.0}, {"description": "Delivery", "amount": 150.0}]}},
        {"doc_id": "invoice/b", "doc_type": "invoice", "source": "invoice/b.pdf",
         "fields": {"invoice_number": "INV-2", "invoice_date": "2024-06-01", "total": 99.0,
                    "vendor": {"entity_number": "3310-05"}, "line_items": []}}]


async def fake_extract(doc_bytes, doc_type=None):
    """Gets a's rows in the wrong order and b's total wrong, with low confidence on it."""
    assert doc_type == "invoice"
    if doc_bytes == b"a":
        out = {"invoice_number": "inv 1", "invoice_date": "08/05/2024", "total": "R 1 150,00",
               "vendor": {"entity_number": "4471 - 22"},
               "line_items": [{"description": "Delivery", "amount": "150"}, {"description": "SAFE", "amount": 1000}]}
        conf = {"invoice_number": .95, "total": .97, "line_items[0].amount": .9}
    else:
        out = {"invoice_number": "INV-2", "invoice_date": "2024-06-01", "total": 9.9,
               "vendor": {"entity_number": "3310-05"}, "line_items": []}
        conf = {"invoice_number": .95, "total": .3}
    return _result("sha256-of-" + doc_bytes.decode(), out, conf)


def test_score_pipeline_end_to_end(tmp_path):
    docs = {"invoice/a": b"a", "invoice/b": b"b"}
    path, res = asyncio.run(score_pipeline(fake_extract, docs, GOLD, specs=SPECS, runs=2, out=tmp_path))
    s = res.summary
    # b's total is wrong in both runs; everything else matches after normalising
    assert s["fields"]["incorrect"] == 2 and s["fields"]["missing"] == 0
    assert s["record"]["critical_pass"]["mean"] == 0.5          # b's total is critical
    assert s["status"]["accepted_precision"] == 1.0 and s["status"]["errors_accepted"] == 0
    assert s["stability"]["field_agreement"] == 1.0
    assert Path(path).exists()


def test_score_results_from_file(tmp_path):
    results = [{**asyncio.run(fake_extract(k.encode(), "invoice")), "doc_id": f"invoice/{k}"} for k in "ab"]
    (tmp_path / "r.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (tmp_path / "g.jsonl").write_text("".join(json.dumps(g) + "\n" for g in GOLD))
    _, res = score_results(tmp_path / "r.jsonl", tmp_path / "g.jsonl", specs=load_specs(SPECS), out=tmp_path / "o")
    assert res.summary["fields"]["incorrect"] == 1
