"""End-to-end orchestrator runs with fake extractors and the real validators."""

from __future__ import annotations

import asyncio

from conftest import ROOT, tokens_for

from ensemble import Orchestrator, PipelineConfig
from ensemble.sinks import MemorySink
from ensemble.testing import FakeExtractor, FakeValidator, StaticPreprocessor
from ensemble.tools.entity_registry import InMemoryEntityBackend

REGISTRY = InMemoryEntityBackend([{"number": "4471-22", "name": "Acme Pty Ltd", "status": "active"}])


def good_values(pages) -> dict:
    t = lambda *w: tokens_for(pages, *w)  # noqa: E731
    return {
        "vendor.entity_number": ("4471-22", t("4471-22")),
        "vendor.name": ("Acme Pty Ltd", t("ACME", "PTY", "LTD")),
        "invoice_number": ("INV-0042", t("INV-0042")),
        "invoice_date": ("2026-02-03", t("03/02/2026")),
        "currency": ("ZAR", t("ZAR")),
        "line_items[0].description": ("Widget", t("Widget")),
        "line_items[0].quantity": (2.0, t("Widget", "2")[1:]),
        "line_items[0].unit_price": (250.0, t("Widget", "2", "250.00")[2:]),
        "line_items[0].amount": (500.0, t("Widget", "2", "250.00", "500.00")[3:]),
        "line_items[1].description": ("Gadget", t("Gadget")),
        "line_items[1].quantity": (1.0, t("Gadget", "1")[1:]),
        "line_items[1].unit_price": (500.0, t("Gadget", "1", "500.00")[2:]),
        "line_items[1].amount": (500.0, t("Gadget", "1", "500.00", "500.00")[3:]),
        "subtotal": (1000.0, t("1,000.00")),
        "tax": (100.0, t("Tax", "100.00")[1:]),
        "total": (1100.0, t("1,100.00")),
    }


def config(pages, extractor, validators=None) -> PipelineConfig:
    spec = {
        "preprocessors": ["fake.ocr"],
        "doc_types": {
            "invoice": {
                "spec": "specs/invoices.yaml",
                "extractors": ["fake.extract"],
                "validators": validators
                or [
                    "docai.ocr_match",
                    "docai.layout_table",
                    "docai.classifier",
                    "rules.format",
                    {
                        "rules.cross_field": {
                            "rules": [
                                {"sum": "line_items[].amount", "equals": "subtotal"},
                                {"sum": ["subtotal", "tax"], "equals": "total"},
                            ]
                        }
                    },
                    {"api.entity_registry": {"backend": REGISTRY, "name_field": "name"}},
                ],
                "weights": {"docai.ocr_match": 2, "api.entity_registry": 2},
                "max_rounds": 2,
            }
        },
    }
    instances = {"fake.ocr": StaticPreprocessor(pages), "fake.extract": extractor}
    import os

    cwd = os.getcwd()
    os.chdir(ROOT)  # schema path in the dict config is relative to the cwd
    try:
        return PipelineConfig.load(spec, instances=instances)
    finally:
        os.chdir(cwd)


def run(cfg, png, doc_type="invoice", sink=None):
    return asyncio.run(Orchestrator(cfg, render=False, sink=sink).run(png, doc_type=doc_type))


def by_path(result):
    return {f.path: f for f in result.fields}


def test_clean_document_is_accepted_in_one_round(invoice_pages, png_bytes):
    ex = FakeExtractor([good_values(invoice_pages)])
    result = run(config(invoice_pages, ex), png_bytes)
    fields = by_path(result)
    assert result.rounds == 1
    assert {f.status for f in fields.values()} == {"accepted"}, {p: (f.status, f.confidence) for p, f in fields.items()}
    assert result.output["total"] == 1100.0
    assert result.output["line_items"][1]["description"] == "Gadget"
    assert fields["total"].evidence and fields["total"].evidence[0].bbox is not None
    assert len(ex.calls) == 1


def test_misread_value_is_retried_and_others_are_frozen(invoice_pages, png_bytes):
    first = good_values(invoice_pages)
    first["invoice_number"] = ("INV-0024", first["invoice_number"][1])
    second = dict(first, invoice_number=("INV-0042", first["invoice_number"][1]), currency=("USD", []))
    ex = FakeExtractor([first, second])
    result = run(config(invoice_pages, ex), png_bytes)
    fields = by_path(result)

    assert result.rounds == 2
    assert ex.calls[1].retry_paths == ["invoice_number"]
    assert any("INV-0024" in str(f.message) or f.validator == "docai.ocr_match" for f in ex.calls[1].findings)
    assert fields["invoice_number"].value == "INV-0042"
    assert fields["invoice_number"].status == "accepted"
    assert fields["currency"].value == "ZAR"  # frozen: not in retry paths


def test_suggestion_grounded_by_cited_tokens_is_corrected(invoice_pages, png_bytes):
    values = good_values(invoice_pages)
    values["subtotal"] = (1010.0, values["subtotal"][1])  # cited the right token, misread the digits
    result = run(config(invoice_pages, FakeExtractor([values])), png_bytes)
    fields = by_path(result)
    # line items sum to 1000 and the cited token says 1,000.00, so the suggestion is grounded.
    assert fields["subtotal"].status == "corrected"
    assert fields["subtotal"].value == 1000.0
    assert result.output["subtotal"] == 1000.0
    # total was flagged by the wrong subtotal in round 1, and passes once it is corrected.
    assert fields["total"].status == "accepted"


def test_ungrounded_suggestion_is_not_applied(invoice_pages, png_bytes):
    values = good_values(invoice_pages)
    values["total"] = (1010.0, tokens_for(invoice_pages, "Total"))  # cited the label, not the amount
    ex = FakeExtractor([values])
    result = run(config(invoice_pages, ex), png_bytes)
    total = by_path(result)["total"]
    assert total.status == "needs_review"  # failed, then no change in the retry: stop and flag
    assert total.value == 1010.0


def test_worse_retry_keeps_earlier_value(invoice_pages, png_bytes):
    first = good_values(invoice_pages)
    first["invoice_number"] = ("INV-0024", first["invoice_number"][1])
    second = dict(first, invoice_number=("XXX", []))
    result = run(config(invoice_pages, FakeExtractor([first, second])), png_bytes)
    field = by_path(result)["invoice_number"]
    assert field.value == "INV-0024"
    assert field.status == "needs_review"
    assert result.rounds == 2


def test_dropped_line_item_retries_the_whole_list(invoice_pages, png_bytes):
    first = {p: v for p, v in good_values(invoice_pages).items() if not p.startswith("line_items[1]")}
    ex = FakeExtractor([first, good_values(invoice_pages)])
    result = run(config(invoice_pages, ex), png_bytes)
    fields = by_path(result)
    assert "line_items" in ex.calls[1].retry_paths
    assert fields["line_items[1].description"].value == "Gadget"
    assert fields["subtotal"].value == 1000.0  # not "corrected" to the 500 the short list summed to
    assert fields["subtotal"].status == "accepted"
    assert len(result.output["line_items"]) == 2


def test_critical_field_is_not_auto_corrected_but_retried_with_suggestions(invoice_pages, png_bytes):
    first = good_values(invoice_pages)
    first["vendor.entity_number"] = ("4471-23", first["vendor.entity_number"][1])
    ex = FakeExtractor([first, good_values(invoice_pages)])
    result = run(config(invoice_pages, ex), png_bytes)
    assert ex.calls[1].retry_paths == ["vendor.entity_number"]
    suggestions = {f.validator: f.suggested_value for f in ex.calls[1].findings}
    assert suggestions == {"docai.ocr_match": "4471-22", "api.entity_registry": "4471-22"}
    field = by_path(result)["vendor.entity_number"]
    assert (field.value, field.status) == ("4471-22", "accepted")


def test_validator_crash_is_unknown_not_fatal(invoice_pages, png_bytes):
    class Boom(FakeValidator):
        async def run(self, ctx, inp):
            raise RuntimeError("backend down")

    ex = FakeExtractor([good_values(invoice_pages)])
    cfg = config(invoice_pages, ex, validators=["docai.ocr_match"])
    cfg.profiles["invoice"].validators.append(Boom(name="boom"))
    result = run(cfg, png_bytes)
    boom = [f for f in result.findings if f.validator == "boom"]
    assert boom and {f.verdict for f in boom} == {"unknown"}
    assert by_path(result)["total"].status == "accepted"


def test_routes_by_classifier_and_saves_to_sink(invoice_pages, png_bytes):
    sink = MemorySink()
    result = run(
        config(invoice_pages, FakeExtractor([good_values(invoice_pages)])), png_bytes, doc_type=None, sink=sink
    )
    assert result.doc_type == "invoice"
    assert sink.saved and sink.saved[0][0].trace_id == result.trace_id
    names = {s["name"] for s in sink.saved[0][1]}
    assert {"ingest", "fake.ocr", "fake.extract", "docai.ocr_match"} <= names


def test_eval_record_shape(invoice_pages, png_bytes):
    result = run(config(invoice_pages, FakeExtractor([good_values(invoice_pages)])), png_bytes)
    record = result.to_eval_record()
    assert set(record) == {"doc_id", "doc_type", "fields", "field_confidence", "validation"}
    assert record["fields"]["line_items"][0]["amount"] == 500.0
    assert record["field_confidence"]["line_items[0].amount"] == 1.0
