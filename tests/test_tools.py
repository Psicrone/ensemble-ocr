"""Unit tests for Document AI normalisation, matching, rules, registry and entry point."""

from __future__ import annotations

import asyncio
import functools

import pytest
from conftest import ROOT, page_context
from google.cloud import documentai

from ensemble import extract
from ensemble.ingest import ingest, sniff_mime
from ensemble.matching import matches, parse_number
from ensemble.registry import PipelineConfig, expand_env
from ensemble.schema import covers, field_specs
from ensemble.testing import FakeExtractor, StaticPreprocessor
from ensemble.tool import FunctionTool, RunContext, ValidatorInput, call_tool
from ensemble.tools._docai import parse_classification, parse_layout, parse_ocr
from ensemble.tools.docai_ocr import DocAIOcr
from ensemble.tools.entity_registry import EntityLookup, EntityQuery, InMemoryEntityBackend
from ensemble.tools.rules_cross_field import CrossFieldRules
from ensemble.tools.rules_format import FormatRules
from ensemble.types import FieldValue, PageContext, ValidatorOutput

D = documentai.Document


def _anchor(start: int, end: int) -> D.TextAnchor:
    return D.TextAnchor(text_segments=[D.TextAnchor.TextSegment(start_index=start, end_index=end)])


def _poly(x0, y0, x1, y1) -> documentai.BoundingPoly:
    return documentai.BoundingPoly(normalized_vertices=[
        documentai.NormalizedVertex(x=x0, y=y0), documentai.NormalizedVertex(x=x1, y=y0),
        documentai.NormalizedVertex(x=x1, y=y1), documentai.NormalizedVertex(x=x0, y=y1),
    ])  # fmt: skip


def ocr_doc() -> D:
    text = "Total 1,100.00\n"
    tokens = [
        D.Page.Token(
            layout=D.Page.Layout(text_anchor=_anchor(0, 6), bounding_poly=_poly(0.1, 0.8, 0.2, 0.82), confidence=0.99)
        ),
        D.Page.Token(
            layout=D.Page.Layout(text_anchor=_anchor(6, 15), bounding_poly=_poly(0.3, 0.8, 0.4, 0.82), confidence=0.98)
        ),
    ]
    return D(
        text=text, pages=[D.Page(page_number=1, tokens=tokens, dimension=D.Page.Dimension(width=850, height=1100))]
    )


def test_parse_ocr_assigns_token_ids_and_boxes():
    ctx = parse_ocr(ocr_doc())
    assert [(t.id, t.text) for t in ctx.tokens] == [("t0", "Total"), ("t1", "1,100.00")]
    assert ctx.tokens[1].bbox.x0 == pytest.approx(0.3) and ctx.tokens[1].bbox.page == 0
    assert ctx.pages[0].width == 850


def test_parse_layout_tables_and_blocks():
    L = D.DocumentLayout.DocumentLayoutBlock
    cell = lambda t: L.LayoutTableCell(blocks=[L(text_block=L.LayoutTextBlock(text=t))])  # noqa: E731
    row = lambda *ts: L.LayoutTableRow(cells=[cell(t) for t in ts])  # noqa: E731
    doc = D(document_layout=D.DocumentLayout(blocks=[
        L(text_block=L.LayoutTextBlock(text="Invoice", type_="heading-1"), page_span=L.LayoutPageSpan(page_start=1)),
        L(table_block=L.LayoutTableBlock(header_rows=[row("Item", "Amount")], body_rows=[row("Widget", "500.00")]),
          page_span=L.LayoutPageSpan(page_start=2)),
    ]))  # fmt: skip
    ctx = parse_layout(doc)
    assert ctx.pages[0].blocks[0].kind == "heading-1"
    table = ctx.pages[1].tables[0]
    assert table.header_rows == 1 and [c.text for c in table.body[0]] == ["Widget", "500.00"]


def test_parse_classification_picks_top_label():
    doc = D(entities=[D.Entity(type_="invoice", confidence=0.9), D.Entity(type_="receipt", confidence=0.1)])
    result = parse_classification(doc)
    assert (result.label, result.confidence) == ("invoice", pytest.approx(0.9))


def test_docai_preprocessor_with_fake_client(png_bytes):
    class FakeClient:
        def process_document(self, request):
            assert request.name == "projects/p/locations/eu/processors/abc"
            return documentai.ProcessResponse(document=ocr_doc())

    tool = DocAIOcr(processor_id="abc", project="p", location="eu", client=FakeClient())
    _, pages = ingest(png_bytes)
    ctx = RunContext(doc_id="d", doc_type="x", pages=pages)
    out = asyncio.run(tool.run(ctx, tool.Input()))
    assert out.tokens[1].text == "1,100.00"


@pytest.mark.parametrize("token,value", [("1,234.50", 1234.5), ("1.234,50", 1234.5), ("12,50", 12.5),
                                         ("1,234", 1234.0), ("(12.00)", -12.0), ("1'000.00", 1000.0)])  # fmt: skip
def test_parse_number(token, value):
    assert parse_number(token) == value


def test_matching_is_exact_for_ids_and_fuzzy_for_names():
    assert matches("INV-0042", "Invoice INV-0042")[0]
    assert not matches("INV-0024", "INV-0042")[0]
    assert matches("Acme Pty Ltd", "ACME PTY. LTD.")[0]
    assert matches("Acme Pty Limited", "ACME PTY LTD", threshold=0.7)[0]
    assert matches("2026-02-03", "Date 03/02/2026")[0]
    assert matches(1100.0, "Total 1,100.00")[0]
    assert not matches(1010.0, "Total 1,100.00")[0]


def test_format_rules(invoice_schema):
    ctx = RunContext(doc_id="d", doc_type="invoice", pages=PageContext(), schema_model=invoice_schema)
    fields = [FieldValue(path="vendor.entity_number", value="44-1"), FieldValue(path="currency", value="ZAR"),
              FieldValue(path="invoice_date", value="2026-02-31"), FieldValue(path="total", value="12,5x")]  # fmt: skip
    out = asyncio.run(FormatRules().run(ctx, ValidatorInput(fields=fields)))
    verdicts = {f.path: f.verdict for f in out.findings}
    assert verdicts == {"vendor.entity_number": "fail", "currency": "pass", "invoice_date": "fail", "total": "fail"}


def test_cross_field_rules():
    rules = CrossFieldRules(rules=[{"sum": "line_items[].amount", "equals": "subtotal"},
                                   {"product": ["subtotal", "rate"], "equals": "tax"}])  # fmt: skip
    output = {"line_items": [{"amount": 10}, {"amount": 5}], "subtotal": 16, "rate": 0.1, "tax": 1.6}
    ctx = RunContext(doc_id="d", doc_type="x", pages=PageContext())
    out = asyncio.run(rules.run(ctx, ValidatorInput(fields=[], output=output)))
    by = {f.path: f for f in out.findings}
    assert by["subtotal"].verdict == "fail" and by["subtotal"].suggested_value == 15
    assert by["tax"].verdict == "pass"


def test_spec_becomes_schema_with_x_keys(invoice_schema):
    specs = field_specs(invoice_schema)
    assert specs["vendor.entity_number"] == {"x-critical": True, "x-format": r"^\d{4}-\d{2}$"}
    assert specs["currency"] == {"x-enum": ["ZAR", "USD", "EUR"]}
    assert invoice_schema.model_validate({"invoice_date": "2026-01-02"}).invoice_date.year == 2026


def test_x_validators_override_tool_globs():
    from pydantic import BaseModel, Field

    class Doc(BaseModel):
        entity_number: str | None = Field(None, json_schema_extra={"x-validators": ["api.entity_registry"]})
        amount: float | None = None

    class T:
        name = "docai.ocr_match"
        fields = ["*"]

    assert not covers(T(), "entity_number", Doc)  # the field lists its own validators
    assert covers(T(), "amount", Doc)


def test_function_tool_and_entity_lookup():
    backend = InMemoryEntityBackend([{"number": "4471-22", "name": "Acme"}, {"number": "9000-01", "name": "Other"}])
    lookup = EntityLookup(backend=backend)
    ctx = RunContext(doc_id="d", doc_type="x", pages=PageContext())
    out = asyncio.run(lookup.run(ctx, EntityQuery(name="acme")))
    assert out.candidates[0].number == "4471-22"

    def no_po_box(ctx, inp):
        return ValidatorOutput(findings=[])

    tool = FunctionTool("rules.no_po_box", no_po_box, ValidatorInput, ValidatorOutput, kind="validator")
    assert asyncio.run(call_tool(tool, ctx, ValidatorInput(fields=[]))).findings == []


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("ENTITY_API", "https://registry.example")
    assert expand_env({"a": "${ENTITY_API}/v1", "b": ["${MISSING:-x}"]}) == {
        "a": "https://registry.example/v1",
        "b": ["x"],
    }


def test_shipped_profiles_load(monkeypatch):
    monkeypatch.setenv("ENTITY_API", "https://registry.example")
    monkeypatch.setenv("GCP_PROJECT", "proj")
    cfg = PipelineConfig.load(ROOT / "profiles")
    assert set(cfg.profiles) == {"invoice", "generic"}
    invoice = cfg.profiles["invoice"]
    assert [v.name for v in invoice.validators][:3] == ["docai.ocr_match", "docai.layout_table", "docai.classifier"]
    assert [t.name for t in invoice.llm_tools] == ["ocr.region_text", "layout.table", "image.zoom", "lookup.entity"]
    assert invoice.extractors[0].max_tool_calls == 8
    assert [p.name for p in cfg.preprocessors] == ["docai.ocr", "docai.layout", "docai.classify"]


def test_extract_entry_point_with_partial(png_bytes, invoice_schema):
    pages = page_context(["Invoice INV-7"], label=None)
    cfg = PipelineConfig.load(
        {"doc_types": {"invoice": {"schema": invoice_schema, "extractors": ["fx"], "validators": ["docai.ocr_match"]}}},
        instances={"fx": FakeExtractor([{"invoice_number": ("INV-7", ["t1"])}]), "ocr": StaticPreprocessor(pages)},
    )
    cfg.preprocessors.append(StaticPreprocessor(pages))
    run = functools.partial(extract, config=cfg)
    result = asyncio.run(run(png_bytes, doc_type="invoice"))
    assert result.output["invoice_number"] == "INV-7"
    assert result.doc_id == ingest(png_bytes)[0]


def test_sniff_and_render(png_bytes):
    assert sniff_mime(png_bytes) == "image/png"
    _, pages = ingest(png_bytes)
    assert pages.pages[0].image is not None
