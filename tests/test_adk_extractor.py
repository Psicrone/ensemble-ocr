"""The ADK extractor running the real ADK agent loop against scripted models."""

from __future__ import annotations

import asyncio

from conftest import tokens_for

from ensemble.schema import grounded_model
from ensemble.testing import scripted_llm
from ensemble.tool import ExtractorInput, RunContext, call_tool
from ensemble.tools.adk_extractor import GeminiAdkExtractor
from ensemble.tools.docai_ocr import OcrRegionText
from ensemble.types import Finding


def grounded_invoice(pages, number: str = "INV-0042") -> dict:
    leaf = lambda v, *words: {"value": v, "source_tokens": tokens_for(pages, *words) if words else [], "page": 0}  # noqa: E731
    return {
        "vendor": {"entity_number": leaf("4471-22", "4471-22"), "name": leaf("Acme Pty Ltd", "ACME", "PTY", "LTD")},
        "invoice_number": leaf(number, "INV-0042"),
        "invoice_date": leaf("2026-02-03", "03/02/2026"),
        "currency": leaf("ZAR", "ZAR"),
        "line_items": [
            {
                "description": leaf("Widget", "Widget"),
                "quantity": leaf(2),
                "unit_price": leaf(250.0),
                "amount": leaf(500.0),
            },
        ],
        "subtotal": leaf(1000.0, "1,000.00"),
        "tax": leaf(100.0),
        "total": leaf(1100.0, "1,100.00"),
    }


def make_ctx(pages, schema, round_=0) -> RunContext:
    return RunContext(doc_id="abc123", doc_type="invoice", pages=pages, schema_model=schema, round=round_,
                      llm_tools=[OcrRegionText()])  # fmt: skip


def test_reader_calls_tool_then_formatter_emits_grounded_json(invoice_pages, invoice_schema):
    reader = scripted_llm([
        ("call", "ocr_region_text", {"page": 0, "x0": 0, "y0": 0.4, "x1": 1, "y1": 0.6}),
        "total = 1100.00 | tokens: t40 | page: 0",
    ])  # fmt: skip
    formatter = scripted_llm([grounded_invoice(invoice_pages)])
    ex = GeminiAdkExtractor(reader_model=reader, formatter_model=formatter)
    ctx = make_ctx(invoice_pages, invoice_schema)

    out = asyncio.run(ex.run(ctx, ExtractorInput()))

    fields = {f.path: f for f in out.fields}
    assert fields["total"].value == 1100.0
    assert fields["total"].source_tokens == tokens_for(invoice_pages, "1,100.00")
    assert fields["line_items[0].description"].value == "Widget"
    assert out.notes and "total = 1100.00" in out.notes

    # The tool result went back to the reader, and the call was traced.
    second_request = reader.requests[1]
    responses = [p.function_response for c in second_request.contents for p in c.parts if p.function_response]
    assert responses and "token_ids" in responses[0].response
    assert ctx.trace.count("ocr.region_text") == 1
    assert ctx.trace.count("agent.tool_call") == 1

    # Reader saw the OCR listing with token ids and the field list; formatter saw only the draft.
    first_user_text = reader.requests[0].contents[-1].parts[-1].text
    assert "[t0] ACME" in first_user_text and "vendor.entity_number" in first_user_text
    assert "Draft:" in formatter.requests[0].config.system_instruction
    assert formatter.requests[0].config.temperature == 0
    assert formatter.requests[0].config.seed == 7


def test_feedback_round_lists_failing_fields_and_findings(invoice_pages, invoice_schema):
    reader = scripted_llm(["invoice_number = INV-0042 | tokens: t7 | page: 0"])
    formatter = scripted_llm([grounded_invoice(invoice_pages)])
    ex = GeminiAdkExtractor(reader_model=reader, formatter_model=formatter)
    first = asyncio.run(ex.run(make_ctx(invoice_pages, invoice_schema), ExtractorInput()))
    inp = ExtractorInput(
        current=first.fields,
        retry_paths=["invoice_number"],
        findings=[
            Finding(
                validator="docai.ocr_match",
                path="invoice_number",
                verdict="fail",
                message="Value not found in OCR text.",
                suggested_value="INV-0042",
            )
        ],  # fmt: skip
    )
    asyncio.run(ex.run(make_ctx(invoice_pages, invoice_schema, round_=1), inp))
    text = reader.requests[-1].contents[-1].parts[-1].text
    assert "REVIEW ROUND" in text
    assert "* invoice_number" in text and 'Suggested: "INV-0042"' in text


def test_tool_budget_stops_runaway_tool_use(invoice_pages, invoice_schema):
    call = ("call", "ocr_region_text", {"page": 0, "x0": 0, "y0": 0, "x1": 1, "y1": 1})
    reader = scripted_llm([call, call, call, "done"])
    formatter = scripted_llm([grounded_invoice(invoice_pages)])
    ex = GeminiAdkExtractor(reader_model=reader, formatter_model=formatter, max_tool_calls=2)
    ctx = make_ctx(invoice_pages, invoice_schema)
    asyncio.run(ex.run(ctx, ExtractorInput()))
    last = reader.requests[3]
    responses = [p.function_response.response for c in last.contents for p in c.parts if p.function_response]
    assert "budget" in str(responses[-1]).lower()
    spans = [s for s in ctx.trace.spans if s["name"] == "ocr.region_text"]
    assert [s["cache_hit"] for s in spans] == [False, True]  # identical repeat call served from cache


def test_extractor_is_cached_by_input(invoice_pages, invoice_schema):
    reader = scripted_llm(["x"])
    formatter = scripted_llm([grounded_invoice(invoice_pages)])
    ex = GeminiAdkExtractor(reader_model=reader, formatter_model=formatter)
    ctx = make_ctx(invoice_pages, invoice_schema)
    asyncio.run(call_tool(ex, ctx, ExtractorInput()))
    asyncio.run(call_tool(ex, ctx, ExtractorInput()))
    assert len(formatter.requests) == 1


def test_grounded_model_wraps_every_leaf(invoice_schema):
    schema = grounded_model(invoice_schema).model_json_schema()
    assert "source_tokens" in str(schema)
    parsed = grounded_model(invoice_schema).model_validate(
        {
            "vendor": {"entity_number": {"value": "1", "source_tokens": ["t1"], "page": 0}},
            "invoice_number": {"value": "2"},
            "line_items": [{"amount": {"value": 3.0}}],
        }  # fmt: skip
    )
    assert parsed.line_items[0].amount.value == 3.0
