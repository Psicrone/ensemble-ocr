from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ensemble.spec import model_from_spec  # noqa: E402
from ensemble.types import BBox, Cell, ClassifierResult, PageContext, Table, Token  # noqa: E402

INVOICE_LINES = [
    "ACME PTY LTD",
    "Entity No 4471-22",
    "Invoice INV-0042",
    "Date 03/02/2026",
    "Currency ZAR",
    "Widget 2 250.00 500.00",
    "Gadget 1 500.00 500.00",
    "Subtotal 1,000.00",
    "Tax 100.00",
    "Total 1,100.00",
]


def page_context(lines: list[str], tables: list[Table] | None = None, label: str | None = "invoice") -> PageContext:
    """One page; each line is a row of tokens with simple boxes."""
    ctx = PageContext(text="\n".join(lines))
    page = ctx.page(0)
    n = 0
    for row, line in enumerate(lines):
        for col, word in enumerate(line.split()):
            y = 0.05 + row * 0.05
            page.tokens.append(
                Token(
                    id=f"t{n}",
                    text=word,
                    bbox=BBox(page=0, x0=0.05 + col * 0.1, y0=y, x1=0.14 + col * 0.1, y1=y + 0.03),
                )
            )
            n += 1
    page.tables.extend(tables or [])
    if label:
        ctx.classification = ClassifierResult(label=label, confidence=0.97, scores={label: 0.97})
    return ctx


def invoice_table() -> Table:
    rows = [["Item", "Qty", "Unit", "Amount"], ["Widget", "2", "250.00", "500.00"], ["Gadget", "1", "500.00", "500.00"]]
    return Table(page=0, rows=[[Cell(text=c) for c in r] for r in rows], header_rows=1)


def tokens_for(ctx: PageContext, *words: str) -> list[str]:
    """Token ids of the first run of ``words`` in reading order."""
    toks = ctx.tokens
    for i in range(len(toks)):
        if [t.text for t in toks[i : i + len(words)]] == list(words):
            return [t.id for t in toks[i : i + len(words)]]
    raise AssertionError(f"{words} not in fixture")


@pytest.fixture
def invoice_pages() -> PageContext:
    return page_context(INVOICE_LINES, [invoice_table()])


@pytest.fixture
def invoice_schema():
    return model_from_spec(ROOT / "specs/invoices.yaml", "invoice")


@pytest.fixture
def png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(buf, format="PNG")
    return buf.getvalue()
