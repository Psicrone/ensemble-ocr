"""Shared Document AI client and response normalisation.

Everything downstream reads ``PageContext``; only this module touches the
Document AI protobuf types.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..types import BBox, Block, Cell, ClassifierResult, Page, PageContext, Table, Token


class DocumentAIClient:
    """Thin async wrapper around ``DocumentProcessorServiceClient.process_document``.

    Pass ``client=`` to inject a fake in tests.
    """

    def __init__(self, project: str | None = None, location: str = "us", client: Any = None) -> None:
        self.project = project
        self.location = location
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from google.api_core.client_options import ClientOptions
            from google.cloud import documentai

            opts = ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
            self._client = documentai.DocumentProcessorServiceClient(client_options=opts)
        return self._client

    def processor_name(self, processor_id: str, version: str | None = None) -> str:
        if processor_id.startswith("projects/"):
            return processor_id
        base = f"projects/{self.project}/locations/{self.location}/processors/{processor_id}"
        return f"{base}/processorVersions/{version}" if version else base

    async def process(self, content: bytes, mime_type: str, processor_id: str, version: str | None = None) -> Any:
        from google.cloud import documentai

        request = documentai.ProcessRequest(
            name=self.processor_name(processor_id, version),
            raw_document=documentai.RawDocument(content=content, mime_type=mime_type),
        )
        response = await asyncio.to_thread(self.client.process_document, request=request)
        return response.document


def _anchor_text(full_text: str, layout: Any) -> str:
    return "".join(full_text[int(s.start_index) : int(s.end_index)] for s in layout.text_anchor.text_segments)


def _bbox(page: int, poly: Any) -> BBox | None:
    vertices = list(poly.normalized_vertices)
    if not vertices:
        return None
    xs, ys = [v.x for v in vertices], [v.y for v in vertices]
    return BBox(page=page, x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))


def parse_ocr(doc: Any) -> PageContext:
    """Enterprise OCR response -> tokens with ids ``t0, t1, ...`` in reading order."""
    ctx = PageContext(text=doc.text)
    n = 0
    for i, page in enumerate(doc.pages):
        out = ctx.page(i)
        out.width = int(page.dimension.width) or None
        out.height = int(page.dimension.height) or None
        for token in page.tokens:
            out.tokens.append(
                Token(
                    id=f"t{n}",
                    text=_anchor_text(doc.text, token.layout).strip(),
                    bbox=_bbox(i, token.layout.bounding_poly),
                    confidence=token.layout.confidence or None,
                )
            )
            n += 1
    return ctx


def parse_layout(doc: Any) -> PageContext:
    """Layout Parser response -> blocks and tables."""
    ctx = PageContext()
    for block in doc.document_layout.blocks:
        _walk(block, ctx)
    return ctx


def _page_of(block: Any) -> int:
    start = block.page_span.page_start
    return start - 1 if start else 0  # page_span is 1-based


def _text(block: Any) -> str:
    if "text_block" in block:
        parts = [block.text_block.text, *(_text(b) for b in block.text_block.blocks)]
        return " ".join(p for p in parts if p)
    if "table_block" in block:
        return " ".join(c.text for row in _rows(block) for c in row)
    if "list_block" in block:
        return " ".join(_text(b) for e in block.list_block.list_entries for b in e.blocks)
    return ""


def _rows(block: Any) -> list[list[Cell]]:
    table = block.table_block
    return [
        [Cell(text=" ".join(_text(b) for b in cell.blocks).strip()) for cell in row.cells]
        for row in [*table.header_rows, *table.body_rows]
    ]


def _walk(block: Any, ctx: PageContext) -> None:
    page = _page_of(block)
    bbox = _bbox(page, block.bounding_box) if "bounding_box" in block else None
    if "table_block" in block:
        ctx.page(page).tables.append(
            Table(page=page, rows=_rows(block), header_rows=len(block.table_block.header_rows), bbox=bbox)
        )
    elif "list_block" in block:
        ctx.page(page).blocks.append(Block(kind="list", text=_text(block), page=page, bbox=bbox))
    elif "text_block" in block:
        tb = block.text_block
        ctx.page(page).blocks.append(Block(kind=tb.type_ or "text", text=tb.text, page=page, bbox=bbox))
        for child in tb.blocks:
            _walk(child, ctx)


def parse_classification(doc: Any) -> ClassifierResult | None:
    scores: dict[str, float] = {}
    for entity in doc.entities:
        if entity.type_:
            scores[entity.type_] = max(scores.get(entity.type_, 0.0), float(entity.confidence))
    if not scores:
        return None
    label = max(sorted(scores), key=scores.__getitem__)
    return ClassifierResult(label=label, confidence=scores[label], scores=scores)


__all__ = ["DocumentAIClient", "Page", "parse_classification", "parse_layout", "parse_ocr"]
