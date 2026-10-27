"""Shared Pydantic types: the contract between every plug-in and the harness.

Coordinates are normalised to 0..1 so they survive re-rendering. Page indices
are 0-based.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from . import paths

Status = Literal["accepted", "corrected", "failed", "needs_review"]


class BBox(BaseModel):
    page: int
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, other: BBox, slack: float = 0.0) -> bool:
        return (
            self.page == other.page
            and other.x0 >= self.x0 - slack
            and other.y0 >= self.y0 - slack
            and other.x1 <= self.x1 + slack
            and other.y1 <= self.y1 + slack
        )


class Evidence(BaseModel):
    source: str  # tool name, e.g. "docai.ocr", "api.entity_registry"
    text: str | None = None  # OCR text or API-returned value
    bbox: BBox | None = None
    score: float | None = None  # the source's own confidence, 0..1
    raw_ref: str | None = None  # pointer into the trace, not the payload


class FieldValue(BaseModel):
    path: str  # e.g. "lines[2].amount"
    value: Any
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = 0.0  # set by Reconcile
    status: Status = "needs_review"
    source_tokens: list[str] = Field(default_factory=list)  # OCR token ids the model cited
    extractor: str | None = None
    round: int = 0


class Severity(str, Enum):
    info = "info"
    warn = "warn"
    error = "error"


class Finding(BaseModel):
    validator: str
    path: str  # the field it is about ("" for document-level)
    verdict: Literal["pass", "fail", "unknown"]
    severity: Severity = Severity.error
    message: str  # one line, written for the model to read
    suggested_value: Any | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    doc_id: str  # content hash
    doc_type: str
    schema_version: str
    fields: list[FieldValue]
    findings: list[Finding]
    rounds: int
    trace_id: str
    output: dict[str, Any]  # final JSON in schema shape, built from fields

    def to_eval_record(self, doc_id: str | None = None) -> dict[str, Any]:
        """The evalkit JSONL record: nested fields plus per-path confidence."""
        return {
            "doc_id": doc_id or self.doc_id,
            "doc_type": self.doc_type,
            "fields": self.output,
            "field_confidence": {f.path: f.confidence for f in self.fields},
            "validation": {
                "status": {f.path: f.status for f in self.fields},
                "rounds": self.rounds,
                "findings": [f.model_dump(mode="json") for f in self.findings if f.verdict != "pass"],
            },
        }


# --- tool output shapes -------------------------------------------------------


class ExtractorOutput(BaseModel):
    fields: list[FieldValue]
    notes: str | None = None  # e.g. the reader agent's draft, for the trace


class ValidatorOutput(BaseModel):
    findings: list[Finding] = Field(default_factory=list)


# --- pre-pass state (read-only during extraction) ----------------------------


class Token(BaseModel):
    id: str  # "t123", stable within a document
    text: str
    bbox: BBox | None = None
    confidence: float | None = None


class Cell(BaseModel):
    text: str
    bbox: BBox | None = None


class Table(BaseModel):
    page: int
    rows: list[list[Cell]]
    header_rows: int = 0
    bbox: BBox | None = None

    @property
    def body(self) -> list[list[Cell]]:
        return self.rows[self.header_rows :]


class Block(BaseModel):
    kind: str  # "paragraph", "heading-1", "list", ...
    text: str
    page: int
    bbox: BBox | None = None


class ClassifierResult(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float] = Field(default_factory=dict)
    page_labels: list[str] = Field(default_factory=list)


class Page(BaseModel):
    index: int
    image: bytes | None = None  # PNG, rendered at ingest
    width: int | None = None
    height: int | None = None
    tokens: list[Token] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)


class PageContext(BaseModel):
    """What the pre-pass produced. Every tool can read it; none may change it."""

    content: bytes = b""
    mime_type: str = "application/pdf"
    pages: list[Page] = Field(default_factory=list)
    text: str = ""  # full OCR text in reading order
    classification: ClassifierResult | None = None

    def page(self, index: int) -> Page:
        while len(self.pages) <= index:
            self.pages.append(Page(index=len(self.pages)))
        return self.pages[index]

    @property
    def tokens(self) -> list[Token]:
        return [t for p in self.pages for t in p.tokens]

    @property
    def tables(self) -> list[Table]:
        return [t for p in self.pages for t in p.tables]

    def token_map(self) -> dict[str, Token]:
        return {t.id: t for t in self.tokens}

    def tokens_in(self, region: BBox, slack: float = 0.005) -> list[Token]:
        return [t for t in self.page(region.page).tokens if t.bbox and region.contains(t.bbox, slack)]

    def merge(self, other: PageContext) -> None:
        """Fold one preprocessor's partial context into this one."""
        for page in other.pages:
            mine = self.page(page.index)
            mine.tokens.extend(page.tokens)
            mine.blocks.extend(page.blocks)
            mine.tables.extend(page.tables)
            if page.image is not None:
                mine.image, mine.width, mine.height = page.image, page.width, page.height
        if other.text:
            self.text = other.text
        if other.classification is not None:
            self.classification = other.classification


def output_from_fields(fields: list[FieldValue]) -> dict[str, Any]:
    return paths.unflatten({f.path: f.value for f in fields})
