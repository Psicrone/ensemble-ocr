"""Document AI Enterprise OCR: pre-pass, the ``docai.ocr_match`` validator and
the ``ocr.region_text`` agent tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..matching import matches, normalize
from ..tool import PreprocessorInput, RunContext, ValidatorInput, register
from ..types import BBox, Evidence, FieldValue, Finding, PageContext, Severity, Token, ValidatorOutput
from ._docai import DocumentAIClient, parse_ocr


@register
class DocAIOcr:
    """Run Enterprise OCR once and publish tokens (with ids and boxes) to the PageContext."""

    name = "docai.ocr"
    version = "1"
    kind = "preprocessor"
    Input = PreprocessorInput
    Output = PageContext
    llm_callable = False
    fields: list[str] = []

    def __init__(
        self,
        processor_id: str,
        project: str | None = None,
        location: str = "us",
        processor_version: str | None = None,
        client: Any = None,
    ) -> None:
        self.processor_id = processor_id
        self.processor_version = processor_version
        self.client = DocumentAIClient(project, location, client)

    async def run(self, ctx: RunContext, inp: PreprocessorInput) -> PageContext:
        doc = await self.client.process(
            ctx.pages.content, ctx.pages.mime_type, self.processor_id, self.processor_version
        )
        return parse_ocr(doc)


def union_bbox(tokens: list[Token]) -> BBox | None:
    boxes = [t.bbox for t in tokens if t.bbox]
    if not boxes or len({b.page for b in boxes}) > 1:
        return boxes[0] if boxes else None
    return BBox(
        page=boxes[0].page,
        x0=min(b.x0 for b in boxes),
        y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes),
        y1=max(b.y1 for b in boxes),
    )


def find_span(value: Any, tokens: list[Token], threshold: float) -> tuple[list[Token], float]:
    """Best run of consecutive tokens that says ``value`` (empty if none)."""
    width = max(1, len(normalize(str(value)).split()))
    best: tuple[list[Token], float] = ([], 0.0)
    for size in {max(1, width - 1), width, width + 1}:
        for i in range(0, max(1, len(tokens) - size + 1)):
            window = tokens[i : i + size]
            ok, score = matches(value, " ".join(t.text for t in window), threshold)
            if ok and score > best[1]:
                best = (window, score)
                if score == 1.0 and size == width:
                    return best
    return best


@register
class OcrMatch:
    """Check each value against the OCR text at the tokens the model cited.

    pass: the cited tokens say the value (or, with no citation, it is found
    somewhere on the page). fail: it is not on the page at all (error), or the
    citation points at different text (warn).
    """

    name = "docai.ocr_match"
    version = "1"
    kind = "validator"
    Input = ValidatorInput
    Output = ValidatorOutput
    llm_callable = False
    fields: list[str] = ["*"]
    grounding = True  # a pass counts as an external check for x-critical fields

    def __init__(self, threshold: float = 0.85, search_uncited: bool = True) -> None:
        self.threshold = threshold
        self.search_uncited = search_uncited

    async def run(self, ctx: RunContext, inp: ValidatorInput) -> ValidatorOutput:
        token_map = ctx.pages.token_map()
        all_tokens = ctx.pages.tokens
        if not all_tokens:
            return ValidatorOutput(
                findings=[self._finding(f, "unknown", "No OCR tokens available.", Severity.info) for f in inp.fields]
            )
        return ValidatorOutput(
            findings=[self._check(f, token_map, all_tokens) for f in inp.fields if f.value not in (None, "")]
        )

    def _check(self, field: FieldValue, token_map: dict[str, Token], all_tokens: list[Token]) -> Finding:
        cited = [token_map[t] for t in field.source_tokens if t in token_map]
        if cited:
            text = " ".join(t.text for t in cited)
            ok, score = matches(field.value, text, self.threshold)
            if ok:
                return self._finding(field, "pass", "Matches cited OCR tokens.", evidence=self._evidence(cited, score))
        span, score = (
            find_span(field.value, all_tokens, self.threshold) if self.search_uncited or not cited else ([], 0)
        )
        if span and not cited:
            return self._finding(field, "pass", "Found in OCR text.", evidence=self._evidence(span, score))
        if span:
            return self._finding(
                field,
                "fail",
                f"Cited tokens read {text!r}, but the value appears at {', '.join(t.id for t in span)}.",
                Severity.warn,
                evidence=self._evidence(span, score),
            )
        message = "Value not found in OCR text."
        suggestion = None
        if cited and isinstance(field.value, str):
            message = f"Value not found in OCR text; cited tokens read {text!r}."
            suggestion = text
        return self._finding(field, "fail", message, suggested_value=suggestion)

    def _evidence(self, tokens: list[Token], score: float) -> list[Evidence]:
        return [Evidence(source=self.name, text=" ".join(t.text for t in tokens), bbox=union_bbox(tokens), score=score)]

    def _finding(
        self,
        field: FieldValue,
        verdict: str,
        message: str,
        severity: Severity = Severity.error,
        *,
        evidence: list[Evidence] | None = None,
        suggested_value: Any = None,
    ) -> Finding:
        return Finding(
            validator=self.name,
            path=field.path,
            verdict=verdict,
            severity=severity,
            message=message,
            suggested_value=suggested_value,
            evidence=evidence or [],
        )


class RegionIn(BaseModel):
    page: int = Field(description="0-based page index")
    x0: float = Field(description="left edge, 0..1")
    y0: float = Field(description="top edge, 0..1")
    x1: float = Field(description="right edge, 0..1")
    y1: float = Field(description="bottom edge, 0..1")


class RegionOut(BaseModel):
    text: str
    token_ids: list[str]


@register
class OcrRegionText:
    """Read the Document AI OCR text inside a box on a page. Returns the text and its token ids to cite."""

    name = "ocr.region_text"
    version = "1"
    kind = "lookup"
    Input = RegionIn
    Output = RegionOut
    llm_callable = True
    fields: list[str] = []

    async def run(self, ctx: RunContext, inp: RegionIn) -> RegionOut:
        tokens = ctx.pages.tokens_in(BBox(**inp.model_dump()))
        return RegionOut(text=" ".join(t.text for t in tokens), token_ids=[t.id for t in tokens])
