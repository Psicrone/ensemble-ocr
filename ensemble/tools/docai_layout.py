"""Document AI Layout Parser: pre-pass, the ``docai.layout_table`` validator and
the ``layout.table`` agent tool."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from ..matching import matches
from ..tool import PreprocessorInput, RunContext, ValidatorInput, register
from ..types import Evidence, FieldValue, Finding, PageContext, Severity, Table, ValidatorOutput
from ._docai import DocumentAIClient, parse_layout


@register
class DocAILayout:
    """Run the Layout Parser once and publish blocks and tables to the PageContext."""

    name = "docai.layout"
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
        return parse_layout(doc)


def _list_prefix(path: str) -> tuple[str, int] | None:
    """``lines[2].amount`` -> (``lines``, 2)."""
    head, sep, _ = path.partition("[")
    if not sep:
        return None
    index = path[len(head) + 1 : path.index("]")]
    return head, int(index)


@register
class LayoutTable:
    """Check line items against Layout Parser tables: row count and per-row values.

    Covers fields under list paths (default glob ``*[].*``). The table compared
    is the one that contains the most extracted values.
    """

    name = "docai.layout_table"
    version = "1"
    kind = "validator"
    Input = ValidatorInput
    Output = ValidatorOutput
    llm_callable = False
    fields: list[str] = ["*[].*"]

    def __init__(self, threshold: float = 0.85, fields: list[str] | None = None) -> None:
        self.threshold = threshold
        if fields:
            self.fields = fields

    async def run(self, ctx: RunContext, inp: ValidatorInput) -> ValidatorOutput:
        tables = ctx.pages.tables
        groups: dict[str, dict[int, list[FieldValue]]] = defaultdict(lambda: defaultdict(list))
        for f in inp.fields:
            if (key := _list_prefix(f.path)) is not None:
                groups[key[0]][key[1]].append(f)
        findings: list[Finding] = []
        for list_path, items in groups.items():
            if not tables:
                findings += [self._f(f.path, "unknown", "No tables detected by the Layout Parser.", Severity.info)
                             for row in items.values() for f in row]  # fmt: skip
                continue
            table = max(tables, key=lambda t: self._overlap(t, items))
            findings += self._check_list(list_path, items, table)
        return ValidatorOutput(findings=findings)

    def _overlap(self, table: Table, items: dict[int, list[FieldValue]]) -> int:
        text = " ".join(c.text for row in table.body for c in row)
        return sum(
            matches(f.value, text, self.threshold)[0] for row in items.values() for f in row if f.value is not None
        )

    def _check_list(self, list_path: str, items: dict[int, list[FieldValue]], table: Table) -> list[Finding]:
        out: list[Finding] = []
        rows = table.body
        if len(items) != len(rows):
            out.append(
                self._f(
                    list_path,
                    "fail",
                    f"Extracted {len(items)} items but the table on page {table.page} has {len(rows)} rows.",
                    suggested=len(rows),
                )
            )
        for index, fields in sorted(items.items()):
            row = rows[index] if index < len(rows) else []
            row_text = " ".join(c.text for c in row)
            for f in fields:
                if f.value in (None, ""):
                    continue
                ok, score = matches(f.value, row_text, self.threshold)
                if ok:
                    out.append(self._f(f.path, "pass", f"Matches table row {index}.",
                                       evidence=[Evidence(source=self.name, text=row_text, bbox=table.bbox, score=score)]))  # fmt: skip
                else:
                    out.append(self._f(f.path, "fail", f"Not in table row {index} ({row_text[:80]!r}).", Severity.warn))
        return out

    def _f(self, path: str, verdict: str, message: str, severity: Severity = Severity.error, *,
           suggested: Any = None, evidence: list[Evidence] | None = None) -> Finding:  # fmt: skip
        return Finding(validator=self.name, path=path, verdict=verdict, severity=severity, message=message,
                       suggested_value=suggested, evidence=evidence or [])  # fmt: skip


class TableIn(BaseModel):
    page: int | None = Field(None, description="0-based page; omit for any page")
    index: int = Field(0, description="which table on that page (or in the document), 0-based")


class TableOut(BaseModel):
    found: bool
    page: int | None = None
    header_rows: int = 0
    rows: list[list[str]] = Field(default_factory=list)


@register
class LayoutTableLookup:
    """Get a Layout Parser table as rows of cell text (header rows first)."""

    name = "layout.table"
    version = "1"
    kind = "lookup"
    Input = TableIn
    Output = TableOut
    llm_callable = True
    fields: list[str] = []

    async def run(self, ctx: RunContext, inp: TableIn) -> TableOut:
        tables = [t for t in ctx.pages.tables if inp.page is None or t.page == inp.page]
        if inp.index >= len(tables):
            return TableOut(found=False)
        t = tables[inp.index]
        return TableOut(found=True, page=t.page, header_rows=t.header_rows, rows=[[c.text for c in r] for r in t.rows])
