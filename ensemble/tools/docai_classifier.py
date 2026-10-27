"""Document AI Classifier: pre-pass (``docai.classify``) and the
``docai.classifier`` validator that checks the chosen profile."""

from __future__ import annotations

from typing import Any

from ..tool import PreprocessorInput, RunContext, ValidatorInput, register
from ..types import Evidence, Finding, PageContext, Severity, ValidatorOutput
from ._docai import DocumentAIClient, parse_classification


@register
class DocAIClassify:
    """Run the Classifier processor once and publish the document type and scores."""

    name = "docai.classify"
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
        return PageContext(classification=parse_classification(doc))


@register
class ClassifierCheck:
    """Document-level check that the classifier agrees with the profile in use.

    A confident disagreement is an error (wrong schema applied); a weak one
    only a warning.
    """

    name = "docai.classifier"
    version = "1"
    kind = "validator"
    Input = ValidatorInput
    Output = ValidatorOutput
    llm_callable = False
    fields: list[str] = []  # document-level: the orchestrator always runs it once

    def __init__(self, min_confidence: float = 0.6, aliases: dict[str, str] | None = None) -> None:
        self.min_confidence = min_confidence
        self.aliases = aliases or {}

    async def run(self, ctx: RunContext, inp: ValidatorInput) -> ValidatorOutput:
        result = ctx.pages.classification
        if result is None:
            return ValidatorOutput(findings=[self._f("unknown", "No classifier result.", Severity.info)])
        label = self.aliases.get(result.label, result.label)
        evidence = [Evidence(source=self.name, text=result.label, score=result.confidence)]
        if label == ctx.doc_type:
            return ValidatorOutput(findings=[self._f("pass", f"Classifier agrees: {label}.", evidence=evidence)])
        confident = result.confidence >= self.min_confidence
        return ValidatorOutput(
            findings=[
                self._f(
                    "fail",
                    f"Classifier says {label!r} ({result.confidence:.2f}) but the {ctx.doc_type!r} profile was applied.",
                    Severity.error if confident else Severity.warn,
                    evidence=evidence,
                    suggested=label,
                )
            ]
        )

    def _f(self, verdict: str, message: str, severity: Severity = Severity.error, *,
           evidence: list[Evidence] | None = None, suggested: str | None = None) -> Finding:  # fmt: skip
        return Finding(validator=self.name, path="", verdict=verdict, severity=severity, message=message,
                       suggested_value=suggested, evidence=evidence or [])  # fmt: skip
