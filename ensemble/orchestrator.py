"""Stages 1-8: ingest, pre-pass, route, extract, validate, reconcile, feedback, emit.

Plain Python on purpose: the control flow (what is retried, when to stop) is
deterministic and testable with fake tools, and no model sees it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Protocol

from pydantic import ValidationError

from .cache import Cache
from .ingest import ingest
from .reconcile import group_findings, reconcile_field, sort_findings
from .registry import DocTypeProfile, PipelineConfig
from .schema import covers, schema_version
from .tool import ExtractorInput, PreprocessorInput, RunContext, ValidatorInput, call_tool
from .trace import Tracer
from .types import ExtractionResult, FieldValue, Finding, Severity, output_from_fields

log = logging.getLogger(__name__)

DOC = ""  # path of document-level findings


class ResultSink(Protocol):
    async def save(self, result: ExtractionResult, spans: list[dict[str, Any]]) -> None: ...


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith((prefix + ".", prefix + "["))


def group_by_validator(findings: list[Finding]) -> dict[str, list[Finding]]:
    out: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        out[f.validator].append(f)
    return dict(out)


def _is_doc_level(tool: Any) -> bool:
    return not getattr(tool, "fields", None)


class Orchestrator:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        sink: ResultSink | None = None,
        cache: Cache | None = None,
        render: bool = True,
        dpi: int = 300,
    ) -> None:
        self.config = config
        self.sink = sink
        self.cache = cache or Cache()  # shared across documents: same input, same output
        self.render = render
        self.dpi = dpi

    async def run(self, content: bytes, doc_type: str | None = None, mime_type: str | None = None) -> ExtractionResult:
        started = time.monotonic()
        tracer = Tracer()

        # 1. Ingest
        with tracer.span("ingest"):
            doc_id, pages = ingest(content, mime_type, render=self.render, dpi=self.dpi)
        ctx = RunContext(doc_id=doc_id, doc_type=doc_type or "unknown", pages=pages, trace=tracer, cache=self.cache)

        # 2. Pre-pass (parallel, once)
        doc_findings = await self._prepass(ctx)

        # 3. Route
        profile, flagged = self.config.route(doc_type, ctx.pages.classification)
        ctx.doc_type = profile.name
        ctx.schema_model = profile.schema_model
        ctx.schema_version = schema_version(profile.schema_model)
        ctx.llm_tools = list(profile.llm_tools)
        if flagged:
            label = ctx.pages.classification.label if ctx.pages.classification else "none"
            doc_findings.append(Finding(validator="route", path=DOC, verdict="unknown", severity=Severity.warn,
                                        message=f"Routed to fallback profile {profile.name!r} (classifier: {label})."))  # fmt: skip

        # 4-7. Extract, validate, reconcile, feedback
        best, findings_by_path, rounds = await self._loop(ctx, profile, started)

        # 8. Emit
        fields = [
            f.model_copy(update={"status": "needs_review"}) if f.status == "failed" else f
            for f in sorted(best.values(), key=lambda f: f.path)
        ]
        findings = sort_findings(
            doc_findings + [f for per in findings_by_path.values() for fs in per.values() for f in fs]
        )
        result = ExtractionResult(
            doc_id=doc_id,
            doc_type=profile.name,
            schema_version=ctx.schema_version,
            fields=fields,
            findings=findings,
            rounds=rounds,
            trace_id=tracer.trace_id,
            output=self._output(profile, fields),
        )
        if self.sink is not None:
            await self.sink.save(result, tracer.spans)
        return result

    async def _prepass(self, ctx: RunContext) -> list[Finding]:
        tools = self.config.preprocessors
        results = await asyncio.gather(*(call_tool(t, ctx, PreprocessorInput()) for t in tools), return_exceptions=True)
        findings = []
        for t, r in zip(tools, results, strict=True):
            if isinstance(r, BaseException):
                log.warning("preprocessor %s failed: %r", t.name, r)
                findings.append(Finding(validator=t.name, path=DOC, verdict="unknown", severity=Severity.warn,
                                        message=f"{t.name} failed ({type(r).__name__}); its checks will be unknown."))  # fmt: skip
            else:
                ctx.pages.merge(r)
        return findings

    async def _loop(
        self, ctx: RunContext, profile: DocTypeProfile, started: float
    ) -> tuple[dict[str, FieldValue], dict[str, dict[str, list[Finding]]], int]:
        best: dict[str, FieldValue] = {}
        # path -> validator -> findings; a re-run of a validator on a path replaces its old findings.
        by_path: dict[str, dict[str, list[Finding]]] = defaultdict(dict)
        retry: list[str] = []
        rounds = 0

        for rnd in range(profile.max_rounds + 1):
            ctx.round = rnd
            rounds = rnd + 1
            relevant = [
                f for p in retry for path, per in by_path.items() if _under(path, p) for fs in per.values() for f in fs
            ]
            inp = ExtractorInput(
                current=sorted(best.values(), key=lambda f: f.path),
                retry_paths=retry,
                findings=sort_findings([f for f in relevant if f.verdict != "pass"]),
            )
            candidates = await self._extract(ctx, profile, inp)
            if candidates is None:
                if rnd == 0:
                    raise RuntimeError("every extractor failed on the first round")
                break

            if rnd > 0:  # only retried fields may change; the rest is frozen
                candidates = {p: c for p, c in candidates.items() if any(_under(p, r) for r in retry)}
                list_prefixes = [r for r in retry if r not in best]  # e.g. "lines": re-extract the whole list
                for r in list_prefixes:
                    for p in [p for p in best if _under(p, r)]:
                        best.pop(p)
                        by_path.pop(p, None)
            previous = dict(best)
            previous_findings = {p: dict(by_path.get(p, {})) for p in candidates}

            # Validate (parallel)
            current = {p: c[0] for p, c in candidates.items()}
            merged = {**best, **current}
            await self._validate(ctx, profile, list(current.values()), list(merged.values()), by_path)

            # Try validator suggestions against the grounding checks
            corrections = await self._corrections(ctx, profile, current, merged, by_path)

            # Reconcile every field (frozen ones too: document-level findings may have changed)
            for path in sorted(set(merged)):
                cands = candidates.get(path) or [best[path]]
                path_findings = [f for fs in by_path.get(path, {}).values() for f in fs]
                old = previous.get(path)
                if (
                    path not in candidates
                    and old is not None
                    and old.status == "corrected"
                    and not any(f.verdict == "fail" and f.severity == Severity.error for f in path_findings)
                ):
                    continue  # a frozen correction stands unless a check now fails it
                field = reconcile_field(
                    cands,
                    path_findings,
                    schema=profile.schema_model,
                    weights=profile.weights,
                    thresholds=profile.thresholds,
                    grounding=self._grounding(profile),
                    correction=corrections.get(path),
                )
                if field.status == "corrected" and path in corrections:
                    # The findings now describe the corrected value, not the one the model read.
                    by_path[path].update(group_by_validator(corrections[path][1]))
                if rnd > 0 and path in candidates and old is not None and field.confidence <= old.confidence:
                    # A value may only change if it scores higher; otherwise keep the earlier round.
                    best[path] = old
                    by_path[path] = previous_findings.get(path, {})
                else:
                    best[path] = field

            changed = set(best) != set(previous) or any(best[p].value != previous[p].value for p in best)
            failing = sorted(
                {p for p, f in best.items() if f.status not in ("accepted", "corrected")}
                | {p for p, per in by_path.items() if p != DOC and p not in best and any(
                    f.verdict == "fail" and f.severity == Severity.error for fs in per.values() for f in fs)}
            )  # fmt: skip
            ctx.trace.event("round", round=rnd, failing=failing, changed=changed)
            over_budget = profile.budget_s is not None and time.monotonic() - started > profile.budget_s
            if not failing or rnd == profile.max_rounds or (rnd > 0 and not changed) or over_budget:
                break
            retry = failing
        return best, by_path, rounds

    async def _extract(
        self, ctx: RunContext, profile: DocTypeProfile, inp: ExtractorInput
    ) -> dict[str, list[FieldValue]] | None:
        outs = await asyncio.gather(*(call_tool(e, ctx, inp) for e in profile.extractors), return_exceptions=True)
        candidates: dict[str, list[FieldValue]] = defaultdict(list)
        ok = False
        for e, out in zip(profile.extractors, outs, strict=True):
            if isinstance(out, BaseException):
                log.warning("extractor %s failed: %r", e.name, out)
                ctx.trace.event("extractor.error", tool=e.name, error=repr(out), round=ctx.round)
                continue
            ok = True
            for f in out.fields:
                candidates[f.path].append(f)
        return dict(candidates) if ok else None

    async def _validate(
        self,
        ctx: RunContext,
        profile: DocTypeProfile,
        fields: list[FieldValue],
        all_fields: list[FieldValue],
        by_path: dict[str, dict[str, list[Finding]]],
    ) -> None:
        output = output_from_fields(all_fields)
        jobs = []
        for v in profile.validators:
            if _is_doc_level(v):
                jobs.append((v, ValidatorInput(fields=all_fields, output=output)))
            elif covered := [f for f in fields if covers(v, f.path, profile.schema_model)]:
                jobs.append((v, ValidatorInput(fields=covered, output=output)))
        results = await asyncio.gather(*(call_tool(v, ctx, i) for v, i in jobs))
        for (v, inp), res in zip(jobs, results, strict=True):
            if _is_doc_level(v):
                for per in by_path.values():
                    per.pop(v.name, None)
            else:
                for f in inp.fields:
                    by_path[f.path].pop(v.name, None)
            for path, fs in group_findings(res.findings).items():
                by_path[path][v.name] = fs

    def _grounding(self, profile: DocTypeProfile) -> set[str]:
        return {v.name for v in profile.validators if getattr(v, "grounding", False)}

    async def _corrections(
        self,
        ctx: RunContext,
        profile: DocTypeProfile,
        current: dict[str, FieldValue],
        merged: dict[str, FieldValue],
        by_path: dict[str, dict[str, list[Finding]]],
    ) -> dict[str, tuple[Any, list[Finding]]]:
        grounding = [v for v in profile.validators if getattr(v, "grounding", False)]
        out: dict[str, tuple[Any, list[Finding]]] = {}
        for path, field in merged.items():
            suggestions = [
                f.suggested_value
                for fs in by_path.get(path, {}).values()
                for f in fs
                if f.verdict == "fail" and f.suggested_value is not None and f.suggested_value != field.value
            ]
            if not suggestions or (path not in current and ctx.round > 0):
                continue
            # Keep the original citation: a suggestion is only grounded if the tokens the
            # model pointed at actually say it (a number found elsewhere proves little).
            trial = field.model_copy(update={"value": suggestions[0]})
            output = output_from_fields([trial if f.path == path else f for f in merged.values()])
            checks: list[Finding] = []
            for v in grounding:
                if covers(v, path, profile.schema_model):
                    res = await call_tool(v, ctx, ValidatorInput(fields=[trial], output=output))
                    checks += [f for f in res.findings if f.path == path]
            passed = any(f.verdict == "pass" for f in checks)
            if passed and not any(f.verdict == "fail" and f.severity == Severity.error for f in checks):
                out[path] = (suggestions[0], checks)
        return out

    def _output(self, profile: DocTypeProfile, fields: list[FieldValue]) -> dict[str, Any]:
        raw = output_from_fields(fields)
        try:
            return profile.schema_model.model_validate(raw).model_dump(mode="json")
        except ValidationError:
            return raw
