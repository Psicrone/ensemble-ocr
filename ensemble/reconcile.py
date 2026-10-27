"""Reconcile: votes and findings -> one value, confidence and status per field.

Deterministic, no model calls, so it can be replayed from stored tool outputs.

Confidence is the weighted pass rate of the field's checks:

    c = sum(w_i * p_i) / sum(w_i)      (pass = 1, fail = 0, unknown left out)

With more than one extractor, agreement counts as one more vote (weight
``agreement``). Status rules, applied in order:

1. any ``error`` fail on an ``x-critical`` field          -> failed
   (an absent optional field with no ``error`` fails -> accepted)
2. a suggested value that then passed a grounding check   -> corrected
3. c >= accept, no ``error`` fails, and (for ``x-critical`` fields) at least
   one passing grounding check                             -> accepted
4. c >= review -> needs_review, otherwise                  -> failed
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from pydantic import BaseModel

from .schema import is_critical, is_required
from .types import FieldValue, Finding, Severity

AGREEMENT = "agreement"


class Thresholds(BaseModel):
    accept: float = 0.85
    review: float = 0.6


def _key(value: Any) -> str:
    return repr(value)


def vote(candidates: list[FieldValue]) -> tuple[FieldValue, float | None]:
    """Majority value across extractors (ties: earliest extractor wins).

    Returns the winning candidate and the agreement score (None for one extractor).
    """
    if len(candidates) == 1:
        return candidates[0], None
    counts = Counter(_key(c.value) for c in candidates)
    top = max(counts.values())
    winner = next(c for c in candidates if counts[_key(c.value)] == top)
    return winner, top / len(candidates)


def confidence(findings: list[Finding], weights: dict[str, float], agreement: float | None, prior: float) -> float:
    num = den = 0.0
    for f in findings:
        if f.verdict == "unknown":
            continue
        w = weights.get(f.validator, 1.0)
        num += w * (1.0 if f.verdict == "pass" else 0.0)
        den += w
    if agreement is not None:
        w = weights.get(AGREEMENT, 1.0)
        num += w * agreement
        den += w
    return round(num / den, 4) if den else prior


def reconcile_field(
    candidates: list[FieldValue],
    findings: list[Finding],
    *,
    schema: type[BaseModel] | None,
    weights: dict[str, float],
    thresholds: Thresholds,
    grounding: set[str],
    correction: tuple[Any, list[Finding]] | None = None,
    prior: float = 0.5,
) -> FieldValue:
    chosen, agreement = vote(candidates)
    path = chosen.path
    c = confidence(findings, weights, agreement, prior)
    error_fail = any(f.verdict == "fail" and f.severity == Severity.error for f in findings)
    grounded = any(f.verdict == "pass" and f.validator in grounding for f in findings)
    critical = is_critical(schema, path)
    evidence = [e for f in findings if f.verdict == "pass" for e in f.evidence]
    base = chosen.model_copy(update={"confidence": c, "evidence": evidence})

    if critical and error_fail:
        return base.model_copy(update={"status": "failed"})
    if chosen.value is None and not is_required(schema, path) and not error_fail:
        # Absence cannot be grounded; it is accepted unless a check says the value is there.
        return base.model_copy(update={"status": "accepted"})
    if correction is not None:
        value, checks = correction
        c2 = confidence(checks, weights, None, prior)
        ev = [e for f in checks if f.verdict == "pass" for e in f.evidence]
        return base.model_copy(update={"value": value, "status": "corrected", "confidence": c2, "evidence": ev})
    if c >= thresholds.accept and not error_fail and (grounded or not critical):
        return base.model_copy(update={"status": "accepted"})
    if c >= thresholds.review:
        return base.model_copy(update={"status": "needs_review"})
    return base.model_copy(update={"status": "failed"})


def group_findings(findings: list[Finding]) -> dict[str, list[Finding]]:
    by_path: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_path[f.path].append(f)
    return by_path


def sort_findings(findings: list[Finding]) -> list[Finding]:
    """Stable order, so prompts built from findings are stable run to run."""
    return sorted(findings, key=lambda f: (f.path, f.validator, f.verdict, f.message))
