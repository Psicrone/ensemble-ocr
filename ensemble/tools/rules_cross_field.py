"""``rules.cross_field``: arithmetic and consistency between fields.

Rules come from the profile YAML::

    - rules.cross_field:
        rules:
          - {sum: "lines[].amount", equals: subtotal}
          - {sum: [subtotal, tax], equals: total}
          - {product: [subtotal, tax_rate], equals: tax, tolerance: 0.02}
"""

from __future__ import annotations

import fnmatch
from typing import Any

from pydantic import BaseModel

from ..paths import flatten, generic
from ..tool import RunContext, ValidatorInput, register
from ..types import Finding, Severity, ValidatorOutput


class Rule(BaseModel):
    sum: str | list[str] | None = None
    product: list[str] | None = None
    equals: str
    tolerance: float = 0.01


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


@register
class CrossFieldRules:
    """A wrong digit rarely survives arithmetic: line totals must add up, tax must equal base x rate."""

    name = "rules.cross_field"
    version = "1"
    kind = "validator"
    Input = ValidatorInput
    Output = ValidatorOutput
    llm_callable = False
    fields: list[str] = []  # reads the whole output; findings land on each rule's `equals` path

    def __init__(self, rules: list[dict[str, Any]] | None = None) -> None:
        self.rules = [Rule.model_validate(r) for r in rules or []]

    async def run(self, ctx: RunContext, inp: ValidatorInput) -> ValidatorOutput:
        flat = flatten(inp.output)
        return ValidatorOutput(findings=[f for r in self.rules if (f := self._apply(r, flat)) is not None])

    def _values(self, spec: str | list[str], flat: dict[str, Any]) -> list[float | None]:
        patterns = [spec] if isinstance(spec, str) else spec
        out: list[float | None] = []
        for pattern in patterns:
            hits = [v for p, v in flat.items() if p == pattern or fnmatch.fnmatchcase(generic(p), pattern)]
            out.extend(_number(v) for v in hits) if hits else out.append(None)
        return out

    def _apply(self, rule: Rule, flat: dict[str, Any]) -> Finding | None:
        target = _number(flat.get(rule.equals))
        if rule.sum is not None:
            operands, label = self._values(rule.sum, flat), "sum"
            computed = sum(v for v in operands if v is not None)
        elif rule.product is not None:
            operands, label = self._values(rule.product, flat), "product"
            computed = 1.0
            for v in operands:
                computed *= v if v is not None else 1.0
        else:
            return None
        if target is None or not operands or any(v is None for v in operands):
            return Finding(validator=self.name, path=rule.equals, verdict="unknown", severity=Severity.info,
                           message=f"Cannot check {label} rule: missing or non-numeric values.")  # fmt: skip
        computed = round(computed, 6)
        if abs(computed - target) <= rule.tolerance:
            return Finding(
                validator=self.name, path=rule.equals, verdict="pass", message=f"Equals the {label} of its parts."
            )
        what = rule.sum if rule.sum is not None else rule.product
        return Finding(
            validator=self.name,
            path=rule.equals,
            verdict="fail",
            severity=Severity.error,
            message=f"{rule.equals} is {target:g} but the {label} of {what} is {computed:g}.",
            suggested_value=round(computed, 2),
        )
