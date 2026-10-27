"""``rules.format``: type, regex, range and date constraints declared on the schema."""

from __future__ import annotations

import re
from typing import Any

from pydantic import TypeAdapter, ValidationError

from ..paths import generic
from ..schema import leaf_types, spec_for
from ..tool import RunContext, ValidatorInput, register
from ..types import FieldValue, Finding, Severity, ValidatorOutput


@register
class FormatRules:
    """Checks every covered field against its declared type and ``x-format``
    (regex), ``x-min`` / ``x-max`` (numbers or ISO dates) and ``x-enum``."""

    name = "rules.format"
    version = "1"
    kind = "validator"
    Input = ValidatorInput
    Output = ValidatorOutput
    llm_callable = False
    fields: list[str] = ["*"]

    async def run(self, ctx: RunContext, inp: ValidatorInput) -> ValidatorOutput:
        types = leaf_types(ctx.schema_model) if ctx.schema_model else {}
        findings = [self._check(f, types.get(generic(f.path)), spec_for(ctx.schema_model, f.path)) for f in inp.fields]
        return ValidatorOutput(findings=[f for f in findings if f is not None])

    def _check(self, field: FieldValue, annotation: Any, spec: dict[str, Any]) -> Finding | None:
        if field.value is None:
            return None
        value = field.value
        if annotation is not None:
            try:
                value = TypeAdapter(annotation).validate_python(field.value)
            except ValidationError as exc:
                return self._fail(
                    field, f"Not a valid {getattr(annotation, '__name__', annotation)}: {exc.errors()[0]['msg']}."
                )
        checked = False
        if (pattern := spec.get("x-format")) is not None:
            checked = True
            if not re.fullmatch(pattern, str(field.value)):
                return self._fail(field, f"{field.value!r} does not match the format {pattern}.")
        if (choices := spec.get("x-enum")) is not None:
            checked = True
            if value not in choices:
                return self._fail(field, f"{field.value!r} is not one of {choices}.")
        for key, op, word in (("x-min", lambda a, b: a >= b, "below"), ("x-max", lambda a, b: a <= b, "above")):
            if (bound := spec.get(key)) is not None:
                checked = True
                bound = TypeAdapter(type(value)).validate_python(bound) if not isinstance(value, str) else bound
                if not op(value, bound):
                    return self._fail(field, f"{field.value!r} is {word} the allowed {key[2:]} {bound}.")
        if not checked and annotation is None:
            return None
        return Finding(validator=self.name, path=field.path, verdict="pass", message="Format OK.")

    def _fail(self, field: FieldValue, message: str) -> Finding:
        return Finding(validator=self.name, path=field.path, verdict="fail", severity=Severity.error, message=message)
