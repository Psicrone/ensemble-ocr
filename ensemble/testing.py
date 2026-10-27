"""Fakes for running the whole pipeline offline: no model, no Document AI, no network."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable
from typing import Any

from pydantic import PrivateAttr

from .tool import ExtractorInput, PreprocessorInput, RunContext, ValidatorInput
from .types import ExtractorOutput, FieldValue, Finding, PageContext, ValidatorOutput


class FakeExtractor:
    """Returns scripted values per round: ``rounds[i]`` maps path -> value or (value, [token ids])."""

    kind = "extractor"
    version = "1"
    Input = ExtractorInput
    Output = ExtractorOutput
    llm_callable = False
    fields: list[str] = []

    def __init__(self, rounds: list[dict[str, Any]], name: str = "fake.extract") -> None:
        self.name = name
        self.rounds = rounds
        self.calls: list[ExtractorInput] = []

    async def run(self, ctx: RunContext, inp: ExtractorInput) -> ExtractorOutput:
        self.calls.append(inp)
        script = self.rounds[min(ctx.round, len(self.rounds) - 1)]
        fields = []
        for path, spec in script.items():
            value, tokens = spec if isinstance(spec, tuple) else (spec, [])
            fields.append(
                FieldValue(path=path, value=value, source_tokens=tokens, extractor=self.name, round=ctx.round)
            )
        return ExtractorOutput(fields=fields)


class FakeValidator:
    """Fails the paths in ``fail`` (path -> message or (message, suggestion)); passes everything else."""

    kind = "validator"
    version = "1"
    Input = ValidatorInput
    Output = ValidatorOutput
    llm_callable = False

    def __init__(self, name: str = "fake.check", fail: dict[str, Any] | None = None,
                 fields: list[str] | None = None, grounding: bool = False) -> None:  # fmt: skip
        self.name = name
        self.fail = fail or {}
        self.fields = fields or ["*"]
        self.grounding = grounding

    async def run(self, ctx: RunContext, inp: ValidatorInput) -> ValidatorOutput:
        out = []
        for f in inp.fields:
            spec = self.fail.get(f.path)
            if callable(spec):
                spec = spec(f)
            if spec is None:
                out.append(Finding(validator=self.name, path=f.path, verdict="pass", message="ok"))
            else:
                message, suggestion = spec if isinstance(spec, tuple) else (spec, None)
                out.append(Finding(validator=self.name, path=f.path, verdict="fail", message=message,
                                   suggested_value=suggestion))  # fmt: skip
        return ValidatorOutput(findings=out)


class StaticPreprocessor:
    """Publishes a fixed PageContext (e.g. OCR tokens from a fixture)."""

    kind = "preprocessor"
    version = "1"
    Input = PreprocessorInput
    Output = PageContext
    llm_callable = False
    fields: list[str] = []

    def __init__(self, context: PageContext, name: str = "fake.ocr") -> None:
        self.name = name
        self.context = context

    async def run(self, ctx: RunContext, inp: PreprocessorInput) -> PageContext:
        return self.context


def scripted_llm(turns: list[Any] | Callable[[Any], Any], model: str = "scripted") -> Any:
    """An ADK ``BaseLlm`` that replays responses, for testing the real ADK agent loop.

    Each turn is a ``str`` (text answer), a ``dict`` (JSON answer) or a
    ``("call", name, args)`` tuple (a function call). Or pass a function of the
    ``LlmRequest`` that returns one of those.
    """
    from google.adk.models import BaseLlm, LlmResponse
    from google.adk.models._capabilities import LlmCapabilities
    from google.genai import types

    class ScriptedLlm(BaseLlm):
        _turns: Any = PrivateAttr()
        _i: int = PrivateAttr(default=0)
        _requests: list[Any] = PrivateAttr(default_factory=list)

        @property
        def capabilities(self) -> LlmCapabilities:
            return LlmCapabilities(output_schema_and_tools=True)

        @property
        def requests(self) -> list[Any]:
            return self._requests

        async def generate_content_async(self, llm_request: Any, stream: bool = False) -> AsyncGenerator[Any, None]:
            self._requests.append(llm_request)
            if callable(self._turns):
                turn = self._turns(llm_request)
            else:
                turn = self._turns[min(self._i, len(self._turns) - 1)]
                self._i += 1
            if isinstance(turn, tuple) and turn[0] == "call":
                part = types.Part.from_function_call(name=turn[1], args=turn[2])
            else:
                part = types.Part.from_text(text=turn if isinstance(turn, str) else json.dumps(turn))
            yield LlmResponse(
                content=types.Content(role="model", parts=[part]),
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=10, candidates_token_count=5
                ),
            )

    llm = ScriptedLlm(model=model)
    llm._turns = turns
    return llm
