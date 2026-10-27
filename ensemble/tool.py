"""The one plug-in interface: an async ``Tool`` with typed input and output.

A tool's ``kind`` decides where the orchestrator calls it:

* ``preprocessor`` runs once per document and returns a partial ``PageContext``
* ``extractor``    returns ``ExtractorOutput`` (field values with citations)
* ``validator``    returns ``ValidatorOutput`` (findings, never edits values)
* ``lookup``       is only called by the agent (``llm_callable = True``)

Any tool with ``llm_callable = True`` can also be offered to the extraction
agent; its ``Input`` model becomes the function declaration.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
from collections.abc import Callable
from typing import Any, ClassVar, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .cache import Cache
from .trace import Tracer
from .types import ExtractorOutput, FieldValue, Finding, PageContext, Severity, ValidatorOutput

log = logging.getLogger(__name__)

I = TypeVar("I", bound=BaseModel)
O = TypeVar("O", bound=BaseModel)

ToolKind = Literal["preprocessor", "extractor", "validator", "lookup"]


class RunContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    doc_id: str
    doc_type: str
    pages: PageContext
    round: int = 0
    trace: Tracer = Field(default_factory=Tracer)
    cache: Cache = Field(default_factory=Cache)
    schema_model: type[BaseModel] | None = None
    schema_version: str = ""
    # Names of the tools the profile offers to the agent in this run.
    llm_tools: list[Any] = Field(default_factory=list)


@runtime_checkable
class Tool(Protocol[I, O]):
    name: ClassVar[str]
    version: ClassVar[str]
    kind: ClassVar[ToolKind]
    Input: ClassVar[type[BaseModel]]
    Output: ClassVar[type[BaseModel]]
    llm_callable: ClassVar[bool]
    fields: ClassVar[list[str]]

    async def run(self, ctx: RunContext, inp: I) -> O: ...


# --- standard inputs ----------------------------------------------------------


class ValidatorInput(BaseModel):
    """Fields a validator covers (already filtered by the orchestrator)."""

    fields: list[FieldValue]
    output: dict[str, Any] = Field(default_factory=dict)  # whole document, for cross-field rules


class ExtractorInput(BaseModel):
    # Empty on the first round. On feedback rounds: the current values and the
    # findings for the fields that must be re-read; everything else is frozen.
    current: list[FieldValue] = Field(default_factory=list)
    retry_paths: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)


class PreprocessorInput(BaseModel):
    pass


# --- FunctionTool: wrap any callable ------------------------------------------


class FunctionTool:
    """Build a tool from a plain function ``fn(ctx, inp) -> Output`` (sync or async).

    >>> lookup = FunctionTool("lookup.vat", check_vat, VatIn, VatOut, kind="lookup", llm_callable=True)
    """

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any],
        Input: type[BaseModel],
        Output: type[BaseModel],
        kind: ToolKind = "lookup",
        *,
        version: str = "1",
        llm_callable: bool = False,
        fields: list[str] | None = None,
        description: str | None = None,
    ) -> None:
        self.name = name
        self.fn = fn
        self.Input = Input
        self.Output = Output
        self.kind = kind
        self.version = version
        self.llm_callable = llm_callable
        self.fields = fields or []
        self.__doc__ = description or inspect.getdoc(fn) or name

    async def run(self, ctx: RunContext, inp: BaseModel) -> BaseModel:
        result = self.fn(ctx, inp)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, self.Output) else self.Output.model_validate(result)


# --- registry -----------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(cls: Any = None, *, name: str | None = None) -> Any:
    """Class decorator: ``@register`` makes a tool constructible by name from YAML."""

    def add(c: Any) -> Any:
        _REGISTRY[name or c.name] = c
        return c

    return add(cls) if cls is not None else add


def create(name: str, **params: Any) -> Any:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no tool registered as {name!r}; known: {sorted(_REGISTRY)}") from None
    return factory(**params)


def registered() -> list[str]:
    return sorted(_REGISTRY)


def describe(tool: Any) -> str:
    return (getattr(tool, "description", None) or inspect.getdoc(tool) or tool.name).strip().split("\n\n")[0]


# --- calling a tool: cache, trace, failure policy -----------------------------


def input_hash(inp: BaseModel) -> str:
    payload = json.dumps(inp.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


async def call_tool(tool: Any, ctx: RunContext, inp: BaseModel, *, timeout_s: float | None = None) -> BaseModel:
    """Run a tool with caching and tracing.

    A validator that raises or times out returns one ``unknown`` finding instead
    of stopping the pipeline. Other kinds re-raise.
    """
    timeout_s = timeout_s if timeout_s is not None else getattr(tool, "timeout_s", None)
    key = (tool.name, tool.version, ctx.doc_id, input_hash(inp))
    with ctx.trace.span(tool.name, version=tool.version, input_hash=key[3], round=ctx.round) as span:
        cached = ctx.cache.get(key)
        if cached is not None:
            span["cache_hit"] = True
            return cached
        try:
            result = await asyncio.wait_for(tool.run(ctx, inp), timeout_s)
        except Exception as exc:  # noqa: BLE001 - plug-in failures must not stop the run
            span["error"] = repr(exc)
            if tool.kind != "validator":
                raise
            log.warning("validator %s failed: %r", tool.name, exc)
            reason = "timed out" if isinstance(exc, TimeoutError) else f"errored ({type(exc).__name__})"
            return ValidatorOutput(
                findings=[
                    Finding(
                        validator=tool.name,
                        path=f.path,
                        verdict="unknown",
                        severity=Severity.info,
                        message=f"{tool.name} {reason}; no verdict.",
                    )
                    for f in getattr(inp, "fields", [])
                ]
            )
        span["cache_hit"] = False
        if isinstance(result, ValidatorOutput):
            span["verdicts"] = {v: sum(f.verdict == v for f in result.findings) for v in ("pass", "fail", "unknown")}
        ctx.cache.set(key, result)
        return result


__all__ = [
    "ExtractorInput",
    "ExtractorOutput",
    "FunctionTool",
    "PreprocessorInput",
    "RunContext",
    "Tool",
    "ToolKind",
    "ValidatorInput",
    "ValidatorOutput",
    "call_tool",
    "create",
    "describe",
    "register",
    "registered",
]
