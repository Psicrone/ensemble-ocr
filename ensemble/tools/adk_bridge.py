"""Bridge plug-in ``Tool``s into Google ADK.

``as_adk_tool(tool, ctx)`` exposes any ``llm_callable`` tool to an ADK agent:
its ``Input`` model is the function declaration and its docstring the
description, so one implementation serves as both a pipeline plug-in and an
agent tool. Calls go through ``call_tool``, so they are cached and traced like
every other tool call.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from ..tool import RunContext, call_tool, describe


def adk_name(tool_name: str) -> str:
    """``ocr.region_text`` -> ``ocr_region_text`` (safe for every Gemini endpoint)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", tool_name)


def _to_parts(value: Any) -> Any:
    """Turn ``image_png`` bytes into an image Part; ADK forwards Parts to the model as media."""
    from google.genai import types

    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "image_png" and isinstance(item, bytes):
                out["image"] = types.Part.from_bytes(data=item, mime_type="image/png")
            else:
                out[key] = _to_parts(item)
        return out
    if isinstance(value, list):
        return [_to_parts(v) for v in value]
    return value


def as_adk_tool(tool: Any, ctx: RunContext) -> Any:
    from google.adk.tools import BaseTool
    from google.genai import types

    class PluginTool(BaseTool):
        def __init__(self) -> None:
            super().__init__(name=adk_name(tool.name), description=describe(tool))
            self.plugin = tool

        def _get_declaration(self) -> types.FunctionDeclaration:
            schema = tool.Input.model_json_schema()
            schema.pop("title", None)
            return types.FunctionDeclaration(
                name=self.name, description=self.description, parameters_json_schema=schema
            )

        async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
            try:
                inp = tool.Input.model_validate(args or {})
            except ValidationError as exc:
                return {"error": f"bad arguments: {exc.errors()[0]['msg']}"}
            try:
                result = await call_tool(tool, ctx, inp)
            except Exception as exc:  # noqa: BLE001 - report to the model, don't crash the run
                return {"error": f"{tool.name} failed: {type(exc).__name__}"}
            return _to_parts(result.model_dump())

    return PluginTool()


class ToolBudget:
    """``before_tool_callback`` that caps tool calls per extraction run."""

    def __init__(self, max_calls: int, ctx: RunContext) -> None:
        self.max_calls = max_calls
        self.ctx = ctx
        self.calls = 0

    def __call__(self, *, tool: Any, args: dict[str, Any], tool_context: Any) -> dict[str, Any] | None:
        self.calls += 1
        self.ctx.trace.event("agent.tool_call", tool=tool.name, args=args, n=self.calls, round=self.ctx.round)
        if self.calls > self.max_calls:
            return {"error": f"Tool budget of {self.max_calls} calls used up. Answer with what you have."}
        return None
