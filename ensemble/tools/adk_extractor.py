"""``gemini.extract``: the extraction agent, built on Google ADK.

Two agents run in sequence for each extraction round:

* **reader** looks at the page images and the OCR text (with token ids), may
  call the profile's ``llm_tools``, and writes a draft: one line per field
  with the value, the token ids it came from and the page.
* **formatter** has no tools and turns the draft into JSON constrained to the
  grounded schema (every leaf is ``{value, source_tokens, page}``).

Both run at temperature 0 with a fixed seed. ``RunConfig(max_llm_calls)``
caps model calls and a ``before_tool_callback`` caps tool calls. Each run gets
a fresh in-memory session, so nothing leaks between documents or rounds. The
feedback loop is not an ADK ``LoopAgent``; the orchestrator calls this tool
again with the failing fields.

Vertex AI is selected the usual ADK way: ``GOOGLE_GENAI_USE_VERTEXAI=TRUE``,
``GOOGLE_CLOUD_PROJECT`` and ``GOOGLE_CLOUD_LOCATION`` (or pass ``project`` /
``location`` here).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from ..schema import field_specs, grounded_model, ungrounded
from ..tool import ExtractorInput, RunContext, register
from ..types import ExtractorOutput, FieldValue, PageContext
from .adk_bridge import ToolBudget, as_adk_tool

READER_INSTRUCTION = """\
You extract data from business documents with complete fidelity.

You get the page images, the OCR text of every page as `[token_id] text`
pairs, and the list of fields to extract. For each field:
- Read the value from the page exactly as printed (normalise dates to
  YYYY-MM-DD and amounts to plain numbers).
- Cite the OCR token ids the value comes from, and the 0-based page index.
- If the field is not on the document, say it is absent. Never guess.
Use the tools when the OCR text is unclear, a table is complex, or an entity
needs checking.

Write your answer as one line per field:
<path> = <value> | tokens: t1,t2 | page: 0
Use paths like `vendor.name` and `lines[0].amount`."""

FORMATTER_INSTRUCTION = """\
Convert the draft extraction below into the required JSON schema. Copy values,
token ids and pages exactly as the draft gives them; do not add, infer or
change anything. Use null for fields the draft marks absent.

Draft:
{draft?}"""


def _ocr_listing(pages: PageContext, max_tokens: int) -> str:
    lines: list[str] = []
    count = 0
    for page in pages.pages:
        lines.append(f"--- page {page.index} ---")
        lines.append(" ".join(f"[{t.id}] {t.text}" for t in page.tokens[: max(0, max_tokens - count)]))
        count += len(page.tokens)
        if count >= max_tokens:
            lines.append("(OCR listing truncated; use ocr_region_text for the rest)")
            break
    return "\n".join(lines)


def _field_listing(ctx: RunContext) -> str:
    assert ctx.schema_model is not None
    rows = []
    for path, spec in field_specs(ctx.schema_model).items():
        fmt = f" (format {spec['x-format']})" if "x-format" in spec else ""
        rows.append(f"- {path}{fmt}")
    schema = json.dumps(ctx.schema_model.model_json_schema(), indent=None)
    return "Fields:\n" + "\n".join(rows) + f"\n\nJSON schema of the result:\n{schema}"


def _feedback(inp: ExtractorInput) -> str:
    if not inp.retry_paths:
        return ""
    current = {f.path: f.value for f in inp.current}
    out = [
        "",
        "REVIEW ROUND. Independent checks disagreed with these fields. Re-read ONLY these from the page, "
        "and keep every other field exactly as it is:",
    ]
    for path in sorted(inp.retry_paths):
        out.append(f"* {path} (current value: {json.dumps(current.get(path), default=str)})")
        for f in sorted((f for f in inp.findings if f.path == path), key=lambda f: (f.validator, f.message)):
            hint = f" Suggested: {json.dumps(f.suggested_value, default=str)}." if f.suggested_value is not None else ""
            out.append(f"    - {f.validator}: {f.message}{hint}")
    out.append("\nCurrent values of all fields (keep the ones not listed above):")
    out += [f"{f.path} = {json.dumps(f.value, default=str)} | tokens: {','.join(f.source_tokens)}" for f in inp.current]
    return "\n".join(out)


@register
class GeminiAdkExtractor:
    """Gemini on Vertex AI via Google ADK: tool-using reader, then a schema-bound formatter."""

    name = "gemini.extract"
    version = "1"
    kind = "extractor"
    Input = ExtractorInput
    Output = ExtractorOutput
    llm_callable = False
    fields: list[str] = []

    def __init__(
        self,
        reader_model: Any = "gemini-2.5-pro",
        formatter_model: Any = "gemini-2.5-flash",
        temperature: float = 0.0,
        seed: int = 7,
        max_llm_calls: int = 10,
        max_tool_calls: int = 8,
        max_ocr_tokens: int = 6000,
        send_images: bool = True,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        # Model ids are placeholders until the evalkit baseline picks them.
        # Either may also be an ADK BaseLlm instance (tests pass a scripted fake).
        self.reader_model = reader_model
        self.formatter_model = formatter_model
        self.temperature = temperature
        self.seed = seed
        self.max_llm_calls = max_llm_calls
        self.max_tool_calls = max_tool_calls
        self.max_ocr_tokens = max_ocr_tokens
        self.send_images = send_images
        if project:
            os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project)
        if location:
            os.environ.setdefault("GOOGLE_CLOUD_LOCATION", location)

    def build_agent(self, ctx: RunContext) -> Any:
        from google.adk.agents import LlmAgent, SequentialAgent
        from google.genai import types

        gen_cfg = types.GenerateContentConfig(temperature=self.temperature, seed=self.seed)
        reader = LlmAgent(
            name="reader",
            model=self.reader_model,
            instruction=READER_INSTRUCTION,
            tools=[as_adk_tool(t, ctx) for t in ctx.llm_tools],
            before_tool_callback=ToolBudget(self.max_tool_calls, ctx),
            generate_content_config=gen_cfg,
            output_key="draft",
        )
        formatter = LlmAgent(
            name="formatter",
            model=self.formatter_model,
            instruction=FORMATTER_INSTRUCTION,
            output_schema=grounded_model(ctx.schema_model),
            generate_content_config=gen_cfg,
            output_key="extraction",
            include_contents="none",
        )
        return SequentialAgent(name="extract", sub_agents=[reader, formatter])

    def build_message(self, ctx: RunContext, inp: ExtractorInput) -> Any:
        from google.genai import types

        parts: list[types.Part] = []
        if self.send_images:
            images = [p.image for p in ctx.pages.pages if p.image]
            if images:
                parts += [types.Part.from_bytes(data=img, mime_type="image/png") for img in images]
            elif ctx.pages.content:
                parts.append(types.Part.from_bytes(data=ctx.pages.content, mime_type=ctx.pages.mime_type))
        text = "\n\n".join(
            s for s in (
                f"Document type: {ctx.doc_type}",
                _field_listing(ctx),
                "OCR text:\n" + _ocr_listing(ctx.pages, self.max_ocr_tokens),
                _feedback(inp),
            ) if s
        )  # fmt: skip
        parts.append(types.Part.from_text(text=text))
        return types.Content(role="user", parts=parts)

    async def run(self, ctx: RunContext, inp: ExtractorInput) -> ExtractorOutput:
        from google.adk.agents.run_config import RunConfig
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        sessions = InMemorySessionService()
        runner = Runner(app_name="ensemble", agent=self.build_agent(ctx), session_service=sessions)
        session_id = f"{ctx.doc_id[:16]}-r{ctx.round}-{uuid.uuid4().hex[:8]}"
        await sessions.create_session(app_name="ensemble", user_id="pipeline", session_id=session_id)

        with ctx.trace.span("agent.run", extractor=self.name, round=ctx.round) as span:
            usage = {"prompt_tokens": 0, "output_tokens": 0}
            llm_calls = 0
            async for event in runner.run_async(
                user_id="pipeline",
                session_id=session_id,
                new_message=self.build_message(ctx, inp),
                run_config=RunConfig(max_llm_calls=self.max_llm_calls),
            ):
                if event.usage_metadata:
                    llm_calls += 1
                    usage["prompt_tokens"] += event.usage_metadata.prompt_token_count or 0
                    usage["output_tokens"] += event.usage_metadata.candidates_token_count or 0
                if event.error_message:
                    ctx.trace.event("agent.error", author=event.author, message=event.error_message)
            span.update(usage, llm_calls=llm_calls)

        session = await sessions.get_session(app_name="ensemble", user_id="pipeline", session_id=session_id)
        state = session.state if session else {}
        raw = state.get("extraction")
        if raw is None:
            raise RuntimeError("formatter produced no extraction")
        data = json.loads(raw) if isinstance(raw, str) else raw
        grounded = grounded_model(ctx.schema_model).model_validate(data)
        fields = [
            FieldValue(
                path=path, value=value, source_tokens=tokens, extractor=self.name, round=ctx.round, evidence=[]
            )  # fmt: skip
            for path, value, tokens, _page in ungrounded(grounded.model_dump(mode="json"))
        ]
        return ExtractorOutput(fields=fields, notes=state.get("draft"))
