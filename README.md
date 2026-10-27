# ensemble-ocr: validated document extraction

A Gemini vision agent (Google ADK) extracts fields into a typed schema. Independent checkers then score every field: Document AI OCR, Layout Parser, the Classifier, format and cross-field rules, and your backend entity API. Disagreements go back to the agent for a bounded retry. Every model, checker and lookup is a plug-in behind one `Tool` interface, so a new check is a class plus a line of YAML.

This follows the *Ensemble extraction architecture* doc.

```
ingest -> pre-pass (OCR, layout, classifier) -> route -> extract (ADK agent)
       -> validate (parallel) -> reconcile -> feedback (failing fields only) -> emit
```

## Quick start

```bash
pip install -e ".[render,eval,dev]"
pytest                                    # runs fully offline, no GCP needed
```

```python
import asyncio
from ensemble import extract

result = asyncio.run(extract(open("invoice.pdf", "rb").read(), doc_type="invoice", config="profiles"))
result.output  # final JSON in schema shape
result.fields  # per-field value, confidence, status, evidence (bbox + source)
result.findings  # what every validator said
result.to_eval_record()  # one JSONL record for evalkit
```

To score it, bind the config and pass it to `ensemble.evaluation.score_pipeline`: `functools.partial(extract, config=PipelineConfig.load("profiles"))`.

### Environment

| Variable | Used for |
|---|---|
| `GOOGLE_GENAI_USE_VERTEXAI=TRUE`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` | Gemini via ADK on Vertex AI |
| `GCP_PROJECT`, `DOCAI_LOCATION` | Document AI processors |
| `DOCAI_OCR_PROCESSOR`, `DOCAI_LAYOUT_PROCESSOR`, `DOCAI_CLASSIFIER_PROCESSOR` | Processor ids |
| `ENTITY_API` | Base URL of the entity registry |

## Layout

```
ensemble/
  types.py          BBox, Evidence, FieldValue, Finding, ExtractionResult, PageContext
  tool.py           Tool protocol, RunContext, FunctionTool, @register, call_tool (cache + trace)
  registry.py       YAML profiles -> DocTypeProfile; routing
  orchestrator.py   stages 1-8, feedback loop, stopping rules
  reconcile.py      confidence and status rules
  extract.py        public entry point: extract(doc_bytes, doc_type=None, config=...)
  spec.py           doc-type spec -> Pydantic schema with x-critical / x-format / x-enum
  evaluation/       spec loader, labelling CLI, ExtractionResult adapter, score_pipeline (uses evalkit)
  discovery/        Gemini proposes per-class field specs and pre-fills labels (see its README)
  schema.py         x-* field declarations, grounded {value, source_tokens, page} wrapper
  ingest.py         content hash, MIME sniffing, page rendering
  cache.py trace.py sinks.py testing.py
  tools/
    adk_extractor.py   gemini.extract: reader agent (tools) -> formatter agent (output_schema)
    adk_bridge.py      as_adk_tool(): any llm_callable Tool becomes an ADK tool
    docai_ocr.py       docai.ocr (pre-pass), docai.ocr_match (validator), ocr.region_text (agent tool)
    docai_layout.py    docai.layout, docai.layout_table, layout.table
    docai_classifier.py docai.classify, docai.classifier
    rules_format.py    rules.format
    rules_cross_field.py rules.cross_field
    entity_registry.py api.entity_registry, lookup.entity (HTTP or in-memory backend)
    image_zoom.py      image.zoom
profiles/            pipeline.yaml (pre-pass, routing), one YAML per doc type
specs/               doc-type specs: extraction schema, labels and scoring all come from these
tests/               offline: fake tools, recorded Document AI protos, scripted ADK models
```

## Schemas come from the doc-type spec

A profile points at the same document-type spec used for labelling and scoring (`ensemble.evaluation.specs`, see README-evaluation.md), so there is one definition per doc type:

```yaml
doc_types:
  invoice:
    spec: ../specs/invoices.yaml      # doc type defaults to the profile name; override with spec_doc_type
```

`critical` becomes `x-critical`, `pattern` becomes `x-format`, `values` becomes `x-enum`, and `required` means an empty value is never accepted silently. Every leaf is nullable in the extraction schema. A profile can instead name a Pydantic model (`schema: package.module:Model`), which can also pin validators per field with `x-validators`.

## Adding a check

```python
from pydantic import BaseModel
from ensemble import register
from ensemble.tool import ValidatorInput
from ensemble.types import Finding, ValidatorOutput


@register
class AbnChecksum:
    name = "rules.abn_checksum"
    version = "1"
    kind = "validator"
    Input = ValidatorInput
    Output = ValidatorOutput
    llm_callable = False
    fields = ["*.abn", "abn"]

    async def run(self, ctx, inp):
        return ValidatorOutput(
            findings=[
                Finding(
                    validator=self.name,
                    path=f.path,
                    verdict="pass" if valid(f.value) else "fail",
                    message="ABN checksum OK." if valid(f.value) else f"ABN {f.value} fails its checksum.",
                )
                for f in inp.fields
            ]
        )
```

Then add `- rules.abn_checksum` under the doc type's `validators:`. To wrap an existing function or HTTP call without a class, use `FunctionTool(name, fn, Input, Output, kind)`. A tool with `llm_callable = True` listed under `llm_tools:` is also offered to the agent.

Plug-in rules: same input gives the same output (results are cached by tool name, version, document hash and input hash); failures become `unknown` findings, never exceptions; finding messages are one line written for the model, because they are sent back verbatim in the feedback round.

## How a field gets its status

Confidence is the weighted pass rate of the field's checks (weights per validator in the profile; `unknown` is left out; extractor agreement is one more vote when there are several extractors). Then, in order:

1. An `error` fail on an `x-critical` field gives `failed`. Critical fields are never auto-corrected; the agent re-reads them with the findings and suggestions.
2. A validator's `suggested_value` that passes a grounding check (OCR at the tokens the model cited, or the backend) gives `corrected`.
3. Confidence at or above `thresholds.accept` with no `error` fails gives `accepted` (critical fields also need a passing grounding check).
4. At or above `thresholds.review` gives `needs_review`, below it `failed`.

Failed and below-threshold fields go back to the agent, which is told to re-read only those; everything else is frozen. A retried value replaces the old one only if it scores higher. The loop stops when nothing is failing, after `max_rounds`, when a round changes nothing, or when `budget_s` runs out. Anything still failing is emitted as `needs_review` with its findings.

## Not done yet

- No live GCP run yet: Document AI, Vertex and the entity API are mocked in tests. Processor ids, region and the entity API contract are open questions in the architecture doc.
- `specs/invoices.yaml` is an example until the real doc types are defined.
- Traces are plain dicts (stored with `MongoSink`); OpenTelemetry export to Cloud Trace is not wired.
- No replay mode yet (re-run Reconcile from stored tool outputs).
- ADK warns that `SequentialAgent` is deprecated in favour of `Workflow`, which cannot yet hold `LlmAgent` sub-agents; revisit when it can.
