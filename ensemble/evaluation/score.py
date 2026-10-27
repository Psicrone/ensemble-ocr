"""Score the extraction pipeline against labelled documents with evalkit."""
from __future__ import annotations

from pathlib import Path

from evalkit import evaluate
from evalkit.structured import ascore_callable

from .adapter import ACCEPT_STATUSES, REVIEW_STATUSES, extraction_result_adapter
from .specs import DocTypeSpec, load_specs

ID_COL, CATEGORY_COL = "doc_id", "doc_type"


def eval_schemas(specs) -> dict:
    """``{doc_type: evalkit Schema}`` from specs (a path, a dict, or loaded specs)."""
    if specs is None:
        return {}
    if not (isinstance(specs, dict) and all(isinstance(v, DocTypeSpec) for v in specs.values())):
        specs = load_specs(specs)
    return {t: s.eval_schema() for t, s in specs.items()}


def _options(specs, kwargs):
    return {"schema": eval_schemas(specs), "adapter": extraction_result_adapter,
            "accept_statuses": ACCEPT_STATUSES, "review_statuses": REVIEW_STATUSES,
            "id_col": ID_COL, "category_col": CATEGORY_COL, **kwargs}


async def score_pipeline(extract, documents: dict, gold, *, specs=None, runs: int = 1, out=None, **kwargs):
    """Run ``extract`` on each labelled document and score the results.

    ``extract`` is ``async (doc_bytes, doc_type=None) -> ExtractionResult``: the pipeline's
    entry point with its config bound via ``functools.partial``. ``documents`` maps gold
    ``doc_id`` to a file path or bytes (``labels.documents_from_gold`` builds it). The gold
    ``doc_type`` is passed through, so classification errors don't hide extraction scores;
    pass ``call=`` (see ``evalkit.structured.ascore_callable``) to let the pipeline
    classify instead. With ``runs > 1`` the report includes run-to-run stability.
    Returns ``(report_path, evalkit EvalResult)``.
    """
    inputs = {k: v if isinstance(v, (bytes, bytearray)) else Path(v).read_bytes()
              for k, v in documents.items()}
    call = kwargs.pop("call", None) or (lambda fn, data, g: fn(data, doc_type=g.get(CATEGORY_COL)))
    return await ascore_callable(extract, inputs, gold, runs=runs, call=call, out=out,
                                 **_options(specs, kwargs))


def score_results(results, gold, *, specs=None, out=None, **kwargs):
    """Score saved results (JSONL of ExtractionResult with the gold ``doc_id``) against gold."""
    return evaluate(results, "structured", gold=gold, out=out, embedder=None, return_result=True,
                    **_options(specs, kwargs))
