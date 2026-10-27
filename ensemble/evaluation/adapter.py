"""Read ``ensemble.types.ExtractionResult`` for evalkit."""
from __future__ import annotations

from evalkit.structured import to_record

# Reconcile's statuses: what counts as auto-accepted, and what goes to a person.
ACCEPT_STATUSES = ("accepted", "corrected")
REVIEW_STATUSES = ("needs_review", "failed")


def _strip_root(path: str) -> str:
    return path[2:] if path.startswith("$.") else path.lstrip("$")


def extraction_result_adapter(record) -> tuple[dict, dict]:
    """``ExtractionResult`` (model or dumped dict) -> (values, {path: {confidence, status}}).

    Values come from ``output``; confidence and status from ``fields[]`` (``FieldValue``),
    whose ``path`` is relative to ``output`` (``line_items[2].amount``). A record without
    ``output`` is read as an evalkit envelope (``{"fields": {...}, "field_meta": {...}}``).
    """
    r = to_record(record)
    if not isinstance(r.get("output"), dict):
        from evalkit.structured import default_adapter
        return default_adapter(r, "fields", ["doc_id", "doc_type", "source"])
    meta = {}
    for fv in r.get("fields") or []:
        if isinstance(fv, dict) and "path" in fv:
            meta[_strip_root(fv["path"])] = {"confidence": fv.get("confidence"), "status": fv.get("status")}
    return r["output"], meta
