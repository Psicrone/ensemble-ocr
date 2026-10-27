"""Discover document classes and fields from unlabelled samples, and draft their labels.

Outputs are in the evalkit labelling format (``specs/<class>.yaml``, ``prelabels.jsonl``)
and are written next to the documents (``DOCS/_discovery`` by default), never into a repo.
Run ``python -m ensemble.discovery --help``.
"""

from .discover import classify, clean_spec, load_specs, prelabel, propose, scan
from .model import Blob, CachedModel, GeminiJsonModel, JsonModel

__all__ = [
    "classify",
    "propose",
    "prelabel",
    "clean_spec",
    "load_specs",
    "scan",
    "Blob",
    "JsonModel",
    "GeminiJsonModel",
    "CachedModel",
]
