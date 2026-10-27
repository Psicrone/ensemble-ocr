"""Evaluation of the extraction pipeline, built on evalkit.

specs    document-type specs: one field list per doc type -> extractor JSON Schema and
         evalkit comparators (runtime-safe: no evalkit import at module level)
labels   labelling sheets from a documents folder, import back to gold, checks
adapter  ExtractionResult -> evalkit (values, per-field confidence and status)
score    score_pipeline (run extract() and score) and score_results (score saved results)
"""
from .specs import DocTypeSpec, load_specs, starter_spec

__all__ = ["DocTypeSpec", "load_specs", "starter_spec"]
