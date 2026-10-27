"""Ensemble document extraction: a vision agent checked by independent validators."""

from . import tools  # noqa: F401 - registers built-in tools
from .extract import extract
from .orchestrator import Orchestrator
from .registry import DocTypeProfile, PipelineConfig
from .tool import FunctionTool, RunContext, Tool, register
from .types import BBox, Evidence, ExtractionResult, FieldValue, Finding, PageContext, Severity

__all__ = [
    "BBox",
    "DocTypeProfile",
    "Evidence",
    "ExtractionResult",
    "FieldValue",
    "Finding",
    "FunctionTool",
    "Orchestrator",
    "PageContext",
    "PipelineConfig",
    "RunContext",
    "Severity",
    "Tool",
    "extract",
    "register",
]
