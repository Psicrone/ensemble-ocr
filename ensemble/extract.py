"""Public entry point.

    result = await extract(pdf_bytes, doc_type="invoice", config="profiles/")

``config`` is a ``PipelineConfig``, an ``Orchestrator``, or a path to a YAML
file or directory (default: ``$ENSEMBLE_CONFIG`` or ``./profiles``). Bind it
with ``functools.partial`` to get the ``extract(doc_bytes, doc_type=...)``
callable the evalkit harness expects.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .orchestrator import Orchestrator
from .registry import PipelineConfig
from .types import ExtractionResult


@lru_cache(maxsize=8)
def _orchestrator_for(path: str) -> Orchestrator:
    return Orchestrator(PipelineConfig.load(path))


async def extract(
    doc_bytes: bytes,
    doc_type: str | None = None,
    config: PipelineConfig | Orchestrator | str | Path | None = None,
    *,
    mime_type: str | None = None,
) -> ExtractionResult:
    if isinstance(config, Orchestrator):
        orchestrator = config
    elif isinstance(config, PipelineConfig):
        orchestrator = Orchestrator(config)
    else:
        path = config or os.environ.get("ENSEMBLE_CONFIG", "profiles")
        orchestrator = _orchestrator_for(str(Path(path).resolve()))
    return await orchestrator.run(doc_bytes, doc_type=doc_type, mime_type=mime_type)
