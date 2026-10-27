"""Load YAML profiles and resolve tools by name.

A config directory holds any number of YAML files; their ``preprocessors``,
``routing`` and ``doc_types`` sections are merged. ``${VAR}`` and
``${VAR:-default}`` are expanded from the environment::

    preprocessors:
      - docai.ocr: {processor_id: ${DOCAI_OCR_PROCESSOR}, project: ${GCP_PROJECT}}
    doc_types:
      invoice:
        spec: ../specs/invoices.yaml        # evalkit doc-type spec (or schema: module:PydanticModel)
        extractors: [gemini.extract]
        llm_tools: [lookup.entity, ocr.region_text]
        validators:
          - docai.ocr_match
          - api.entity_registry: {base_url: ${ENTITY_API}, timeout_s: 3}
        thresholds: {accept: 0.85, review: 0.6}
        max_rounds: 2
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from . import tool as tool_mod
from . import tools as _builtin_tools  # noqa: F401 - registers the built-in tools
from .reconcile import Thresholds
from .spec import model_from_spec
from .types import ClassifierResult

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def load_schema(ref: str | type[BaseModel], base_dir: Path) -> type[BaseModel]:
    """``pkg.module:Class`` or ``relative/file.py:Class``."""
    if isinstance(ref, type):
        return ref
    target, _, attr = ref.partition(":")
    if target.endswith(".py"):
        path = (base_dir / target).resolve()
        name = f"ensemble_schema_{path.stem}"
        module = sys.modules.get(name)
        if module is None:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load schema module {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, attr)


def build_tools(entries: list[Any], instances: dict[str, Any]) -> list[Any]:
    """``["name", {"name": {params}}]`` -> tool instances (``instances`` wins by name)."""
    out = []
    for entry in entries or []:
        name, params = (entry, {}) if isinstance(entry, str) else next(iter(entry.items()))
        out.append(instances[name] if name in instances else tool_mod.create(name, **(params or {})))
    return out


class DocTypeProfile(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    schema_model: type[BaseModel]
    extractors: list[Any]
    llm_tools: list[Any] = Field(default_factory=list)
    validators: list[Any] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    max_rounds: int = 2
    budget_s: float | None = None


class PipelineConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    preprocessors: list[Any] = Field(default_factory=list)
    profiles: dict[str, DocTypeProfile]
    route_min_confidence: float = 0.6
    fallback: str | None = "generic"

    @classmethod
    def load(cls, source: str | Path | dict[str, Any], *, instances: dict[str, Any] | None = None) -> PipelineConfig:
        """Load from a YAML file, a directory of YAML files, or an already-parsed dict.

        ``instances`` maps tool names to pre-built objects (fakes in tests, or
        tools that need constructor arguments YAML cannot express).
        """
        instances = instances or {}
        if isinstance(source, dict):
            docs, base = [source], Path.cwd()
        else:
            path = Path(source)
            files = sorted(path.glob("*.y*ml")) if path.is_dir() else [path]
            docs = [yaml.safe_load(f.read_text()) or {} for f in files]
            base = path if path.is_dir() else path.parent
        merged: dict[str, Any] = {"preprocessors": [], "doc_types": {}, "routing": {}}
        for doc in docs:
            doc = expand_env(doc)
            merged["preprocessors"] += doc.get("preprocessors", [])
            merged["doc_types"].update(doc.get("doc_types", {}))
            merged["routing"].update(doc.get("routing", {}))
        profiles = {
            name: DocTypeProfile(
                name=name,
                schema_model=(
                    model_from_spec(base / spec["spec"], spec.get("spec_doc_type", name))
                    if "spec" in spec
                    else load_schema(spec["schema"], base)
                ),
                extractors=build_tools(spec.get("extractors", []), instances),
                llm_tools=build_tools(spec.get("llm_tools", []), instances),
                validators=build_tools(spec.get("validators", []), instances),
                weights=spec.get("weights", {}),
                thresholds=Thresholds(**spec.get("thresholds", {})),
                max_rounds=spec.get("max_rounds", 2),
                budget_s=spec.get("budget_s"),
            )
            for name, spec in merged["doc_types"].items()
        }
        routing = merged["routing"]
        return cls(
            preprocessors=build_tools(merged["preprocessors"], instances),
            profiles=profiles,
            route_min_confidence=routing.get("min_confidence", 0.6),
            fallback=routing.get("fallback", "generic"),
        )

    def route(self, doc_type: str | None, classification: ClassifierResult | None) -> tuple[DocTypeProfile, bool]:
        """Pick a profile. Returns (profile, flagged); flagged means a low-confidence or fallback route."""
        if doc_type:
            if doc_type not in self.profiles:
                raise KeyError(f"no profile for doc_type {doc_type!r}; have {sorted(self.profiles)}")
            return self.profiles[doc_type], False
        if classification and classification.label in self.profiles:
            if classification.confidence >= self.route_min_confidence:
                return self.profiles[classification.label], False
        if self.fallback and self.fallback in self.profiles:
            return self.profiles[self.fallback], True
        if len(self.profiles) == 1:
            return next(iter(self.profiles.values())), True
        raise LookupError("cannot route document: no confident classification and no fallback profile")
