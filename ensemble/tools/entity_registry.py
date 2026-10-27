"""Backend entity checks: the ``api.entity_registry`` validator and the
``lookup.entity`` agent tool.

Both talk to an ``EntityBackend``. ``HttpEntityBackend`` is a generic REST
adapter; implement the two-method protocol for anything else (gRPC, a MongoDB
mirror of the registry, an in-memory dict in tests).
"""

from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ..matching import normalize
from ..paths import get
from ..tool import RunContext, ValidatorInput, register
from ..types import Evidence, FieldValue, Finding, Severity, ValidatorOutput


class EntityRecord(BaseModel):
    number: str
    name: str | None = None
    status: str | None = None  # "active", "inactive", ...
    address: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EntityBackend(Protocol):
    async def get(self, number: str) -> EntityRecord | None: ...

    async def search(self, query: str, limit: int = 3) -> list[EntityRecord]: ...


class HttpEntityBackend:
    """``GET {base_url}{get_path}`` for one entity, ``GET {base_url}{search_path}`` for candidates.

    Expects JSON objects with at least ``number`` (rename with ``field_map``).
    A 404 means "not found". Responses are cached per run.
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 3.0,
        headers: dict[str, str] | None = None,
        get_path: str = "/entities/{number}",
        search_path: str = "/entities?q={query}&limit={limit}",
        field_map: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.headers = headers or {}
        self.get_path = get_path
        self.search_path = search_path
        self.field_map = field_map or {}
        self._cache: dict[tuple[str, str], Any] = {}

    def _record(self, data: dict[str, Any]) -> EntityRecord:
        mapped = {ours: data.get(theirs) for ours, theirs in self.field_map.items()}
        return EntityRecord.model_validate({**data, **{k: v for k, v in mapped.items() if v is not None}})

    async def _get_json(self, path: str) -> Any:
        import httpx

        if ("GET", path) in self._cache:
            return self._cache[("GET", path)]
        async with httpx.AsyncClient(timeout=self.timeout_s, headers=self.headers) as client:
            response = await client.get(self.base_url + path)
        data = None if response.status_code == 404 else response.raise_for_status().json()
        self._cache[("GET", path)] = data
        return data

    async def get(self, number: str) -> EntityRecord | None:
        from urllib.parse import quote

        data = await self._get_json(self.get_path.format(number=quote(number, safe="")))
        return self._record(data) if data else None

    async def search(self, query: str, limit: int = 3) -> list[EntityRecord]:
        from urllib.parse import quote

        data = await self._get_json(self.search_path.format(query=quote(query, safe=""), limit=limit))
        items = data.get("items", data) if isinstance(data, dict) else data or []
        return [self._record(item) for item in items][:limit]


class InMemoryEntityBackend:
    """Dict-backed backend for tests and local runs."""

    def __init__(self, records: list[EntityRecord | dict[str, Any]]) -> None:
        self.records = {r.number: r for r in (EntityRecord.model_validate(x) for x in records)}

    async def get(self, number: str) -> EntityRecord | None:
        return self.records.get(number)

    async def search(self, query: str, limit: int = 3) -> list[EntityRecord]:
        q = normalize(query)

        def score(r: EntityRecord) -> float:
            return max(SequenceMatcher(None, q, normalize(r.number)).ratio(),
                       SequenceMatcher(None, q, normalize(r.name or "")).ratio())  # fmt: skip

        return sorted(self.records.values(), key=score, reverse=True)[:limit]


def make_backend(backend: EntityBackend | None, base_url: str | None, **http: Any) -> EntityBackend:
    if backend is not None:
        return backend
    if not base_url:
        raise ValueError("entity registry needs either backend= or base_url=")
    return HttpEntityBackend(base_url, **http)


@register
class EntityRegistryCheck:
    """The entity number must exist, be active, and belong to the extracted name.

    On a miss, the nearest registry matches go back to the model as suggestions.
    ``name_field`` is the sibling field holding the entity's name (relative to
    the number's parent object), e.g. ``"entity_name"``.
    """

    name = "api.entity_registry"
    version = "1"
    kind = "validator"
    Input = ValidatorInput
    Output = ValidatorOutput
    llm_callable = False
    fields: list[str] = ["entity_number", "*.entity_number"]
    grounding = True

    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float = 3.0,
        backend: EntityBackend | None = None,
        name_field: str | None = "entity_name",
        name_threshold: float = 0.8,
        active_statuses: tuple[str, ...] = ("active",),
        fields: list[str] | None = None,
        **http: Any,
    ) -> None:
        self.backend = make_backend(backend, base_url, timeout_s=timeout_s, **http)
        self.timeout_s = timeout_s * 2  # per validator run; call_tool enforces it
        self.name_field = name_field
        self.name_threshold = name_threshold
        self.active_statuses = active_statuses
        if fields:
            self.fields = fields

    async def run(self, ctx: RunContext, inp: ValidatorInput) -> ValidatorOutput:
        checks = [self._check(f, inp.output) for f in inp.fields if f.value not in (None, "")]
        results = await asyncio.gather(*checks)
        return ValidatorOutput(findings=[f for group in results for f in group])

    async def _check(self, field: FieldValue, output: dict[str, Any]) -> list[Finding]:
        number = str(field.value).strip()
        record = await self.backend.get(number)
        if record is None:
            candidates = await self.backend.search(number)
            names = ", ".join(c.number + (f" ({c.name})" if c.name else "") for c in candidates) or "none"
            return [self._f(field.path, "fail", f"Entity {number} not found; closest matches: {names}.",
                            suggested=candidates[0].number if candidates else None,
                            evidence=[Evidence(source=self.name, text=c.number, score=None) for c in candidates])]  # fmt: skip
        evidence = [Evidence(source=self.name, text=record.number, raw_ref=f"entity:{record.number}")]
        if record.status and record.status.lower() not in self.active_statuses:
            return [self._f(field.path, "fail", f"Entity {number} exists but is {record.status}.", evidence=evidence)]
        out = [self._f(field.path, "pass", f"Entity {number} found ({record.name or 'no name'}).", evidence=evidence)]
        if self.name_field and record.name:
            parent, _, _ = field.path.rpartition(".")
            name_path = f"{parent}.{self.name_field}" if parent else self.name_field
            extracted = get(output, name_path)
            if extracted:
                score = SequenceMatcher(None, normalize(str(extracted)), normalize(record.name)).ratio()
                if score >= self.name_threshold:
                    out.append(self._f(name_path, "pass", "Name matches the registry.",
                                       evidence=[Evidence(source=self.name, text=record.name, score=score)]))  # fmt: skip
                else:
                    out.append(self._f(name_path, "fail", f"Registry name for {number} is {record.name!r}.",
                                       Severity.warn, suggested=record.name,
                                       evidence=[Evidence(source=self.name, text=record.name, score=score)]))  # fmt: skip
        return out

    def _f(self, path: str, verdict: str, message: str, severity: Severity = Severity.error, *,
           suggested: Any = None, evidence: list[Evidence] | None = None) -> Finding:  # fmt: skip
        return Finding(validator=self.name, path=path, verdict=verdict, severity=severity, message=message,
                       suggested_value=suggested, evidence=evidence or [])  # fmt: skip


class EntityQuery(BaseModel):
    number: str | None = Field(None, description="entity number to look up exactly")
    name: str | None = Field(None, description="entity name to search for")


class EntityCandidates(BaseModel):
    candidates: list[EntityRecord]


@register
class EntityLookup:
    """Query the entity registry by number (exact) or name (fuzzy). Returns candidate records."""

    name = "lookup.entity"
    version = "1"
    kind = "lookup"
    Input = EntityQuery
    Output = EntityCandidates
    llm_callable = True
    fields: list[str] = []

    def __init__(self, base_url: str | None = None, timeout_s: float = 3.0,
                 backend: EntityBackend | None = None, **http: Any) -> None:  # fmt: skip
        self.backend = make_backend(backend, base_url, timeout_s=timeout_s, **http)

    async def run(self, ctx: RunContext, inp: EntityQuery) -> EntityCandidates:
        if inp.number and (record := await self.backend.get(inp.number)):
            return EntityCandidates(candidates=[record])
        query = inp.name or inp.number or ""
        return EntityCandidates(candidates=await self.backend.search(query) if query else [])
