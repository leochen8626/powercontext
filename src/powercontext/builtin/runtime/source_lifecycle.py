# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Source-driven availability propagation for Memory and Topic Memory evidence.

Each scope keeps a caller-registered dependency DAG over captured Sources,
Memory entry citations, and Topic Memory artifact references. Withdrawing or
restoring a Source propagates through per-node ``all``/``any`` support
policies: unsupported Memory entries are deactivated through the native Memory
service (keeping their bodies and history), unsupported Topic Memory
artifacts leave native Topic search and Context assembly, and everything
returns when support is restored. Registration, events, and state persist in
dedicated tables, so restarts and event replays observe the same results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import insert, select

from powercontext.artifacts import ArtifactRef
from powercontext.artifacts.models import MemoryCitation
from powercontext.builtin.persistence.tables import (
    SOURCE_LIFECYCLE_EVENT_TABLE,
    SOURCE_LIFECYCLE_NODE_TABLE,
)
from powercontext.builtin.sources import validate_scope_id
from powercontext.sources.models import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.context import PowerContext, BuiltinArtifacts, BuiltinSources
    from powercontext.builtin.runtime.application import BuiltinRuntime

__all__ = [
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "SourceLifecycleScopeService",
    "filter_inactive_topic_hits",
]

_LifecycleReference = Union[SourceRef, MemoryCitation, ArtifactRef]

_REGISTER_REASON = "source-lifecycle"


class SourceLifecycleError(ValueError):
    """A source-lifecycle request violated the dependency contract."""


class SourceLifecycleEvent(BaseModel):
    """One explicit availability change for a registered Source node."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    revision: int = Field(strict=True, ge=1)
    sequence: int = Field(strict=True, ge=1)
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def _require_visible_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be blank")
        return value


@dataclass(frozen=True)
class _Node:
    """One registered evidence node and its immutable binding."""

    node_id: str
    kind: str
    parents: tuple[str, ...]
    policy: str
    reference: _LifecycleReference
    revision: int | None


def _dump_reference(kind: str, reference: _LifecycleReference) -> str:
    return json.dumps({"kind": kind, "payload": reference.model_dump(mode="json")}, ensure_ascii=False)


def _load_reference(kind: str, encoded: str) -> _LifecycleReference:
    payload = json.loads(encoded)["payload"]
    if kind == "source":
        return SourceRef.model_validate(payload)
    if kind == "memory":
        return MemoryCitation.model_validate(payload)
    return ArtifactRef.model_validate(payload)


def _check_node_id(node_id: str) -> str:
    if not isinstance(node_id, str) or not node_id.strip():
        raise SourceLifecycleError("node-id")
    return node_id


def _check_parents(parents: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(parents, str) or not isinstance(parents, (tuple, list)):
        raise SourceLifecycleError("parents")
    ordered = tuple(parents)
    if not ordered:
        raise SourceLifecycleError("parents")
    for parent in ordered:
        _check_node_id(parent)
    if len(set(ordered)) != len(ordered):
        raise SourceLifecycleError("parents")
    return ordered


def _check_policy(policy: str) -> str:
    if policy not in ("all", "any"):
        raise SourceLifecycleError("policy")
    return policy


def _check_positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SourceLifecycleError(field)
    return value


def _availability(nodes: dict[str, _Node], events: dict[str, list[tuple[int, int, bool, str]]]) -> dict[str, bool]:
    """Compute current availability for every node in dependency order."""

    available: dict[str, bool] = {}
    for node_id, node in nodes.items():
        if node.kind == "source":
            history = events.get(node_id) or ()
            available[node_id] = history[-1][2] if history else True
    remaining = {node_id: node for node_id, node in nodes.items() if node.kind != "source"}
    while remaining:
        progressed = False
        for node_id in list(remaining):
            node = remaining[node_id]
            if not all(parent in available for parent in node.parents):
                continue
            support = [available[parent] for parent in node.parents]
            available[node_id] = all(support) if node.policy == "all" else any(support)
            del remaining[node_id]
            progressed = True
        if not progressed:
            # Registered graphs are acyclic by construction; a cycle can only
            # come from out-of-band mutation, and unsupported is the safe read.
            for node_id in remaining:
                available[node_id] = False
            break
    return available


async def _load_state(
    database,
    scope_id: str,
) -> tuple[dict[str, _Node], dict[str, list[tuple[int, int, bool, str]]]]:
    """Read registered nodes and applied lifecycle events for one scope."""

    async with database.transaction() as connection:
        node_rows = (
            await connection.execute(
                select(SOURCE_LIFECYCLE_NODE_TABLE).where(SOURCE_LIFECYCLE_NODE_TABLE.c.scope_id == scope_id)
            )
        ).mappings().all()
        event_rows = (
            await connection.execute(
                select(SOURCE_LIFECYCLE_EVENT_TABLE)
                .where(SOURCE_LIFECYCLE_EVENT_TABLE.c.scope_id == scope_id)
                .order_by(SOURCE_LIFECYCLE_EVENT_TABLE.c.node_id, SOURCE_LIFECYCLE_EVENT_TABLE.c.sequence)
            )
        ).mappings().all()
    nodes: dict[str, _Node] = {}
    for row in node_rows:
        nodes[row["node_id"]] = _Node(
            node_id=row["node_id"],
            kind=row["kind"],
            parents=tuple(json.loads(row["parents"])),
            policy=row["policy"],
            reference=_load_reference(row["kind"], row["reference"]),
            revision=row["lifecycle_revision"],
        )
    events: dict[str, list[tuple[int, int, bool, str]]] = {}
    for row in event_rows:
        events.setdefault(row["node_id"], []).append(
            (row["revision"], row["sequence"], bool(row["available"]), row["reason"])
        )
    return nodes, events


async def filter_inactive_topic_hits(runtime: "BuiltinRuntime", scope_id: str, result: BaseModel) -> BaseModel:
    """Drop Topic Memory hits whose lifecycle support is currently withdrawn."""

    database = runtime._provider.database
    nodes, events = await _load_state(database, validate_scope_id(scope_id))
    if not nodes:
        return result
    available = _availability(nodes, events)
    inactive = {
        node.reference.artifact_id
        for node in nodes.values()
        if node.kind == "topic" and not available[node.node_id]
    }
    if not inactive:
        return result
    hits = tuple(
        hit
        for hit in result.hits
        if not (hit.artifact_ref.family == "topic-memory" and hit.artifact_ref.artifact_id in inactive)
    )
    if len(hits) == len(result.hits):
        return result
    return result.model_copy(update={"hits": hits})


class SourceLifecycleScopeService:
    """Register evidence dependencies and propagate availability in one scope."""

    def __init__(self, runtime: "BuiltinRuntime", scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = validate_scope_id(scope_id)

    async def register_source(self, node_id: str, source_ref: SourceRef, /, *, revision: int = 1) -> None:
        """Register one captured Source with its lifecycle revision."""

        _check_node_id(node_id)
        if not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("source-ref")
        _check_positive_int(revision, "revision")
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            await self._validate_source_readable(source_ref)
            node = _Node(node_id=node_id, kind="source", parents=(), policy="none", reference=source_ref, revision=revision)
            await self._store_node(node)
            # Sources start available; nothing native to propagate.

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: tuple[str, ...],
        policy: str,
    ) -> None:
        """Register one Memory entry citation under a support policy."""

        _check_node_id(node_id)
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("citation")
        checked_parents = _check_parents(parents)
        _check_policy(policy)
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            nodes, events = await _load_state(self._database, self.scope_id)
            self._check_binding(nodes, node_id, checked_parents)
            await self._validate_citation_readable(citation)
            node = _Node(
                node_id=node_id,
                kind="memory",
                parents=checked_parents,
                policy=policy,
                reference=citation,
                revision=None,
            )
            await self._store_node(node)
            # Registration computes state immediately from current availability.
            registered = dict(nodes)
            registered[node_id] = node
            available = _availability(registered, events)
            if not available[node_id]:
                await self._set_memory_state(citation, "inactive", _REGISTER_REASON)

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        /,
        *,
        parents: tuple[str, ...],
        policy: str,
    ) -> None:
        """Register one Topic Memory artifact reference under a support policy."""

        _check_node_id(node_id)
        if not isinstance(reference, ArtifactRef):
            raise SourceLifecycleError("artifact-ref")
        if reference.family != "topic-memory":
            raise SourceLifecycleError("topic-family")
        checked_parents = _check_parents(parents)
        _check_policy(policy)
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            nodes, _ = await _load_state(self._database, self.scope_id)
            self._check_binding(nodes, node_id, checked_parents)
            await self._validate_topic_readable(reference)
            await self._store_node(
                _Node(
                    node_id=node_id,
                    kind="topic",
                    parents=checked_parents,
                    policy=policy,
                    reference=reference,
                    revision=None,
                )
            )
            # Unsupported topics are withheld from native search by the
            # lifecycle filter, so registration needs no native write.

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        """Apply one Source availability event and return all available node ids."""

        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event")
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            nodes, events = await _load_state(self._database, self.scope_id)
            node = nodes.get(event.node_id)
            if node is None or node.kind != "source":
                raise SourceLifecycleError("unknown-source")
            if event.revision != node.revision:
                raise SourceLifecycleError("revision-mismatch")
            history = events.get(event.node_id, [])
            if history:
                last_revision, last_sequence, last_available, last_reason = history[-1]
                if event.sequence < last_sequence:
                    raise SourceLifecycleError("sequence-regressed")
                if event.sequence == last_sequence:
                    if (
                        last_available == event.available
                        and last_reason == event.reason
                        and last_revision == event.revision
                    ):
                        # Successful replay is idempotent and adds no revisions.
                        return self._active_ids(nodes, events)
                    raise SourceLifecycleError("sequence-conflict")
            before = _availability(nodes, events)
            prospective = {key: list(value) for key, value in events.items()}
            prospective.setdefault(event.node_id, []).append(
                (event.revision, event.sequence, event.available, event.reason)
            )
            after = _availability(nodes, prospective)
            await self._propagate(nodes, before, after, event.reason)
            await self._append_event(event)
            return self._active_ids(nodes, prospective)

    async def resolve(self, node_id: str, /, *, purpose: str) -> _LifecycleReference:
        """Return the original reference, gated by purpose and availability."""

        if purpose not in ("generation", "audit"):
            raise SourceLifecycleError("purpose")
        nodes, events = await _load_state(self._database, self.scope_id)
        node = nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError("unknown-node")
        if purpose == "audit":
            # Audit keeps reading the original reference and its historical body.
            return node.reference
        if not _availability(nodes, events).get(node_id, False):
            raise SourceLifecycleError("unavailable")
        return node.reference

    @property
    def _database(self):
        return self._runtime._provider.database

    @staticmethod
    def _active_ids(nodes: dict[str, _Node], events: dict[str, list[tuple[int, int, bool, str]]]) -> list[str]:
        return sorted(node_id for node_id, supported in _availability(nodes, events).items() if supported)

    @staticmethod
    def _check_binding(nodes: dict[str, _Node], node_id: str, parents: tuple[str, ...]) -> None:
        for parent in parents:
            if parent == node_id:
                raise SourceLifecycleError("self-parent")
            if parent not in nodes:
                raise SourceLifecycleError("missing-parent")

    async def _store_node(self, node: _Node) -> None:
        """Insert one node; an identical re-registration is idempotent."""

        async with self._database.transaction() as connection:
            row = (
                await connection.execute(
                    select(SOURCE_LIFECYCLE_NODE_TABLE).where(
                        SOURCE_LIFECYCLE_NODE_TABLE.c.scope_id == self.scope_id,
                        SOURCE_LIFECYCLE_NODE_TABLE.c.node_id == node.node_id,
                    )
                )
            ).mappings().first()
            if row is not None:
                existing = _Node(
                    node_id=row["node_id"],
                    kind=row["kind"],
                    parents=tuple(json.loads(row["parents"])),
                    policy=row["policy"],
                    reference=_load_reference(row["kind"], row["reference"]),
                    revision=row["lifecycle_revision"],
                )
                if existing != node:
                    raise SourceLifecycleError("identity-conflict")
                return
            await connection.execute(
                insert(SOURCE_LIFECYCLE_NODE_TABLE).values(
                    scope_id=self.scope_id,
                    node_id=node.node_id,
                    kind=node.kind,
                    parents=json.dumps(list(node.parents), ensure_ascii=False),
                    policy=node.policy,
                    reference=_dump_reference(node.kind, node.reference),
                    lifecycle_revision=node.revision,
                )
            )

    async def _append_event(self, event: SourceLifecycleEvent) -> None:
        async with self._database.transaction() as connection:
            await connection.execute(
                insert(SOURCE_LIFECYCLE_EVENT_TABLE).values(
                    scope_id=self.scope_id,
                    node_id=event.node_id,
                    sequence=event.sequence,
                    revision=event.revision,
                    available=event.available,
                    reason=event.reason,
                )
            )

    async def _validate_source_readable(self, source_ref: SourceRef) -> None:
        try:
            await self._runtime._records().get_source(self.scope_id, source_ref.source_type, source_ref.source_id)
        except Exception as error:
            raise SourceLifecycleError("source-unreadable") from error

    async def _validate_topic_readable(self, reference: ArtifactRef) -> None:
        try:
            await self._runtime._records().get_artifact_revision(
                self.scope_id,
                reference.family,
                reference.artifact_id,
                reference.revision,
            )
        except Exception as error:
            raise SourceLifecycleError("topic-unreadable") from error

    async def _validate_citation_readable(self, citation: MemoryCitation) -> None:
        async with self._runtime._context(self.scope_id) as context:
            if citation.memory_ref.artifact_id != context.artifacts.memory_artifact_id:
                raise SourceLifecycleError("memory-scope")
            try:
                await context.artifacts.memory.validate_citation(citation)
            except Exception as error:
                raise SourceLifecycleError("citation-unreadable") from error

    async def _propagate(
        self,
        nodes: dict[str, _Node],
        before: dict[str, bool],
        after: dict[str, bool],
        reason: str,
    ) -> None:
        """Sync native Memory entry states for every node whose support changed.

        A failure partway leaves the event unrecorded, so retrying the same
        event recomputes and completes the propagation; already-synced entries
        are no-ops and never create new Memory revisions.
        """

        changed = [
            node
            for node in nodes.values()
            if node.kind == "memory" and before[node.node_id] != after[node.node_id]
        ]
        if not changed:
            return
        async with self._runtime._context(self.scope_id) as context:
            for node in changed:
                target = "active" if after[node.node_id] else "inactive"
                await self._set_memory_state_with(context, node.reference, target, reason)

    async def _set_memory_state(self, citation: MemoryCitation, target_state: str, reason: str) -> None:
        async with self._runtime._context(self.scope_id) as context:
            await self._set_memory_state_with(context, citation, target_state, reason)

    @staticmethod
    async def _set_memory_state_with(
        context: "PowerContext[BuiltinSources, BuiltinArtifacts, BuiltinTriggers]",
        citation: MemoryCitation,
        target_state: str,
        reason: str,
    ) -> None:
        service = context.artifacts.memory
        artifact_id = context.artifacts.memory_artifact_id
        if citation.memory_ref.artifact_id != artifact_id:
            raise SourceLifecycleError("memory-scope")
        try:
            current = await service.head(artifact_id)
        except Exception as error:
            raise SourceLifecycleError("memory-missing") from error
        item = next(
            (entry for entry in current.content.manifest.entries if entry.entry_id == citation.entry_id),
            None,
        )
        if item is None or item.entry_version_id != citation.entry_version_id:
            raise SourceLifecycleError("entry-version")
        if item.state == target_state:
            # State already matches: no new Memory revision is created.
            return
        entry = next(
            (version for version in await service.entries(current) if version.entry_id == citation.entry_id),
            None,
        )
        if entry is None:
            raise SourceLifecycleError("entry-version")
        if target_state == "inactive":
            await service.forget(current, entries=(entry,), reason=reason)
        else:
            await service.reactivate(current, entries=(entry,), reason=reason)


class SourceLifecycleApplication:
    """Select the scoped source-lifecycle service."""

    def __init__(self, runtime: "BuiltinRuntime") -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> SourceLifecycleScopeService:
        return SourceLifecycleScopeService(self._runtime, scope_id)
