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

"""Pi session history Source for the pi-history task (source-02).

Captures the complete raw visible events of a selected branch of a Pi v3
session, preserving the evidence before any compaction. Source identity is
stable and determined only by (session_id, event_id); ingestion is incremental
and its progress is persisted through the ConnectorLifecycle checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    AdapterSourceDefinition,
    TEXT_EVIDENCE_PROJECTION_KEY,
    TextEvidence,
)
from powercontext.sources.connectors import (
    Connector,
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.models import Source, SourceMaterialization


class PiSessionInput(BaseModel):
    """A complete Pi v3 session document (UTF-8 JSONL) with an optional leaf."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on a selected branch, stored as durable evidence."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


class PiEventSource(Source):
    """One captured Pi event, stored as durable evidence."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Stable identity: pi:{encoded session_id}:{encoded event_id}."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty trimmed string")
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id must be a non-empty trimmed string")
    source_id = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("source_id exceeds the maximum allowed length")
    return source_id


# Event types that produce a visible text event. Everything else (control
# nodes such as model_change / thinking_level_change) only keeps the graph
# connected and never yields evidence text.
_VISIBLE_TYPES = {"message", "compaction", "branch_summary", "custom_message"}


def _join_text_blocks(blocks: Any) -> str:
    """Concatenate visible text blocks; toolCall blocks become a compact tag."""
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif btype == "toolCall":
            name = block.get("name")
            call_id = block.get("id")
            arguments = block.get("arguments") or {}
            args_json = json.dumps(
                arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            parts.append(f"[tool_call {name} {call_id}] {args_json}")
        # thinking, image and any other block kinds are ignored.
    return "\n".join(parts)


def _event_body(record: dict[str, Any]) -> str | None:
    """Return the visible body for a record, or None when it is not an event."""
    etype = record.get("type")
    if etype == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            raise ValueError("message record requires a 'message' object")
        role = message.get("role")
        if not isinstance(role, str):
            raise ValueError("message requires a string 'role'")
        if role == "toolResult":
            tool_name = message.get("toolName")
            tool_call_id = message.get("toolCallId")
            text = _join_text_blocks(message.get("content"))
            return f"[tool_result {tool_name} {tool_call_id}] {text}"
        if role == "bashExecution":
            command = message.get("command")
            output = message.get("output")
            return f"$ {command}\n{output}"
        return _join_text_blocks(message.get("content"))
    if etype in ("compaction", "branch_summary"):
        summary = record.get("summary")
        if not isinstance(summary, str):
            raise ValueError(f"{etype} requires a string 'summary'")
        return summary
    if etype == "custom_message":
        return _join_text_blocks(record.get("content"))
    # Control or unknown node: keep connectivity, no visible event.
    return None


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse a Pi v3 session and return the selected branch's visible events.

    The first non-blank line must be the session header (type=session,
    version=3, non-empty id). Every later record needs a unique id and a
    parentId that is null or points to an already-seen node. Any malformed
    input raises ValueError without returning a partial result.
    """
    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"content is not valid UTF-8: {exc}") from exc

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty document")

    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON line: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("each line must be a JSON object")
        records.append(parsed)

    header = records[0]
    if header.get("type") != "session" or header.get("version") != 3:
        raise ValueError("first record must be a Pi session header (type=session, version=3)")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session header requires a non-empty trimmed id")

    nodes: dict[str, dict[str, Any]] = {}
    parent_of: dict[str, str | None] = {}
    for record in records[1:]:
        rid = record.get("id")
        if not isinstance(rid, str) or not rid.strip():
            raise ValueError("each record requires a non-empty trimmed string id")
        if rid in nodes:
            raise ValueError(f"duplicate id: {rid!r}")
        parent = record.get("parentId")
        if parent is not None:
            if not isinstance(parent, str):
                raise ValueError(f"parentId must be null or a string for {rid!r}")
            if parent not in nodes:
                raise ValueError(f"parent {parent!r} not found for {rid!r}")
        nodes[rid] = record
        parent_of[rid] = parent

    leaf = value.leaf_id if value.leaf_id is not None else records[-1]["id"]
    if leaf not in nodes:
        raise ValueError(f"leaf {leaf!r} not found")

    chain: list[str] = []
    seen_in_chain: set[str] = set()
    current: str | None = leaf
    while current is not None:
        if current in seen_in_chain:
            raise ValueError("cycle detected in session graph")
        seen_in_chain.add(current)
        chain.append(current)
        current = parent_of[current]

    events: list[PiEventInput] = []
    for rid in reversed(chain):
        record = nodes[rid]
        etype = record.get("type")
        if etype not in _VISIBLE_TYPES:
            continue
        body = _event_body(record)
        if body is None or not body.strip():
            # Ignore events that carry no visible body.
            continue
        if etype == "message":
            role = record["message"]["role"]
        elif etype in ("compaction", "branch_summary"):
            role = etype
        else:  # custom_message
            role = "custom"
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=rid,
                parent_id=parent_of[rid],
                event_type=etype if isinstance(etype, str) else "",
                role=role if isinstance(role, str) else "",
                content=f"{role}: {body}",
                record=record,
            )
        )
    return tuple(events)


class _PiEventAdapter:
    input_class = PiEventInput
    name = "pi-event"
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        identity = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=identity,
            source_id=identity,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class _PiEventTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pi-event",
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    _PiEventAdapter(),
    version="1",
    projections=(_PiEventTextEvidenceProjection(),),
)


def _record_hash(record: dict[str, Any]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PiSessionConnector(Connector):
    """Incrementally ingest the selected branch of a Pi session file.

    The connector re-parses the whole file on every run and only submits the
    branch's events that have not been imported yet, keyed by event id. The
    durable checkpoint stores each imported event id together with a hash of
    its raw record, so a rewritten (mutated) event is detected and the run
    fails without advancing the checkpoint.
    """

    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        content = self._path.read_bytes()
        events = parse_pi_session(PiSessionInput(content=content, leaf_id=self._leaf_id))

        imported: dict[str, str] = {}
        previous = session.checkpoint
        if isinstance(previous, dict):
            stored = previous.get("imported")
            if isinstance(stored, dict):
                imported = {str(key): str(value) for key, value in stored.items()}

        new_imported = dict(imported)
        session_id: str | None = None
        for event in events:
            session_id = event.session_id
            record_hash = _record_hash(event.record)
            prior = new_imported.get(event.event_id)
            if prior is not None:
                if prior != record_hash:
                    raise ValueError(
                        f"imported event {event.event_id!r} was mutated; "
                        f"refusing to advance checkpoint"
                    )
                continue
            await session.submit(event.event_id, "pi-event", event)
            new_imported[event.event_id] = record_hash

        checkpoint = {"session_id": session_id, "imported": new_imported}
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)
