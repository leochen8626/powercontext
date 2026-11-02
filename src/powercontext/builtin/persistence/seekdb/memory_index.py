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

"""seekdb native memory search index using one unified hybrid candidate statement.

seekdb exposes both its full-text retrieval scan and its vector index through the
same query engine. In ``hybrid`` mode a single statement therefore returns the
full-text and vector candidates together (a ``UNION ALL`` of two native index
scans), and the application layer still fuses and reads exact bodies afterwards.
This keeps candidate retrieval to exactly one seekdb statement while preserving
the two native index types the planner exposes as ``TEXT RETRIEVAL SCAN`` and
``VECTOR INDEX``.
"""

from __future__ import annotations

from hashlib import sha256

from pyobvector import VECTOR, VectorIndex
from sqlalchemy import (
    BigInteger,
    Column,
    MetaData,
    Table,
    UniqueConstraint,
    bindparam,
    delete,
    insert,
    select,
    text,
)
from sqlalchemy.dialects.mysql import match
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import (
    CapabilityNotSupportedError,
    EmbeddingProfile,
    MemoryCapabilities,
    MemoryProjection,
    MemorySearchChannels,
    MemorySearchRequest,
)
from powercontext.builtin.artifacts.memory.canonical import embedding_content_hash, validate_embedding
from powercontext.builtin.persistence.memory_index import memory_channel_hits
from powercontext.builtin.persistence.tables import (
    ARTIFACT_HEADS_TABLE,
    MAX_ARTIFACT_ID_LENGTH,
    MAX_MEMORY_ENTRY_ID_LENGTH,
    MAX_MEMORY_HASH_LENGTH,
    MAX_SCOPE_ID_LENGTH,
    MEMORY_ENTRY_HEADS_TABLE,
    identity_string,
)
from powercontext.builtin.persistence.tags import memory_tag_sql

_SEEKDB_FTS_INDEX_NAME = "ix_pc_memory_entry_heads_fts"
_SEEKDB_VECTOR_TABLE_NAME = "pc_memory_vector_entries"
_SEEKDB_VECTOR_INDEX_NAME = "ix_pc_memory_vector_entries_embedding"

# fmt: off
_FTS_SQL = """
SELECT
    'fts' AS channel,
    h.memory_artifact_id,
    h.head_revision,
    h.entry_id,
    h.entry_version_id,
    v.text,
    NULL AS distance
FROM pc_memory_entry_heads AS h
JOIN pc_memory_entry_versions AS v
  ON v.scope_id = h.scope_id
 AND v.memory_artifact_id = h.memory_artifact_id
 AND v.entry_version_id = h.entry_version_id
WHERE h.scope_id = :scope_id
  AND h.memory_artifact_id IN :memory_artifact_ids
  AND MATCH(h.searchable_text) AGAINST (:analyzed_query)
"""
_VECTOR_SQL = """
SELECT
    'vector' AS channel,
    m.memory_artifact_id,
    m.head_revision,
    m.entry_id,
    m.entry_version_id,
    v.text,
    l2_distance(m.embedding, :query_vector) AS distance
FROM pc_memory_vector_entries AS m
JOIN pc_memory_entry_versions AS v
  ON v.scope_id = m.scope_id
 AND v.memory_artifact_id = m.memory_artifact_id
 AND v.entry_version_id = m.entry_version_id
WHERE m.scope_id = :scope_id
  AND m.memory_artifact_id IN :memory_artifact_ids
"""
# fmt: on

_FTS_ORDER = (
    "ORDER BY MATCH(h.searchable_text) AGAINST (:analyzed_query) DESC, "
    "h.memory_artifact_id, h.entry_id, h.entry_version_id "
    "LIMIT :candidate_limit"
)
# The untagged tail mirrors the frozen two-query path bit for bit: APPROXIMATE is
# what routes the ordering through the native HNSW vector index (EXPLAIN shows
# VECTOR INDEX). Tagged searches drop APPROXIMATE exactly like the frozen tagged
# path and add identity tiebreakers for repeat-query stability.
_VECTOR_ORDER_UNTAGGED = "ORDER BY l2_distance(m.embedding, :query_vector) APPROXIMATE LIMIT :candidate_limit"
_VECTOR_ORDER_TAGGED = (
    "ORDER BY l2_distance(m.embedding, :query_vector) ASC, "
    "m.memory_artifact_id, m.entry_id, m.entry_version_id "
    "LIMIT :candidate_limit"
)


def _key_hashes(tag_filter: object) -> tuple[bytes, ...]:
    return tuple(sha256(key.encode("utf-8")).digest() for key in tag_filter.keys)


def _embedding_hash(profile: EmbeddingProfile, entry_hash: str) -> str:
    return embedding_content_hash(
        profile_id=profile.profile_id,
        model=profile.model,
        dimension=profile.dimension,
        distance=profile.distance,
        normalization=profile.normalization,
        entry_content_hash=entry_hash,
    )


class SeekDBMemoryIndex:
    """Native seekdb full-text + vector index with a single hybrid candidate statement."""

    def __init__(self, profile: EmbeddingProfile | None) -> None:
        if profile is None:
            self.profile: EmbeddingProfile | None = None
            self.capabilities = MemoryCapabilities(fts=True, tag_filter=True)
        else:
            if profile.dimension < 1 or profile.distance != "l2" or profile.normalization != "unit":
                raise CapabilityNotSupportedError(
                    "vector",
                    "seekdb requires a positive unit-normalized L2 embedding profile",
                )
            self.profile = profile
            self.capabilities = MemoryCapabilities(
                fts=True,
                vector=True,
                hybrid=True,
                tag_filter=True,
                embedding_profile=profile,
            )
        metadata = MetaData()
        self.table = Table(
            _SEEKDB_VECTOR_TABLE_NAME,
            metadata,
            Column("vector_id", BigInteger, primary_key=True, autoincrement=True),
            Column("scope_id", identity_string(MAX_SCOPE_ID_LENGTH), nullable=False),
            Column("memory_artifact_id", identity_string(MAX_ARTIFACT_ID_LENGTH), nullable=False),
            Column("head_revision", BigInteger, nullable=False),
            Column("entry_id", identity_string(MAX_MEMORY_ENTRY_ID_LENGTH), nullable=False),
            Column("entry_version_id", identity_string(MAX_MEMORY_ENTRY_ID_LENGTH), nullable=False),
            Column("entry_content_hash", identity_string(MAX_MEMORY_HASH_LENGTH), nullable=False),
            Column("embedding_content_hash", identity_string(MAX_MEMORY_HASH_LENGTH), nullable=False),
            Column("embedding", VECTOR(profile.dimension) if profile is not None else VECTOR(1), nullable=False),
            UniqueConstraint(
                "scope_id",
                "memory_artifact_id",
                "entry_id",
                name="uq_pc_memory_vector_entries_head",
            ),
        )
        self.tables: tuple[Table, ...] = (self.table,) if profile is not None else ()
        self._vector_index = (
            VectorIndex(
                _SEEKDB_VECTOR_INDEX_NAME,
                self.table.c.embedding,
                params="distance=l2,type=hnsw",
            )
            if profile is not None
            else None
        )

    # -- DDL ------------------------------------------------------------------

    async def initialize(self, connection: AsyncConnection, /) -> None:
        if connection.dialect.name != "mysql":
            raise CapabilityNotSupportedError("seekdb-fts")
        count = await connection.scalar(
            text(
                """
                SELECT COUNT(*)
                FROM information_schema.statistics
                WHERE table_schema = DATABASE()
                  AND table_name = 'pc_memory_entry_heads'
                  AND index_name = :index_name
                """
            ),
            {"index_name": _SEEKDB_FTS_INDEX_NAME},
        )
        if count == 0:
            await connection.exec_driver_sql(
                f"CREATE FULLTEXT INDEX {_SEEKDB_FTS_INDEX_NAME} "
                f"ON pc_memory_entry_heads (searchable_text) WITH PARSER SPACE"
            )
        await connection.execute(
            select(MEMORY_ENTRY_HEADS_TABLE.c.entry_version_id)
            .where(match(MEMORY_ENTRY_HEADS_TABLE.c.searchable_text, against="powercontext"))
            .limit(1)
        )
        if self._vector_index is not None:
            await connection.run_sync(
                lambda sync_connection: self._vector_index.create(sync_connection, checkfirst=True)
            )

    # -- writes ----------------------------------------------------------------

    async def replace(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> None:
        if self.profile is None:
            return
        await connection.execute(
            delete(self.table).where(
                self.table.c.scope_id == scope_id,
                self.table.c.memory_artifact_id == memory_ref.artifact_id,
            )
        )
        await self.upsert(connection, scope_id, memory_ref, projections)

    async def delete(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        entry_ids: tuple[str, ...],
        /,
    ) -> None:
        if not entry_ids or self.profile is None:
            return
        await connection.execute(
            delete(self.table).where(
                self.table.c.scope_id == scope_id,
                self.table.c.memory_artifact_id == memory_ref.artifact_id,
                self.table.c.entry_id.in_(entry_ids),
            )
        )

    async def upsert(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> None:
        if self.profile is None:
            return
        values = []
        for projection in projections:
            if projection.embedding is None or projection.embedding_content_hash is None:
                continue
            entry = projection.entry_version
            values.append({
                "scope_id": scope_id,
                "memory_artifact_id": memory_ref.artifact_id,
                "head_revision": memory_ref.revision,
                "entry_id": entry.entry_id,
                "entry_version_id": entry.entry_version_id,
                "entry_content_hash": entry.entry_content_hash,
                "embedding_content_hash": projection.embedding_content_hash,
                "embedding": validate_embedding(
                    projection.embedding,
                    dimension=self.profile.dimension,
                ),
            })
        if values:
            await connection.execute(insert(self.table), values)

    # -- search ----------------------------------------------------------------

    async def search(
        self,
        connection: AsyncConnection,
        scope_id: str,
        request: MemorySearchRequest,
        /,
    ) -> MemorySearchChannels:
        if request.mode == "fts":
            return MemorySearchChannels(fts=await self._search(connection, scope_id, request, "fts"))
        if request.mode == "vector":
            if not await self.vector_complete(connection, scope_id, request.memories, request.embedding_profile):
                raise CapabilityNotSupportedError("vector")
            return MemorySearchChannels(vector=await self._search(connection, scope_id, request, "vector"))
        if request.mode == "hybrid":
            if self.profile is None or request.embedding_profile != self.profile or request.query_vector is None:
                raise CapabilityNotSupportedError("vector")
            if not await self.vector_complete(connection, scope_id, request.memories, self.profile):
                raise CapabilityNotSupportedError("vector")
            fts_rows: list[object] = []
            vector_rows: list[object] = []
            executed = await connection.execute(
                self._hybrid_sql(request),
                self._params(scope_id, request, "hybrid"),
            )
            for row in executed.mappings():
                if row["channel"] == "vector":
                    vector_rows.append(row)
                else:
                    fts_rows.append(row)
            return MemorySearchChannels(
                fts=memory_channel_hits(fts_rows, request.memories),
                vector=memory_channel_hits(vector_rows, request.memories),
            )
        return MemorySearchChannels()

    async def _search(
        self,
        connection: AsyncConnection,
        scope_id: str,
        request: MemorySearchRequest,
        channel: str,
    ) -> tuple[object, ...]:
        sql = self._fts_sql(request) if channel == "fts" else self._vector_sql(request)
        rows = (await connection.execute(sql, self._params(scope_id, request, channel))).mappings()
        return memory_channel_hits(rows, request.memories)

    # -- projection integrity --------------------------------------------------

    async def vector_complete(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memories: tuple[ArtifactRef, ...],
        profile: EmbeddingProfile,
        /,
    ) -> bool:
        if self.profile is None or profile != self.profile:
            return False
        for memory in memories:
            current = await connection.scalar(
                select(ARTIFACT_HEADS_TABLE.c.revision).where(
                    ARTIFACT_HEADS_TABLE.c.scope_id == scope_id,
                    ARTIFACT_HEADS_TABLE.c.family == "memory",
                    ARTIFACT_HEADS_TABLE.c.artifact_id == memory.artifact_id,
                )
            )
            if current is None or int(current) != memory.revision:
                raise CapabilityNotSupportedError("head")
            heads = (
                await connection.execute(
                    select(
                        MEMORY_ENTRY_HEADS_TABLE.c.entry_id,
                        MEMORY_ENTRY_HEADS_TABLE.c.entry_version_id,
                        MEMORY_ENTRY_HEADS_TABLE.c.entry_content_hash,
                    ).where(
                        MEMORY_ENTRY_HEADS_TABLE.c.scope_id == scope_id,
                        MEMORY_ENTRY_HEADS_TABLE.c.memory_artifact_id == memory.artifact_id,
                    )
                )
            ).all()
            metadata = (
                await connection.execute(
                    select(
                        self.table.c.entry_id,
                        self.table.c.entry_version_id,
                        self.table.c.entry_content_hash,
                        self.table.c.embedding_content_hash,
                    ).where(
                        self.table.c.scope_id == scope_id,
                        self.table.c.memory_artifact_id == memory.artifact_id,
                    )
                )
            ).all()
            expected = {(str(row[0]), str(row[1]), str(row[2])) for row in heads}
            actual = {(str(row[0]), str(row[1]), str(row[2])) for row in metadata}
            if actual != expected:
                return False
            if any(str(row[3]) != _embedding_hash(self.profile, str(row[2])) for row in metadata):
                return False
        return True

    async def hydrate(
        self,
        connection: AsyncConnection,
        scope_id: str,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> tuple[MemoryProjection, ...]:
        if self.profile is None:
            return projections
        hydrated: list[MemoryProjection] = []
        for projection in projections:
            entry = projection.entry_version
            row = (
                await connection.execute(
                    select(
                        self.table.c.entry_content_hash,
                        self.table.c.embedding_content_hash,
                        self.table.c.embedding,
                    ).where(
                        self.table.c.scope_id == scope_id,
                        self.table.c.memory_artifact_id == entry.memory_artifact_id,
                        self.table.c.entry_id == entry.entry_id,
                        self.table.c.entry_version_id == entry.entry_version_id,
                    )
                )
            ).one_or_none()
            if (
                row is None
                or str(row[0]) != entry.entry_content_hash
                or str(row[1]) != _embedding_hash(self.profile, entry.entry_content_hash)
            ):
                hydrated.append(projection)
                continue
            embedding = validate_embedding(
                tuple(float(value) for value in row[2]),
                dimension=self.profile.dimension,
            )
            hydrated.append(
                MemoryProjection(
                    entry_version=entry,
                    searchable_text=projection.searchable_text,
                    embedding=embedding,
                    embedding_content_hash=str(row[1]),
                )
            )
        return tuple(hydrated)

    # -- SQL compilation -------------------------------------------------------

    def _tag_fragment(self, alias: str, request: MemorySearchRequest) -> str:
        if request.tag_filter is None:
            return ""
        return memory_tag_sql(alias)

    def _vector_order(self, request: MemorySearchRequest) -> str:
        return _VECTOR_ORDER_TAGGED if request.tag_filter is not None else _VECTOR_ORDER_UNTAGGED

    def _fts_sql(self, request: MemorySearchRequest) -> object:
        body = _FTS_SQL + self._tag_fragment("h", request) + "\n" + _FTS_ORDER
        return text(body).bindparams(
            bindparam("memory_artifact_ids", expanding=True),
            *self._tag_bindparams(request),
        )

    def _vector_sql(self, request: MemorySearchRequest) -> object:
        body = _VECTOR_SQL + self._tag_fragment("m", request) + "\n" + self._vector_order(request)
        return text(body).bindparams(
            bindparam("memory_artifact_ids", expanding=True),
            bindparam("query_vector", type_=VECTOR(self.profile.dimension)),
            *self._tag_bindparams(request),
        )

    def _hybrid_sql(self, request: MemorySearchRequest) -> object:
        fts = (_FTS_SQL + self._tag_fragment("h", request) + "\n" + _FTS_ORDER).rstrip().rstrip(";")
        vector = (_VECTOR_SQL + self._tag_fragment("m", request) + "\n" + self._vector_order(request))
        vector = vector.rstrip().rstrip(";")
        hybrid = f"(\n{fts}\n) UNION ALL (\n{vector}\n)"
        return text(hybrid).bindparams(
            bindparam("memory_artifact_ids", expanding=True),
            bindparam("query_vector", type_=VECTOR(self.profile.dimension)),
            *self._tag_bindparams(request),
        )

    def _tag_bindparams(self, request: MemorySearchRequest) -> tuple[object, ...]:
        if request.tag_filter is None:
            return ()
        return (
            bindparam("tag_keys", expanding=True),
            bindparam("tag_hashes", expanding=True),
        )

    def _params(self, scope_id: str, request: MemorySearchRequest, channel: str) -> dict[str, object]:
        params: dict[str, object] = {
            "scope_id": scope_id,
            "memory_artifact_ids": tuple(memory.artifact_id for memory in request.memories),
            "analyzed_query": request.analyzed_query,
            "candidate_limit": request.candidate_limit,
        }
        if request.tag_filter is not None:
            params["tag_keys"] = request.tag_filter.keys
            params["tag_hashes"] = _key_hashes(request.tag_filter)
            params["tag_minimum"] = len(request.tag_filter.keys) if request.tag_filter.match == "all" else 1
        if channel in {"vector", "hybrid"}:
            params["query_vector"] = validate_embedding(
                request.query_vector,
                dimension=self.profile.dimension,
            )
        return params
