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

"""HTTP scoring-service reranker for Memory search candidates.

This adapts an external HTTP reranking service to the existing
``MemoryReranker`` contract. Candidates keep their one-based global rank; the
service scores each batch and the reranker performs one global selection across
the full candidate pool. Only the selection and order change — candidate
identity (entry id, entry version id, text) is never rewritten.
"""

from __future__ import annotations

import json
import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Score candidates through an HTTP reranking service and reorder them.

    The reranker splits the candidate pool into fixed-size batches, sends each
    batch's original global ranks and texts to ``endpoint`` via the caller-
    supplied ``httpx.AsyncClient``, validates the returned scores, and selects
    the top ``limit`` ranks globally. The number of HTTP requests is reported
    through ``InferenceUsage.requests``.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("rerank batch_size must be greater than zero")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Return original ranks ordered by service score, identity preserved."""
        candidate_count = len(candidates)
        if candidate_count < 1:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if limit < 1 or limit > candidate_count:
            raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003

        scores: dict[int, float] = {}
        requests = 0
        for batch in self._build_batches(candidate_count):
            requests += 1
            body: dict[str, Any] = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidates[rank - 1].text} for rank in batch
                ],
            }
            response = await self._client.post(self._endpoint, json=body)
            response.raise_for_status()
            scores.update(self._parse_response(response, set(batch)))

        selected_ranks = self._select_global(scores, limit)
        return MemoryRerankDecision(
            selected_ranks=tuple(selected_ranks),
            usage=InferenceUsage(requests=requests),
        )

    def _build_batches(self, candidate_count: int) -> list[tuple[int, ...]]:
        batches: list[tuple[int, ...]] = []
        for start in range(0, candidate_count, self._batch_size):
            end = min(start + self._batch_size, candidate_count)
            batches.append(tuple(range(start + 1, end + 1)))
        return batches

    @staticmethod
    def _parse_response(response: httpx.Response, expected_ranks: set[int]) -> dict[int, float]:
        try:
            raw = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise ValueError("rerank service returned invalid JSON") from exc

        if not isinstance(raw, dict) or set(raw.keys()) != {"scores"}:
            raise ValueError("rerank response must be an object with exactly 'scores'")
        items = raw["scores"]
        if not isinstance(items, list):
            raise ValueError("rerank 'scores' must be a list")

        seen: dict[int, float] = {}
        for item in items:
            if not isinstance(item, dict) or set(item.keys()) != {"rank", "score"}:
                raise ValueError(
                    "rerank score entry must be an object with exactly 'rank' and 'score'"
                )
            rank = item["rank"]
            score = item["score"]
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
                raise ValueError("rerank score rank must be a positive integer")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("rerank score must be a finite number")
            if rank in seen:
                raise ValueError("rerank response contains a duplicate rank")
            seen[rank] = float(score)

        if set(seen) != expected_ranks:
            raise ValueError("rerank response ranks must exactly cover the requested candidates")
        return seen

    @staticmethod
    def _select_global(scores: dict[int, float], limit: int) -> list[int]:
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [rank for rank, _ in ranked[:limit]]


__all__ = ["HTTPMemoryReranker"]
