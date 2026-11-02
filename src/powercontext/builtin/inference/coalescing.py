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

"""Concurrent request coalescing and bounded caching for embedding models.

Multiple callers that submit the same texts on one event loop share a single
in-flight base-model request per distinct text, successful vectors are reused
through a bounded LRU cache, and every caller keeps its own view of order,
errors, cancellation, and provider usage.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Sequence

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel

__all__ = ["CoalescingEmbeddingModel"]


def _consume_outcome(future: "asyncio.Future[_SharedBatch]") -> None:
    """Mark a settled waiter's exception as retrieved so unobserved failures stay quiet."""

    if not future.cancelled():
        future.exception()


class _SharedBatch:
    """One deduplicated in-flight base request plus the callers waiting on it."""

    __slots__ = ("texts", "vectors", "usage", "error", "waiters")

    def __init__(self, texts: tuple[str, ...]) -> None:
        self.texts = texts
        self.vectors: dict[str, EmbeddingVector] = {}
        self.usage: InferenceUsage | None = None
        self.error: BaseException | None = None
        self.waiters: list[asyncio.Future[_SharedBatch]] = []


class CoalescingEmbeddingModel:
    """Wrap an embedding model with cross-call request coalescing and an LRU cache.

    Identical texts from concurrent callers share one in-flight base request;
    successful vectors are cached in a bounded LRU (``cache_size=0`` disables
    the cross-call cache while in-flight sharing still applies). Cancelling one
    caller never cancels the shared base request or the results of other
    waiters, failed batches are never cached so they can be retried, and each
    caller's usage sums the distinct base batches it actually waited on.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")
        if cache_size < 0:
            raise ValueError("cache_size must not be negative")
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, _SharedBatch] = {}
        self._workers: set[asyncio.Task[None]] = set()

    async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
        """Return one ordered vector per input text, sharing work across callers."""

        ordered = tuple(texts)
        if not ordered:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        cache_hits: dict[str, EmbeddingVector] = {}
        batch_by_text: dict[str, _SharedBatch] = {}
        waited: dict[_SharedBatch, asyncio.Future[_SharedBatch]] = {}
        fresh: list[str] = []
        seen: set[str] = set()

        for text in ordered:
            if text in cache_hits or text in batch_by_text or text in seen:
                continue
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                cache_hits[text] = cached
                continue
            batch = self._inflight.get(text)
            if batch is not None:
                self._join(batch, waited)
                batch_by_text[text] = batch
                continue
            seen.add(text)
            fresh.append(text)

        for start in range(0, len(fresh), self._max_batch_size):
            batch = _SharedBatch(tuple(fresh[start : start + self._max_batch_size]))
            for text in batch.texts:
                self._inflight[text] = batch
                batch_by_text[text] = batch
            self._join(batch, waited)
            self._schedule(batch)

        if waited:
            await asyncio.gather(*waited.values())

        vectors = tuple(
            cache_hits[text] if text in cache_hits else batch_by_text[text].vectors[text]
            for text in ordered
        )
        return EmbeddingResult(vectors=vectors, usage=self._aggregate(waited))

    def _join(self, batch: _SharedBatch, waited: dict[_SharedBatch, asyncio.Future[_SharedBatch]]) -> None:
        """Attach one waiter future to a batch, at most once per call."""

        if batch in waited:
            return
        future: asyncio.Future[_SharedBatch] = asyncio.get_running_loop().create_future()
        future.add_done_callback(_consume_outcome)
        batch.waiters.append(future)
        waited[batch] = future

    def _schedule(self, batch: _SharedBatch) -> None:
        worker = asyncio.create_task(self._run(batch))
        self._workers.add(worker)
        worker.add_done_callback(self._on_worker_done)

    def _on_worker_done(self, worker: asyncio.Task[None]) -> None:
        self._workers.discard(worker)
        if not worker.cancelled():
            worker.exception()

    async def _run(self, batch: _SharedBatch) -> None:
        try:
            if self._coalesce_seconds > 0:
                await asyncio.sleep(self._coalesce_seconds)
            result = await self._model.embed(batch.texts)
            vectors = tuple(result.vectors)
            if len(vectors) != len(batch.texts):
                raise ValueError(
                    f"embedding model returned {len(vectors)} vectors for {len(batch.texts)} texts"
                )
        except BaseException as error:
            self._settle(batch, error=error)
            raise
        self._settle(
            batch,
            vectors=dict(zip(batch.texts, vectors, strict=True)),
            usage=result.usage,
        )

    def _settle(
        self,
        batch: _SharedBatch,
        *,
        vectors: dict[str, EmbeddingVector] | None = None,
        usage: InferenceUsage | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Release a batch's texts and resolve every waiter, synchronously."""

        for text in batch.texts:
            if self._inflight.get(text) is batch:
                del self._inflight[text]
        if error is None:
            if self._cache_size > 0:
                for text, vector in vectors.items():  # type: ignore[union-attr]
                    self._cache[text] = vector
                    self._cache.move_to_end(text)
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
            batch.vectors = vectors or {}
            batch.usage = usage
            for future in batch.waiters:
                if not future.done():
                    future.set_result(batch)
        else:
            batch.error = error
            for future in batch.waiters:
                if not future.done():
                    future.set_exception(error)
        batch.waiters.clear()

    def _aggregate(self, waited: dict[_SharedBatch, asyncio.Future[_SharedBatch]]) -> InferenceUsage:
        """Sum provider usage once per distinct batch this call waited on."""

        requests = 0
        input_tokens: int | None = None
        output_tokens: int | None = None
        for batch in waited:
            usage = batch.usage
            if usage is None:
                continue
            requests += usage.requests
            if usage.input_tokens is not None:
                input_tokens = (0 if input_tokens is None else input_tokens) + usage.input_tokens
            if usage.output_tokens is not None:
                output_tokens = (0 if output_tokens is None else output_tokens) + usage.output_tokens
        return InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)
