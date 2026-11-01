"""Incremental context transport for PowerContext.

This module implements the ``context-02-incremental-pack`` wire contract. A full
:class:`ContextPack` carries the entire prepared context split into deterministic,
independently-decodable UTF-8 chunks plus a SHA-256 digest and the exact serialized
byte count. After the first transmission a :class:`ContextPackDelta` lets the receiver
rebuild the next full pack from the previous one by keeping a prefix of chunks and
appending a suffix, without any shared in-process state.

The wire format is the receiver's interop contract:

* Pack JSON (sorted keys, ``,``/``:`` separators, no extra whitespace) contains only
  ``schema``, ``items`` (Base64 string array), ``digest`` and ``byte_count``.
* Delta JSON contains only ``schema``, ``base_digest``, ``target_digest``,
  ``prefix_items`` and ``appended`` (Base64 string array). ``byte_count`` is exposed as
  an attribute but is *not* written into the delta JSON.
* The pack digest is ``sha256:<hex>`` over the canonical JSON of ``{schema, items}``.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Iterable

__all__ = (
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
)

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"

_PACK_FIELDS = ("schema", "items", "digest", "byte_count")
_DELTA_FIELDS = ("schema", "base_digest", "target_digest", "prefix_items", "appended")


class ContextPackError(Exception):
    """Raised on protocol or chunking validation failures."""


def _b64_encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64_decode(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except Exception as error:  # noqa: BLE001 - any decode failure is a protocol error
        raise ContextPackError(f"invalid base64 chunk: {error}") from error


def _digest_of(items: tuple[bytes, ...]) -> str:
    unsigned = {
        "schema": PACK_SCHEMA,
        "items": [_b64_encode(item) for item in items],
    }
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


class ContextPack:
    """A complete, self-describing chunked context payload."""

    __slots__ = ("items", "digest")

    def __init__(self, items: Iterable[bytes], digest: str) -> None:
        normalized: list[bytes] = []
        for item in items:
            if not isinstance(item, bytes):
                raise ContextPackError("ContextPack items must be bytes")
            normalized.append(item)
        self.items: tuple[bytes, ...] = tuple(normalized)
        self.digest: str = digest

    # -- construction -----------------------------------------------------

    @classmethod
    def build(cls, items: Iterable[bytes], /) -> "ContextPack":
        """Build a complete pack from a tuple of chunk bytes."""
        item_tuple = tuple(items)
        for item in item_tuple:
            if not isinstance(item, bytes):
                raise ContextPackError("ContextPack items must be bytes")
        return cls(item_tuple, _digest_of(item_tuple))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = 512) -> "ContextPack":
        """Deterministically chunk UTF-8 ``content`` into valid, bounded pieces.

        Every chunk decodes as valid UTF-8 and is at most ``chunk_bytes`` long. The
        concatenation of the chunks reproduces the original content exactly. An empty
        ``content`` produces an empty pack. A positive ``chunk_bytes`` that cannot form
        a legal character boundary raises :class:`ContextPackError`.
        """
        if chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be a positive integer")
        if not isinstance(content, str):
            raise ContextPackError("content must be a str")
        data = content.encode("utf-8")
        if not data:
            return cls.build(())
        chunks: list[bytes] = []
        index = 0
        total = len(data)
        while index < total:
            end = min(index + chunk_bytes, total)
            while end > index:
                try:
                    data[index:end].decode("utf-8")
                    break
                except UnicodeDecodeError:
                    end -= 1
            else:
                # No non-empty valid UTF-8 chunk fits within the requested limit.
                raise ContextPackError("chunk_bytes too small for a character boundary")
            chunks.append(data[index:end])
            index = end
        return cls.build(tuple(chunks))

    # -- serialization ----------------------------------------------------

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        items_b64 = [_b64_encode(item) for item in self.items]
        # byte_count == len(to_bytes()) including the field itself. The number of
        # digits in byte_count affects the length by only a few bytes, so iterate to
        # the fixed point: len == L0 + 14 + len(str(len)).
        base_length = len(
            json.dumps(
                {"schema": PACK_SCHEMA, "items": items_b64, "digest": self.digest},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        candidate = base_length + 14 + len(str(base_length + 14))
        for _ in range(8):
            next_candidate = base_length + 14 + len(str(candidate))
            if next_candidate == candidate:
                candidate = next_candidate
                break
            candidate = next_candidate
        payload = {
            "schema": PACK_SCHEMA,
            "items": items_b64,
            "digest": self.digest,
            "byte_count": candidate,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes, /) -> "ContextPack":
        try:
            payload = json.loads(data.decode("utf-8"))
        except Exception as error:  # noqa: BLE001
            raise ContextPackError(f"invalid pack JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ContextPackError("pack payload must be a JSON object")
        if set(payload.keys()) != set(_PACK_FIELDS):
            raise ContextPackError("pack payload has unexpected fields")
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError("pack schema mismatch")
        raw_items = payload["items"]
        if not isinstance(raw_items, list) or not all(isinstance(v, str) for v in raw_items):
            raise ContextPackError("pack items must be a list of base64 strings")
        items = tuple(_b64_decode(value) for value in raw_items)
        digest = payload["digest"]
        if not isinstance(digest, str):
            raise ContextPackError("pack digest must be a string")
        byte_count = payload["byte_count"]
        if not isinstance(byte_count, int) or isinstance(byte_count, bool):
            raise ContextPackError("pack byte_count must be an integer")
        pack = cls(items, _digest_of(items))
        if pack.digest != digest:
            raise ContextPackError("pack digest mismatch")
        if pack.byte_count != byte_count:
            raise ContextPackError("pack byte_count mismatch")
        return pack

    # -- dunder -----------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContextPack):
            return NotImplemented
        return self.items == other.items and self.digest == other.digest

    def __hash__(self) -> int:
        return hash((self.items, self.digest))

    def __repr__(self) -> str:
        return f"ContextPack(items={len(self.items)}, digest={self.digest!r})"


class ContextPackDelta:
    """An incremental update that rebuilds a target pack from a base pack."""

    __slots__ = ("base_digest", "target_digest", "prefix_items", "appended")

    def __init__(
        self,
        base_digest: str,
        target_digest: str,
        prefix_items: int,
        appended: Iterable[bytes],
    ) -> None:
        self.base_digest = base_digest
        self.target_digest = target_digest
        self.prefix_items = prefix_items
        self.appended: tuple[bytes, ...] = tuple(appended)

    # -- construction -----------------------------------------------------

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack, /) -> "ContextPackDelta":
        """Select a valid common prefix and carry the remaining target suffix.

        Any valid prefix works; we use the longest common prefix to maximize reuse.
        """
        prefix = 0
        for base_item, target_item in zip(base.items, target.items):
            if base_item == target_item:
                prefix += 1
            else:
                break
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    # -- serialization ----------------------------------------------------

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        payload = {
            "schema": DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": [_b64_encode(item) for item in self.appended],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes, /) -> "ContextPackDelta":
        try:
            payload = json.loads(data.decode("utf-8"))
        except Exception as error:  # noqa: BLE001
            raise ContextPackError(f"invalid delta JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ContextPackError("delta payload must be a JSON object")
        if set(payload.keys()) != set(_DELTA_FIELDS):
            raise ContextPackError("delta payload has unexpected fields")
        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError("delta schema mismatch")
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        if not isinstance(base_digest, str) or not isinstance(target_digest, str):
            raise ContextPackError("delta digests must be strings")
        prefix_items = payload["prefix_items"]
        if not isinstance(prefix_items, int) or isinstance(prefix_items, bool):
            raise ContextPackError("delta prefix_items must be a non-negative integer")
        if prefix_items < 0:
            raise ContextPackError("delta prefix_items must not be negative")
        raw_appended = payload["appended"]
        if not isinstance(raw_appended, list) or not all(isinstance(v, str) for v in raw_appended):
            raise ContextPackError("delta appended must be a list of base64 strings")
        appended = tuple(_b64_decode(value) for value in raw_appended)
        return cls(base_digest, target_digest, prefix_items, appended)

    # -- dunder -----------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContextPackDelta):
            return NotImplemented
        return (
            self.base_digest == other.base_digest
            and self.target_digest == other.target_digest
            and self.prefix_items == other.prefix_items
            and self.appended == other.appended
        )

    def __hash__(self) -> int:
        return hash((self.base_digest, self.target_digest, self.prefix_items, self.appended))

    def __repr__(self) -> str:
        return (
            f"ContextPackDelta(prefix_items={self.prefix_items}, "
            f"appended={len(self.appended)})"
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta, /) -> ContextPack:
    """Validate ``delta`` against ``base`` and return the rebuilt target pack."""
    if not isinstance(base, ContextPack):
        raise ContextPackError("base must be a ContextPack")
    if not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta must be a ContextPackDelta")
    if delta.base_digest != base.digest:
        raise ContextPackError("delta base_digest does not match base pack")
    prefix = delta.prefix_items
    if prefix < 0 or prefix > len(base.items):
        raise ContextPackError("delta prefix_items out of range for base pack")
    target_items = base.items[:prefix] + tuple(delta.appended)
    target = ContextPack.build(target_items)
    if target.digest != delta.target_digest:
        raise ContextPackError("delta target_digest does not match rebuilt pack")
    return target
