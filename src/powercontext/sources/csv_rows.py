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

"""CSV row evidence Source for the table-evidence task (source-01)."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.sources import (
    AdapterSourceDefinition,
    SourceDefinition,
    TEXT_EVIDENCE_PROJECTION_KEY,
    TextEvidence,
)
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.models import Source, SourceMaterialization


class CSVDocumentInput(BaseModel):
    """A complete offline CSV document with a stable identity."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """The exact identity of one row within a CSV document."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """One captured CSV row, stored as durable evidence."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Stable identity: csv:{encoded document_id}:{encoded primary_key_value}."""
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain leading or trailing whitespace")
    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("source_id exceeds the maximum allowed length")
    return source_id


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Decode and validate a CSV document, one CSVRowInput per data row.

    Raises ValueError on any encoding, CSV-format, or data-validation failure.
    The whole document must parse and validate before any row is returned.
    """
    document_id = value.document_id
    primary_key = value.primary_key

    if not document_id:
        raise ValueError("document_id must not be empty")
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain leading or trailing whitespace")
    if not primary_key:
        raise ValueError("primary_key must not be empty")

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"content is not valid UTF-8: {exc}") from exc

    try:
        rows = list(csv.reader(io.StringIO(text), strict=True))
    except csv.Error as exc:
        raise ValueError(f"invalid CSV structure: {exc}") from exc
    if not rows:
        raise ValueError("empty file: no rows")

    header = rows[0]
    if len(header) == 0:
        raise ValueError("empty header")

    seen_columns: set[str] = set()
    for column in header:
        if column == "" or column.isspace():
            raise ValueError("header fields must be non-empty and non-whitespace-only")
        if column in seen_columns:
            raise ValueError(f"duplicate header field: {column!r}")
        seen_columns.add(column)

    if primary_key not in header:
        raise ValueError(f"primary_key {primary_key!r} is not a header column")

    pk_index = header.index(primary_key)
    result: list[CSVRowInput] = []
    seen_keys: set[str] = set()

    for raw in rows[1:]:
        if len(raw) == 0:
            # A completely empty physical record after the header is ignored.
            continue
        if len(raw) != len(header):
            raise ValueError(
                f"row has {len(raw)} fields but the header has {len(header)}"
            )
        pk_value = raw[pk_index]
        if pk_value == "" or pk_value.isspace():
            raise ValueError("primary key value must not be empty or whitespace-only")
        if pk_value in seen_keys:
            raise ValueError(f"duplicate primary key value: {pk_value!r}")
        seen_keys.add(pk_value)
        result.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=pk_value,
                columns=tuple(header),
                values=tuple(raw),
            )
        )

    return tuple(result)


class _CSVRowAdapter:
    input_class = CSVRowInput
    name = "csv-row"
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        identity = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=identity,
            source_id=identity,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class _CSVRowTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values))
        content = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        return TextEvidence(
            source_type="csv-row",
            source_id=source.source_id,
            content=content,
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = (
    AdapterSourceDefinition(
        _CSVRowAdapter(),
        version="1",
        projections=(_CSVRowTextEvidenceProjection(),),
    )
)
