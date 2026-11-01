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

"""PDF page evidence source for source-03-pdf-retrieval.

Parses a complete original PDF into per-physical-page captured Sources, projects each
page through the standard text-evidence projection, and answers natural-language
finance questions by searching an already-populated scoped Memory via the public
Memory API, returning exact per-page citations.
"""

from __future__ import annotations

import hashlib
import io
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.artifacts.models import MemoryCitation
from powercontext.builtin.runtime.models import SearchMemoryRequest
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.adapters import SourceAdapter
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class PDFDocumentInput(BaseModel):
    """A complete original PDF to be parsed in full."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical page of a parsed PDF."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


def _pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Stable, collision-free identity from (document_id, document_sha256, page_number)."""
    source_id = f"pdf:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("PDF page source_id exceeds the maximum allowed length")
    return source_id


class PDFPageSource(Source):
    """Captured Source for a single PDF page."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of the PDF in order, returning one input per page.

    The identity of each page is anchored to the whole-file SHA-256, so a revised PDF
    produces new identities while old references keep pointing at the original revision.
    Empty/invalid/corrupt input fails atomically with ``ValueError`` and never yields a
    partial result. Encrypted PDFs carrying an empty-user-password permission marker are
    readable without a supplied password.
    """
    document_id = value.document_id
    if not document_id or not document_id.strip():
        raise ValueError("document_id must not be empty or whitespace")
    content = value.content
    if not content:
        raise ValueError("PDF content must not be empty")
    document_sha256 = hashlib.sha256(content).hexdigest()
    try:
        reader = _open_reader(content)
        pages: list[PDFPageInput] = []
        for index, page in enumerate(reader.pages, start=1):
            pages.append(
                PDFPageInput(
                    document_id=document_id,
                    document_sha256=document_sha256,
                    page_number=index,
                    text=page.extract_text() or "",
                )
            )
        if not pages:
            raise ValueError("PDF parsed with zero pages")
        return tuple(pages)
    except ValueError:
        raise
    except Exception as error:  # pypdf raises many exception types for bad input
        raise ValueError(f"failed to parse PDF: {error}") from error


def _open_reader(content: bytes):
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    if reader.is_encrypted:
        try:
            if reader.decrypt("") == 0:
                raise ValueError("PDF requires a non-empty password")
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"unable to decrypt PDF: {error}") from error
    return reader


class _PDFPageAdapter:
    """Resolve/Read the pdf-page Source around ``PDFPageInput``."""

    name = "pdf-page"
    input_class = PDFPageInput
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput) -> PDFPageSource:
        source_id = _pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            definition_version="1",
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class _PDFPageTextEvidenceProjection:
    """Standard text-evidence projection for pdf-page Sources."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource) -> TextEvidence:
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition = AdapterSourceDefinition(
    _PDFPageAdapter(),
    version="1",
    projections=(_PDFPageTextEvidenceProjection(),),
)


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Search an already-populated scoped Memory and return exact per-page citations.

    Each returned ``MemoryCitation`` anchors a distinct physical page; the caller verifies
    distinctness by resolving every citation back to its ``(document_id, page_number)``.
    """
    result = await memory.search(SearchMemoryRequest(query=question, limit=limit, mode="auto"))
    citations: list[MemoryCitation] = []
    for hit in result.hits:
        citations.append(
            MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
        )
        if len(citations) >= limit:
            break
    return tuple(citations)


__all__ = [
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "PDF_PAGE_SOURCE_DEFINITION",
    "search_pdf_pages",
]
