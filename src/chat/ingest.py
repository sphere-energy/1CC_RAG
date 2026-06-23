"""
Document ingestion pipeline.

Flow:
  1. Download PDF from the provided URL via httpx (presigned S3 or public URL).
  2. Parse text page-by-page with pypdf.
  3. Build a two-level hierarchy:
       • Macro chunks  (~1 500–2 000 chars) – logical sections.
       • Micro chunks  (~300–500 chars)     – paragraphs / sentences within each macro.
  4. Generate Cohere embeddings via BedrockClient for every chunk.
  5. Upsert all points into the Qdrant collection used by the retriever.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from typing import Any

import httpx
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from src.chat.llm import BedrockClient
from src.chat.schemas import DocumentIngestRequest
from src.core.config import Settings

logger = logging.getLogger(__name__)

MACRO_TARGET_CHARS: int = 1800
MACRO_OVERLAP_CHARS: int = 200
MICRO_TARGET_CHARS: int = 400
MICRO_OVERLAP_CHARS: int = 50


class DocumentIngestService:
    """Orchestrates the full ingestion pipeline for a single document."""

    def __init__(self, llm_client: BedrockClient, settings: Settings) -> None:
        self.llm_client = llm_client
        self.settings = settings
        self.qdrant = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=settings.qdrant_timeout_seconds,
            check_compatibility=False,
        )
        self.collection_name = settings.qdrant_collection_name

    def ingest(self, req: DocumentIngestRequest) -> dict[str, Any]:
        """
        Run the full ingestion pipeline synchronously.

        Returns a summary dict with ``total_points`` and ``macro_count``.
        """
        logger.info(
            "Ingestion started",
            extra={
                "legislation_id": str(req.legislation_id),
                "document_id": str(req.document_id),
                "title": req.title,
            },
        )

        # 1. Download PDF to a temporary file.
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        self._download_pdf(req.file_url, tmp_path)

        # 2. Extract text.
        pages = self._extract_text(tmp_path)
        full_text = "\n\n".join(pages)

        if not full_text.strip():
            logger.warning(
                "No extractable text found in PDF (document_id=%s)", req.document_id
            )
            return {"total_points": 0, "macro_count": 0}

        # 3. Build chunk hierarchy.
        macro_groups = self._build_macro_chunks(full_text)

        # 4. Embed & collect Qdrant points.
        points: list[PointStruct] = []
        common_payload: dict[str, Any] = {
            "title": req.title,
            "document_id": str(req.document_id),
            "legislation_id": str(req.legislation_id),
            "publication_date": req.publication_date,
            "source_origin": "uploaded_legislation",
        }

        for macro_text, macro_id, micro_pairs in macro_groups:
            macro_vec = self.llm_client.generate_embedding(macro_text)
            child_ids = [m_id for _, m_id in micro_pairs]

            points.append(
                PointStruct(
                    id=macro_id,
                    vector=macro_vec,
                    payload={
                        **common_payload,
                        "text": macro_text,
                        "chunk_level": "macro",
                        "chunk_id": macro_id,
                        "children": child_ids,
                    },
                )
            )

            for idx, (micro_text, micro_id) in enumerate(micro_pairs):
                micro_vec = self.llm_client.generate_embedding(micro_text)
                prev_id = micro_pairs[idx - 1][1] if idx > 0 else None
                next_id = (
                    micro_pairs[idx + 1][1] if idx < len(micro_pairs) - 1 else None
                )

                points.append(
                    PointStruct(
                        id=micro_id,
                        vector=micro_vec,
                        payload={
                            **common_payload,
                            "text": micro_text,
                            "chunk_level": "micro",
                            "chunk_id": micro_id,
                            "prev_chunk_id": prev_id,
                            "next_chunk_id": next_id,
                        },
                    )
                )

        # 5. Upsert into Qdrant in batches to stay within payload limits.
        batch_size = 64
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.qdrant.upsert(collection_name=self.collection_name, points=batch)
            logger.info(
                "Upserted batch %d/%d (%d points) for document_id=%s",
                i // batch_size + 1,
                (len(points) + batch_size - 1) // batch_size,
                len(batch),
                req.document_id,
            )

        result = {"total_points": len(points), "macro_count": len(macro_groups)}
        logger.info("Ingestion complete: %s", result)

        # 6. Notify KMS that ingestion succeeded.
        self._notify_kms(str(req.document_id), "completed")

        return result

    def _notify_kms(self, document_id: str, status: str) -> None:
        """PATCH the KMS ingest-status endpoint so the UI reflects the final state."""
        base_url = self.settings.kms_callback_url
        if not base_url:
            return
        url = f"{base_url.rstrip('/')}/api/v1/upload/ingest-status/{document_id}"
        headers = {"Content-Type": "application/json"}
        if self.settings.kms_internal_api_key:
            headers["X-Internal-API-Key"] = self.settings.kms_internal_api_key
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.patch(url, json={"status": status}, headers=headers)
                resp.raise_for_status()
            logger.info(
                "KMS ingest-status callback succeeded: document_id=%s status=%s",
                document_id,
                status,
            )
        except Exception as exc:
            logger.warning(
                "KMS ingest-status callback failed: document_id=%s status=%s error=%s",
                document_id,
                status,
                exc,
            )

    def _download_pdf(self, url: str, dest: str) -> None:
        """Download the PDF at *url* and write it to *dest*."""
        logger.info("Downloading PDF from %s", url)
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
        with open(dest, "wb") as fh:
            fh.write(response.content)
        logger.info("PDF downloaded (%d bytes)", len(response.content))

    def _extract_text(self, pdf_path: str) -> list[str]:
        """Return a list of non-empty page texts extracted by pypdf."""
        reader = PdfReader(pdf_path)
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
        logger.info("Extracted text from %d pages", len(pages))
        return pages

    def _build_macro_chunks(
        self,
        text: str,
    ) -> list[tuple[str, str, list[tuple[str, str]]]]:
        """
        Split *text* into macro chunks, each further split into micro chunks.

        Returns a list of ``(macro_text, macro_uuid, [(micro_text, micro_uuid), ...])``.
        """
        macro_groups: list[tuple[str, str, list[tuple[str, str]]]] = []
        raw_macros = self._split_text(text, MACRO_TARGET_CHARS, MACRO_OVERLAP_CHARS)

        for macro_text in raw_macros:
            if not macro_text.strip():
                continue
            macro_id = str(uuid.uuid4())
            raw_micros = self._split_text(
                macro_text, MICRO_TARGET_CHARS, MICRO_OVERLAP_CHARS
            )
            micro_pairs = [(mt, str(uuid.uuid4())) for mt in raw_micros if mt.strip()]
            macro_groups.append((macro_text, macro_id, micro_pairs))

        return macro_groups

    @staticmethod
    def _split_text(text: str, target: int, overlap: int) -> list[str]:
        """
        Greedily split *text* into segments of at most *target* characters,
        preferring paragraph (double-newline), then single-newline, then
        sentence (". ") boundaries before hard-cutting.

        Consecutive segments share *overlap* trailing characters for context.
        """
        chunks: list[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + target, length)

            if end < length:
                boundary = text.rfind("\n\n", start, end)
                if boundary > start + target // 2:
                    end = boundary + 2
                else:
                    boundary = text.rfind("\n", start, end)
                    if boundary > start + target // 2:
                        end = boundary + 1
                    else:
                        boundary = text.rfind(". ", start, end)
                        if boundary > start + target // 2:
                            end = boundary + 2

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            advance = end - start - overlap
            if advance <= 0:
                advance = max(1, end - start)
            start += advance

        return chunks
