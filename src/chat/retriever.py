import logging
import random
import time
from typing import Any

from pybreaker import CircuitBreakerError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from src.core.circuit_breaker import get_qdrant_breaker
from src.core.config import Settings
from src.core.exceptions import QdrantException

logger = logging.getLogger(__name__)


class QdrantRetriever:
    """
    Retriever class for interacting with Qdrant vector database.
    Implements macro-micro chunk retrieval strategy with circuit breaker protection.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the Qdrant retriever.

        Args:
            settings (Settings): Application settings.

        Raises:
            QdrantException: If client initialization fails.
        """
        try:
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=settings.qdrant_timeout_seconds,
            )
            self.collection_name = settings.qdrant_collection_name
            self.breaker = get_qdrant_breaker()
            self.settings = settings
            logger.info("Qdrant client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Qdrant client: %s", e)
            raise QdrantException(
                message="Failed to initialize Qdrant client",
                detail={"error": str(e)},
            )

    def retrieve(
        self,
        query_embedding: list[float],
        user_query: str,
        top_k_macro: int = 5,
        per_macro_k: int = 3,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Retrieve relevant context using macro-micro strategy with circuit breaker protection.

        Args:
            query_embedding (List[float]): The query embedding vector.
            top_k_macro (int): Number of macro chunks to retrieve. Defaults to 5.
            per_macro_k (int): Number of micro chunks to retrieve per macro chunk. Defaults to 3.

        Returns:
            List[Dict[str, Any]]: A list of retrieved documents with metadata.

        Raises:
            QdrantException: If retrieval fails or circuit is open.
        """
        try:
            return self.breaker.call(
                self._retrieve_impl,
                query_embedding,
                user_query,
                top_k_macro,
                per_macro_k,
            )
        except CircuitBreakerError as e:
            logger.error("Circuit breaker open for Qdrant: %s", e)
            raise QdrantException(
                message="Vector database temporarily unavailable. Please try again later.",
                detail={"circuit_breaker": "open"},
            )

    def _retrieve_impl(
        self,
        query_embedding: list[float],
        user_query: str,
        top_k_macro: int,
        per_macro_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Internal implementation of retrieval."""
        try:
            # Stage 1: Macro retrieval
            macro_points = self._retrieve_macro_chunks(
                query_embedding,
                top_k=top_k_macro,
            )

            # Stage 2: Micro retrieval
            micro_points = self._retrieve_micro_chunks(
                query_embedding,
                macro_points,
                per_macro_k=per_macro_k,
            )

            # Stage 3: Merge and Expand
            all_points = macro_points + micro_points
            expanded_points = self._expand_context(all_points)

            # Sort by score
            sorted_points = sorted(
                expanded_points,
                key=lambda p: p.score if p.score else 0.0,
                reverse=True,
            )

            # Format results
            results = []
            for p in sorted_points[:10]:  # Limit to top 10 unique results
                if p.payload:
                    score = p.score if p.score else 0.0
                    results.append(
                        {
                            "text": p.payload.get("text", ""),
                            "title": p.payload.get("title"),
                            "document_id": p.payload.get("document_id"),
                            "publication_date": p.payload.get("publication_date"),
                            "chunk_level": p.payload.get("chunk_level"),
                            "chunk_id": p.payload.get("chunk_id"),
                            "score": score,
                        },
                    )

            reranked = self._hybrid_rerank(user_query=user_query, docs=results)
            diagnostics = {
                "retrieved_k": len(reranked),
                "rerank_scores": [doc["score"] for doc in reranked[:5]],
            }
            logger.info("Retrieved %d documents", len(results))
            return reranked, diagnostics

        except UnexpectedResponse as e:
            logger.error("Qdrant UnexpectedResponse: %s", e)
            raise QdrantException(
                message="Failed to retrieve documents from Qdrant",
                detail={
                    "error": str(e),
                    "status_code": e.status_code if hasattr(e, "status_code") else None,
                },
            )
        except Exception as e:
            logger.error("Error during retrieval: %s", e)
            raise QdrantException(
                message="Failed to retrieve documents",
                detail={"error": str(e)},
            )

    def _retrieve_macro_chunks(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[Any]:
        """Retrieve top k macro chunks."""
        try:
            results = self._with_retries(
                self.client.query_points,
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="chunk_level",
                            match=MatchValue(value="macro"),
                        ),
                    ],
                ),
                limit=top_k,
            ).points
            return results
        except Exception as e:
            logger.error("Failed to retrieve macro chunks: %s", e)
            raise

    def _retrieve_micro_chunks(
        self,
        query_vector: list[float],
        macro_results: list[Any],
        per_macro_k: int,
    ) -> list[Any]:
        """Retrieve micro chunks related to the retrieved macro chunks."""
        micro_results = []
        for macro_point in macro_results:
            if not macro_point.payload:
                continue

            children = macro_point.payload.get("children", [])
            if not children:
                continue

            try:
                micro_results.extend(
                    self._with_retries(
                        self.client.query_points,
                        collection_name=self.collection_name,
                        query=query_vector,
                        query_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="chunk_id",
                                    match=MatchAny(any=children),
                                ),
                            ],
                        ),
                        limit=per_macro_k,
                    ).points,
                )
            except Exception as e:
                logger.warning(
                    "Failed to retrieve micro chunks for macro %s: %s",
                    macro_point.id,
                    e,
                )
                continue
        return micro_results

    def _expand_context(self, points: list[Any]) -> list[Any]:
        """Expand context by retrieving neighboring chunks."""
        expanded = []
        seen: set[Any] = set()

        # Add original points first
        for p in points:
            if p.id not in seen:
                expanded.append(p)
                seen.add(p.id)

        # Retrieve neighbors
        for p in points:
            if not p.payload:
                continue

            prev_id = p.payload.get("prev_chunk_id")
            next_id = p.payload.get("next_chunk_id")

            for neighbor_id in [prev_id, next_id]:
                if neighbor_id and neighbor_id not in seen:
                    try:
                        neighbor = self._with_retries(
                            self.client.retrieve,
                            collection_name=self.collection_name,
                            ids=[neighbor_id],
                            with_payload=True,
                        )
                        if neighbor:
                            expanded.extend(neighbor)
                            seen.add(neighbor_id)
                    except Exception as e:
                        logger.warning(
                            "Failed to retrieve neighbor %s: %s",
                            neighbor_id,
                            e,
                        )

        return expanded

    def _hybrid_rerank(
        self,
        user_query: str,
        docs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        query_terms = {term for term in user_query.lower().split() if len(term) > 2}
        reranked: list[dict[str, Any]] = []
        for doc in docs:
            text = (doc.get("text") or "").lower()
            lexical_hits = sum(1 for term in query_terms if term in text)
            hybrid_score = float(doc.get("score", 0.0)) + (0.03 * lexical_hits)
            updated = dict(doc)
            updated["score"] = round(hybrid_score, 6)
            reranked.append(updated)

        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked

    def retrieve_by_document(
        self,
        document_id: str | None = None,
        title: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Retrieve all chunks for a specific document using filter-based scroll (no vector similarity).

        Args:
            document_id: Filter chunks by exact document_id value.
            title: Filter chunks by exact title value.

        Returns:
            Tuple of (list of chunk dicts, diagnostics dict).

        Raises:
            QdrantException: If retrieval fails or circuit is open.
        """
        try:
            return self.breaker.call(
                self._retrieve_by_document_impl,
                document_id,
                title,
            )
        except CircuitBreakerError as e:
            logger.error("Circuit breaker open for Qdrant: %s", e)
            raise QdrantException(
                message="Vector database temporarily unavailable. Please try again later.",
                detail={"circuit_breaker": "open"},
            )

    def _retrieve_by_document_impl(
        self,
        document_id: str | None,
        title: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Internal implementation of document-filtered scroll retrieval."""
        conditions = []
        if document_id:
            conditions.append(
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
            )
        if title:
            conditions.append(
                FieldCondition(key="title", match=MatchValue(value=title)),
            )

        scroll_filter = Filter(must=conditions) if conditions else None

        all_records = []
        next_page_offset = None
        while True:
            records, next_page_offset = self._with_retries(
                self.client.scroll,
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=100,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False,
            )
            all_records.extend(records)
            if next_page_offset is None:
                break

        results: list[dict[str, Any]] = []
        for record in all_records:
            if record.payload:
                results.append(
                    {
                        "text": record.payload.get("text", ""),
                        "title": record.payload.get("title"),
                        "document_id": record.payload.get("document_id"),
                        "publication_date": record.payload.get("publication_date"),
                        "chunk_level": record.payload.get("chunk_level"),
                        "chunk_id": record.payload.get("chunk_id"),
                        "score": 1.0,  # No similarity score — full document pinned
                    },
                )

        # Sort by chunk_id to preserve logical reading order
        results.sort(key=lambda x: x.get("chunk_id") or "")

        diagnostics = {
            "retrieved_k": len(results),
            "rerank_scores": [],
            "citation_coverage": 1.0 if results else 0.0,
            "pinned_document": True,
            "pinned_document_id": document_id,
            "pinned_title": title,
        }
        logger.info(
            "Retrieved %d chunks for document (id=%s, title=%s)",
            len(results),
            document_id,
            title,
        )
        return results, diagnostics

    def _with_retries(self, func, *args, **kwargs):
        attempts = self.settings.qdrant_retries
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                retryable = (
                    status_code in {429, 500, 502, 503, 504} or status_code is None
                )
                if not retryable or attempt == attempts:
                    break

                backoff = min(
                    self.settings.external_backoff_max_seconds,
                    self.settings.external_backoff_base_seconds * (2 ** (attempt - 1)),
                )
                sleep_for = backoff + random.uniform(0, 0.2)
                logger.warning(
                    "Qdrant transient failure on attempt %d/%d, retrying in %.2fs",
                    attempt,
                    attempts,
                    sleep_for,
                )
                time.sleep(sleep_for)

        raise last_error
