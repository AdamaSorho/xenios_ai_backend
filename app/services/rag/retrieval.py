"""Retrieval service for semantic search using pgvector."""

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.logging import get_logger
from app.schemas.rag import SearchResult
from app.services.rag.openai_client import OpenAIEmbeddingClient

logger = get_logger(__name__)


class RetrievalService:
    """
    Service for retrieving relevant context using semantic search.

    Uses pgvector for approximate nearest neighbor search with
    cosine similarity and configurable threshold filtering.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.openai_client = OpenAIEmbeddingClient()
        self.settings = get_settings()
        self.similarity_threshold = self.settings.rag_similarity_threshold

    async def retrieve_context(
        self,
        client_id: UUID,
        query: str,
        max_items: int = 10,
        source_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve relevant context for RAG using semantic search.

        Args:
            client_id: Client to search embeddings for
            query: Query text to find similar content
            max_items: Maximum number of results to return
            source_types: Optional filter by source types

        Returns:
            List of SearchResult ordered by relevance (descending)
        """
        # Generate embedding for query
        query_embedding = await self.openai_client.generate_embedding(query)

        # Execute vector search
        raw_results = await self._vector_search(
            client_id=client_id,
            query_embedding=query_embedding,
            limit=max_items,
            source_types=source_types,
        )

        # Filter by threshold and build results
        results: list[SearchResult] = []
        for row in raw_results:
            # Threshold check (already done in query, but double-check)
            if row["similarity"] < self.similarity_threshold:
                continue

            # Fetch full content if needed (for single-record types)
            content = await self._fetch_source_content(row)
            if content is None:
                # Source was deleted or inaccessible
                continue

            results.append(
                SearchResult(
                    source_type=row["source_type"],
                    source_id=row["source_id"],
                    content=content,
                    relevance_score=row["similarity"],
                    metadata=row["metadata"] or {},
                )
            )

        logger.info(
            "Retrieved context",
            client_id=str(client_id),
            query_length=len(query),
            results_count=len(results),
            threshold=self.similarity_threshold,
        )

        return results

    async def _vector_search(
        self,
        client_id: UUID,
        query_embedding: list[float],
        limit: int,
        source_types: list[str] | None,
    ) -> list[dict]:
        """
        Execute pgvector similarity search.

        Uses cosine distance operator (<=>). Similarity = 1 - distance.
        """
        # Build source type filter
        source_filter = ""
        params: dict = {
            "client_id": str(client_id),
            "limit": limit,
            "threshold": self.similarity_threshold,
        }

        if source_types:
            source_filter = "AND source_type = ANY(:source_types)"
            params["source_types"] = source_types

        # pgvector requires the embedding as a string representation
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        params["embedding"] = embedding_str

        query = text(f"""
            SELECT
                id,
                source_type,
                source_id,
                source_table,
                content_text,
                metadata,
                1 - (embedding <=> :embedding::vector) as similarity
            FROM ai_backend.embeddings
            WHERE client_id = :client_id
            {source_filter}
            AND 1 - (embedding <=> :embedding::vector) >= :threshold
            ORDER BY embedding <=> :embedding::vector
            LIMIT :limit
        """)

        result = await self.db.execute(query, params)
        rows = result.fetchall()

        return [
            {
                "id": row.id,
                "source_type": row.source_type,
                "source_id": row.source_id,
                "source_table": row.source_table,
                "content_text": row.content_text,
                "metadata": row.metadata,
                "similarity": float(row.similarity),
            }
            for row in rows
        ]

    async def _fetch_source_content(self, result: dict) -> str | None:
        """
        Fetch full source content based on type.

        For aggregated types (profile, metric_summary, message_thread),
        we use the stored content_text directly.

        For single-record types, we could join back to source tables,
        but for simplicity we use stored content_text which is already
        a comprehensive summary.

        Returns None if source is deleted or inaccessible.
        """
        source_type = result["source_type"]
        source_table = result["source_table"]
        source_id = result["source_id"]
        content_text = result["content_text"]

        # For aggregated types, always use stored content
        if source_type in ["health_profile", "health_metric_summary", "message_thread"]:
            return content_text

        # For single-record types, verify source still exists
        # (defense in depth - embedding should be updated when source deleted)
        if source_table and ":" not in source_id:  # UUID source_id
            try:
                # Check if source record still exists
                check_query = text(f"""
                    SELECT 1 FROM {source_table}
                    WHERE id = :source_id
                    LIMIT 1
                """)
                check_result = await self.db.execute(check_query, {"source_id": source_id})
                if check_result.scalar_one_or_none() is None:
                    logger.warning(
                        "Source record not found",
                        source_type=source_type,
                        source_id=source_id,
                        source_table=source_table,
                    )
                    return None
            except Exception as e:
                # Table might not exist or other error - use cached content
                logger.debug(
                    "Could not verify source, using cached content",
                    source_type=source_type,
                    error=str(e),
                )

        return content_text

    async def search_similar(
        self,
        client_id: UUID,
        text: str,
        limit: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Simple similarity search returning source IDs and scores.

        Useful for deduplication checks.
        """
        embedding = await self.openai_client.generate_embedding(text)
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        query = text("""
            SELECT
                source_id,
                1 - (embedding <=> :embedding::vector) as similarity
            FROM ai_backend.embeddings
            WHERE client_id = :client_id
            ORDER BY embedding <=> :embedding::vector
            LIMIT :limit
        """)

        result = await self.db.execute(
            query,
            {
                "client_id": str(client_id),
                "embedding": embedding_str,
                "limit": limit,
            },
        )

        return [(row.source_id, float(row.similarity)) for row in result.fetchall()]
