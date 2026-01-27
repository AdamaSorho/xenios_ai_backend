"""OpenAI embeddings client for RAG system."""

from openai import AsyncOpenAI

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAIEmbeddingError(Exception):
    """Exception raised for OpenAI embedding errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class OpenAIEmbeddingClient:
    """
    Client for OpenAI embeddings API.

    Uses text-embedding-ada-002 model which produces 1536-dimension vectors.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.rag_embedding_model
        self.dimensions = self.settings.rag_embedding_dimensions

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            1536-dimension embedding vector

        Raises:
            OpenAIEmbeddingError: On API errors
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            embedding = response.data[0].embedding

            logger.debug(
                "Generated embedding",
                model=self.model,
                text_length=len(text),
                dimensions=len(embedding),
            )

            return embedding

        except Exception as e:
            logger.error(
                "Failed to generate embedding",
                model=self.model,
                error=str(e),
            )
            raise OpenAIEmbeddingError(f"Failed to generate embedding: {e}") from e

    async def generate_embeddings_batch(
        self, texts: list[str], batch_size: int = 100
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Maximum texts per API call (OpenAI limit is ~2048)

        Returns:
            List of embedding vectors in same order as input

        Raises:
            OpenAIEmbeddingError: On API errors
        """
        if not texts:
            return []

        try:
            all_embeddings: list[list[float]] = []

            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )

                # Sort by index to maintain order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                batch_embeddings = [item.embedding for item in sorted_data]
                all_embeddings.extend(batch_embeddings)

            logger.info(
                "Generated batch embeddings",
                model=self.model,
                total_texts=len(texts),
                batches=len(range(0, len(texts), batch_size)),
            )

            return all_embeddings

        except Exception as e:
            logger.error(
                "Failed to generate batch embeddings",
                model=self.model,
                batch_size=len(texts),
                error=str(e),
            )
            raise OpenAIEmbeddingError(f"Failed to generate batch embeddings: {e}") from e
