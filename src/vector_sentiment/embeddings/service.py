import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from vector_sentiment.config.settings import get_settings


class EmbeddingService:
    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 128,
        normalize: bool = True,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding.model_name
        self.batch_size = batch_size
        self.normalize = normalize

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info(
            f"Model loaded: {self.model_name}, embedding_dim={self.get_embedding_dimension()}"
        )

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        sentences: list[str],
        batch_size: int | None = None,
        normalize_embeddings: bool | None = None,
    ) -> np.ndarray:
        if not sentences:
            return np.array([])

        batch_size = batch_size or self.batch_size
        normalize = normalize_embeddings if normalize_embeddings is not None else self.normalize

        logger.debug(
            f"Encoding {len(sentences)} sentences with batch_size={batch_size}, "
            f"normalize={normalize}"
        )

        embeddings = self.model.encode(
            sentences=sentences,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        logger.debug(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def encode_to_named_vector_format(
        self,
        sentences: list[str],
        batch_size: int | None = None,
        normalize_embeddings: bool | None = None,
    ) -> dict[str, list[list[float]]]:
        embeddings = self.encode(
            sentences=sentences,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )

        # Convert to named vector format (as per PDF requirements)
        named_vectors = {self.model_name: [arr.tolist() for arr in embeddings]}

        return named_vectors

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
