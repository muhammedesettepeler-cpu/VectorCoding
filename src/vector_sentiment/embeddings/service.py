"""Embedding service using SentenceTransformers.

This module provides embedding generation using the SentenceTransformers library.
It supports batch encoding with normalization and provides the named vector format
required for Qdrant.

ALTERNATIVE: FastEmbed (not implemented)
---------------------------------------
FastEmbed is a faster, more memory-efficient alternative to SentenceTransformers.
It provides:
- Lower memory footprint (no PyTorch dependency)
- Faster inference (ONNX optimization)
- Same model compatibility

Usage example (not implemented in this project):
```python
from fastembed import TextEmbedding

model = TextEmbedding(model_name="all-MiniLM-L6-v2")
embeddings = list(model.embed(texts))
```

Benefits over SentenceTransformers:
- 2-3x faster inference
- 50% less memory usage
- Better for production deployments

Trade-offs:
- Less flexibility for custom models
- Smaller model selection
"""

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Embedding generation service using SentenceTransformers.

    This class wraps SentenceTransformer to provide consistent embedding
    generation with batching and normalization support.

    Attributes:
        model_name: Name of the SentenceTransformer model
        model: The loaded SentenceTransformer instance
        batch_size: Default batch size for encoding
        normalize: Whether to normalize embeddings

    Example:
        >>> service = EmbeddingService("all-MiniLM-L6-v2")
        >>> embeddings = service.encode(["Hello world", "Test sentence"])
        >>> print(embeddings.shape)  # (2, 384)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 128,
        normalize: bool = True,
    ) -> None:
        """Initialize embedding service.

        Args:
            model_name: Name of the SentenceTransformer model
            batch_size: Default batch size for encoding
            normalize: Whether to L2-normalize embeddings
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Model loaded: {model_name}, embedding_dim={self.get_embedding_dimension()}")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embedding vectors.

        Returns:
            Embedding vector dimension

        Example:
            >>> service = EmbeddingService("all-MiniLM-L6-v2")
            >>> print(service.get_embedding_dimension())  # 384
        """
        return self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        sentences: list[str],
        batch_size: int | None = None,
        normalize_embeddings: bool | None = None,
    ) -> np.ndarray:
        """Encode sentences to embedding vectors.

        This method generates embeddings for a list of sentences using the
        configured SentenceTransformer model.

        Args:
            sentences: List of text sentences to encode
            batch_size: Batch size for encoding (uses default if None)
            normalize_embeddings: Whether to normalize (uses default if None)

        Returns:
            NumPy array of shape (len(sentences), embedding_dim)

        Example:
            >>> service = EmbeddingService()
            >>> texts = ["Good product", "Bad service", "Neutral review"]
            >>> embeddings = service.encode(texts)
            >>> print(embeddings.shape)  # (3, 384)
        """
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
        """Encode sentences to named vector format for Qdrant.

        This method generates the specific named vector format required by Qdrant
        as specified in the case study requirements:

        vectors = {
            embedding_model_name: [
                arr.tolist()
                for arr in model.encode(
                    sentences=data,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                )
            ]
        }

        Args:
            sentences: List of text sentences to encode
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to normalize

        Returns:
            Dictionary with model name as key and list of vectors as value

        Example:
            >>> service = EmbeddingService("all-MiniLM-L6-v2")
            >>> texts = ["Hello", "World"]
            >>> named_vectors = service.encode_to_named_vector_format(texts)
            >>> print(list(named_vectors.keys()))  # ['all-MiniLM-L6-v2']
            >>> print(len(named_vectors['all-MiniLM-L6-v2']))  # 2
        """
        embeddings = self.encode(
            sentences=sentences,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )

        # Convert to named vector format (as per PDF requirements)
        named_vectors = {self.model_name: [arr.tolist() for arr in embeddings]}

        return named_vectors

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to embedding vector.

        Args:
            text: Text to encode

        Returns:
            NumPy array of shape (embedding_dim,)

        Example:
            >>> service = EmbeddingService()
            >>> embedding = service.encode_single("Test sentence")
            >>> print(embedding.shape)  # (384,)
        """
        return self.encode([text])[0]
