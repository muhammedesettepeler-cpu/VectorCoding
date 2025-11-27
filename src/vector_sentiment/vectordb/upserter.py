"""Vector upserter with named vector format.

This module provides vector upload functionality to Qdrant using the upsert operation
with named vector format and batch processing for memory efficiency.

IMPORTANT: This module uses client.upsert() and NOT client.add() as per requirements.
"""

from collections.abc import Generator
from typing import Any

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models


class VectorUpserter:
    """Vector upserter for batch operations.

    This class handles batch vector upload to Qdrant with named vector format,
    payload attachment, and progress tracking.

    The named vector format uses a dictionary with the embedding model name as key:
    vectors = {
        embedding_model_name: [vector1.tolist(), vector2.tolist(), ...]
    }

    Attributes:
        client: QdrantClient instance
        collection_name: Name of the target collection

    Example:
        >>> upserter = VectorUpserter(client, "sentiment_vectors")
        >>> upserter.upsert_batch(
        ...     vectors=embeddings,
        ...     payloads=[{"label": "positive", "text": "Great!"}],
        ...     vector_name="all-MiniLM-L6-v2",
        ...     start_id=0
        ... )
    """

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        """Initialize vector upserter.

        Args:
            client: QdrantClient instance
            collection_name: Name of the collection to upload to
        """
        self.client = client
        self.collection_name = collection_name
        logger.info(f"Initialized VectorUpserter for collection '{collection_name}'")

    def upsert_batch(
        self,
        vectors: np.ndarray | list[list[float]],
        payloads: list[dict[str, Any]],
        vector_name: str,
        start_id: int = 0,
    ) -> int:
        """Upsert a batch of vectors with named vector format.

        This method uploads vectors using the named vector format as specified in
        the case study requirements:

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
            vectors: Array or list of embedding vectors
            payloads: List of metadata dictionaries (one per vector)
            vector_name: Name of the embedding model (used as vector name)
            start_id: Starting ID for the batch

        Returns:
            Number of vectors uploaded

        Raises:
            ValueError: If vectors and payloads have different lengths

        Example:
            >>> vectors = np.array([[0.1, 0.2], [0.3, 0.4]])
            >>> payloads = [{"label": "positive"}, {"label": "negative"}]
            >>> upserter.upsert_batch(
            ...     vectors=vectors,
            ...     payloads=payloads,
            ...     vector_name="all-MiniLM-L6-v2",
            ...     start_id=0
            ... )
            2
        """
        # Convert vectors to list if numpy array
        if isinstance(vectors, np.ndarray):
            vector_list = [arr.tolist() for arr in vectors]
        else:
            vector_list = vectors

        # Validate inputs
        if len(vector_list) != len(payloads):
            raise ValueError(
                f"Vectors and payloads length mismatch: {len(vector_list)} vs {len(payloads)}"
            )

        batch_size = len(vector_list)

        logger.debug(
            f"Upserting batch of {batch_size} vectors with name '{vector_name}', "
            f"starting at ID {start_id}"
        )

        # Create named vector format (as per PDF requirements)
        named_vectors = {vector_name: vector_list}

        # Generate ID range for this batch
        ids = list(range(start_id, start_id + batch_size))

        # Create points with named vectors
        points = [
            models.PointStruct(
                id=point_id,
                vector=named_vectors,
                payload=payload,
            )
            for point_id, payload in zip(ids, payloads, strict=True)
        ]

        # Upsert to Qdrant (NOT using add() as per requirements)
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        logger.debug(
            f"Successfully upserted {batch_size} vectors "
            f"(IDs {start_id}-{start_id + batch_size - 1})"
        )

        return batch_size

    def upsert_generator(
        self,
        vector_generator: Generator[np.ndarray | list[list[float]], None, None],
        payload_generator: Generator[list[dict[str, Any]], None, None],
        vector_name: str,
        batch_size: int = 128,
    ) -> int:
        """Upsert vectors from generators.

        This method accepts generators for memory-efficient batch processing
        of large datasets.

        Args:
            vector_generator: Generator yielding batches of vectors
            payload_generator: Generator yielding batches of payloads
            vector_name: Name of the embedding model
            batch_size: Size of each batch

        Returns:
            Total number of vectors uploaded

        Example:
            >>> def vector_gen():
            ...     for batch in data_batches:
            ...         yield model.encode(batch)
            >>>
            >>> def payload_gen():
            ...     for batch in data_batches:
            ...         yield [{"label": item.label} for item in batch]
            >>>
            >>> total = upserter.upsert_generator(
            ...     vector_gen(), payload_gen(), "all-MiniLM-L6-v2"
            ... )
        """
        total_uploaded = 0
        current_id = 0

        logger.info("Starting batch upload from generators")

        try:
            for vectors, payloads in zip(vector_generator, payload_generator, strict=True):
                count = self.upsert_batch(
                    vectors=vectors,
                    payloads=payloads,
                    vector_name=vector_name,
                    start_id=current_id,
                )

                total_uploaded += count
                current_id += count

                if total_uploaded % 1000 == 0:
                    logger.info(f"Progress: {total_uploaded} vectors uploaded")

        except Exception as e:
            logger.error(f"Error during batch upload: {e}")
            raise

        logger.info(f"Completed upload: {total_uploaded} total vectors")
        return total_uploaded
