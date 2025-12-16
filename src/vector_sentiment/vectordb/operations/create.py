"""Point creation operations for Qdrant.

This module handles creating and upserting points to Qdrant collections.
Separated from update operations for single responsibility.
"""

from collections.abc import Generator
from typing import Any

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models


class PointCreator:
    """Handles point creation and upsert operations.

    This class is responsible ONLY for creating/upserting points to Qdrant.
    It uses the named vector format for compatibility with multiple embedding models.

    Attributes:
        client: QdrantClient instance
        collection_name: Name of the target collection

    Example:
        >>> creator = PointCreator(client, "sentiment_vectors")
        >>> creator.upsert_points(
        ...     vectors=embeddings,
        ...     payloads=[{"label": "positive", "text": "Great!"}],
        ...     vector_name="all-MiniLM-L6-v2",
        ...     start_id=0
        ... )
    """

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        """Initialize point creator.

        Args:
            client: QdrantClient instance
            collection_name: Name of the collection to upload to
        """
        self.client = client
        self.collection_name = collection_name
        logger.info(f"Initialized PointCreator for collection '{collection_name}'")

    def upsert_points(
        self,
        vectors: np.ndarray | list[list[float]],
        payloads: list[dict[str, Any]],
        vector_name: str,
        start_id: int = 0,
        shard_key_selector: str | int | None = None,
        sparse_vectors: list | None = None,
        sparse_vector_name: str | None = None,
    ) -> int:
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

        if sparse_vectors and len(sparse_vectors) != len(vector_list):
            raise ValueError(
                f"Sparse vectors length mismatch: {len(sparse_vectors)} vs {len(vector_list)}"
            )

        batch_size = len(vector_list)

        logger.debug(
            f"Upserting batch of {batch_size} points with vector '{vector_name}', "
            f"starting at ID {start_id}"
        )

        # Generate ID range for this batch
        ids = list(range(start_id, start_id + batch_size))

        # Create points with named vectors
        points = []
        for i, (point_id, vector, payload) in enumerate(
            zip(ids, vector_list, payloads, strict=True)
        ):
            # Build vector dict
            vector_dict: dict = {vector_name: vector}

            # Add sparse vector if provided
            if sparse_vectors and sparse_vector_name:
                sparse_vec = sparse_vectors[i]
                vector_dict[sparse_vector_name] = models.SparseVector(
                    indices=sparse_vec.indices,
                    values=sparse_vec.values,
                )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector_dict,
                    payload=payload,
                )
            )

        # Upsert to Qdrant with shard key routing
        if shard_key_selector is not None:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                shard_key_selector=shard_key_selector,
            )
            logger.debug(
                f"Successfully upserted {batch_size} points to shard '{shard_key_selector}' "
                f"(IDs {start_id}-{start_id + batch_size - 1})"
            )
        else:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.debug(
                f"Successfully upserted {batch_size} points "
                f"(IDs {start_id}-{start_id + batch_size - 1})"
            )

        return batch_size

    def upsert_points_from_generator(
        self,
        vector_generator: Generator[np.ndarray | list[list[float]], None, None],
        payload_generator: Generator[list[dict[str, Any]], None, None],
        vector_name: str,
        batch_size: int = 128,
    ) -> int:
        total_uploaded = 0
        current_id = 0

        logger.info("Starting batch upload from generators")

        try:
            for vectors, payloads in zip(vector_generator, payload_generator, strict=True):
                count = self.upsert_points(
                    vectors=vectors,
                    payloads=payloads,
                    vector_name=vector_name,
                    start_id=current_id,
                )

                total_uploaded += count
                current_id += count

                if total_uploaded % 1000 == 0:
                    logger.info(f"Progress: {total_uploaded} points uploaded")

        except Exception as e:
            logger.error(f"Error during batch upload: {e}")
            raise

        logger.info(f"Completed upload: {total_uploaded} total points")
        return total_uploaded
