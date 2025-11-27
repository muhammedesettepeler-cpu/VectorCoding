"""Collection management for Qdrant.

This module provides collection creation, deletion, and configuration management
for Qdrant vector storage.
"""

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models


class CollectionManager:
    """Manager for Qdrant collection operations.

    This class handles creation, deletion, and configuration of Qdrant collections
    including vector configuration and distance metrics.

    Attributes:
        client: QdrantClient instance

    Example:
        >>> manager = CollectionManager(client)
        >>> manager.create_collection("vectors", vector_size=384)
    """

    def __init__(self, client: QdrantClient) -> None:
        """Initialize collection manager.

        Args:
            client: QdrantClient instance
        """
        self.client = client
        logger.info("Initialized CollectionManager")

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        vector_name: str,
        distance: str = "Cosine",
        on_disk_payload: bool = False,
    ) -> None:
        """Create a new collection with specified configuration.

        This method creates a collection with named vector configuration.
        Named vectors allow storing multiple vector types in the same collection.

        Args:
            collection_name: Name of the collection to create
            vector_size: Dimension of the vectors
            vector_name: Name for the vector (e.g., model name)
            distance: Distance metric ("Cosine", "Euclid", or "Dot")
            on_disk_payload: Whether to store payload on disk

        Raises:
            ValueError: If distance metric is invalid

        Example:
            >>> manager = CollectionManager(client)
            >>> manager.create_collection(
            ...     collection_name="sentiment_vectors",
            ...     vector_size=384,
            ...     vector_name="all-MiniLM-L6-v2",
            ...     distance="Cosine"
            ... )
        """
        # Validate distance metric
        valid_distances = {"Cosine", "Euclid", "Dot"}
        if distance not in valid_distances:
            raise ValueError(f"Distance must be one of {valid_distances}")

        # Map string distance to Qdrant Distance enum
        distance_map = {
            "Cosine": models.Distance.COSINE,
            "Euclid": models.Distance.EUCLID,
            "Dot": models.Distance.DOT,
        }

        logger.info(
            f"Creating collection '{collection_name}' with vector_name='{vector_name}', "
            f"size={vector_size}, distance={distance}"
        )

        # Create collection with named vectors
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                vector_name: models.VectorParams(
                    size=vector_size,
                    distance=distance_map[distance],
                    on_disk=on_disk_payload,
                )
            },
        )

        logger.info(f"Collection '{collection_name}' created successfully")

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection.

        Args:
            collection_name: Name of the collection to delete

        Example:
            >>> manager = CollectionManager(client)
            >>> manager.delete_collection("old_vectors")
        """
        logger.warning(f"Deleting collection '{collection_name}'")
        self.client.delete_collection(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' deleted")

    def recreate_collection(
        self,
        collection_name: str,
        vector_size: int,
        vector_name: str,
        distance: str = "Cosine",
    ) -> None:
        """Delete and recreate a collection.

        This is useful for resetting a collection with fresh configuration.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of the vectors
            vector_name: Name for the vector
            distance: Distance metric

        Example:
            >>> manager = CollectionManager(client)
            >>> manager.recreate_collection("vectors", 384, "model-name")
        """
        # Check if collection exists
        try:
            self.client.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' exists, deleting...")
            self.delete_collection(collection_name)
        except Exception:
            logger.info(f"Collection '{collection_name}' does not exist")

        # Create new collection
        self.create_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            vector_name=vector_name,
            distance=distance,
        )

    def get_collection_info(self, collection_name: str) -> models.CollectionInfo:
        """Get detailed collection information.

        Args:
            collection_name: Name of the collection

        Returns:
            CollectionInfo with details about the collection

        Example:
            >>> manager = CollectionManager(client)
            >>> info = manager.get_collection_info("vectors")
            >>> print(f"Points count: {info.points_count}")
        """
        info = self.client.get_collection(collection_name)
        logger.debug(f"Collection '{collection_name}' info: {info}")
        return info

    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: str = "keyword",
    ) -> None:
        """Create an index on a payload field.

        Payload indexes improve query performance when filtering by metadata.
        This is important for production deployments with large datasets.

        Args:
            collection_name: Name of the collection
            field_name: Name of the payload field to index
            field_schema: Type of index ("keyword", "integer", "float", "geo")

        Example:
            >>> manager = CollectionManager(client)
            >>> # Create index on 'label' field for fast filtering
            >>> manager.create_payload_index("vectors", "label", "keyword")
        """
        schema_map = {
            "keyword": models.PayloadSchemaType.KEYWORD,
            "integer": models.PayloadSchemaType.INTEGER,
            "float": models.PayloadSchemaType.FLOAT,
            "geo": models.PayloadSchemaType.GEO,
        }

        if field_schema not in schema_map:
            raise ValueError(f"Field schema must be one of {list(schema_map.keys())}")

        logger.info(
            f"Creating {field_schema} index on '{field_name}' in collection '{collection_name}'"
        )

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_map[field_schema],
        )

        logger.info(f"Payload index created on '{field_name}'")
