"""Collection lifecycle management for Qdrant.

This module handles collection creation, configuration, and information retrieval.
Separated from deletion and index management for single responsibility.
"""

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models


class CollectionManager:
    """Manager for Qdrant collection lifecycle operations.

    This class handles creation and configuration of Qdrant collections.
    Deletion operations are in delete.py, index operations are in index_manager.py.

    Attributes:
        client: QdrantClient instance

    Example:
        >>> manager = CollectionManager(client)
        >>> manager.create_collection(
        ...     "sentiment_vectors",
        ...     vector_size=384,
        ...     vector_name="all-MiniLM-L6-v2"
        ... )
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
        shard_key_field: str | None = None,
        shard_number: int = 4,
        sparse_vector_name: str | None = None,
    ) -> None:
        """Create a new collection with specified configuration.

        This method creates a collection with named vector configuration.
        Named vectors allow storing multiple vector types in the same collection.

        Shard keys enable multi-tenancy and horizontal scaling by partitioning
        data across shards based on a key field (e.g., tenant_id, user_id).

        Args:
            collection_name: Name of the collection to create
            vector_size: Dimension of the vectors
            vector_name: Name for the vector (e.g., model name)
            distance: Distance metric ("Cosine", "Euclid", or "Dot")
            on_disk_payload: Whether to store payload on disk
            shard_key_field: Field name for shard key (enables custom sharding)
            shard_number: Number of shards (used with custom sharding)

        Raises:
            ValueError: If distance metric is invalid

        Example:
            >>> # Standard collection
            >>> manager.create_collection(
            ...     collection_name="sentiment_vectors",
            ...     vector_size=384,
            ...     vector_name="all-MiniLM-L6-v2",
            ... )
            >>>
            >>> # Multi-tenant collection with shard keys
            >>> manager.create_collection(
            ...     collection_name="multi_tenant_vectors",
            ...     vector_size=384,
            ...     vector_name="all-MiniLM-L6-v2",
            ...     shard_key_field="tenant_id",
            ...     shard_number=8,
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

        if sparse_vector_name:
            logger.info(f"Sparse vector enabled: '{sparse_vector_name}'")

        # Build sparse vectors config if enabled
        sparse_vectors_config = None
        if sparse_vector_name:
            sparse_vectors_config = {sparse_vector_name: models.SparseVectorParams()}

        # Create collection with named vectors
        if shard_key_field:
            # Create collection with custom sharding for multi-tenancy
            logger.info(
                f"Creating collection with shard key '{shard_key_field}' and {shard_number} shards"
            )
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=distance_map[distance],
                        on_disk=on_disk_payload,
                    )
                },
                sparse_vectors_config=sparse_vectors_config,
                sharding_method=models.ShardingMethod.CUSTOM,
                shard_number=shard_number,
            )
            logger.info(
                f"Collection '{collection_name}' created with custom sharding "
                f"(shard_key: {shard_key_field})"
            )
        else:
            # Standard collection creation with configurable shards
            logger.info(f"Creating collection '{collection_name}' with {shard_number} shard(s)")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=distance_map[distance],
                        on_disk=on_disk_payload,
                    )
                },
                sparse_vectors_config=sparse_vectors_config,
                shard_number=shard_number,
            )
            logger.info(
                f"Collection '{collection_name}' created successfully with {shard_number} shard(s)"
            )

    def recreate_collection(
        self,
        collection_name: str,
        vector_size: int,
        vector_name: str,
        distance: str = "Cosine",
        sparse_vector_name: str | None = None,
    ) -> None:
        """Delete and recreate a collection.

        This is useful for resetting a collection with fresh configuration.
        Note: This uses CollectionDeleter internally.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of the vectors
            vector_name: Name for the vector
            distance: Distance metric
            sparse_vector_name: Name for sparse vector (for hybrid search)

        Example:
            >>> manager.recreate_collection("vectors", 384, "model-name")
            >>> # With hybrid search support
            >>> manager.recreate_collection("vectors", 384, "model-name", sparse_vector_name="sparse")
        """
        # Import here to avoid circular dependency
        from vector_sentiment.vectordb.operations.delete import CollectionDeleter

        # Check if collection exists and delete
        deleter = CollectionDeleter(self.client)
        deleter.delete_collection_if_exists(collection_name)

        # Create new collection
        self.create_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            vector_name=vector_name,
            distance=distance,
            sparse_vector_name=sparse_vector_name,
        )

    def get_collection_info(self, collection_name: str) -> models.CollectionInfo | None:
        """Get detailed collection information.

        Args:
            collection_name: Name of the collection

        Returns:
            CollectionInfo with details about the collection, or None if not found

        Example:
            >>> info = manager.get_collection_info("vectors")
            >>> if info:
            ...     print(f"Points count: {info.points_count}")
        """
        try:
            info = self.client.get_collection(collection_name)
            logger.debug(f"Collection '{collection_name}' info: {info}")
            return info
        except Exception as e:
            logger.warning(f"Could not get collection '{collection_name}': {e}")
            return None

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists, False otherwise

        Example:
            >>> if manager.collection_exists("sentiment_vectors"):
            ...     print("Collection exists")
        """
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            exists = collection_name in collection_names

            logger.debug(f"Collection '{collection_name}' exists: {exists}")
            return exists

        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False
