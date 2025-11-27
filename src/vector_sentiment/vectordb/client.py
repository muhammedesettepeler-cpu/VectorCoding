"""Qdrant client wrapper with connection management.

This module provides a wrapper around QdrantClient with health checks,
error handling, and context manager support.
"""

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

from vector_sentiment.config.settings import QdrantSettings


class QdrantClientWrapper:
    """Wrapper for Qdrant client with connection management.

    This class provides a managed Qdrant client with health checks and
    proper resource cleanup.

    Attributes:
        client: The underlying QdrantClient instance
        settings: Qdrant connection settings

    Example:
        >>> settings = QdrantSettings()
        >>> with QdrantClientWrapper(settings) as qd_client:
        ...     collections = qd_client.client.get_collections()
    """

    def __init__(self, settings: QdrantSettings) -> None:
        """Initialize Qdrant client wrapper.

        Args:
            settings: Qdrant connection settings
        """
        self.settings = settings
        self._client: QdrantClient | None = None

        logger.info(
            f"Initializing Qdrant client: host={settings.host}, "
            f"port={settings.port}, prefer_grpc={settings.prefer_grpc}"
        )

    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client instance.

        Returns:
            QdrantClient instance
        """
        if self._client is None:
            self._client = QdrantClient(
                host=self.settings.host,
                port=self.settings.port,
                grpc_port=self.settings.grpc_port,
                prefer_grpc=self.settings.prefer_grpc,
                api_key=self.settings.api_key,
                timeout=self.settings.timeout,
            )
            logger.info("Qdrant client connection established")

        return self._client

    def health_check(self) -> bool:
        """Check if Qdrant server is healthy.

        Returns:
            True if server is healthy, False otherwise

        Example:
            >>> wrapper = QdrantClientWrapper(settings)
            >>> if wrapper.health_check():
            ...     print("Qdrant is healthy")
        """
        try:
            # Try to get collections as a health check
            self.client.get_collections()
            logger.info("Qdrant health check passed")
            return True

        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    def get_collection_info(self, collection_name: str) -> models.CollectionInfo | None:
        """Get collection information.

        Args:
            collection_name: Name of the collection

        Returns:
            CollectionInfo if collection exists, None otherwise
        """
        try:
            return self.client.get_collection(collection_name)
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
            >>> wrapper = QdrantClientWrapper(settings)
            >>> if wrapper.collection_exists("my_vectors"):
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

    def close(self) -> None:
        """Close the Qdrant client connection.

        This method should be called when done with the client to free resources.
        """
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Qdrant client connection closed")

    def __enter__(self) -> "QdrantClientWrapper":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        """Exit context manager and close connection."""
        self.close()
