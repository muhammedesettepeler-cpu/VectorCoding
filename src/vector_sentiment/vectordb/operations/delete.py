"""Point and collection deletion operations for Qdrant.

This module handles all delete operations for Qdrant collections and points.
"""

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models


class PointDeleter:
    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name
        logger.info(f"Initialized PointDeleter for collection '{collection_name}'")

    def delete_points(self, point_ids: list[int]) -> None:
        logger.info(f"Deleting {len(point_ids)} points")

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=point_ids),
        )

        logger.info(f"Successfully deleted {len(point_ids)} points")

    def delete_points_by_filter(self, filter_condition: models.Filter) -> None:
        logger.info("Deleting points by filter")

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=filter_condition),
        )

        logger.info("Successfully deleted points by filter")


class CollectionDeleter:
    def __init__(self, client: QdrantClient) -> None:
        self.client = client
        logger.info("Initialized CollectionDeleter")

    def delete_collection(self, collection_name: str) -> None:
        logger.warning(f"Deleting collection '{collection_name}'")
        self.client.delete_collection(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' deleted")

    def delete_collection_if_exists(self, collection_name: str) -> bool:
        try:
            self.client.get_collection(collection_name)
            self.delete_collection(collection_name)
            return True
        except Exception:
            logger.info(f"Collection '{collection_name}' does not exist, skipping deletion")
            return False
