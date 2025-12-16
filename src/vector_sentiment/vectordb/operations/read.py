"""Point and collection read operations for Qdrant.

This module handles all read/retrieval operations from Qdrant collections.
"""

from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models


class PointReader:
    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name
        logger.info(f"Initialized PointReader for collection '{collection_name}'")

    def get_point_by_id(self, point_id: int) -> dict[str, Any] | None:
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )

            if points:
                return {
                    "id": points[0].id,
                    "payload": points[0].payload,
                }
            return None

        except Exception as e:
            logger.error(f"Error retrieving point {point_id}: {e}")
            return None

    def get_points_by_ids(
        self,
        point_ids: list[int],
        with_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=point_ids,
                with_payload=True,
                with_vectors=with_vectors,
            )

            results = []
            for point in points:
                result = {
                    "id": point.id,
                    "payload": point.payload,
                }
                if with_vectors and hasattr(point, "vector"):
                    result["vector"] = point.vector
                results.append(result)

            logger.debug(f"Retrieved {len(results)} points")
            return results

        except Exception as e:
            logger.error(f"Error retrieving points: {e}")
            return []

    def get_collection_info(self) -> models.CollectionInfo | None:
        try:
            info = self.client.get_collection(self.collection_name)
            logger.debug(f"Collection '{self.collection_name}' info: {info}")
            return info

        except Exception as e:
            logger.warning(f"Could not get collection '{self.collection_name}': {e}")
            return None

    def collection_exists(self) -> bool:
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            exists = self.collection_name in collection_names

            logger.debug(f"Collection '{self.collection_name}' exists: {exists}")
            return exists

        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def count_points(self) -> int:
        try:
            info = self.get_collection_info()
            if info:
                return info.points_count
            return 0

        except Exception as e:
            logger.error(f"Error counting points: {e}")
            return 0
