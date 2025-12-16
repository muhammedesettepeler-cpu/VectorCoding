from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models


class PointScroller:
    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name
        logger.info(f"Initialized PointScroller for collection '{collection_name}'")

    def scroll_all(
        self,
        batch_size: int = 100,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[list[dict[str, Any]]]:
        offset = None
        total_scrolled = 0

        logger.info(f"Starting scroll with batch_size={batch_size}")

        batches = []
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )

            if not points:
                break

            batch = []
            for point in points:
                point_data: dict[str, Any] = {"id": point.id}
                if with_payload:
                    point_data["payload"] = point.payload
                if with_vectors and hasattr(point, "vector"):
                    point_data["vector"] = point.vector
                batch.append(point_data)

            batches.append(batch)
            total_scrolled += len(points)

            if offset is None:
                break

        logger.info(f"Scrolled through {total_scrolled} total points")
        return batches

    def scroll_with_filter(
        self,
        filter_condition: models.Filter,
        batch_size: int = 100,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[list[dict[str, Any]]]:
        offset = None
        total_scrolled = 0

        logger.info(f"Starting filtered scroll with batch_size={batch_size}")

        batches = []
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=batch_size,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )

            if not points:
                break

            batch = []
            for point in points:
                point_data: dict[str, Any] = {"id": point.id}
                if with_payload:
                    point_data["payload"] = point.payload
                if with_vectors and hasattr(point, "vector"):
                    point_data["vector"] = point.vector
                batch.append(point_data)

            batches.append(batch)
            total_scrolled += len(points)

            if offset is None:
                break

        logger.info(f"Scrolled through {total_scrolled} filtered points")
        return batches
