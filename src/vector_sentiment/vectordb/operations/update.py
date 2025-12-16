
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient


class PointUpdater:

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name
        logger.info(f"Initialized PointUpdater for collection '{collection_name}'")

    def update_payload(
        self,
        point_id: int,
        payload: dict[str, Any],
    ) -> None:
        logger.debug(f"Updating payload for point {point_id}")

        self.client.set_payload(
            collection_name=self.collection_name,
            payload=payload,
            points=[point_id],
        )

        logger.debug(f"Successfully updated payload for point {point_id}")

    def update_payload_field(
        self,
        point_id: int,
        field_name: str,
        field_value: Any,  # noqa: ANN401
    ) -> None:

        logger.debug(f"Updating field '{field_name}' for point {point_id}")

        self.client.set_payload(
            collection_name=self.collection_name,
            payload={field_name: field_value},
            points=[point_id],
        )

        logger.debug(f"Successfully updated field '{field_name}' for point {point_id}")

    def update_payload_batch(
        self,
        point_ids: list[int],
        payloads: list[dict[str, Any]],
    ) -> None:

        if len(point_ids) != len(payloads):
            raise ValueError(
                f"Point IDs and payloads length mismatch: {len(point_ids)} vs {len(payloads)}"
            )

        logger.debug(f"Updating payloads for {len(point_ids)} points")

        for point_id, payload in zip(point_ids, payloads, strict=True):
            self.update_payload(point_id=point_id, payload=payload)

        logger.info(f"Successfully updated {len(point_ids)} points")

    def delete_payload_field(
        self,
        point_id: int,
        field_names: list[str],
    ) -> None:
        logger.debug(f"Deleting fields {field_names} from point {point_id}")

        self.client.delete_payload(
            collection_name=self.collection_name,
            keys=field_names,
            points=[point_id],
        )

        logger.debug(f"Successfully deleted fields from point {point_id}")
