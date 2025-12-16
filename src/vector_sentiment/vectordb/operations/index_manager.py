"""Index management operations for Qdrant.

This module handles payload index creation, deletion, and management.
Separated from collection management for single responsibility.
"""

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models


class IndexManager:
    def __init__(self, client: QdrantClient) -> None:
        self.client = client
        logger.info("Initialized IndexManager")

    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: str = "keyword",
    ) -> None:
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

    def delete_payload_index(
        self,
        collection_name: str,
        field_name: str,
    ) -> None:
        logger.info(f"Deleting index on '{field_name}' in collection '{collection_name}'")

        self.client.delete_payload_index(
            collection_name=collection_name,
            field_name=field_name,
        )

        logger.info(f"Deleted payload index on '{field_name}'")

    def list_collection_indexes(self, collection_name: str) -> dict:
        try:
            info = self.client.get_collection(collection_name)
            payload_schema = info.config.params.payload_schema or {}

            logger.debug(f"Collection '{collection_name}' has {len(payload_schema)} indexes")
            return payload_schema

        except Exception as e:
            logger.error(f"Error listing indexes for '{collection_name}': {e}")
            return {}
