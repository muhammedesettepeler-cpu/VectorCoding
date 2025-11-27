"""Configuration module."""

from vector_sentiment.config.constants import (
    BATCH_SIZE_DEFAULT,
    COLLECTION_NAME_DEFAULT,
    DISTANCE_METRIC,
    EMBEDDING_MODEL_DEFAULT,
    VECTOR_SIZE_DEFAULT,
)
from vector_sentiment.config.settings import Settings, get_settings

__all__ = [
    "Settings",
    "get_settings",
    "BATCH_SIZE_DEFAULT",
    "COLLECTION_NAME_DEFAULT",
    "DISTANCE_METRIC",
    "EMBEDDING_MODEL_DEFAULT",
    "VECTOR_SIZE_DEFAULT",
]
