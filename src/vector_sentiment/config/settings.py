"""Application settings using Pydantic.

This module provides environment-based configuration management using Pydantic
settings. All settings can be configured via environment variables or .env file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from vector_sentiment.config.constants import (
    BATCH_SIZE_DEFAULT,
    COLLECTION_NAME_DEFAULT,
    DISTANCE_METRIC,
    EMBEDDING_MODEL_DEFAULT,
    LOG_LEVEL_DEFAULT,
    LOG_RETENTION,
    LOG_ROTATION,
    NORMALIZE_EMBEDDINGS,
    QDRANT_GRPC_PORT_DEFAULT,
    QDRANT_HOST_DEFAULT,
    QDRANT_PORT_DEFAULT,
    QDRANT_PREFER_GRPC_DEFAULT,
    QDRANT_TIMEOUT,
    SEARCH_LIMIT_DEFAULT,
    SEARCH_SCORE_THRESHOLD_DEFAULT,
    VECTOR_SIZE_DEFAULT,
)


class QdrantSettings(BaseSettings):
    url: str | None = Field(default=None, description="Qdrant URL (for cloud connections)")
    host: str = Field(default=QDRANT_HOST_DEFAULT, description="Qdrant host")
    port: int = Field(default=QDRANT_PORT_DEFAULT, description="Qdrant HTTP port")
    grpc_port: int = Field(default=QDRANT_GRPC_PORT_DEFAULT, description="Qdrant gRPC port")
    prefer_grpc: bool = Field(
        default=QDRANT_PREFER_GRPC_DEFAULT, description="Prefer gRPC over HTTP"
    )
    api_key: str | None = Field(default=None, description="API key for authentication")
    timeout: int = Field(default=QDRANT_TIMEOUT, description="Connection timeout in seconds")

    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("port", "grpc_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port numbers are in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class EmbeddingSettings(BaseSettings):
    model_name: str = Field(
        default=EMBEDDING_MODEL_DEFAULT, description="Sentence transformer model name"
    )
    sparse_model_name: str = Field(
        default="prithivida/Splade_PP_en_v1",
        description="SPLADE sparse embedding model name",
    )
    batch_size: int = Field(default=BATCH_SIZE_DEFAULT, description="Batch size for encoding")
    normalize: bool = Field(default=NORMALIZE_EMBEDDINGS, description="Normalize embeddings")
    vector_size: int = Field(default=VECTOR_SIZE_DEFAULT, description="Vector dimension")

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is positive."""
        if v <= 0:
            raise ValueError("Batch size must be positive")
        return v


class DataSettings(BaseSettings):
    parquet_path: Path = Field(
        default=Path("data/sentiment.parquet"), description="Path to parquet data"
    )

    model_config = SettingsConfigDict(env_prefix="DATA_", case_sensitive=False)


class CollectionSettings(BaseSettings):
    name: str = Field(default=COLLECTION_NAME_DEFAULT, description="Collection name")
    vector_size: int = Field(default=VECTOR_SIZE_DEFAULT, description="Vector dimension")
    distance_metric: str = Field(default=DISTANCE_METRIC, description="Distance metric")

    model_config = SettingsConfigDict(env_prefix="COLLECTION_", case_sensitive=False)

    @field_validator("distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        allowed = {"Cosine", "Euclid", "Dot"}
        if v not in allowed:
            raise ValueError(f"Distance metric must be one of {allowed}")
        return v


class SearchSettings(BaseSettings):
    default_limit: int = Field(default=SEARCH_LIMIT_DEFAULT, description="Default search limit")
    score_threshold: float = Field(
        default=SEARCH_SCORE_THRESHOLD_DEFAULT, description="Score threshold"
    )

    model_config = SettingsConfigDict(env_prefix="SEARCH_", case_sensitive=False)

    @field_validator("default_limit")
    @classmethod
    def validate_limit(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Search limit must be positive")
        return v

    @field_validator("score_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Score threshold must be between 0 and 1")
        return v


class LoggingSettings(BaseSettings):
    level: str = Field(default=LOG_LEVEL_DEFAULT, description="Log level")
    file_path: Path = Field(default=Path("logs/vector_sentiment.log"), description="Log file path")
    rotation: str = Field(default=LOG_ROTATION, description="Log rotation")
    retention: str = Field(default=LOG_RETENTION, description="Log retention")

    model_config = SettingsConfigDict(env_prefix="LOG_", case_sensitive=False)

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        allowed = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()


class Settings(BaseSettings):
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    collection: CollectionSettings = Field(default_factory=CollectionSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
