"""Application settings using Pydantic.

This module provides environment-based configuration management using Pydantic
settings. All settings can be configured via environment variables or .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

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
    """Qdrant connection settings.

    Attributes:
        host: Qdrant server hostname
        port: HTTP API port
        grpc_port: gRPC API port
        prefer_grpc: Whether to prefer gRPC over HTTP
        api_key: Optional API key for authentication
        timeout: Connection timeout in seconds
    """

    host: str = Field(default=QDRANT_HOST_DEFAULT, description="Qdrant host")
    port: int = Field(default=QDRANT_PORT_DEFAULT, description="Qdrant HTTP port")
    grpc_port: int = Field(default=QDRANT_GRPC_PORT_DEFAULT, description="Qdrant gRPC port")
    prefer_grpc: bool = Field(
        default=QDRANT_PREFER_GRPC_DEFAULT, description="Prefer gRPC over HTTP"
    )
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    timeout: int = Field(default=QDRANT_TIMEOUT, description="Connection timeout in seconds")

    model_config = SettingsConfigDict(env_prefix="QDRANT_", case_sensitive=False)

    @field_validator("port", "grpc_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port numbers are in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class EmbeddingSettings(BaseSettings):
    """Embedding model settings.

    Attributes:
        model_name: Name of the SentenceTransformer model
        batch_size: Batch size for encoding
        normalize: Whether to normalize embeddings
        vector_size: Dimension of embedding vectors
    """

    model_name: str = Field(
        default=EMBEDDING_MODEL_DEFAULT, description="Sentence transformer model name"
    )
    batch_size: int = Field(default=BATCH_SIZE_DEFAULT, description="Batch size for encoding")
    normalize: bool = Field(default=NORMALIZE_EMBEDDINGS, description="Normalize embeddings")
    vector_size: int = Field(default=VECTOR_SIZE_DEFAULT, description="Vector dimension")

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", case_sensitive=False)

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is positive."""
        if v <= 0:
            raise ValueError("Batch size must be positive")
        return v


class DataSettings(BaseSettings):
    """Data processing settings.

    Attributes:
        parquet_path: Path to parquet data file
        remove_stopwords: Whether to remove stopwords during preprocessing
        remove_punctuation: Whether to remove punctuation
        lowercase: Whether to convert text to lowercase
    """

    parquet_path: Path = Field(
        default=Path("data/sentiment.parquet"), description="Path to parquet data"
    )
    remove_stopwords: bool = Field(default=True, description="Remove stopwords")
    remove_punctuation: bool = Field(default=True, description="Remove punctuation")
    lowercase: bool = Field(default=True, description="Convert to lowercase")

    model_config = SettingsConfigDict(env_prefix="PREPROCESSING_", case_sensitive=False)


class CollectionSettings(BaseSettings):
    """Collection configuration settings.

    Attributes:
        name: Name of the Qdrant collection
        distance_metric: Distance metric for similarity
    """

    name: str = Field(default=COLLECTION_NAME_DEFAULT, description="Collection name")
    distance_metric: str = Field(default=DISTANCE_METRIC, description="Distance metric")

    model_config = SettingsConfigDict(env_prefix="COLLECTION_", case_sensitive=False)

    @field_validator("distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        """Validate distance metric is supported."""
        allowed = {"Cosine", "Euclid", "Dot"}
        if v not in allowed:
            raise ValueError(f"Distance metric must be one of {allowed}")
        return v


class SearchSettings(BaseSettings):
    """Search operation settings.

    Attributes:
        default_limit: Default number of results to return
        score_threshold: Minimum score threshold for results
    """

    default_limit: int = Field(default=SEARCH_LIMIT_DEFAULT, description="Default search limit")
    score_threshold: float = Field(
        default=SEARCH_SCORE_THRESHOLD_DEFAULT, description="Score threshold"
    )

    model_config = SettingsConfigDict(env_prefix="SEARCH_", case_sensitive=False)

    @field_validator("default_limit")
    @classmethod
    def validate_limit(cls, v: int) -> int:
        """Validate limit is positive."""
        if v <= 0:
            raise ValueError("Search limit must be positive")
        return v

    @field_validator("score_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Score threshold must be between 0 and 1")
        return v


class LoggingSettings(BaseSettings):
    """Logging configuration settings.

    Attributes:
        level: Logging level
        file_path: Path to log file
        rotation: Log file rotation setting
        retention: Log file retention setting
    """

    level: str = Field(default=LOG_LEVEL_DEFAULT, description="Log level")
    file_path: Path = Field(default=Path("logs/vector_sentiment.log"), description="Log file path")
    rotation: str = Field(default=LOG_ROTATION, description="Log rotation")
    retention: str = Field(default=LOG_RETENTION, description="Log retention")

    model_config = SettingsConfigDict(env_prefix="LOG_", case_sensitive=False)

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate log level is valid."""
        allowed = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()


class Settings(BaseSettings):
    """Main application settings.

    This class aggregates all configuration settings and provides a single
    point of access for application configuration.

    Attributes:
        qdrant: Qdrant connection settings
        embedding: Embedding model settings
        data: Data processing settings
        collection: Collection configuration
        search: Search operation settings
        logging: Logging configuration
    """

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
    """Get cached application settings.

    This function returns a cached instance of Settings to avoid
    repeatedly reading environment variables and configuration files.

    Returns:
        Settings instance with all configuration
    """
    return Settings()
