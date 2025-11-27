"""Application constants.

This module contains all constant values used throughout the application,
including default configurations, field names, and preprocessing patterns.
"""

import re
from typing import Final

# Collection Configuration
COLLECTION_NAME_DEFAULT: Final[str] = "sentiment_vectors"
VECTOR_SIZE_DEFAULT: Final[int] = 384  # all-MiniLM-L6-v2 dimension
DISTANCE_METRIC: Final[str] = "Cosine"

# Embedding Configuration
EMBEDDING_MODEL_DEFAULT: Final[str] = "all-MiniLM-L6-v2"
BATCH_SIZE_DEFAULT: Final[int] = 128
NORMALIZE_EMBEDDINGS: Final[bool] = True

# Data Field Names
FIELD_TEXT: Final[str] = "text"
FIELD_LABEL: Final[str] = "label"
FIELD_SENTENCE: Final[str] = "sentence"

# Sentiment Labels
LABEL_POSITIVE: Final[str] = "positive"
LABEL_NEGATIVE: Final[str] = "negative"
LABEL_NEUTRAL: Final[str] = "neutral"

# Search Configuration
SEARCH_LIMIT_DEFAULT: Final[int] = 10
SEARCH_SCORE_THRESHOLD_DEFAULT: Final[float] = 0.7
SEARCH_WITH_PAYLOAD: Final[bool] = True
SEARCH_WITH_VECTORS: Final[bool] = False

# Preprocessing Patterns
PUNCTUATION_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^\w\s]")
MULTIPLE_SPACES_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")

# Common English Stopwords (subset for performance)
STOPWORDS: Final[set[str]] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "will",
    "with",
    "the",
    "this",
    "but",
    "they",
    "have",
}

# Logging Configuration
LOG_FORMAT: Final[str] = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)
LOG_ROTATION: Final[str] = "10 MB"
LOG_RETENTION: Final[str] = "7 days"
LOG_LEVEL_DEFAULT: Final[str] = "INFO"

# Qdrant Configuration
QDRANT_HOST_DEFAULT: Final[str] = "localhost"
QDRANT_PORT_DEFAULT: Final[int] = 6333
QDRANT_GRPC_PORT_DEFAULT: Final[int] = 6334
QDRANT_PREFER_GRPC_DEFAULT: Final[bool] = True
QDRANT_TIMEOUT: Final[int] = 30

# Parquet Reading Configuration
PARQUET_BATCH_SIZE: Final[int] = 256
PARQUET_USE_THREADS: Final[bool] = True
