"""Pydantic schemas for data validation.

This module defines all data models used throughout the application,
providing type safety and validation for inputs and outputs.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class SentimentRecord(BaseModel):
    """Sentiment data record from dataset.

    Represents a single sentiment analysis record with text and label.
    The text field can be named 'text' or 'sentence' in the source data.

    Attributes:
        text: The text content for sentiment analysis
        label: Sentiment label (positive, negative, neutral)
    """

    text: str = Field(..., min_length=1, description="Text content")
    label: str = Field(..., min_length=1, description="Sentiment label")

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Validate text is not empty after stripping."""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validate and normalize label."""
        normalized = v.lower().strip()
        allowed_labels = {"positive", "negative", "neutral"}

        if normalized not in allowed_labels:
            # Allow numeric labels (0, 1, 2) and convert them
            if normalized in {"0", "1", "2"}:
                label_map = {"0": "negative", "1": "neutral", "2": "positive"}
                return label_map[normalized]
            # If not in allowed set, log warning but allow it
            # This provides flexibility for different datasets

        return normalized

    model_config = {"extra": "ignore"}


class VectorPoint(BaseModel):
    """Vector point to be stored in Qdrant.

    Represents a vector embedding with its associated metadata payload.

    Attributes:
        id: Unique identifier for the vector point
        vector: The embedding vector as list of floats
        payload: Metadata associated with the vector (e.g., label, original text)
    """

    id: int = Field(..., ge=0, description="Point ID")
    vector: list[float] = Field(..., min_length=1, description="Embedding vector")
    payload: dict[str, Any] = Field(default_factory=dict, description="Metadata payload")

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, v: list[float]) -> list[float]:
        """Validate vector contains valid floats."""
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Vector must contain only numeric values")
        return v


class FilterOptions(BaseModel):
    """Filter options for search queries.

    Defines filtering criteria for vector search operations.

    Attributes:
        label: Optional label to filter by (e.g., 'positive', 'negative')
        score_threshold: Minimum similarity score threshold
        limit: Maximum number of results to return
    """

    label: str | None = Field(default=None, description="Label filter")
    score_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Score threshold"
    )
    limit: int = Field(default=10, ge=1, le=1000, description="Result limit")

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str | None) -> str | None:
        """Normalize label if provided."""
        return v.lower().strip() if v else None


class SearchQuery(BaseModel):
    """Search query model.

    Represents a search query with text and optional filters.

    Attributes:
        query_text: Text to search for
        filters: Optional filter criteria
    """

    query_text: str = Field(..., min_length=1, description="Query text")
    filters: FilterOptions | None = Field(default=None, description="Filter options")

    @field_validator("query_text")
    @classmethod
    def validate_query_text(cls, v: str) -> str:
        """Validate query text is not empty."""
        if not v.strip():
            raise ValueError("Query text cannot be empty")
        return v.strip()


class SearchResult(BaseModel):
    """Search result model.

    Represents a single search result with score and metadata.

    Attributes:
        id: Point ID from Qdrant
        score: Similarity score
        label: Sentiment label from payload
        text: Original text from payload (if available)
    """

    id: int = Field(..., description="Point ID")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    label: str = Field(..., description="Sentiment label")
    text: str | None = Field(default=None, description="Original text")

    def __str__(self) -> str:
        """String representation of search result."""
        text_preview = f" - {self.text[:50]}..." if self.text else ""
        return f"Result(id={self.id}, score={self.score:.4f}, label={self.label}{text_preview})"
