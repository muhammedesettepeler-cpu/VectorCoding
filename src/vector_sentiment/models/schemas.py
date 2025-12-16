"""Pydantic schemas for data validation.

This module defines all data models used throughout the application,
providing type safety and validation for inputs and outputs.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class SentimentRecord(BaseModel):
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

        if normalized not in allowed_labels and normalized in {"0", "1", "2"}:
            # Allow numeric labels (0, 1, 2) and convert them
            label_map = {"0": "negative", "1": "neutral", "2": "positive"}
            return label_map[normalized]
            # If not in allowed set, log warning but allow it
            # This provides flexibility for different datasets

        return normalized

    model_config = {"extra": "ignore"}


class VectorPoint(BaseModel):
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
    id: int = Field(..., description="Point ID")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    label: str = Field(..., description="Sentiment label")
    text: str | None = Field(default=None, description="Original text")

    def __str__(self) -> str:
        """String representation of search result."""
        text_preview = f" - {self.text[:50]}..." if self.text else ""
        return f"Result(id={self.id}, score={self.score:.4f}, label={self.label}{text_preview})"
