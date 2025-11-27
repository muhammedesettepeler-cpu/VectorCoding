"""Data validation utilities.

This module provides data validation using Pydantic models and tracks
validation statistics for monitoring data quality.
"""

from typing import Any

from loguru import logger
from pydantic import ValidationError

from vector_sentiment.models.schemas import SentimentRecord


class DataValidator:
    """Data validator using Pydantic models.

    This class validates data against Pydantic schemas and tracks validation
    statistics including success/failure counts and error patterns.

    Attributes:
        valid_count: Number of valid records processed
        invalid_count: Number of invalid records encountered
        error_summary: Dictionary of error types and their counts

    Example:
        >>> validator = DataValidator()
        >>> record = validator.validate_sentiment_record({"text": "test", "label": "positive"})
        >>> print(validator.get_stats())
    """

    def __init__(self) -> None:
        """Initialize data validator."""
        self.valid_count = 0
        self.invalid_count = 0
        self.error_summary: dict[str, int] = {}

        logger.info("Initialized DataValidator")

    def validate_sentiment_record(
        self,
        data: dict[str, Any],
    ) -> SentimentRecord | None:
        """Validate a sentiment record.

        Args:
            data: Dictionary containing record data

        Returns:
            SentimentRecord if valid, None if invalid

        Example:
            >>> validator = DataValidator()
            >>> record = validator.validate_sentiment_record({
            ...     "text": "Great product",
            ...     "label": "positive"
            ... })
        """
        try:
            record = SentimentRecord(**data)
            self.valid_count += 1
            return record

        except ValidationError as e:
            self.invalid_count += 1

            # Track error types
            for error in e.errors():
                error_type = error["type"]
                self.error_summary[error_type] = self.error_summary.get(error_type, 0) + 1

            logger.warning(f"Validation failed: {e}")
            return None

    def validate_batch(
        self,
        data_batch: list[dict[str, Any]],
    ) -> list[SentimentRecord]:
        """Validate a batch of records.

        Args:
            data_batch: List of dictionaries containing record data

        Returns:
            List of valid SentimentRecord instances (invalid records are filtered)

        Example:
            >>> validator = DataValidator()
            >>> batch = [
            ...     {"text": "Good", "label": "positive"},
            ...     {"text": "", "label": "invalid"},  # Will be filtered
            ... ]
            >>> valid_records = validator.validate_batch(batch)
        """
        valid_records = []

        for data in data_batch:
            record = self.validate_sentiment_record(data)
            if record is not None:
                valid_records.append(record)

        return valid_records

    def get_stats(self) -> dict[str, Any]:
        """Get validation statistics.

        Returns:
            Dictionary containing validation statistics

        Example:
            >>> validator = DataValidator()
            >>> # ... validate some records ...
            >>> stats = validator.get_stats()
            >>> print(f"Valid: {stats['valid_count']}")
        """
        total = self.valid_count + self.invalid_count
        valid_rate = (self.valid_count / total * 100) if total > 0 else 0.0

        return {
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "total_count": total,
            "valid_rate": valid_rate,
            "error_summary": self.error_summary.copy(),
        }

    def log_stats(self) -> None:
        """Log validation statistics.

        Example:
            >>> validator = DataValidator()
            >>> # ... validate some records ...
            >>> validator.log_stats()
        """
        stats = self.get_stats()

        logger.info(
            f"Validation Stats - Valid: {stats['valid_count']}, "
            f"Invalid: {stats['invalid_count']}, "
            f"Rate: {stats['valid_rate']:.2f}%"
        )

        if stats["error_summary"]:
            logger.info(f"Error Summary: {stats['error_summary']}")

    def reset(self) -> None:
        """Reset validation statistics.

        Example:
            >>> validator = DataValidator()
            >>> # ... validate some records ...
            >>> validator.reset()
            >>> assert validator.valid_count == 0
        """
        self.valid_count = 0
        self.invalid_count = 0
        self.error_summary = {}
        logger.debug("Reset validation statistics")
