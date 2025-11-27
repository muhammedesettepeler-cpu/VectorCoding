"""Parquet data loader with generator pattern.

This module provides memory-efficient data loading from Parquet files using
generators to prevent RAM overflow with large datasets.
"""

from pathlib import Path
from types import TracebackType
from typing import Generator

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from vector_sentiment.config.constants import FIELD_LABEL, FIELD_SENTENCE, FIELD_TEXT
from vector_sentiment.models.schemas import SentimentRecord


class ParquetDataLoader:
    """Memory-efficient Parquet data loader using generator pattern.

    This class reads Parquet files in batches to prevent memory overflow.
    It uses pyarrow for efficient columnar data reading.

    Attributes:
        file_path: Path to the Parquet file
        batch_size: Number of rows to read per batch

    Example:
        >>> loader = ParquetDataLoader("data/sentiment.parquet", batch_size=128)
        >>> for batch in loader.iter_batches():
        ...     process_batch(batch)
    """

    def __init__(self, file_path: Path, batch_size: int = 256) -> None:
        """Initialize Parquet data loader.

        Args:
            file_path: Path to Parquet file
            batch_size: Number of rows per batch

        Raises:
            FileNotFoundError: If parquet file does not exist
            ValueError: If batch_size is not positive
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        self.file_path = file_path
        self.batch_size = batch_size
        self._parquet_file: pq.ParquetFile | None = None

        logger.info(f"Initialized ParquetDataLoader for {file_path} with batch_size={batch_size}")

    def _get_parquet_file(self) -> pq.ParquetFile:
        """Get or create ParquetFile instance.

        Returns:
            ParquetFile instance for reading
        """
        if self._parquet_file is None:
            self._parquet_file = pq.ParquetFile(self.file_path)
        return self._parquet_file

    def get_total_rows(self) -> int:
        """Get total number of rows in the Parquet file.

        Returns:
            Total row count
        """
        parquet_file = self._get_parquet_file()
        return int(parquet_file.metadata.num_rows)

    def iter_batches(self) -> Generator[pd.DataFrame, None, None]:
        """Iterate over Parquet file in batches.

        This generator yields pandas DataFrames of the specified batch size,
        preventing the entire dataset from being loaded into memory at once.

        Yields:
            DataFrame containing a batch of rows

        Example:
            >>> loader = ParquetDataLoader("data.parquet", batch_size=128)
            >>> for batch_df in loader.iter_batches():
            ...     print(f"Processing {len(batch_df)} rows")
        """
        parquet_file = self._get_parquet_file()
        total_rows = self.get_total_rows()

        logger.info(f"Starting batch iteration over {total_rows} rows")

        batch_number = 0
        for batch in parquet_file.iter_batches(batch_size=self.batch_size):
            batch_df = batch.to_pandas()
            batch_number += 1

            logger.debug(
                f"Yielding batch {batch_number} with {len(batch_df)} rows "
                f"({batch_number * self.batch_size}/{total_rows})"
            )

            yield batch_df

        logger.info(f"Completed iteration over {batch_number} batches")

    def iter_records(
        self,
        text_field: str = FIELD_TEXT,
        label_field: str = FIELD_LABEL,
    ) -> Generator[SentimentRecord, None, None]:
        """Iterate over validated sentiment records.

        This generator validates each record using Pydantic models and yields
        only valid records. Invalid records are logged and skipped.

        Args:
            text_field: Name of the text column (default: 'text')
            label_field: Name of the label column (default: 'label')

        Yields:
            Validated SentimentRecord instances

        Example:
            >>> loader = ParquetDataLoader("data.parquet")
            >>> for record in loader.iter_records():
            ...     print(f"{record.label}: {record.text[:50]}")
        """
        valid_count = 0
        invalid_count = 0

        for batch_df in self.iter_batches():
            # Handle different field naming conventions
            text_col = text_field if text_field in batch_df.columns else FIELD_SENTENCE

            if text_col not in batch_df.columns:
                logger.error(
                    f"Neither '{text_field}' nor '{FIELD_SENTENCE}' found in columns: "
                    f"{list(batch_df.columns)}"
                )
                raise ValueError("Text column not found in data")

            if label_field not in batch_df.columns:
                logger.error(f"Label column '{label_field}' not found in columns")
                raise ValueError(f"Label column '{label_field}' not found")

            for _, row in batch_df.iterrows():
                try:
                    record = SentimentRecord(
                        text=str(row[text_col]),
                        label=str(row[label_field]),
                    )
                    valid_count += 1
                    yield record

                except Exception as e:
                    invalid_count += 1
                    logger.warning(f"Invalid record skipped: {e}")

        logger.info(f"Completed record iteration: {valid_count} valid, {invalid_count} invalid")

    def close(self) -> None:
        """Close the Parquet file handle.

        This method should be called when done reading to free resources.
        """
        if self._parquet_file is not None:
            self._parquet_file = None
            logger.debug("Closed Parquet file")

    def __enter__(self) -> "ParquetDataLoader":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager and close file."""
        self.close()
