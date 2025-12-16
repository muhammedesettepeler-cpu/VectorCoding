from collections.abc import Generator
from pathlib import Path
from types import TracebackType

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from vector_sentiment.config.constants import FIELD_LABEL, FIELD_SENTENCE, FIELD_TEXT
from vector_sentiment.models.schemas import SentimentRecord


class ParquetDataLoader:
    def __init__(self, file_path: Path, batch_size: int = 256) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        self.file_path = file_path
        self.batch_size = batch_size
        self._parquet_file: pq.ParquetFile | None = None

        logger.info(f"Initialized ParquetDataLoader for {file_path} with batch_size={batch_size}")

    def _get_parquet_file(self) -> pq.ParquetFile:
        if self._parquet_file is None:
            self._parquet_file = pq.ParquetFile(self.file_path)
        return self._parquet_file

    def get_total_rows(self) -> int:
        parquet_file = self._get_parquet_file()
        return int(parquet_file.metadata.num_rows)

    def iter_batches(self) -> Generator[pd.DataFrame, None, None]:
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

    def extract_batch(
        self,
        batch_df: pd.DataFrame,
        text_column: str,
        label_column: str | None = None,
        metadata_columns: list[str] | None = None,
    ) -> tuple[list[str], list[str] | None, dict[str, list]]:
        """Extract text, labels, and metadata from a batch DataFrame."""
        # Extract texts
        if text_column not in batch_df.columns:
            raise ValueError(f"Text column '{text_column}' not found in batch")
        texts = batch_df[text_column].astype(str).tolist()

        # Extract labels if specified
        labels = None
        if label_column:
            if label_column not in batch_df.columns:
                raise ValueError(f"Label column '{label_column}' not found in batch")
            labels = batch_df[label_column].astype(str).tolist()

        # Extract metadata
        metadata: dict[str, list] = {}
        if metadata_columns:
            for col in metadata_columns:
                if col not in batch_df.columns:
                    logger.warning(f"Metadata column '{col}' not found, skipping")
                    continue
                metadata[col] = batch_df[col].astype(str).tolist()

        return texts, labels, metadata

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
