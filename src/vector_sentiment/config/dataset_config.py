"""Dataset configuration models.

Defines configuration structure for datasets with flexible column mappings.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class DatasetConfig(BaseModel):
    # Dataset metadata
    name: str = Field(..., description="Dataset name (identifier)")
    description: str = Field(..., description="Dataset description")
    data_file: str = Field(..., description="Parquet file name (relative to config dir)")

    # Column mappings
    text_column: str = Field(..., description="Column containing text data")
    label_column: str | None = Field(None, description="Column containing labels (optional)")
    metadata_columns: list[str] = Field(
        default_factory=list,
        description="Additional columns to include in payload",
    )

    # Collection settings
    collection_name: str = Field(..., description="Qdrant collection name")
    recreate: bool = Field(False, description="Force recreation of collection")

    # Embedding settings
    enable_sparse: bool = Field(False, description="Enable SPLADE sparse vectors for hybrid search")
    batch_size: int | None = Field(
        None, description="Batch size for processing (uses global default if None)"
    )

    @field_validator("text_column")
    @classmethod
    def text_column_not_empty(cls, v: str) -> str:
        """Validate text column is not empty."""
        if not v or not v.strip():
            raise ValueError("text_column cannot be empty")
        return v

    @field_validator("collection_name")
    @classmethod
    def collection_name_valid(cls, v: str) -> str:
        """Validate collection name is valid."""
        if not v or not v.strip():
            raise ValueError("collection_name cannot be empty")
        # Qdrant collection name rules
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "collection_name must contain only alphanumeric characters, "
                "underscores, and hyphens"
            )
        return v

    @classmethod
    def from_yaml(cls, path: Path | str) -> "DatasetConfig":
        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open() as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty or invalid YAML file: {config_path}")

        return cls(**data)

    @classmethod
    def from_master_config(cls, path: Path | str, scenario: str | None = None) -> "DatasetConfig":
        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open() as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty or invalid YAML file: {config_path}")

        # Check if this is a master config (has 'scenarios' key)
        if "scenarios" not in data:
            # Not a master config, treat as regular config
            return cls(**data)

        # Determine which scenario to load
        if scenario is None:
            scenario = data.get("active_scenario")
            if not scenario:
                raise ValueError(
                    "No scenario specified and no active_scenario in config. "
                    "Either specify --scenario or set active_scenario in master config."
                )

        # Get scenario data
        scenarios = data.get("scenarios", {})
        if scenario not in scenarios:
            available = ", ".join(scenarios.keys())
            raise ValueError(
                f"Scenario '{scenario}' not found in master config. "
                f"Available scenarios: {available}"
            )

        scenario_data = scenarios[scenario]
        return cls(**scenario_data)

    def get_data_path(self, config_dir: Path | str) -> Path:
        config_dir = Path(config_dir)
        data_path = config_dir / self.data_file

        if not data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n(relative to config dir: {config_dir})"
            )

        return data_path

    def get_all_columns(self) -> list[str]:
        columns = [self.text_column]

        if self.label_column:
            columns.append(self.label_column)

        columns.extend(self.metadata_columns)

        return columns
