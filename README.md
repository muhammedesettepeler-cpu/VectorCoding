# Vector Sentiment Search

A production-ready Qdrant vector database implementation for sentiment analysis with advanced features including Pydantic validation, modular architecture, and comprehensive search capabilities.

## Features

- Memory-efficient Parquet data loading with generator pattern
- Text preprocessing pipeline (stopwords, punctuation, normalization)
- SentenceTransformer embeddings with named vector format
- Qdrant vector database integration with collection management
- Advanced search with filters, score thresholds, and recommendations
- Pydantic validation for type safety
- Comprehensive logging with Loguru
- CLI interface for operations
- Production-ready architecture

## Requirements

- Python 3.11.8 or higher
- Docker and Docker Compose (for Qdrant)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd model_assingnment
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

### 4. Setup Qdrant with Docker

```bash
docker-compose up -d
```

Verify Qdrant is running:
```bash
curl http://localhost:6333/healthz
```

### 5. Configure environment

```bash
cp .env.example .env
```

Edit `.env` to customize settings if needed.

## Usage

### CLI Commands

#### Check status

```bash
vector-sentiment status
```

#### Ingest data

```bash
vector-sentiment ingest --data-path data/sentiment.parquet --batch-size 128
```

Options:
- `--data-path`: Path to parquet file (default from config)
- `--batch-size`: Processing batch size (default: 128)
- `--recreate`: Recreate collection if exists

#### Search for similar vectors

```bash
# Basic search
vector-sentiment search "Great product quality" --limit 10

# Search with label filter
vector-sentiment search "Amazing experience" --label positive --limit 5

# Search with score threshold
vector-sentiment search "Good quality" --score-threshold 0.8 --limit 10
```

#### Get recommendations

```bash
# Recommend using point IDs
vector-sentiment recommend --positive-ids "1,2,3" --negative-ids "10,11" --limit 5

# Recommend using labels
vector-sentiment recommend --positive-label positive --negative-label negative --limit 10
```

### Programmatic Usage

```python
from pathlib import Path
from vector_sentiment.config.settings import get_settings
from vector_sentiment.embeddings.service import EmbeddingService
from vector_sentiment.vectordb.client import QdrantClientWrapper
from vector_sentiment.search.searcher import SearchService

# Load settings
settings = get_settings()

# Initialize services
embedding_service = EmbeddingService(settings.embedding.model_name)

with QdrantClientWrapper(settings.qdrant) as qdrant:
    search_service = SearchService(
        client=qdrant.client,
        collection_name=settings.collection.name,
        embedding_service=embedding_service,
        vector_name=settings.embedding.model_name,
    )
    
    # Search
    results = search_service.search(
        query_text="Excellent product!",
        filter_label="positive",
        limit=10,
    )
    
    for result in results:
        print(f"Score: {result.score:.4f}, Label: {result.label}")
```

## Project Structure

```
vector-sentiment-search/
├── pyproject.toml          # Project configuration and dependencies
├── docker-compose.yml      # Qdrant service configuration
├── .env.example            # Environment variables template
├── README.md               # This file
├── src/vector_sentiment/
│   ├── config/             # Configuration management
│   │   ├── settings.py     # Pydantic settings
│   │   └── constants.py    # Application constants
│   ├── models/             # Pydantic data models
│   │   └── schemas.py      # Validation schemas
│   ├── data/               # Data processing
│   │   ├── loader.py       # Parquet loader with generators
│   │   ├── preprocessor.py # Text preprocessing
│   │   └── validator.py    # Data validation
│   ├── vectordb/           # Qdrant operations
│   │   ├── client.py       # Client wrapper
│   │   ├── collection.py   # Collection management
│   │   ├── upserter.py     # Vector upload
│   │   └── concepts.py     # Advanced concepts documentation
│   ├── embeddings/         # Embedding generation
│   │   └── service.py      # SentenceTransformer wrapper
│   ├── search/             # Search operations
│   │   ├── searcher.py     # Vector search
│   │   └── recommender.py  # Recommendations
│   ├── utils/              # Utilities
│   │   ├── logger.py       # Logging setup
│   │   └── helpers.py      # Helper functions
│   └── cli/                # Command-line interface
│       └── app.py          # CLI commands
├── data/                   # Data directory
├── logs/                   # Log files
└── examples/               # Example scripts
```

## Advanced Qdrant Concepts

This project demonstrates understanding of advanced Qdrant features:

### Quantization
Memory optimization through vector compression. See [src/vector_sentiment/vectordb/concepts.py](file:///home/esettepeler/Desktop/model_assingnment/src/vector_sentiment/vectordb/concepts.py) for detailed documentation on:
- Scalar quantization (4x memory reduction)
- Product quantization (8x-64x reduction)
- Trade-offs and use cases

### Shard Key Selectors
Critical for production deployments with horizontal scaling. Detailed documentation covers:
- Multi-tenant data isolation
- Geographic distribution
- Workload isolation patterns
- Production architecture examples

### Payload Indexing
Query performance optimization through metadata indexing on frequently filtered fields.

### Batch Operations
Improved throughput with batch search and recommend operations.

View full documentation:
```bash
python -m vector_sentiment.vectordb.concepts
```

## Data Preparation

### Download Dataset

Example using HuggingFace datasets:

```python
from datasets import load_dataset
import pandas as pd

# Load sentiment dataset
dataset = load_dataset("sentiment140", split="train[:10000]")

# Convert to pandas and save as parquet
df = pd.DataFrame(dataset)
df = df.rename(columns={"text": "text", "sentiment": "label"})
df.to_parquet("data/sentiment.parquet")
```

### Data Format

Expected parquet columns:
- `text` or `sentence`: Text content
- `label`: Sentiment label (positive/negative/neutral or 0/1/2)

## Development

### Code Quality

Run linting:
```bash
ruff check .
ruff format .
```

Run type checking:
```bash
mypy src/

```

### Testing

```bash
pytest
```

## Configuration

Key environment variables in `.env`:

```bash
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_PREFER_GRPC=true

# Embedding
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=128

# Collection
COLLECTION_NAME=sentiment_vectors
VECTOR_SIZE=384

# Search
SEARCH_DEFAULT_LIMIT=10
SEARCH_SCORE_THRESHOLD=0.7
```

## Architecture Decisions

### Named Vector Format
Implements the exact format specified in project requirements:
```python
vectors = {
    embedding_model_name: [
        arr.tolist()
        for arr in model.encode(sentences=data, batch_size=batch_size, normalize_embeddings=True)
    ]
}
```

### Generator Pattern
Uses `pyarrow.parquet.ParquetFile.iter_batches()` for memory-efficient data loading, preventing RAM overflow with large datasets.

### Upsert vs Add
Uses `client.upsert()` instead of `client.add()` as per project requirements for proper vector insertion.

###  Pydantic Validation
All data flows validated with Pydantic models ensuring type safety and data quality.

## Troubleshooting

### Qdrant Connection Failed
```bash
# Check if Qdrant is running
docker-compose ps

# View logs
docker-compose logs qdrant

# Restart if needed
docker-compose restart qdrant
```

### Memory Issues During Ingestion
Reduce batch size:
```bash
vector-sentiment ingest --batch-size 64
```

### Import Errors
Ensure package is installed in editable mode:
```bash
pip install -e .
```

## License

MIT

## Contact

For questions or issues, please open an issue in the repository.
