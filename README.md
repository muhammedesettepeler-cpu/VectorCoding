# 🚀 Vectora: Scenario-Driven Vector Search Engine

Vectora is a production-ready, configurable **Vector Search & Sentiment Analysis** system built on **Qdrant** and **Sentence Transformers**. It adopts a **Scenario-Driven Architecture**, allowing you to switch between use cases (e.g., Sentiment Analysis, Product Reviews, Support Tickets) instantly via a central configuration file.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg) ![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red.svg) ![License](https://img.shields.io/badge/license-MIT-green)

---

## 🌟 Key Features

*   **⚡ Scenario-Driven Design**: Switch entire datasets and logic by changing a single line in `master_config.yaml`.
*   **🔍 Hybrid Search**: Combines **Dense Vectors** (Semantic similarity) and **Sparse Vectors (SPLADE)** (Keyword matching) for superior retrieval accuracy.
*   **🛡️ Robust & Type-Safe**: Fully typed with **Pydantic** for configuration validation and **Mypy** strict typing.
*   **🚀 Production Ready**: Features generator-based memory-efficient data loading, batched processing, and rigorous error handling.
*   **📊 Comprehensive Analytics**: Built-in tools to analyze vector distributions, calculate statistics, and visualize data health.
*   **🐳 Containerized**: One-command setup with Docker Compose.

---

## 🛠️ Installation

### 1. Clone & Setup
```bash
git clone <repository-url>
cd vectora

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

### 2. Start Vector Database
Start a local Qdrant instance using Docker:
```bash
docker-compose up -d
```
*Verify running at: `http://localhost:6333`*

### 3. Environment Configuration
Create your `.env` file from the template:
```bash
cp .env.example .env
```
*No changes usually needed for local development.*

---

## ⚙️ Configuration (The Core)

This project uses a **Centralized Configuration** approach. Control everything from `data_dir/master_config.yaml`.

**Example `master_config.yaml`:**
```yaml
# 🎯 ACTIVE SCENARIO: Change this to switch contexts!
active_scenario: sentiment_analysis 

scenarios:
  sentiment_analysis:
    name: "Sentiment Analysis"
    description: "Twitter sentiment dataset"
    data_file: "sentiment/data.parquet"
    
    # Map your data columns to internal schema
    text_column: "text"
    label_column: "label"
    
    # Vector Search Settings
    collection_name: "sentiment_collection"
    enable_sparse: true  # Enable Hybrid Search
    
  product_reviews:
    name: "Product Reviews"
    data_file: "reviews/data.parquet"
    text_column: "review_body" # Flexible column mapping
    label_column: "star_rating"
```

---

## 🎬 Scenarios & Usage

All scripts automatically read the `active_scenario` from `master_config.yaml`. You don't need to pass arguments every time.

### 1. 📥 Ingestion (Load Data)
Reads your Parquet file, generates Dense & Sparse embeddings, and indexes them in Qdrant.
```bash
python scenarios/ingest.py
# Optional: --recreate to force fresh start
```

### 2. 🔎 Search (Semantic & Hybrid)
Perform semantic searches on your indexed data.
```bash
# Standard Semantic Search
python scenarios/search.py --query "Unresponsive customer service"

# ⚡ Hybrid Search (Dense + Keyword match)
python scenarios/search.py --query "Error code 503" --hybrid
```

### 3. 💡 Recommendations
Find similar items or generate recommendations based on positive/negative examples.
```bash
# Recommend content similar to positive examples
python scenarios/recommend.py --positive-ids "10, 25, 42"

# Filtered recommendation
python scenarios/recommend.py --positive-label "positive" --limit 5
```

### 4. 📈 Analytics
Inspect your collection health, label distribution, and vector statistics.
```bash
python scenarios/analytics.py
```

---

## 🏗️ Project Structure

```
vectora/
├── data_dir/                  # 📂 CENTRAL DATA & CONFIG
│   ├── master_config.yaml     # 🧠 The Brain: All scenarios defined here
│   ├── sentiment/             # Dataset folder
│   │   └── data.parquet       # Actual data file
│   └── product_reviews/       # Another dataset...
│
├── scenarios/                 # 🎬 EXECUTABLE SCRIPTS
│   ├── ingest.py              # Data ETL & Indexing
│   ├── search.py              # Search Interface
│   ├── recommend.py           # Recommendation Engine
│   └── analytics.py           # Analysis Tools
│
├── src/vector_sentiment/      # 🧠 CORE LIBRARY
│   ├── config/                # Pydantic Settings
│   ├── embeddings/            # Dense (BGE) & Sparse (SPLADE) Models
│   ├── vectordb/              # Qdrant Operations Module
│   └── ...
└── ...
```

---

## 🧩 Advanced Concepts Used

*   **Named Vector Format**: Compliant with generic vector inputs for flexibility.
*   **Hybrid Search**: Using `BGE-Small` (Dense) + `SPLADE` (Sparse) for optimal retrieval.
*   **Generator Pattern**: Efficiently processes massive datasets without memory spikes.
*   **Modular Design**: Separation of concerns between Logic (`src`), Configuration (`yaml`), and Execution (`scenarios`).

---

## 🤝 Contributing

Contributions are welcome! Please run linting before submitting PRs:
```bash
ruff check .
black .
mypy .
```

---

**Built with 💙 by Advanced Agentic Coding Team**
