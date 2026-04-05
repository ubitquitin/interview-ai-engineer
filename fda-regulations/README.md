## Project Overview

A RAG-enabled agent for fetching historical FDA warning letters and comparing the most relevant violations against your input processes.

1. **Data Ingestion & Vectorization**: FDA warning letters were scraped using `selectolax.lexbor` HTML parser and coerced into structured Pydantic outputs. Filtered down to pharma/drug-related warning letters, then split into unique violations per letter and vectorized into ChromaDB for RAG.

2. **Multi-Agent Pipeline**: Input text is processed by a schematizer agent that extracts entities into structured Pydantic classes. A second RAG-enabled agent fetches the most relevant vectorized violations from ChromaDB and evaluates risk against the input entity schema.

3. **Analysis & Services**: Analysis script generates visualizations and summary statistics. Docker container spins up services, with each agent exposed as standalone endpoints plus a final pipeline endpoint.

## Directory Structure

```
fda-regulations/
├── src/
│   ├── app.py              # Main agent pipeline orchestration
│   ├── models.py           # Pydantic schemas for entities
│   └── tools/
│       └── rag_tool.py     # ChromaDB RAG tool for fetching letters
├── data/
│   ├── ingest.py           # Scrapes FDA warning letters
│   ├── schematizer.py      # LLM-based entity extraction
│   ├── analysis.py         # Generates visualizations & stats
│   ├── vector_db/          # ChromaDB vector store
│   └── *.jsonl             # Data pipeline outputs
├── tests/
│   ├── run_pipeline.py     # End-to-end integration demo
│   ├── test_models.py      # Pydantic model unit tests
│   └── test_tools.py       # RAG tool unit tests (mocked)
├── docker-compose.yml      # ChromaDB service definition
└── pyproject.toml          # uv dependencies & config
```

## Quick Start

### 1. Install & Run Ollama with llama3.2:3b

**macOS:**
```bash
brew install ollama  # Version 0.20.2 required
ollama serve         # Keep running in terminal
ollama run llama3.2:3b
```

**WSL/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve         # Keep running in terminal
ollama run llama3.2:3b
```

### 2. Start Docker Services
```bash
docker compose up --build  # Takes ~1-2 min
```

### 3. Install uv & Run Pipelines

**macOS/Linux/WSL:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Run integration pipeline (2-10 min):**
```bash
uv run tests/run_pipeline.py
```

**Run analysis pipeline:**
```bash
uv run data/analysis.py
```

**Run unit tests (fast):**
```bash
uv run pytest
```

**Run tests with coverage:**
```bash
uv run pytest --cov=src --cov-report=term-missing
```