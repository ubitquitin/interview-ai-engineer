FROM python:3.13-slim

WORKDIR /app

# Install uv for faster dependency installation
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --no-dev

# Copy source code
COPY src/ ./src/

# Copy vector database (needed for RAG tool)
COPY data/vector_db/ ./data/vector_db/

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run the unified FastAPI app
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]