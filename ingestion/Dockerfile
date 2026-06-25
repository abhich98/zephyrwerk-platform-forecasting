FROM python:3.12-slim

WORKDIR /app

# Copy dependency manifest first to leverage Docker layer caching
COPY pyproject.toml README.md ./
COPY ingestion/ ./ingestion/
COPY orchestration/ ./orchestration/

# Install build tools and project dependencies
RUN pip install --no-cache-dir .

# Copy application source
COPY . .

# Default entrypoint
CMD ["python", "orchestration/run_pipeline.py"]