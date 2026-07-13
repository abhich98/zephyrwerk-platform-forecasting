FROM python:3.12-slim

WORKDIR /app

# Copy dependency manifest first to leverage Docker layer caching
COPY pyproject.toml README.md ./

# Copy application source
COPY ml/ ./ml/
COPY db/ ./db/

# Install build tools and project dependencies
RUN pip install --no-cache-dir .

# Default entrypoint
CMD ["python", "-m", "ml.train_price_model"]