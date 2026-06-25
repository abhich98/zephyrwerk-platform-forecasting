FROM python:3.12-slim

WORKDIR /app

# Install dbt
RUN pip install --no-cache-dir dbt-postgres

# Copy dbt project
COPY dbt/ ./dbt/

WORKDIR /app/dbt

# Run dbt run then dbt test
CMD ["/bin/sh", "-c", "dbt run --profiles-dir . && dbt test --profiles-dir ."]