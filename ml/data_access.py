import pandas as pd
import logging
from sqlalchemy import text

from db.database import engine

logger = logging.getLogger(__name__)


def load_features(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """
    Load ml features from the database for the given date range.

    Args:
        start_date (str | None): The start date for filtering features (inclusive).
        end_date (str | None): The end date for filtering features (inclusive).

    Returns:
        pd.DataFrame: A DataFrame containing the loaded features.
    """
    # Placeholder for actual database loading logic
    # This should include connecting to the database, executing a query,
    # and returning the results as a pandas DataFrame.
    logger.info(f"Loading ml features from {start_date} to {end_date}")

    try:
        query = "SELECT * FROM analytics.fct_ml_features WHERE 1=1"
        params: dict = {}

        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY timestamp ASC"

        with engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params)
            logger.info(f"Loaded {len(df)} features from the database.")

        return df.set_index("timestamp").sort_index()
    except Exception:
        logger.exception("Failed to load features")
        raise

if __name__ == "__main__":
    # Example usage
    df = load_features(start_date="2023-04-16")
    print(df.head())
    print(df.shape, df.index.dtype, df.index.min(), df.index.max()) 
    print(df.isna().sum().sort_values(ascending=False).head(10))
    print(df["nuclear_mw"].describe())
    print(df[df["nuclear_mw"].notna()]["nuclear_mw"].value_counts().head())