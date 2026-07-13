import logging

import pandas as pd
import pytest

import ml.data_access as data_access
from unittest.mock import patch, MagicMock

# Mock settings before importing ml.data_access
@pytest.fixture(scope="session", autouse=True)
def mock_settings():
    with patch.dict('os.environ', {
        'ZEPHYRWERK_RDS_HOST': 'localhost',
        'ZEPHYRWERK_RDS_PORT': '5432',
        'ZEPHYRWERK_RDS_DATABASE': 'test_db',
        'ZEPHYRWERK_RDS_USER': 'test_user',
        'ZEPHYRWERK_RDS_PASSWORD': 'test_password',
        'ZEPHYRWERK_DASHBOARD_API_URL': 'http://localhost:8000',
    }):
        yield

class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


@pytest.fixture
def fake_engine(monkeypatch):
    monkeypatch.setattr(data_access.engine, "connect", lambda: FakeConnection())


@pytest.fixture
def capture_read_sql(monkeypatch):
    calls = []

    def fake_read_sql_query(query, conn, params=None):
        calls.append({"query": str(query), "params": params})
        return pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-01"]),
            "value": [2.0, 1.0],
        })

    monkeypatch.setattr(data_access.pd, "read_sql_query", fake_read_sql_query)
    return calls


def test_load_features_no_date_filters(fake_engine, capture_read_sql):
    data_access.load_features()

    query = capture_read_sql[0]["query"]
    assert "WHERE 1=1" in query
    assert "start_date" not in query
    assert "end_date" not in query
    assert "ORDER BY timestamp ASC" in query
    assert capture_read_sql[0]["params"] == {}


def test_load_features_with_start_date_only(fake_engine, capture_read_sql):
    data_access.load_features(start_date="2024-01-01")

    call = capture_read_sql[0]
    assert "AND timestamp >= :start_date" in call["query"]
    assert "AND timestamp <= :end_date" not in call["query"]
    assert call["params"] == {"start_date": "2024-01-01"}


def test_load_features_with_end_date_only(fake_engine, capture_read_sql):
    data_access.load_features(end_date="2024-02-01")

    call = capture_read_sql[0]
    assert "AND timestamp <= :end_date" in call["query"]
    assert "AND timestamp >= :start_date" not in call["query"]
    assert call["params"] == {"end_date": "2024-02-01"}


def test_load_features_with_both_dates(fake_engine, capture_read_sql):
    data_access.load_features(start_date="2024-01-01", end_date="2024-02-01")

    call = capture_read_sql[0]
    assert "AND timestamp >= :start_date" in call["query"]
    assert "AND timestamp <= :end_date" in call["query"]
    assert call["params"] == {"start_date": "2024-01-01", "end_date": "2024-02-01"}


def test_load_features_indexes_and_sorts_by_timestamp(fake_engine, capture_read_sql):
    df = data_access.load_features()

    assert df.index.name == "timestamp"
    assert df.index.is_monotonic_increasing


def test_load_features_logs_and_reraises_on_failure(fake_engine, monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(data_access.pd, "read_sql_query", boom)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="db exploded"):
            data_access.load_features()

    assert "Failed to load features" in caplog.text
