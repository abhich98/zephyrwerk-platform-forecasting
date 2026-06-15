import logging
import os
from datetime import date
from enum import Enum

import pandas as pd
import pyarrow.dataset as ds
from pyarrow.fs import S3FileSystem

from ingestion.smard_client import ENERGY_SOURCE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SEASON(Enum):
    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"

SEASON_BY_MONTH = {
    12: SEASON.WINTER, 1: SEASON.WINTER, 2: SEASON.WINTER,
    3: SEASON.SPRING, 4: SEASON.SPRING, 5: SEASON.SPRING,
    6: SEASON.SUMMER, 7: SEASON.SUMMER, 8: SEASON.SUMMER,
    9: SEASON.AUTUMN, 10: SEASON.AUTUMN, 11: SEASON.AUTUMN,
}

SEASON_ORDER = [SEASON.WINTER, SEASON.SPRING, SEASON.SUMMER, SEASON.AUTUMN]


# Pumped hydro is a storage technology.
# It consumes electricity to pump water up, generates by letting it back down. 
# The energy was already counted when it was originally generated 
# (and most of it came from the grid mix, not specifically renewable sources)
class ENERGY_GROUP(Enum):
    RENEWABLE = "renewable"
    FOSSIL = "fossil"
    NUCLEAR = "nuclear"
    STORAGE = "storage" 

SIGNAL_GROUPS = {
    ENERGY_GROUP.RENEWABLE : [
        ENERGY_SOURCE.WIND_ONSHORE,
        ENERGY_SOURCE.WIND_OFFSHORE,
        ENERGY_SOURCE.SOLAR,
        ENERGY_SOURCE.BIOMASS,
        ENERGY_SOURCE.HYDROPOWER,
        ENERGY_SOURCE.OTHER_RENEWABLE # geothermal, landfill gas, small hydro
    ],
    ENERGY_GROUP.FOSSIL : [
        ENERGY_SOURCE.BROWN_COAL,
        ENERGY_SOURCE.HARD_COAL,
        ENERGY_SOURCE.NATURAL_GAS,
        ENERGY_SOURCE.OTHER_CONVENTIONAL # mostly oil and minor fossil sources
    ],
    ENERGY_GROUP.NUCLEAR : [
        ENERGY_SOURCE.NUCLEAR
    ],
    ENERGY_GROUP.STORAGE : [
        ENERGY_SOURCE.PUMPED_STORAGE
    ]
}

def _get_filesystem():
    endpoint = os.environ.get("AWS_ENDPOINT_URL")

    fs_kwargs = {"region": os.environ.get("AWS_REGION", "eu-central-1")}
    if endpoint:
        # LocalStack 3.x doesn't return the AWS-style response checksums that
        # newer aws-sdk-cpp (bundled in pyarrow) validates on GET. Disable
        # validation for local dev only — real AWS keeps it on by default.
        os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "WHEN_REQUIRED")
        os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "WHEN_REQUIRED")

        # strip scheme — pyarrow expects "host:port"
        fs_kwargs["access_key"] = os.environ.get("AWS_ACCESS_KEY_ID", "test")
        fs_kwargs["secret_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")
        fs_kwargs["endpoint_override"] = endpoint.removeprefix("http://").removeprefix("https://")
        fs_kwargs["scheme"] = "http" if endpoint.startswith("http://") else "https"

    return S3FileSystem(**fs_kwargs)

def _filter_timestamp(dataset, start: date, end: date) -> pd.DataFrame:
    # prune at the partition level — avoids opening files outside the window
    year_filter = (ds.field("year") >= start.year) & (ds.field("year") <= end.year)
    df = dataset.to_table(filter=year_filter).to_pandas()

    # tighten to exact timestamp bounds (partition pruning is year-level only)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df.loc[df["timestamp"].between(start_ts, end_ts)].reset_index(drop=True)
    return df

def load_smard(start: date = date(2019, 1, 1), end: date = date(2025, 12, 31)) -> pd.DataFrame:
    logger.info("Loading SMARD data (%s → %s)", start, end)
    bucket = os.environ["ZEPHYRWERK_AWS_BUCKET_NAME"]
    filesystem = _get_filesystem()

    dataset = ds.dataset(
        f"{bucket}/raw/smard",
        filesystem=filesystem,
        format="parquet",
        partitioning="hive",
    )

    df = _filter_timestamp(dataset, start, end)
    logger.info("Loaded %d SMARD rows across %d signals", len(df), df["signal"].nunique())
    return df[["timestamp", "signal", "value", "unit"]]

def load_weather(start: date = date(2019, 1, 1), end: date = date(2025, 12, 31)) -> pd.DataFrame:
    logger.info("Loading weather data (%s → %s)", start, end)
    bucket = os.environ["ZEPHYRWERK_AWS_BUCKET_NAME"]
    filesystem = _get_filesystem()

    dataset = ds.dataset(
        f"{bucket}/raw/weather",
        filesystem=filesystem,
        format="parquet",
        partitioning="hive",
    )

    df = _filter_timestamp(dataset, start, end)
    logger.info("Loaded %d weather rows", len(df))
    return df[["timestamp", "region", "signal_type", "value", "unit"]]


def to_berlin_time(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Converting %d timestamps to Europe/Berlin", len(df))
    df = df.copy()
    berlin = df["timestamp"].dt.tz_convert("Europe/Berlin")
    df["hour_local"] = berlin.dt.hour
    df["date_local"] = berlin.dt.date
    df["dow"]        = berlin.dt.dayofweek   # 0=Mon … 6=Sun
    df["month_local"]      = berlin.dt.month
    df["season"]     = pd.Categorical(
        df["month_local"].map(SEASON_BY_MONTH),
        categories=SEASON_ORDER,
        ordered=True
    )
    return df

def pivot_smard(df: pd.DataFrame) -> pd.DataFrame:
    signals = df["signal"].nunique()
    logger.info("Pivoting %d rows x %d signals to wide format", len(df), signals)

    wide = df.pivot(index="timestamp", columns="signal", values="value")
    wide.columns.name = None
    return wide.reset_index()

def smard_wide_local(
    start: date = date(2019, 1, 1),
    end: date = date(2025, 12, 31),
) -> pd.DataFrame:
    """SMARD signals as columns, indexed by hourly UTC timestamp, with local
    time helpers (hour_local, date_local, dow, month_local, season) appended.
    Use for analytical notebooks. For raw data quality checks, call load_smard()
    directly and skip the transformations."""
    return load_smard(start, end).pipe(pivot_smard).pipe(to_berlin_time)