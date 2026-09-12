import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import joblib

from ml.training_utils import ModelType

logger = logging.getLogger(__name__)

_s3_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None
        )
    return _s3_client


def _reset_client_cache() -> None:
    """Drop the cached boto3 client so the next call rebuilds it. Used by tests to isolate mocked backends."""
    global _s3_client
    _s3_client = None


def _get_bucket_name() -> str:
    bucket = os.environ.get("ZEPHYRWERK_AWS_BUCKET_NAME")
    if bucket is None:
        raise ValueError("ZEPHYRWERK_AWS_BUCKET_NAME environment variable is not set.")
    return bucket


def save_pipeline(pipeline, model_type: ModelType, metadata: dict | None = None) -> str:
    """
    Serialize pipeline with joblib and upload to S3.
    Writes to two locations:
      - s3://.../models/{model_name}/latest/{model_name}.joblib  (overwritten)
      - s3://.../models/{model_name}/archive/{YYYYMMDD-HHMMSS}/{model_name}.joblib
    Returns the S3 URI of the archive copy.
    """
    bucket = _get_bucket_name()
    s3 = _get_s3_client()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_name = f"{model_type.value}_forecast"
    archive_prefix = f"models/{model_name}/archive/{timestamp}"
    latest_prefix = f"models/{model_name}/latest"

    archive_model_key = f"{archive_prefix}/{model_name}.joblib"
    archive_meta_key = f"{archive_prefix}/metadata.json"
    latest_model_key = f"{latest_prefix}/{model_name}.joblib"
    latest_meta_key = f"{latest_prefix}/metadata.json"

    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / f"{model_name}.joblib"
        joblib.dump(pipeline, local_path)
        s3.upload_file(str(local_path), bucket, archive_model_key)

    if metadata is None:
        logger.warning(
            f"Saving {model_name} without metadata — training report will be missing"
        )
        metadata = {}

    s3.put_object(
        Bucket=bucket,
        Key=archive_meta_key,
        Body=json.dumps(metadata, indent=2).encode("utf-8"),
    )

    # latest/ always mirrors the archive copy just written
    s3.copy_object(
        Bucket=bucket,
        Key=latest_model_key,
        CopySource={"Bucket": bucket, "Key": archive_model_key},
    )
    s3.copy_object(
        Bucket=bucket,
        Key=latest_meta_key,
        CopySource={"Bucket": bucket, "Key": archive_meta_key},
    )

    logger.info(f"Saved pipeline {model_name} to s3://{bucket}/{archive_model_key}")
    return f"s3://{bucket}/{archive_model_key}"


def load_pipeline(model_type: ModelType, version: str = "latest") -> tuple:
    """
    Download and deserialize a pipeline from S3.
    version: "latest" or an archive timestamp like "20260702-143012".
    Returns a (pipeline, metadata) tuple.
    """
    if version != "latest" and not re.match(r"^\d{8}-\d{6}$", version):
        raise ValueError(
            f"version must be 'latest' or format YYYYMMDD-HHMMSS, got: {version}"
        )

    bucket = _get_bucket_name()
    s3 = _get_s3_client()

    model_name = f"{model_type.value}_forecast"
    prefix = (
        f"models/{model_name}/latest"
        if version == "latest"
        else f"models/{model_name}/archive/{version}"
    )
    model_key = f"{prefix}/{model_name}.joblib"
    meta_key = f"{prefix}/metadata.json"

    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / f"{model_name}.joblib"
        s3.download_file(bucket, model_key, str(local_path))
        pipeline = joblib.load(local_path)

    meta_obj = s3.get_object(Bucket=bucket, Key=meta_key)
    metadata = json.loads(meta_obj["Body"].read())

    logger.info(f"Loaded pipeline {model_name} from s3://{bucket}/{model_key}")
    return pipeline, metadata


def save_best_hyperparameters(
    model_type: ModelType,
    params: dict,
    metadata: dict | None = None,
    wandb_run_id: str | None = None,
) -> str:
    """
    Save tuned hyperparameters as a small, dedicated artifact.
    Mirrors save_pipeline's latest/archive convention:
      - s3://.../models/{model_name}/hyperparameters/latest/params.json
      - s3://.../models/{model_name}/hyperparameters/archive/{YYYYMMDD-HHMMSS}/params.json
    Returns the S3 URI of the archive copy.
    """
    bucket = _get_bucket_name()
    s3 = _get_s3_client()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_name = f"{model_type.value}_forecast"
    archive_prefix = f"models/{model_name}/hyperparameters/archive/{timestamp}"
    latest_prefix = f"models/{model_name}/hyperparameters/latest"

    archive_key = f"{archive_prefix}/params.json"
    latest_key = f"{latest_prefix}/params.json"

    payload = {
        "model_type": model_type.value,
        "params": params,
        "tuned_at": datetime.now(timezone.utc).isoformat(),
        "wandb_run_id": wandb_run_id,
        **(metadata or {}),
    }
    body = json.dumps(payload, indent=2).encode("utf-8")

    s3.put_object(Bucket=bucket, Key=archive_key, Body=body)
    s3.copy_object(
        Bucket=bucket,
        Key=latest_key,
        CopySource={"Bucket": bucket, "Key": archive_key},
    )

    logger.info(
        f"Saved best hyperparameters for {model_name} to s3://{bucket}/{archive_key}"
    )
    return f"s3://{bucket}/{archive_key}"


def load_best_hyperparameters(model_type: ModelType, version: str = "latest") -> dict:
    """
    Download tuned hyperparameters from S3.
    version: "latest" or an archive timestamp like "20260702-143012".
    Returns the full payload dict (params, tuned_at, wandb_run_id, ...).
    """
    if version != "latest" and not re.match(r"^\d{8}-\d{6}$", version):
        raise ValueError(
            f"version must be 'latest' or format YYYYMMDD-HHMMSS, got: {version}"
        )

    bucket = _get_bucket_name()
    s3 = _get_s3_client()

    model_name = f"{model_type.value}_forecast"
    prefix = (
        f"models/{model_name}/hyperparameters/latest"
        if version == "latest"
        else f"models/{model_name}/hyperparameters/archive/{version}"
    )
    params_key = f"{prefix}/params.json"

    params_obj = s3.get_object(Bucket=bucket, Key=params_key)
    payload = json.loads(params_obj["Body"].read())

    logger.info(
        f"Loaded best hyperparameters for {model_name} from s3://{bucket}/{params_key}"
    )
    return payload
