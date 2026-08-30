import logging
from datetime import datetime, timezone

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline

from ml.data_access import load_hourly_price_model_features
from ml.features.feature_engineering import (
    HourlyPriceModelFeatureEngineer,
    drop_incomplete_days,
    split_x_y,
    temporal_split,
)
from ml.s3_model_io import save_pipeline
from ml.training_utils import (
    ModelType,
    create_ml_model,
    create_preprocessor,
    draw_predictions,
    save_report,
    test_model,
)
from ml.wandb_tracking import (
    log_optional_artifacts,
    log_training_report,
    start_wandb_run,
)

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
logger = logging.getLogger(__name__)


def run_hourly_price_model_training(raw, wandb_track: bool = True):
    """
    Train the price model and save the report to a JSON file.

    Args:
        raw (pd.DataFrame): Raw features DataFrame.
    """
    model_type = ModelType.PRICE_HOURLY
    train_start_time = datetime.now(timezone.utc)
    report: dict[str, object] = {
            "model": f"{model_type.value}_forecast",
            "trained_at": train_start_time.isoformat(),
        }

    if wandb_track:
        tracking_run = start_wandb_run(
            run_name=f"{report['model']}-training-{train_start_time.strftime('%Y%m%d-%H%M%S')}",
            group=report['model'],
        )
        logger.info("Started W&B run with ID: %s", tracking_run.id)

    logger.info("Starting training for model type: %s with %d rows and %d columns raw data", model_type.value, *raw.shape)
    # Preprocess the data
    for col in raw.columns:
        missing_count = raw[col].isna().sum()
        if missing_count > 0:
            logger.info("Column %s has %d missing values", col, missing_count)
            raw[col] = raw[col].ffill(limit=2)
            raw[col] = raw[col].bfill(limit=2)
            logger.info("Column %s has %d missing values after near fill", col, raw[col].isna().sum())

    raw = drop_incomplete_days(raw)
    X_raw, y = split_x_y(raw, model_type=model_type)

    X_trainval, X_test, y_trainval, y_test = temporal_split(X_raw, y, holdout_start_date="2026-01-01")

    report["n_train"] = len(X_trainval)
    report["n_test"] = len(X_test)
    report["train_window"] = {
        "start": str(X_trainval.index.min()),
        "end": str(X_trainval.index.max())
        }
    report["test_window"] = {"start": str(X_test.index.min()), "end": str(X_test.index.max())}

    preprocessor = create_preprocessor()
    model = create_ml_model()

    pipeline = Pipeline([
        ("engineer", HourlyPriceModelFeatureEngineer()),
        ("preprocess", preprocessor),
        ("model", model),
    ])

    tscv = TimeSeriesSplit(n_splits=5, gap=24)

    logger.info("Starting cross-validation with %d splits and gap of 24 hours", tscv.get_n_splits())
    logger.info("Training data shape: %s, Test data shape: %s", X_trainval.shape, X_test.shape)
    cv_scores = cross_val_score(
        pipeline, X_trainval, y_trainval,
        cv=tscv, scoring="neg_mean_absolute_error", n_jobs=1,
    )
    cv_mae = -cv_scores  # sklearn returns negatives for consistency across scorers
    report["cv_mae_mean"] = float(cv_mae.mean())
    report["cv_mae_std"] = float(cv_mae.std())
    report["cv_mae_per_fold"] = cv_mae.tolist()

    pipeline.fit(X_trainval, y_trainval)
    report["hyperparameters"] = model.get_params()
    report["n_features"] = pipeline.named_steps["model"].n_features_in_

    y_pred = pipeline.predict(X_test)
    test_baseline_pred = X_test["price_lag_24h"]  # yesterday, same hour
    
    holdout_report, baseline_report = test_model(y_pred, y_test, test_baseline_pred, mode=model_type)
    report["holdout"] = holdout_report
    report["baseline_persistence"] = baseline_report

    save_report(report, mode=model_type)
    draw_predictions(y_pred, y_test, mode=model_type)

    s3_uri = save_pipeline(pipeline, model_type=model_type, metadata=report)
    logger.info(f"Model saved to {s3_uri}")

    if wandb_track:
        log_training_report(tracking_run, report)
        # log_optional_artifacts(tracking_run, pipeline, model_type)
        tracking_run.summary["model_s3_uri"] = s3_uri
        tracking_run.finish()


if __name__ == "__main__":

    raw = load_hourly_price_model_features(
        start_date="2023-05-01",
        end_date_exclusive="2026-03-31",
        filter_by_local_timestamp=True,
    )
    logger.info("Loaded data with %d rows and %d columns", *raw.shape)

    run_hourly_price_model_training(raw, wandb_track=True)