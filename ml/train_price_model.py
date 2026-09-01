import logging
from datetime import datetime, timezone

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline

from ml.data_access import (
    load_hourly_price_model_features,
    load_quarter_hour_price_model_features,
)
from ml.features.feature_engineering import (
    HourlyPriceModelFeatureEngineer,
    QuarterHourPriceModelFeatureEngineer,
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
from ml.evaluate import regression_metrics
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

STAGE1_START_DATE = "2023-05-01"
STAGE2_START_DATE = "2025-10-01"
HOLDOUT_START_DATE = "2026-01-01"
WEEKLY_PREDICTION_DAYS = 7
WEEK_START_DAY_IDX = 0 # Monday


def _create_price_pipeline(engineer):
    """Create a fresh point-forecast pipeline for one training window."""
    return Pipeline([
        ("engineer", engineer),
        ("preprocess", create_preprocessor()),
        ("model", create_ml_model()),
    ])


def _fill_short_feature_gaps(raw: pd.DataFrame, omit_columns: list[str] | None = None) -> pd.DataFrame:
    """Fill only short gaps while preserving the realized target unchanged."""
    filled = raw.copy()
    for column in filled.columns:
        if omit_columns and column in omit_columns:
            continue
        if not pd.api.types.is_numeric_dtype(filled[column]):
            continue
        if filled[column].isna().any():
            logger.info("Column %s has %d missing values before near fill", column, filled[column].isna().sum())
            filled[column] = filled[column].ffill(limit=3)
            logger.info("Column %s has %d missing values after near fill", column, filled[column].isna().sum())

    return filled


def _predict_stage1_for_window(
    hourly_raw: pd.DataFrame,
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
) -> tuple[Pipeline, pd.Series]:
    """Fit Stage 1 only on rows before a weekly prediction window."""

    train_raw = hourly_raw.loc[hourly_raw.index < prediction_start]
    X_train, y_train = split_x_y(train_raw, model_type=ModelType.PRICE_HOURLY)
    pipeline = _create_price_pipeline(HourlyPriceModelFeatureEngineer())
    pipeline.fit(X_train, y_train)

    window_raw = hourly_raw.loc[
        (hourly_raw.index >= prediction_start) & (hourly_raw.index < prediction_end)
    ]
    X_window, _ = split_x_y(window_raw, model_type=ModelType.PRICE_HOURLY)
    predictions = pd.Series(pipeline.predict(X_window), index=X_window.index, name="stage1_prediction")
    return pipeline, predictions


def _broadcast_hourly_predictions(
    quarter_hour_index: pd.DatetimeIndex,
    hourly_predictions: pd.Series,
) -> pd.Series:
    """Broadcast each hourly Stage 1 point forecast to its four quarter-hours."""
    hourly_by_start = hourly_predictions.copy()
    hourly_by_start.index = pd.DatetimeIndex(hourly_by_start.index).floor("h")
    hourly_prediction_map = hourly_by_start.to_dict()
    quarter_hours = pd.DatetimeIndex(quarter_hour_index).floor("h")
    return pd.Series(
        quarter_hours.map(hourly_prediction_map),
        index=quarter_hour_index,
        name="stage1_prediction",
    )


def run_quarter_hourly_price_model_training(
    hourly_raw: pd.DataFrame,
    quarter_hourly_raw: pd.DataFrame,
    wandb_track: bool = True,
) -> None:
    """Backtest Stage 2 in weekly expanding windows and save the last pipelines.

    Each weekly window is predicted by fresh Stage 1 and Stage 2 pipelines fit only
    on prior delivery days. Stage 2 learns the deviation from Stage 1's historical
    out-of-sample point prediction, then reconstructs quarter-hour prices.
    """
    model_type = ModelType.PRICE_QUARTER_HOURLY
    train_start_time = datetime.now(timezone.utc)
    report: dict[str, object] = {
        "model": f"{model_type.value}_forecast",
        "trained_at": train_start_time.isoformat(),
        "backtest": "weekly_expanding_window",
        "stage2_train_start": STAGE2_START_DATE,
        "holdout_start": HOLDOUT_START_DATE,
        "prediction_window_days": WEEKLY_PREDICTION_DAYS,
    }

    tracking_run = None
    if wandb_track:
        tracking_run = start_wandb_run(
            run_name=f"{report['model']}-training-{train_start_time.strftime('%Y%m%d-%H%M%S')}",
            group=str(report["model"]),
        )

    logger.info("*"*10 + "Filling short gaps and dropping incomplete days in hourly data" + "*"*10)
    hourly_raw = drop_incomplete_days(_fill_short_feature_gaps(hourly_raw, omit_columns=["price_eur_mwh"]))
    logger.info("*"*10 + "Filling short gaps and dropping incomplete days in quarter-hourly data" + "*"*10)
    quarter_hourly_raw = drop_incomplete_days(_fill_short_feature_gaps(quarter_hourly_raw, omit_columns=["price_eur_mwh"]))

    holdout_start = pd.Timestamp(HOLDOUT_START_DATE)
    final_end = quarter_hourly_raw.index.max().normalize() + pd.Timedelta(days=1)
    training_week_starts = pd.date_range(
        start=pd.Timestamp(STAGE2_START_DATE),
        end=holdout_start,
        freq=f"{WEEKLY_PREDICTION_DAYS}D",
        inclusive="left",
    )
    holdout_week_starts = pd.date_range(
        start=holdout_start,
        end=final_end,
        freq=f"{WEEKLY_PREDICTION_DAYS}D",
        inclusive="left",
    )
    weekly_starts = training_week_starts.append(holdout_week_starts)

    annotated_windows: list[pd.DataFrame] = []
    predictions: list[pd.Series] = []
    baseline_predictions: list[pd.Series] = []
    actuals: list[pd.Series] = []
    weekly_reports: list[dict[str, object]] = []
    final_stage1_pipeline = None
    final_stage2_pipeline = None

    for prediction_start in weekly_starts:
        prediction_end = min(prediction_start + pd.Timedelta(days=WEEKLY_PREDICTION_DAYS), final_end)
        quarter_window = quarter_hourly_raw.loc[
            (quarter_hourly_raw.index >= prediction_start) & (quarter_hourly_raw.index < prediction_end)
        ].copy()
        if quarter_window.empty:
            continue

        stage1_pipeline, stage1_predictions = _predict_stage1_for_window(
            hourly_raw,
            prediction_start,
            prediction_end,
        )
        quarter_window["stage1_prediction"] = _broadcast_hourly_predictions(
            pd.DatetimeIndex(quarter_window.index),
            stage1_predictions,
        )
        # quarter_window = drop_incomplete_days(quarter_window)
        if quarter_window.empty:
            logger.warning("Skipping %s because its quarter-hour rows are incomplete", prediction_start.date())
            continue

        if prediction_start < holdout_start:
            annotated_windows.append(quarter_window)
            continue

        # Stage 2 training uses all prior Stage 1 predictions and actuals to learn the deviation
        stage2_training_rows = pd.concat(annotated_windows).sort_index()
        stage2_training_rows = stage2_training_rows.loc[stage2_training_rows.index < prediction_start]
        if stage2_training_rows.empty:
            raise ValueError(f"No Stage 2 training rows available before {prediction_start.date()}")

        X_stage2_train, stage2_price_train = split_x_y(
            stage2_training_rows,
            model_type=ModelType.PRICE_QUARTER_HOURLY,
        )
        stage2_deviation_train = stage2_price_train - X_stage2_train.pop("stage1_prediction")

        X_stage2_window, quarter_actual = split_x_y(
            quarter_window,
            model_type=ModelType.PRICE_QUARTER_HOURLY,
        )
        stage1_window_prediction = X_stage2_window.pop("stage1_prediction")

        stage2_pipeline = _create_price_pipeline(QuarterHourPriceModelFeatureEngineer())
        stage2_pipeline.fit(X_stage2_train, stage2_deviation_train)
        reconstructed_price = pd.Series(
            stage1_window_prediction.to_numpy() + stage2_pipeline.predict(X_stage2_window),
            index=quarter_actual.index,
            name="price_eur_mwh_prediction",
        )

        weekly_report = regression_metrics(quarter_actual, reconstructed_price)
        weekly_report.update(
            {
                "prediction_start": prediction_start.isoformat(),
                "prediction_end_exclusive": prediction_end.isoformat(),
                "n_stage1_train": int((hourly_raw.index < prediction_start).sum()),
                "n_stage2_train": len(X_stage2_train),
                "n_predicted": len(reconstructed_price),
            }
        )
        weekly_reports.append(weekly_report)
        predictions.append(reconstructed_price)
        baseline_predictions.append(quarter_window["price_lag_96qh"])
        actuals.append(quarter_actual)
        annotated_windows.append(quarter_window)

        final_stage1_pipeline = stage1_pipeline
        final_stage2_pipeline = stage2_pipeline
        logger.info(
            "Stage 2 window %s to %s (%s/%s days): MAE %.3f",
            prediction_start.date(),
            (prediction_end - pd.Timedelta(days=1)).date(),
            quarter_window.index.floor("D").nunique(),
            WEEKLY_PREDICTION_DAYS,
            weekly_report["mae"],
        )

    if not predictions or final_stage1_pipeline is None or final_stage2_pipeline is None:
        raise ValueError("No complete Stage 2 holdout windows were available for weekly training")

    holdout_predictions = pd.concat(predictions).sort_index()
    holdout_baseline_predictions = pd.concat(baseline_predictions).sort_index()
    holdout_actuals = pd.concat(actuals).sort_index()

    holdout_report, baseline_report = test_model(holdout_predictions, holdout_actuals, holdout_baseline_predictions, mode=model_type)
    # report["holdout"] = regression_metrics(holdout_actuals, holdout_predictions)
    report["holdout"] = holdout_report
    report["baseline_persistence"] = baseline_report

    report["n_holdout"] = len(holdout_actuals)
    report["weekly_holdout"] = weekly_reports
    report["n_features"] = final_stage2_pipeline.named_steps["model"].n_features_in_
    report["hyperparameters"] = final_stage2_pipeline.named_steps["model"].get_params()

    save_report(report, mode=model_type)
    draw_predictions(holdout_predictions, holdout_actuals, mode=model_type)

    stage1_s3_uri = save_pipeline(final_stage1_pipeline, model_type=ModelType.PRICE_HOURLY)
    stage2_s3_uri = save_pipeline(final_stage2_pipeline, model_type=model_type, metadata=report)
    logger.info("Saved weekly Stage 1 pipeline to %s", stage1_s3_uri)
    logger.info("Saved weekly Stage 2 pipeline to %s", stage2_s3_uri)

    if tracking_run is not None:
        log_training_report(tracking_run, report)
        tracking_run.summary["stage1_model_s3_uri"] = stage1_s3_uri
        tracking_run.summary["stage2_model_s3_uri"] = stage2_s3_uri
        tracking_run.finish()


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
            group=str(report['model']),
        )
        logger.info("Started W&B run with ID: %s", tracking_run.id)

    logger.info("Starting training for model type: %s with %d rows and %d columns raw data", model_type.value, *raw.shape)
    # Preprocess the data
    raw = _fill_short_feature_gaps(raw, omit_columns=["price_eur_mwh"])
    raw = drop_incomplete_days(raw)
    X_raw, y = split_x_y(raw, model_type=model_type)

    X_trainval, X_test, y_trainval, y_test = temporal_split(X_raw, y, holdout_start_date=HOLDOUT_START_DATE)

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

    hourly_raw = load_hourly_price_model_features(
        start_date=STAGE1_START_DATE,
        end_date_exclusive="2026-04-01",
        filter_by_local_timestamp=True,
    )
    quarter_hourly_raw = load_quarter_hour_price_model_features(
        start_date=STAGE2_START_DATE,
        end_date_exclusive="2026-04-01",
        filter_by_local_timestamp=True,
    )
    logger.info("Loaded hourly data with %d rows and %d columns", *hourly_raw.shape)
    logger.info("Loaded quarter-hour data with %d rows and %d columns", *quarter_hourly_raw.shape)

    # run_hourly_price_model_training(hourly_raw, wandb_track=False)
    run_quarter_hourly_price_model_training(hourly_raw, quarter_hourly_raw, wandb_track=False)