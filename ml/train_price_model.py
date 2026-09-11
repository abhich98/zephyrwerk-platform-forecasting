import logging
from datetime import datetime, timezone

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline

from ml.data_access import (
    load_hourly_price_model_features,
    load_quarter_hourly_price_model_features,
)
from ml.features.feature_engineering import (
    TARGET_COLUMNS,
    BASELINE_PRED_COLUMNS,
    HourlyPriceModelFeatureEngineer,
    QuarterHourPriceModelFeatureEngineer,
    drop_incomplete_days,
    split_x_y,
)
from ml.s3_model_io import save_pipeline
from ml.training_utils import (
    ModelType,
    fill_short_feature_gaps,
    create_ml_model,
    create_preprocessor,
    broadcast_predictions_h2qh,
    draw_predictions,
    save_report,
    evaluate_holdout,
)
from ml.wandb_tracking import (
    log_optional_artifacts,
    start_wandb_run,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

STAGE1_START_DATE = "2023-05-01"
STAGE2_START_DATE = "2025-10-01"
HOLDOUT_START_DATE = "2026-09-01"
HOLDOUT_END_DATE_EXCLUSIVE = "2026-09-11"
WEEKLY_PREDICTION_DAYS = 7
WEEK_START_DAY_IDX = 0  # Monday


def _create_price_pipeline(engineer):
    """Create a fresh point-forecast pipeline for one training window."""
    return Pipeline(
        [
            ("engineer", engineer()),
            ("preprocess", create_preprocessor()),
            ("model", create_ml_model()),
        ]
    )


def _predict_stage1_for_window(
    hourly_raw: pd.DataFrame,
    prediction_start: pd.Timestamp,
    prediction_end_exclusive: pd.Timestamp,
) -> tuple[Pipeline, pd.Series]:
    """Fit Stage 1 only on rows before a weekly prediction window."""

    train_raw = hourly_raw.loc[hourly_raw.index < prediction_start]
    X_train, y_train = split_x_y(train_raw, model_type=ModelType.PRICE_HOURLY)
    pipeline = _create_price_pipeline(HourlyPriceModelFeatureEngineer)
    pipeline.fit(X_train, y_train)

    window_raw = hourly_raw.loc[
        (hourly_raw.index >= prediction_start)
        & (hourly_raw.index < prediction_end_exclusive)
    ]
    X_window, _ = split_x_y(window_raw, model_type=ModelType.PRICE_HOURLY)
    predictions = pd.Series(
        pipeline.predict(X_window), index=X_window.index, name="stage1_prediction"
    )
    return pipeline, predictions


def _build_weekly_starts(
    holdout_start: pd.Timestamp, holdout_end: pd.Timestamp
) -> pd.DatetimeIndex:
    """Build weekly prediction-window starts for warmup and holdout periods."""
    stage2_start = pd.Timestamp(STAGE2_START_DATE)

    training_week_starts = pd.DatetimeIndex([STAGE2_START_DATE]).append(
        pd.date_range(
            start=stage2_start
            + pd.Timedelta(days=WEEKLY_PREDICTION_DAYS - stage2_start.weekday()),
            end=holdout_start,
            freq=f"{WEEKLY_PREDICTION_DAYS}D",
            inclusive="left",
        )
    )

    holdout_week_starts = pd.DatetimeIndex([HOLDOUT_START_DATE]).append(
        pd.date_range(
            start=holdout_start
            + pd.Timedelta(days=WEEKLY_PREDICTION_DAYS - holdout_start.weekday()),
            end=holdout_end,
            freq=f"{WEEKLY_PREDICTION_DAYS}D",
            inclusive="left",
        )
    )

    return pd.DatetimeIndex(training_week_starts.append(holdout_week_starts))


def _build_pred_windows() -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Build prediction-windows (start and end_exclusives) from warmup and holdout periods. They are weekly for now, but could be a different length in the future."""
    stage2_start = pd.Timestamp(STAGE2_START_DATE)
    holdout_start = pd.Timestamp(HOLDOUT_START_DATE)
    holdout_end_ex = pd.Timestamp(HOLDOUT_END_DATE_EXCLUSIVE)

    training_week_starts = pd.DatetimeIndex([STAGE2_START_DATE]).append(
        pd.date_range(
            start=stage2_start
            + pd.Timedelta(days=WEEKLY_PREDICTION_DAYS - stage2_start.weekday()),
            end=holdout_start,
            freq=f"{WEEKLY_PREDICTION_DAYS}D",
            inclusive="left",
        )
    )

    holdout_week_starts = pd.DatetimeIndex([HOLDOUT_START_DATE]).append(
        pd.date_range(
            start=holdout_start
            + pd.Timedelta(days=WEEKLY_PREDICTION_DAYS - holdout_start.weekday()),
            end=holdout_end_ex,
            freq=f"{WEEKLY_PREDICTION_DAYS}D",
            inclusive="left",
        )
    )

    week_starts = pd.DatetimeIndex(
        training_week_starts.append(holdout_week_starts)
    ).sort_values()
    week_ends_ex = week_starts[1:].append(pd.DatetimeIndex([holdout_end_ex]))
    week_ends_ex = week_ends_ex.unique().sort_values()

    return week_starts, week_ends_ex


def _initialize_predictions_store(
    *model_types: ModelType,
) -> dict[str, dict[str, list[pd.Series]]]:
    """Create per-stage containers for predictions, baselines, and actuals."""
    return {
        model_type.value: {
            "predictions": [],
            "baseline_predictions": [],
            "actuals": [],
        }
        for model_type in model_types
    }


def run_price_model_training(
    hourly_raw: pd.DataFrame,
    quarter_hourly_raw: pd.DataFrame,
    wandb_track: bool = True,
) -> None:
    """Backtest Both stages in weekly expanding windows and save the last pipelines.

    Each weekly window is predicted by fresh Stage 1 and Stage 2 pipelines fit only
    on prior delivery days. Stage 2 learns the deviation from Stage 1's historical
    out-of-sample point prediction, then reconstructs quarter-hour prices.
    """
    model_stage1 = ModelType.PRICE_HOURLY
    model_stage2 = ModelType.PRICE_QUARTER_HOURLY
    train_start_time = datetime.now(timezone.utc)

    report: dict[str, object] = {
        "model": "price_forecast",
        "trained_at": train_start_time.isoformat(),
        "backtest": "weekly_expanding_window",
        f"{model_stage1.value}_train_start": STAGE1_START_DATE,
        f"{model_stage2.value}_train_start": STAGE2_START_DATE,
        "holdout_start": HOLDOUT_START_DATE,
        "holdout_end_exclusive": HOLDOUT_END_DATE_EXCLUSIVE,
        "prediction_window_days": WEEKLY_PREDICTION_DAYS,
    }

    tracking_run = None
    if wandb_track:
        tracking_run = start_wandb_run(
            run_name=f"{report['model']}-training-{train_start_time.strftime('%Y%m%d-%H%M%S')}",
            group=str(report["model"]),
        )
        tracking_run.config.update(report)

    logger.info(
        "%sFilling short gaps and dropping incomplete days in hourly data%s",
        "*" * 10,
        "*" * 10,
    )
    hourly_raw = drop_incomplete_days(
        fill_short_feature_gaps(
            hourly_raw, omit_columns=["price_eur_mwh"], verbose=True
        )
    )
    logger.info(
        "%sFilling short gaps and dropping incomplete days in quarter-hourly data%s",
        "*" * 10,
        "*" * 10,
    )
    quarter_hourly_raw = drop_incomplete_days(
        fill_short_feature_gaps(
            quarter_hourly_raw, omit_columns=["price_eur_mwh"], verbose=True
        )
    )

    holdout_start = pd.Timestamp(HOLDOUT_START_DATE)
    holdout_end = pd.Timestamp(HOLDOUT_END_DATE_EXCLUSIVE)
    # weekly_starts = _build_weekly_starts(holdout_start=holdout_start, holdout_end=holdout_end)
    pred_windows = _build_pred_windows()

    annotated_windows: list[pd.DataFrame] = []
    predictions_dict = _initialize_predictions_store(model_stage1, model_stage2)
    weekly_reports: list[dict[str, object]] = []
    final_stage1_pipeline = None
    final_stage2_pipeline = None

    for prediction_start, prediction_end_exc in zip(*pred_windows):
        if prediction_start <= hourly_raw.index.min().normalize():
            continue
        if prediction_start <= quarter_hourly_raw.index.min().normalize():
            continue

        hourly_window = hourly_raw.loc[
            (hourly_raw.index >= prediction_start)
            & (hourly_raw.index < prediction_end_exc)
        ].copy()
        quarter_hourly_window = quarter_hourly_raw.loc[
            (quarter_hourly_raw.index >= prediction_start)
            & (quarter_hourly_raw.index < prediction_end_exc)
        ].copy()

        if hourly_window.empty:
            logger.warning(
                "Skipping %s because its hourly rows are incomplete",
                prediction_start.date(),
            )
            continue

        stage1_pipeline, stage1_predictions = _predict_stage1_for_window(
            hourly_raw,
            prediction_start,
            prediction_end_exc,
        )
        quarter_hourly_window["stage1_prediction"] = broadcast_predictions_h2qh(
            pd.DatetimeIndex(quarter_hourly_window.index),
            stage1_predictions,
        )

        if quarter_hourly_window.empty:
            logger.warning(
                "Skipping %s because its quarter-hourly rows are incomplete",
                prediction_start.date(),
            )
            continue

        annotated_windows.append(quarter_hourly_window)
        if prediction_start < holdout_start:
            continue

        # Collect stage1 results
        predictions_dict[model_stage1.value]["predictions"].append(stage1_predictions)
        predictions_dict[model_stage1.value]["baseline_predictions"].append(
            hourly_window[BASELINE_PRED_COLUMNS[model_stage1]]
        )
        predictions_dict[model_stage1.value]["actuals"].append(
            hourly_window[TARGET_COLUMNS[model_stage1]]
        )

        # Stage 2 training uses all prior Stage 1 predictions and actuals to learn the deviation
        stage2_training_rows = pd.concat(annotated_windows).sort_index()
        stage2_training_rows = stage2_training_rows.loc[
            stage2_training_rows.index < prediction_start
        ]
        if stage2_training_rows.empty:
            raise ValueError(
                f"No Stage 2 training rows available before {prediction_start.date()}"
            )

        X_stage2_train, stage2_price_train = split_x_y(
            stage2_training_rows,
            model_type=model_stage2,
        )
        stage2_deviation_train = stage2_price_train - X_stage2_train.pop(
            "stage1_prediction"
        )

        X_stage2_window, quarter_hourly_actual = split_x_y(
            quarter_hourly_window,
            model_type=model_stage2,
        )
        stage1_window_prediction = X_stage2_window.pop("stage1_prediction")

        stage2_pipeline = _create_price_pipeline(QuarterHourPriceModelFeatureEngineer)
        stage2_pipeline.fit(X_stage2_train, stage2_deviation_train)
        reconstructed_price = pd.Series(
            stage1_window_prediction.to_numpy()
            + stage2_pipeline.predict(X_stage2_window),
            index=quarter_hourly_actual.index,
            name="price_eur_mwh_prediction",
        )

        predictions_dict[model_stage2.value]["predictions"].append(reconstructed_price)
        predictions_dict[model_stage2.value]["baseline_predictions"].append(
            quarter_hourly_window[BASELINE_PRED_COLUMNS[model_stage2]]
        )
        predictions_dict[model_stage2.value]["actuals"].append(quarter_hourly_actual)

        weekly_report: dict[str, object] = {
            "prediction_start": prediction_start.isoformat(),
            "prediction_end_exclusive": prediction_end_exc.isoformat(),
        }
        for stg_val, stg_dict in predictions_dict.items():
            stage_window_report = evaluate_holdout(
                stg_dict["predictions"][-1],
                stg_dict["actuals"][-1],
                stg_dict["baseline_predictions"][-1],
            )[0]
            weekly_report[stg_val] = stage_window_report

            num_prediction_days = 0
            if stg_val == model_stage1.value:
                num_prediction_days = (
                    pd.DatetimeIndex(hourly_window.index).floor("D").nunique()
                )
            if stg_val == model_stage2.value:
                num_prediction_days = (
                    pd.DatetimeIndex(quarter_hourly_window.index).floor("D").nunique()
                )

            logger.info(
                "Stage %s - window %s to %s (exclusive) (%s/%s days): MAE %.3f",
                stg_val,
                prediction_start.date(),
                prediction_end_exc.date(),
                num_prediction_days,
                (prediction_end_exc.date() - prediction_start.date()).days,
                stage_window_report["mae"],
            )

        weekly_reports.append(weekly_report)

        final_stage1_pipeline = stage1_pipeline
        final_stage2_pipeline = stage2_pipeline

    # Consolidating for the entire holdout period
    if (
        not predictions_dict
        or final_stage1_pipeline is None
        or final_stage2_pipeline is None
    ):
        raise ValueError(
            "No complete Stage 2 holdout windows were available for weekly training"
        )

    for stg_val, stg_dict in predictions_dict.items():
        holdout_predictions = pd.concat(stg_dict["predictions"]).sort_index()
        holdout_baseline_predictions = pd.concat(
            stg_dict["baseline_predictions"]
        ).sort_index()
        holdout_actuals = pd.concat(stg_dict["actuals"]).sort_index()

        holdout_report, baseline_report = evaluate_holdout(
            holdout_predictions,
            holdout_actuals,
            holdout_baseline_predictions,
        )
        stg_report = {}
        stg_report["holdout"] = holdout_report
        stg_report["holdout"]["baseline_persistence"] = baseline_report
        stg_report["n_holdout"] = len(holdout_actuals)

        if stg_val == model_stage1.value:
            stg_report["n_features"] = final_stage1_pipeline.named_steps[
                "model"
            ].n_features_in_
            stg_report["hyperparameters"] = final_stage1_pipeline.named_steps[
                "model"
            ].get_params()
        if stg_val == model_stage2.value:
            stg_report["n_features"] = final_stage2_pipeline.named_steps[
                "model"
            ].n_features_in_
            stg_report["hyperparameters"] = final_stage2_pipeline.named_steps[
                "model"
            ].get_params()

        report[f"{stg_val}"] = stg_report
        draw_predictions(
            holdout_predictions,
            holdout_actuals,
            key_word=stg_val + "_weekly_expanding_window",
        )

        if tracking_run is not None:
            tracking_run.log(
                {f"{stg_val}/{key}": val for key, val in stg_report.items()}
            )

    # Save the final report and weekly reports for the entire holdout period
    report["weekly_holdout"] = weekly_reports
    save_report(report, "price_forecast_model_report")

    stage1_s3_uri = save_pipeline(final_stage1_pipeline, model_type=model_stage1)
    stage2_s3_uri = save_pipeline(
        final_stage2_pipeline, model_type=model_stage2, metadata=report
    )
    logger.info("Saved weekly Stage 1 pipeline to %s", stage1_s3_uri)
    logger.info("Saved weekly Stage 2 pipeline to %s", stage2_s3_uri)

    if tracking_run is not None:
        tracking_run.summary["stage1_model_s3_uri"] = stage1_s3_uri
        tracking_run.summary["stage2_model_s3_uri"] = stage2_s3_uri
        tracking_run.finish()


if __name__ == "__main__":

    hourly_raw = load_hourly_price_model_features(
        start_date=STAGE1_START_DATE,
        end_date_exclusive=HOLDOUT_END_DATE_EXCLUSIVE,
        filter_by_local_timestamp=True,
    )
    quarter_hourly_raw = load_quarter_hourly_price_model_features(
        start_date=STAGE2_START_DATE,
        end_date_exclusive=HOLDOUT_END_DATE_EXCLUSIVE,
        filter_by_local_timestamp=True,
    )
    logger.info("Loaded hourly data with %d rows and %d columns", *hourly_raw.shape)
    logger.info(
        "Loaded quarter-hour data with %d rows and %d columns",
        *quarter_hourly_raw.shape,
    )

    run_price_model_training(hourly_raw, quarter_hourly_raw, wandb_track=False)
