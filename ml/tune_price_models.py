import argparse
import logging
from datetime import datetime, timezone
from typing import Any

import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from ml.data_access import (
    load_hourly_price_model_features,
    load_quarter_hour_price_model_features,
)
from ml.features.feature_engineering import (
    BASELINE_PRED_COLUMNS,
    TARGET_COLUMNS,
    HourlyPriceModelFeatureEngineer,
    QuarterHourPriceModelFeatureEngineer,
    drop_incomplete_days,
    split_x_y,
    temporal_split,
)
from ml.s3_model_io import save_pipeline
from ml.training_utils import (
    ModelType,
    fill_short_feature_gaps,
    create_preprocessor,
    broadcast_predictions_h2qh,
    draw_predictions,
    evaluate_holdout,
    save_report,
)
from ml.wandb_tracking import start_wandb_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

STAGE1_START_DATE = "2023-05-01"
STAGE2_START_DATE = "2025-10-01"
HOLDOUT_START_DATE = "2026-01-01"
HOLDOUT_END_DATE_EXCLUSIVE = "2026-04-01"


def _build_xgb_from_params(params: dict[str, Any]) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        min_child_weight=float(params["min_child_weight"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def _suggest_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
    }


def _create_price_pipeline(engineer, model_params: dict[str, Any]) -> Pipeline:
    """Create a fresh point-forecast pipeline for one training window."""
    return Pipeline([
        ("engineer", engineer()),
        ("preprocess", create_preprocessor()),
        ("model", _build_xgb_from_params(model_params)),
    ])


def _create_stage1_pipeline(model_params: dict[str, Any]) -> Pipeline:
    return _create_price_pipeline(HourlyPriceModelFeatureEngineer, model_params)


def _create_stage2_pipeline(model_params: dict[str, Any]) -> Pipeline:
    return _create_price_pipeline(QuarterHourPriceModelFeatureEngineer, model_params)


def _predict_stage1_for_qh_index(
    stage1_pipeline: Pipeline,
    hourly_raw: pd.DataFrame,
    qh_index: pd.DatetimeIndex,
) -> pd.Series:
    needed_hours = pd.DatetimeIndex(qh_index).floor("h").unique().sort_values()
    hourly_window = hourly_raw.loc[hourly_raw.index.isin(needed_hours)]
    if hourly_window.empty:
        return pd.Series(index=qh_index, dtype=float, name="stage1_prediction")

    X_hourly, _ = split_x_y(hourly_window, model_type=ModelType.PRICE_HOURLY)
    hourly_pred = pd.Series(stage1_pipeline.predict(X_hourly), index=X_hourly.index, name="hourly_pred")
    return broadcast_predictions_h2qh(pd.DatetimeIndex(qh_index), hourly_pred)


def _objective_stage1(
    trial: optuna.Trial,
    X_stage1_trainval: pd.DataFrame,
    y_stage1_trainval: pd.Series,
    cv_splits: int,
) -> float:
    params = _suggest_xgb_params(trial)
    pipeline = _create_stage1_pipeline(params)
    splitter = TimeSeriesSplit(n_splits=cv_splits, gap=24)

    fold_mae: list[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_stage1_trainval), start=1):
        X_train = X_stage1_trainval.iloc[train_idx]
        y_train = y_stage1_trainval.iloc[train_idx]
        X_test = X_stage1_trainval.iloc[test_idx]
        y_test = y_stage1_trainval.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        fold_mae.append(float(mae))
        trial.set_user_attr(f"fold_{fold_idx}_mae", float(mae))

    score = float(sum(fold_mae) / len(fold_mae))
    trial.set_user_attr("cv_mae", score)
    return score


def _objective_stage2(
    trial: optuna.Trial,
    hourly_trainval_raw: pd.DataFrame,
    qh_trainval_raw: pd.DataFrame,
    best_stage1_params: dict[str, Any],
    cv_splits: int,
) -> float:
    stage2_params = _suggest_xgb_params(trial)
    qh_splitter = TimeSeriesSplit(n_splits=cv_splits, gap=4)
    fold_mae: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(qh_splitter.split(qh_trainval_raw), start=1):
        qh_train_fold = qh_trainval_raw.iloc[train_idx].copy()
        qh_test_fold = qh_trainval_raw.iloc[test_idx].copy()

        if qh_train_fold.empty or qh_test_fold.empty:
            continue

        stage1_train_end = qh_train_fold.index.max().floor("h")
        hourly_fold_train = hourly_trainval_raw.loc[hourly_trainval_raw.index <= stage1_train_end]
        if hourly_fold_train.empty:
            continue

        X_hourly_fold_train, y_hourly_fold_train = split_x_y(hourly_fold_train, model_type=ModelType.PRICE_HOURLY)
        stage1_fold_pipeline = _create_stage1_pipeline(best_stage1_params)
        stage1_fold_pipeline.fit(X_hourly_fold_train, y_hourly_fold_train)

        qh_train_fold["stage1_prediction"] = _predict_stage1_for_qh_index(
            stage1_fold_pipeline,
            hourly_trainval_raw,
            pd.DatetimeIndex(qh_train_fold.index),
        )
        qh_test_fold["stage1_prediction"] = _predict_stage1_for_qh_index(
            stage1_fold_pipeline,
            hourly_trainval_raw,
            pd.DatetimeIndex(qh_test_fold.index),
        )

        qh_train_fold = qh_train_fold.dropna(subset=["stage1_prediction"])
        qh_test_fold = qh_test_fold.dropna(subset=["stage1_prediction"])

        if qh_train_fold.empty or qh_test_fold.empty:
            continue

        X_stage2_train, y_stage2_train_price = split_x_y(qh_train_fold, model_type=ModelType.PRICE_QUARTER_HOURLY)
        X_stage2_test, y_stage2_test_price = split_x_y(qh_test_fold, model_type=ModelType.PRICE_QUARTER_HOURLY)

        stage1_pred_train = X_stage2_train.pop("stage1_prediction")
        stage1_pred_test = X_stage2_test.pop("stage1_prediction")

        y_stage2_train_dev = y_stage2_train_price - stage1_pred_train

        stage2_pipeline = _create_stage2_pipeline(stage2_params)
        stage2_pipeline.fit(X_stage2_train, y_stage2_train_dev)

        pred_dev = stage2_pipeline.predict(X_stage2_test)
        y_reconstructed = stage1_pred_test.to_numpy() + pred_dev
        mae = mean_absolute_error(y_stage2_test_price, y_reconstructed)
        fold_mae.append(float(mae))
        trial.set_user_attr(f"fold_{fold_idx}_mae", float(mae))

    if not fold_mae:
        return float("inf")

    score = float(sum(fold_mae) / len(fold_mae))
    trial.set_user_attr("cv_mae", score)
    return score


def tune_two_stage_price_models(
    n_trials: int = 40,
    stage1_cv_splits: int = 3,
    stage2_cv_splits: int = 2,
    wandb_track: bool = True,
) -> None:
    train_start_time = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "model": "price_two_stage_hyperparameter_tuning",
        "trained_at": train_start_time.isoformat(),
        "search": "optuna",
        "objective": "mae",
        "n_trials": n_trials,
        "cv_splits_stage1": stage1_cv_splits,
        "cv_splits_stage2": stage2_cv_splits,
        "stage1_start_date": STAGE1_START_DATE,
        "stage2_start_date": STAGE2_START_DATE,
        "holdout_start": HOLDOUT_START_DATE,
        "holdout_end_exclusive": HOLDOUT_END_DATE_EXCLUSIVE,
    }

    tracking_run = None
    if wandb_track:
        tracking_run = start_wandb_run(
            run_name=f"price-two-stage-tuning-{train_start_time.strftime('%Y%m%d-%H%M%S')}",
            group="price_hyperparameter_tuning",
        )
        tracking_run.config.update(report)

    logger.info("Loading raw feature contracts")
    hourly_raw = load_hourly_price_model_features(
        start_date=STAGE1_START_DATE,
        end_date_exclusive=HOLDOUT_END_DATE_EXCLUSIVE,
        filter_by_local_timestamp=True,
    )
    qh_raw = load_quarter_hour_price_model_features(
        start_date=STAGE2_START_DATE,
        end_date_exclusive=HOLDOUT_END_DATE_EXCLUSIVE,
        filter_by_local_timestamp=True,
    )

    hourly_raw = drop_incomplete_days(fill_short_feature_gaps(hourly_raw, omit_columns=["price_eur_mwh"]))
    qh_raw = drop_incomplete_days(fill_short_feature_gaps(qh_raw, omit_columns=["price_eur_mwh"]))

    X_stage1_raw, y_stage1 = split_x_y(hourly_raw, model_type=ModelType.PRICE_HOURLY)
    X_stage1_trainval, X_stage1_holdout, y_stage1_trainval, y_stage1_holdout = temporal_split(
        X_stage1_raw,
        y_stage1,
        holdout_start_date=HOLDOUT_START_DATE,
    )

    qh_trainval_raw, qh_holdout_raw, _, _ = temporal_split(
        qh_raw,
        qh_raw[TARGET_COLUMNS[ModelType.PRICE_QUARTER_HOURLY]],
        holdout_start_date=HOLDOUT_START_DATE,
    )

    if X_stage1_trainval.empty or qh_trainval_raw.empty:
        raise ValueError("Insufficient training rows for tuning after preprocessing")

    logger.info("Starting Stage 1 tuning with %d trials and %d splits", n_trials, stage1_cv_splits)
    stage1_study = optuna.create_study(direction="minimize", study_name="stage1_price_hourly")
    stage1_study.optimize(
        lambda trial: _objective_stage1(trial, X_stage1_trainval, y_stage1_trainval, stage1_cv_splits),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_stage1_params = stage1_study.best_params
    logger.info("Stage 1 best CV MAE: %.4f", stage1_study.best_value)

    logger.info("Starting Stage 2 tuning with %d trials and %d splits", n_trials, stage2_cv_splits)
    stage2_study = optuna.create_study(direction="minimize", study_name="stage2_price_quarter_hourly")
    stage2_study.optimize(
        lambda trial: _objective_stage2(trial, hourly_raw.loc[X_stage1_trainval.index], qh_trainval_raw, best_stage1_params, stage2_cv_splits),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_stage2_params = stage2_study.best_params
    logger.info("Stage 2 best CV MAE: %.4f", stage2_study.best_value)

    final_stage1_pipeline = _create_stage1_pipeline(best_stage1_params)
    final_stage1_pipeline.fit(X_stage1_trainval, y_stage1_trainval)

    stage1_holdout_pred = pd.Series(
        final_stage1_pipeline.predict(X_stage1_holdout),
        index=X_stage1_holdout.index,
        name="price_eur_mwh_prediction",
    )
    stage1_holdout_baseline = X_stage1_holdout[BASELINE_PRED_COLUMNS[ModelType.PRICE_HOURLY]]
    stage1_holdout_report, stage1_baseline_report = evaluate_holdout(
        stage1_holdout_pred,
        y_stage1_holdout,
        stage1_holdout_baseline,
    )

    qh_trainval_with_stage1 = qh_trainval_raw.copy()
    qh_holdout_with_stage1 = qh_holdout_raw.copy()
    qh_trainval_with_stage1["stage1_prediction"] = _predict_stage1_for_qh_index(
        final_stage1_pipeline,
        hourly_raw,
        pd.DatetimeIndex(qh_trainval_with_stage1.index),
    )
    qh_holdout_with_stage1["stage1_prediction"] = _predict_stage1_for_qh_index(
        final_stage1_pipeline,
        hourly_raw,
        pd.DatetimeIndex(qh_holdout_with_stage1.index),
    )

    qh_trainval_with_stage1 = qh_trainval_with_stage1.dropna(subset=["stage1_prediction"])
    qh_holdout_with_stage1 = qh_holdout_with_stage1.dropna(subset=["stage1_prediction"])

    X_stage2_train, y_stage2_train_price = split_x_y(qh_trainval_with_stage1, model_type=ModelType.PRICE_QUARTER_HOURLY)
    X_stage2_holdout, y_stage2_holdout_price = split_x_y(qh_holdout_with_stage1, model_type=ModelType.PRICE_QUARTER_HOURLY)

    stage1_train_for_stage2 = X_stage2_train.pop("stage1_prediction")
    stage1_holdout_for_stage2 = X_stage2_holdout.pop("stage1_prediction")
    y_stage2_train_deviation = y_stage2_train_price - stage1_train_for_stage2

    final_stage2_pipeline = _create_stage2_pipeline(best_stage2_params)
    final_stage2_pipeline.fit(X_stage2_train, y_stage2_train_deviation)

    stage2_holdout_reconstructed = pd.Series(
        stage1_holdout_for_stage2.to_numpy() + final_stage2_pipeline.predict(X_stage2_holdout),
        index=y_stage2_holdout_price.index,
        name="price_eur_mwh_prediction",
    )
    stage2_holdout_baseline = qh_holdout_with_stage1[BASELINE_PRED_COLUMNS[ModelType.PRICE_QUARTER_HOURLY]]
    stage2_holdout_report, stage2_baseline_report = evaluate_holdout(
        stage2_holdout_reconstructed,
        y_stage2_holdout_price,
        stage2_holdout_baseline,
    )

    draw_predictions(stage1_holdout_pred, y_stage1_holdout, key_word="price_hourly_tuned_holdout")
    draw_predictions(stage2_holdout_reconstructed, y_stage2_holdout_price, key_word="price_quarter_hourly_tuned_holdout")

    report["stage1"] = {
        "best_cv_mae": float(stage1_study.best_value),
        "best_params": best_stage1_params,
        "n_train": int(len(X_stage1_trainval)),
        "n_holdout": int(len(X_stage1_holdout)),
        "holdout": stage1_holdout_report,
        "baseline_persistence": stage1_baseline_report,
    }
    report["stage2"] = {
        "best_cv_mae": float(stage2_study.best_value),
        "best_params": best_stage2_params,
        "n_train": int(len(X_stage2_train)),
        "n_holdout": int(len(X_stage2_holdout)),
        "holdout": stage2_holdout_report,
        "baseline_persistence": stage2_baseline_report,
    }

    save_report(report, "price_forecast_hyperparameter_tuning_report")

    stage1_s3_uri = save_pipeline(final_stage1_pipeline, model_type=ModelType.PRICE_HOURLY, metadata=report)
    stage2_s3_uri = save_pipeline(final_stage2_pipeline, model_type=ModelType.PRICE_QUARTER_HOURLY, metadata=report)
    report["stage1"]["best_model_s3_uri"] = stage1_s3_uri
    report["stage2"]["best_model_s3_uri"] = stage2_s3_uri
    save_report(report, "price_forecast_hyperparameter_tuning_report")

    logger.info("Saved tuned Stage 1 model to %s", stage1_s3_uri)
    logger.info("Saved tuned Stage 2 model to %s", stage2_s3_uri)

    if tracking_run is not None:
        tracking_run.log(
            {
                "stage1/best_cv_mae": float(stage1_study.best_value),
                "stage2/best_cv_mae": float(stage2_study.best_value),
                "stage1/holdout_mae": float(stage1_holdout_report["mae"]),
                "stage2/holdout_mae": float(stage2_holdout_report["mae"]),
                "stage1/holdout_rmse": float(stage1_holdout_report["rmse"]),
                "stage2/holdout_rmse": float(stage2_holdout_report["rmse"]),
            }
        )
        tracking_run.summary["stage1_model_s3_uri"] = stage1_s3_uri
        tracking_run.summary["stage2_model_s3_uri"] = stage2_s3_uri
        tracking_run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune two-stage price forecasting models with Optuna")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials per stage")
    parser.add_argument("--cv-splits-hourly", type=int, default=4, help="TimeSeriesSplit folds for CV hourly model")
    parser.add_argument("--cv-splits-quarter-hourly", type=int, default=2, help="TimeSeriesSplit folds for CV quarter-hourly model")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B tracking",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tune_two_stage_price_models(
        n_trials=args.trials,
        stage1_cv_splits=args.cv_splits_hourly,
        stage2_cv_splits=args.cv_splits_quarter_hourly,
        wandb_track=False, #not args.no_wandb
    )
