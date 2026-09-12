from typing import Any, Callable

import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from ml.features.feature_engineering import (
    HourlyPriceModelFeatureEngineer,
    QuarterHourPriceModelFeatureEngineer,
    split_x_y,
)
from ml.training_utils import (
    ModelType,
    broadcast_predictions_h2qh,
    create_price_pipeline,
)


def suggest_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest one XGBoost hyperparameter configuration for an Optuna trial."""
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


def create_hourly_pipeline(model_params: dict[str, Any]) -> Pipeline:
    return create_price_pipeline(HourlyPriceModelFeatureEngineer, model_params)


def create_qh_pipeline(model_params: dict[str, Any]) -> Pipeline:
    return create_price_pipeline(QuarterHourPriceModelFeatureEngineer, model_params)


def predict_hourly_for_qh_index(
    hourly_pipeline: Pipeline,
    hourly_df: pd.DataFrame,
    qh_index: pd.DatetimeIndex,
) -> pd.Series:
    """Predict the required hours and broadcast each value to quarter-hour rows."""
    needed_hours = pd.DatetimeIndex(qh_index).floor("h").unique().sort_values()
    hourly_window = hourly_df.loc[hourly_df.index.isin(needed_hours)]
    if hourly_window.empty:
        return pd.Series(index=qh_index, dtype=float, name="hourly_prediction")

    X_hourly, _ = split_x_y(hourly_window, model_type=ModelType.PRICE_HOURLY)
    hourly_prediction = pd.Series(
        hourly_pipeline.predict(X_hourly),
        index=X_hourly.index,
        name="hourly_prediction",
    )
    return broadcast_predictions_h2qh(pd.DatetimeIndex(qh_index), hourly_prediction)


# Stage 1 and Stage 2 tuning objectives for Optuna, agnostic to the specific ml model types and data.


# Do not modify, this function is used as the objective for Optuna stage 1 tuning
def objective_stage1(
    trial: optuna.Trial,
    cv_splits: int,
    X_stage1_trainval: pd.DataFrame,
    y_stage1_trainval: pd.Series,
    create_stage1_pipeline: Callable[[dict[str, Any]], Pipeline],
) -> float:
    """Return mean time-series CV MAE for a candidate model."""
    params = suggest_xgb_params(trial)
    pipeline = create_stage1_pipeline(params)
    splitter = TimeSeriesSplit(n_splits=cv_splits, gap=24)

    fold_mae: list[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(X_stage1_trainval), start=1
    ):
        X_train = X_stage1_trainval.iloc[train_idx]
        y_train = y_stage1_trainval.iloc[train_idx]
        X_test = X_stage1_trainval.iloc[test_idx]
        y_test = y_stage1_trainval.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        mae = float(mean_absolute_error(y_test, pipeline.predict(X_test)))
        fold_mae.append(mae)
        trial.set_user_attr(f"fold_{fold_idx}_mae", mae)

    score = float(sum(fold_mae) / len(fold_mae))
    trial.set_user_attr("cv_mae", score)
    return score


# Do not modify, this function is used as the objective for Optuna stage 2 tuning
def objective_stage2(
    trial: optuna.Trial,
    cv_splits: int,
    stage1_trainval: pd.DataFrame,
    stage2_trainval: pd.DataFrame,
    best_stage1_params: dict[str, Any],
    stage1_model: ModelType,
    stage2_model: ModelType,
    create_stage1_pipeline: Callable[[dict[str, Any]], Pipeline],
    create_stage2_pipeline: Callable[[dict[str, Any]], Pipeline],
    predict_stage1_for_stage2_index: Callable[
        [Any, pd.DataFrame, pd.DatetimeIndex], pd.Series
    ],
) -> float:
    """Return reconstructed-price CV MAE for a candidate stage 2 model, which uses stage 1 predictions as features."""
    stage2_params = suggest_xgb_params(trial)
    splitter = TimeSeriesSplit(n_splits=cv_splits, gap=4)
    fold_mae: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(stage2_trainval), start=1
    ):
        stage2_train_fold = stage2_trainval.iloc[train_idx].copy()
        stage2_test_fold = stage2_trainval.iloc[test_idx].copy()
        stage1_train_end = stage2_train_fold.index.max().floor("h")
        stage1_train_fold = stage1_trainval.loc[
            stage1_trainval.index <= stage1_train_end
        ]
        if stage1_train_fold.empty:
            continue

        X_stage1_train, y_stage1_train = split_x_y(
            stage1_train_fold, model_type=stage1_model
        )
        stage1_pipeline = create_stage1_pipeline(best_stage1_params)
        stage1_pipeline.fit(X_stage1_train, y_stage1_train)

        for stage2_fold in (stage2_train_fold, stage2_test_fold):
            stage2_fold["stage1_prediction"] = predict_stage1_for_stage2_index(
                stage1_pipeline,
                stage1_trainval,
                pd.DatetimeIndex(stage2_fold.index),
            )

        stage2_train_fold = stage2_train_fold.dropna(subset=["stage1_prediction"])
        stage2_test_fold = stage2_test_fold.dropna(subset=["stage1_prediction"])
        if stage2_train_fold.empty or stage2_test_fold.empty:
            continue

        X_stage2_train, y_stage2_train = split_x_y(stage2_train_fold, stage2_model)
        X_stage2_test, y_stage2_test = split_x_y(stage2_test_fold, stage2_model)
        stage1_prediction_train = X_stage2_train.pop("stage1_prediction")
        stage1_prediction_test = X_stage2_test.pop("stage1_prediction")

        stage2_pipeline = create_stage2_pipeline(stage2_params)
        stage2_pipeline.fit(X_stage2_train, y_stage2_train - stage1_prediction_train)
        stage2_prediction = stage1_prediction_test.to_numpy() + stage2_pipeline.predict(
            X_stage2_test
        )

        mae = float(mean_absolute_error(y_stage2_test, stage2_prediction))
        fold_mae.append(mae)
        trial.set_user_attr(f"fold_{fold_idx}_mae", mae)

    if not fold_mae:
        return float("inf")

    score = float(sum(fold_mae) / len(fold_mae))
    trial.set_user_attr("cv_mae", score)
    return score
