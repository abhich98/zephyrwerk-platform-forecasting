import json
from enum import Enum
import logging
from pathlib import Path
from typing import overload

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ml.evaluate import (
    baseline_persistence,
    full_evaluation_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    PRICE_HOURLY = "price_hourly"
    PRICE_QUARTER_HOURLY = "price_quarter_hourly"
    PRICE = "price"
    WIND = "wind"
    SOLAR = "solar"


def fill_short_feature_gaps(
    raw: pd.DataFrame, omit_columns: list[str] | None = None, verbose: bool = False
) -> pd.DataFrame:
    """Fill only short gaps while preserving the realized target unchanged."""
    filled = raw.copy()
    for column in filled.columns:
        if omit_columns and column in omit_columns:
            continue
        if not pd.api.types.is_numeric_dtype(filled[column]):
            continue
        if filled[column].isna().any():
            if verbose:
                logger.info(
                    "Column %s has %d missing values before near fill",
                    column,
                    filled[column].isna().sum(),
                )
            filled[column] = filled[column].ffill(limit=3)
            if verbose:
                logger.info(
                    "Column %s has %d missing values after near fill",
                    column,
                    filled[column].isna().sum(),
                )

    return filled


def filter_raw_data(X_raw, y_raw, mode: ModelType):
    # Imported here, not at module scope: feature_engineering imports ModelType
    # from this module, so a top-level import back would be circular.

    # Run the transformer ONCE to identify NaN rows, then drop from raw indices
    if mode == ModelType.PRICE:
        from ml.features.feature_engineering import PriceModelFeatureEngineer

        feature_engineer = PriceModelFeatureEngineer()
    elif mode == ModelType.WIND or mode == ModelType.SOLAR:
        from ml.features.feature_engineering import GenerationModelFeatureEngineer

        feature_engineer = GenerationModelFeatureEngineer()

    tmp = feature_engineer.transform(X_raw)
    valid_idx = tmp.dropna().index
    del tmp

    X_raw = X_raw.loc[valid_idx]
    y = y_raw.loc[valid_idx]

    return X_raw, y


def create_preprocessor():
    """
    Create a ColumnTransformer for preprocessing the features.

    Returns:
        ColumnTransformer: A configured ColumnTransformer object.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), make_column_selector(dtype_include="number")),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")  # keep as DataFrame for readability
    return preprocessor


def create_ml_model():
    """
    Create an XGBRegressor model with predefined hyperparameters.

    Returns:
        XGBRegressor: A configured XGBRegressor model.
    """
    xgb_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,  # confirmed to introducing some randomness i.e, eval results vary slightly on repeated runs
        colsample_bytree=1.0,
        min_child_weight=5,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",  # fast histogram-based training
    )
    model = XGBRegressor(**xgb_params)
    return model


def broadcast_predictions_h2qh(
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


@overload
def save_report(report: dict, mode: ModelType) -> None: ...
@overload
def save_report(report: dict, file_name: str) -> None: ...


def save_report(report: dict, target: ModelType | str) -> None:
    name = f"{target.value}_model_report" if isinstance(target, ModelType) else target
    file_path = f"ml/artifacts/{name}.json"
    Path("ml/artifacts").mkdir(exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(report, f, indent=2)


def evaluate_holdout(y_pred, y_true, test_baseline_pred):
    """
    Return the holdout and baseline evaluation reports.

    Args:
        y_pred (pd.Series): Predicted values for the test set.
        y_true (pd.Series): Actual values for the test set.
        test_baseline_pred (pd.Series): Baseline predictions for the test set.

    Returns:
        tuple: A tuple containing the holdout report and baseline report.
    """

    holdout_report = full_evaluation_report(
        y_true,
        y_pred,
        reference=test_baseline_pred,
        include_directional=True,
        include_peak=False,
    )
    baseline_report = baseline_persistence(test_baseline_pred, y_true)

    return holdout_report, baseline_report


def draw_predictions(y_pred, y_test, key_word: str):
    """
    Draw predictions for one of the models, right after y_pred is computed.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 4))
    y_test.plot(ax=ax, label="actual", alpha=0.7)
    pd.Series(y_pred, index=y_test.index).plot(ax=ax, label="predicted", alpha=0.7)
    ax.legend()
    ax.set_title(f"{key_word} — holdout")
    plt.tight_layout()
    plt.savefig(f"ml/artifacts/{key_word}_holdout.png", dpi=100)
