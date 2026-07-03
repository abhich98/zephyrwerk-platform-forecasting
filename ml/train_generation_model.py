import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from ml.data_access import load_features
from ml.evaluate import full_evaluation_report, baseline_persistence
from ml.features.feature_engineering import split_x_y, temporal_split, GenerationModelFeatureEngineer
from ml.s3_model_io import save_pipeline


def train_generation_model(
        raw: pd.DataFrame,
        target: str,                # "wind_total_mw" or "solar_mw"
        output_report_name: str     # e.g. "wind_model_report.json"
) -> None:
    """
    Train a generation model (wind or solar) and save the report to a JSON file.

    Args:
        raw (pd.DataFrame): Raw features DataFrame.
        target (str): Target variable, either "wind_total_mw" or "solar_mw".
        output_report_name (str): Name of the output JSON report file.
    """
    feature_name = target.split("_")[0]  # "wind" or "solar"
    if feature_name == "solar":
        raw["solar_mw_lag_24h"] = raw["solar_mw"].shift(24)
        raw["solar_mw_lag_168h"] = raw["solar_mw"].shift(168)

    X_raw, y_raw = split_x_y(raw, target=target)

    # Run the transformer ONCE to identify NaN rows, then drop from raw indices
    tmp = GenerationModelFeatureEngineer().transform(X_raw)
    valid_idx = tmp.dropna().index
    X_raw = X_raw.loc[valid_idx]
    y = y_raw.loc[valid_idx]
    del tmp

    X_trainval, X_test, y_trainval, y_test = temporal_split(X_raw, y, holdout_days=90)

    print("Train/val date range:", X_trainval.index.min(), "to", X_trainval.index.max())
    print("Test date range:", X_test.index.min(), "to", X_test.index.max())

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), make_column_selector(dtype_include="number")),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")   # keep as DataFrame for readability

    xgb_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",   # fast histogram-based training
    )
    model = XGBRegressor(**xgb_params)

    pipeline = Pipeline([
        ("engineer", GenerationModelFeatureEngineer()),
        ("preprocess", preprocessor),
        ("model", model),
    ])

    tscv = TimeSeriesSplit(n_splits=5, gap=24)

    cv_scores = cross_val_score(
        pipeline, X_trainval, y_trainval,
        cv=tscv, scoring="neg_mean_absolute_error", n_jobs=1,
    )
    cv_mae = -cv_scores  # sklearn returns negatives for consistency across scorers
    print(f"CV MAE per fold: {cv_mae}")
    print(f"CV MAE mean: {cv_mae.mean():.2f} ± {cv_mae.std():.2f} mw")

    pipeline.fit(X_trainval, y_trainval)
    y_pred = pipeline.predict(X_test)

    test_baseline_pred = raw[target].shift(24).loc[X_test.index]

    holdout_report = full_evaluation_report(
        y_test, y_pred, reference=test_baseline_pred,
        include_directional=False, include_peak=True,
    )
    baseline_report = baseline_persistence(test_baseline_pred, y_test)

    print(f"Baseline MAE: {baseline_report['mae']:.2f} mw")
    print(f"Model MAE: {holdout_report['mae']:.2f} mw")
    print(f"Model R^2: {holdout_report['r2']:.4f}")
    print(f"Model Peak MAE: {holdout_report['peak_mae']:.2f} mw")

    report = {
        "model": f"{feature_name}_forecast",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_window": {"start": str(X_trainval.index.min()), "end": str(X_trainval.index.max())},
        "test_window": {"start": str(X_test.index.min()), "end": str(X_test.index.max())},
        "n_train": len(X_trainval),
        "n_test": len(X_test),
        "hyperparameters": xgb_params,
        "cv_mae_mean": float(cv_mae.mean()),
        "cv_mae_std": float(cv_mae.std()),
        "cv_mae_per_fold": cv_mae.tolist(),
        "holdout": holdout_report,
        "baseline_persistence": baseline_report,
        "n_features": pipeline.named_steps["model"].n_features_in_,
    }

    Path("ml/artifacts").mkdir(exist_ok=True)
    with open(f"ml/artifacts/{output_report_name}", "w") as f:
        json.dump(report, f, indent=2)

    import matplotlib.pyplot as plt

    # for one of the models, right after y_pred is computed
    fig, ax = plt.subplots(figsize=(15, 4))
    y_test.plot(ax=ax, label="actual", alpha=0.7)
    pd.Series(y_pred, index=y_test.index).plot(ax=ax, label="predicted", alpha=0.7)
    ax.legend()
    ax.set_title(f"{target} — holdout")
    plt.tight_layout()
    plt.savefig(f"ml/artifacts/{target}_holdout.png", dpi=100)

    # Save the trained model to S3 with metadata
    s3_uri = save_pipeline(pipeline, model_name=f"{feature_name}_forecast", metadata=report)
    print(f"Model saved to {s3_uri}")


raw = load_features(start_date="2019-01-01")   # generation model uses full history
raw["wind_total_mw"] = raw["wind_onshore_mw"] + raw["wind_offshore_mw"]

train_generation_model(raw, "wind_total_mw", "wind_model_report.json")
train_generation_model(raw, "solar_mw", "solar_model_report.json")