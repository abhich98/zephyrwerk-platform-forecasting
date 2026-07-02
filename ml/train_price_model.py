import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from ml.data_access import load_features
from ml.features.feature_engineering import split_x_y, temporal_split, PriceModelFeatureEngineer
from ml.s3_model_io import save_pipeline

raw = load_features(start_date="2023-04-09")   # 7-day buffer before 2023-04-16 for lags

X_raw, y_raw = split_x_y(raw, target="price_eur_mwh")

# Run the transformer ONCE to identify NaN rows, then drop from raw indices
tmp = PriceModelFeatureEngineer().transform(X_raw)
valid_idx = tmp.dropna().index
X_raw = X_raw.loc[valid_idx].loc["2023-04-16":]
y = y_raw.loc[valid_idx].loc["2023-04-16":]

print(X_raw.shape, y.shape)
print(X_raw.isna().sum().sort_values(ascending=False).head(10))
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
    ("engineer", PriceModelFeatureEngineer()),
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
print(f"CV MAE mean: {cv_mae.mean():.2f} ± {cv_mae.std():.2f} EUR/MWh")

pipeline.fit(X_trainval, y_trainval)
y_pred = pipeline.predict(X_test)


test_mae = (y_test - y_pred).abs().mean()
test_baseline_pred = X_test["price_24h_lag"]
baseline_mae = (y_test - test_baseline_pred).abs().mean()

# y_pred is a numpy array; align it with y_test's index for differencing
y_pred_series = pd.Series(y_pred, index=y_test.index)
directional_accuracy = ((y_pred_series.diff() * y_test.diff()) > 0).mean()
baseline_directional_accuracy = ((test_baseline_pred.diff() * y_test.diff()) > 0).mean()

ref = X_test["price_24h_lag"]  # yesterday, same hour
model_up = (y_pred > ref)
actual_up = (y_test > ref)
deviation_direction_accuracy = (model_up == actual_up).mean()

print(f"Baseline MAE: {baseline_mae:.2f} EUR/MWh")
print(f"Model MAE: {test_mae:.2f} EUR/MWh")
print(f"Baseline Directional Accuracy: {baseline_directional_accuracy:.2f}")
print(f"Model Directional Accuracy: {directional_accuracy:.2f}")
print(f"Model Deviation Directional Accuracy: {deviation_direction_accuracy:.2f}")

report = {
    "model": "price_forecast",
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "train_window": {"start": str(X_trainval.index.min()), "end": str(X_trainval.index.max())},
    "test_window": {"start": str(X_test.index.min()), "end": str(X_test.index.max())},
    "n_train": len(X_trainval),
    "n_test": len(X_test),
    "hyperparameters": xgb_params,
    "cv_mae_mean": float(cv_mae.mean()),
    "cv_mae_std": float(cv_mae.std()),
    "cv_mae_per_fold": cv_mae.tolist(),
    "holdout": {
        "mae": float(test_mae),
        "rmse": float(np.sqrt(((y_test - y_pred) ** 2).mean())),
        "directional_accuracy": float(directional_accuracy),
        "deviation_directional_accuracy": float(deviation_direction_accuracy),
    },
    "baseline_persistence": {"mae": float(baseline_mae), "directional_accuracy": float(baseline_directional_accuracy)},
    "n_features": pipeline.named_steps["model"].n_features_in_,
}

Path("ml/artifacts").mkdir(exist_ok=True)
with open("ml/artifacts/price_model_report.json", "w") as f:
    json.dump(report, f, indent=2)

s3_uri = save_pipeline(pipeline, model_name="price_forecast", metadata=report)
print(f"Model saved to {s3_uri}")