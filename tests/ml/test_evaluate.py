import numpy as np
import pytest

from ml.evaluate import (
    baseline_persistence,
    deviation_directional_accuracy,
    directional_accuracy,
    full_evaluation_report,
    peak_mae,
    regression_metrics,
)


def test_regression_metrics_perfect_prediction():
    y_true = [10.0, 20.0, 30.0, 40.0]
    metrics = regression_metrics(y_true, y_true)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_regression_metrics_known_errors():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [2.0, 2.0, 3.0, 6.0]
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(0.75)
    assert metrics["rmse"] == pytest.approx(np.sqrt((1 + 0 + 0 + 4) / 4))


def test_directional_accuracy_all_agree():
    y_true = [1.0, 2.0, 1.0, 3.0]
    y_pred = [10.0, 20.0, 10.0, 30.0]  # moves in the same direction every step
    assert directional_accuracy(y_true, y_pred) == pytest.approx(0.75)  # first diff is NaN -> counts as disagreement


def test_directional_accuracy_all_disagree():
    y_true = [1.0, 2.0, 3.0, 4.0]  # always increasing
    y_pred = [4.0, 3.0, 2.0, 1.0]  # always decreasing
    assert directional_accuracy(y_true, y_pred) == pytest.approx(0.0)


def test_deviation_directional_accuracy():
    reference = [10.0, 10.0, 10.0, 10.0]
    y_true = [12.0, 8.0, 12.0, 8.0]      # up, down, up, down relative to reference
    y_pred = [11.0, 9.0, 9.0, 11.0]      # up, down, down, up -> agrees on first two only
    assert deviation_directional_accuracy(y_true, y_pred, reference) == pytest.approx(0.5)


def test_peak_mae_restricts_to_top_quantile():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 90.0])
    # top_pct=0.9 -> only the single value above the 90th percentile (100.0) is scored
    assert peak_mae(y_true, y_pred, top_pct=0.9) == pytest.approx(10.0)


def test_baseline_persistence():
    reference_series = [10.0, 20.0, 15.0, 25.0]
    y_true = [12.0, 18.0, 20.0, 22.0]
    result = baseline_persistence(reference_series, y_true)
    assert set(result) == {"mae", "rmse", "r2", "directional_accuracy"}
    assert result["mae"] == pytest.approx(3.0)
    assert result["rmse"] == pytest.approx(np.sqrt(10.5))
    assert result["r2"] == pytest.approx(0.25)
    assert result["directional_accuracy"] == pytest.approx(0.5)


def test_full_evaluation_report_respects_flags():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.0, 2.0, 3.0, 4.0]
    reference = [0.0, 0.0, 0.0, 0.0]

    full = full_evaluation_report(y_true, y_pred, reference, include_directional=True, include_peak=True)
    assert set(full) == {"mae", "rmse", "r2", "directional_accuracy", "deviation_directional_accuracy", "peak_mae"}

    minimal = full_evaluation_report(y_true, y_pred, reference, include_directional=False, include_peak=False)
    assert set(minimal) == {"mae", "rmse", "r2"}

    # reference is optional only when directional metrics aren't requested
    full_evaluation_report(y_true, y_pred, include_directional=False, include_peak=False)
    with pytest.raises(ValueError):
        full_evaluation_report(y_true, y_pred, include_directional=True)


def test_length_mismatch_raises_or_documents():
    with pytest.raises((ValueError, Exception)):
        regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0])