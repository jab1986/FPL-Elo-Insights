import math

from ml.uncertainty import ResidualIntervalEstimator


def _records():
    return [
        {"actual_points": 5.0, "predicted_points": 4.5, "position": "Forward"},
        {"actual_points": 6.0, "predicted_points": 5.5, "position": "Forward"},
        {"actual_points": 3.0, "predicted_points": 3.5, "position": "Defender"},
        {"actual_points": 4.0, "predicted_points": 4.2, "position": "Defender"},
    ]


def test_fit_and_apply_generate_bounds():
    estimator = ResidualIntervalEstimator(confidence_levels=(0.8, 0.95))
    estimator.fit(_records())

    summary = estimator.describe()
    assert summary
    assert "global" in summary
    assert set(summary.get("confidence_levels", [])) == {0.8, 0.95}

    augmented = estimator.apply(_records())
    assert augmented
    expected_columns = {
        "predicted_points_lower_p80",
        "predicted_points_upper_p80",
        "predicted_points_lower_p95",
        "predicted_points_upper_p95",
        "predicted_points_expected_error_p95",
        "risk_band",
    }
    for row in augmented:
        assert expected_columns.issubset(row)
        assert not math.isnan(row["predicted_points_lower_p80"])
        assert not math.isnan(row["predicted_points_upper_p95"])
        assert row["risk_band"] in {"low", "medium", "high", "unknown"}
