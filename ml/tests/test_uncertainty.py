import pandas as pd

from ml.uncertainty import ResidualIntervalEstimator


def test_residual_interval_estimator_generates_intervals_and_risk():
    frame = pd.DataFrame(
        {
            "predicted_points": [4.0, 5.0, 6.5, 7.2, 5.5, 4.8, 6.2, 7.8],
            "actual_points": [5.5, 3.0, 7.1, 6.0, 6.2, 4.0, 5.8, 9.0],
            "position": [
                "Forward",
                "Forward",
                "Midfielder",
                "Midfielder",
                "Forward",
                "Defender",
                "Defender",
                "Midfielder",
            ],
        }
    )

    estimator = ResidualIntervalEstimator(confidence_levels=(0.8, 0.95))
    estimator.fit(frame)

    summary = estimator.describe()
    assert "global" in summary
    assert 80 in summary["global"]["levels"]
    assert "error_thresholds" in summary

    enriched = estimator.apply(frame)
    expected_columns = {
        "predicted_points_lower_p80",
        "predicted_points_upper_p80",
        "predicted_points_lower_p95",
        "predicted_points_upper_p95",
        "predicted_points_expected_error_p95",
        "risk_band",
    }
    assert expected_columns.issubset(enriched.columns)
    assert (enriched["predicted_points_upper_p80"] >= enriched["predicted_points_lower_p80"]).all()
    assert set(enriched["risk_band"].dropna().unique()) <= {"low", "medium", "high", "unknown"}


def test_residual_interval_estimator_without_group_column():
    frame = pd.DataFrame(
        {
            "predicted_points": [2.0, 3.5, 4.2, 1.8],
            "actual_points": [1.0, 5.0, 6.1, 0.5],
        }
    )

    estimator = ResidualIntervalEstimator(confidence_levels=(0.8,))
    estimator.fit(frame)

    enriched = estimator.apply(frame)
    assert "predicted_points_lower_p80" in enriched.columns
    assert "predicted_points_upper_p80" in enriched.columns
    assert "risk_band" in enriched.columns
