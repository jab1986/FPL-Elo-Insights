import pandas as pd

from ml import metrics


def test_regression_metrics_basic():
    actual = [2.0, 4.0, 6.0, 8.0]
    predicted = [1.5, 4.5, 5.5, 8.5]

    result = metrics.regression_metrics(actual, predicted)

    assert result["count"] == 4
    assert result["mae"] > 0
    assert result["rmse"] >= result["mae"]
    assert "bias" in result and isinstance(result["bias"], float)
    assert "pearson_r" in result


def test_grouped_metrics_returns_per_group():
    frame = pd.DataFrame(
        {
            "actual": [1, 2, 3, 4],
            "pred": [1.1, 2.1, 2.8, 4.2],
            "group": ["A", "A", "B", "B"],
        }
    )

    result = metrics.grouped_metrics(frame, "actual", "pred", "group")

    assert len(result) == 2
    groups = {entry["group"] for entry in result}
    assert groups == {"A", "B"}


def test_compute_baseline_metrics_produces_expected_keys():
    data = pd.DataFrame(
        {
            "season": ["2024-2025"] * 6,
            "player_id": [1, 1, 1, 2, 2, 2],
            "team_id": [10, 10, 10, 11, 11, 11],
            "gameweek": [1, 2, 3, 1, 2, 3],
            "event_points": [2, 5, 6, 1, 0, 3],
        }
    )

    baselines = metrics.compute_baseline_metrics(data, "event_points")

    assert "previous_match_points" in baselines
    assert "player_cumulative_average" in baselines
    assert "team_history_average" in baselines
    for values in baselines.values():
        assert values["count"] > 0
        assert 0 < values["coverage"] <= 1
