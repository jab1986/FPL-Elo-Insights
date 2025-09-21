import math

from ml.metrics import compute_baseline_metrics, grouped_metrics, regression_metrics


def test_regression_metrics_handles_basic_values():
    actual = [2.0, 3.0, 4.0]
    predicted = [2.5, 2.5, 5.0]

    metrics = regression_metrics(actual, predicted)

    assert metrics["count"] == 3
    assert math.isclose(metrics["mae"], 2 / 3, rel_tol=1e-6)
    assert math.isclose(metrics["bias"], (0.5 - 0.5 + 1.0) / 3, abs_tol=1e-6)
    assert "r2" in metrics


def test_grouped_metrics_aggregates_per_group():
    records = [
        {"group": "A", "actual": 2.0, "pred": 2.5},
        {"group": "A", "actual": 3.0, "pred": 3.5},
        {"group": "B", "actual": 1.0, "pred": 0.5},
    ]

    grouped = grouped_metrics(records, "actual", "pred", "group")

    assert len(grouped) == 2
    for entry in grouped:
        assert entry["count"] >= 1
        assert entry["group"] in {"A", "B"}


def test_compute_baseline_metrics_builds_previous_and_averages():
    dataset = [
        {
            "season": "2024-2025",
            "player_id": 1,
            "team_id": 1,
            "gameweek": 1,
            "event_points": 5.0,
        },
        {
            "season": "2024-2025",
            "player_id": 1,
            "team_id": 1,
            "gameweek": 2,
            "event_points": 6.0,
        },
        {
            "season": "2024-2025",
            "player_id": 2,
            "team_id": 1,
            "gameweek": 1,
            "event_points": 4.0,
        },
        {
            "season": "2024-2025",
            "player_id": 2,
            "team_id": 1,
            "gameweek": 2,
            "event_points": 7.0,
        },
    ]

    baselines = compute_baseline_metrics(dataset, target="event_points")

    assert "previous_match_points" in baselines
    assert baselines["previous_match_points"]["coverage"] > 0
    assert "player_cumulative_average" in baselines
    assert baselines["player_cumulative_average"]["coverage"] > 0
