import csv
import json

import pytest

from ml.config import PipelineConfig
from ml import pipeline as pipeline_module
from ml.pipeline import PointsPredictionPipeline


def _synthetic_dataset():
    rows = []
    teams = {
        1: {"players": [(101, "Forward"), (102, "Midfielder")]},
        2: {"players": [(201, "Defender"), (202, "Midfielder")]},
    }

    for gameweek in range(1, 9):
        for team_id, info in teams.items():
            opponent_team_id = 2 if team_id == 1 else 1
            is_home = (gameweek + team_id) % 2 == 0
            team_score = (team_id + gameweek) % 3 + 1
            opponent_score = (opponent_team_id + gameweek + 1) % 3
            result = "W" if team_score > opponent_score else "L" if team_score < opponent_score else "D"
            base_elo = 1500 + team_id * 30
            opponent_elo = 1480 + opponent_team_id * 25

            for player_id, position in info["players"]:
                rows.append(
                    {
                        "player_id": player_id,
                        "season": "2024-2025",
                        "gameweek": gameweek,
                        "position": position,
                        "team_id": team_id,
                        "opponent_team_id": opponent_team_id,
                        "is_home": is_home,
                        "team_score": team_score,
                        "opponent_score": opponent_score,
                        "result": result,
                        "team_elo": base_elo + gameweek,
                        "opponent_elo": opponent_elo,
                        "selected_by_percent": 15 + player_id * 0.1 + gameweek,
                        "now_cost": 5.0 + player_id * 0.01,
                        "value_form": 0.4 + 0.05 * gameweek,
                        "value_season": 0.9 + 0.05 * gameweek,
                        "form": 2.0 + 0.1 * gameweek,
                        "ep_next": 3.5 + 0.05 * gameweek,
                        "transfers_in_event": 800 + 20 * gameweek,
                        "transfers_out_event": 200 + 8 * gameweek,
                        "transfers_in": 4000 + 50 * gameweek,
                        "transfers_out": 1500 + 30 * gameweek,
                        "dreamteam_count": 2 + (player_id % 2),
                        "expected_goals": 0.2 + 0.03 * gameweek,
                        "expected_assists": 0.1 + 0.02 * gameweek,
                        "expected_goal_involvements": 0.3 + 0.04 * gameweek,
                        "expected_goals_conceded": 1.1 - 0.03 * gameweek,
                        "ict_index": 7.0 + 0.3 * gameweek,
                        "event_points": 3 + (player_id % 5) + (gameweek % 4),
                    }
                )

    return rows


def test_pipeline_runs_and_persists_results(tmp_path, monkeypatch):
    config = PipelineConfig(
        seasons=("2024-2025",),
        min_history_games=2,
        holdout_gameweeks=1,
        event_point_windows=(2,),
        expectation_windows=(2,),
        team_form_windows=(2,),
        cross_validation_folds=2,
        cv_min_train_gameweeks=2,
        output_dir=tmp_path,
    )

    monkeypatch.setattr(pipeline_module, "load_merged_gameweek_data", lambda _config: _synthetic_dataset())

    pipeline = PointsPredictionPipeline(config)
    result = pipeline.run()

    assert result.train_rows > 0
    assert "train" in result.metrics
    assert "bias" in result.metrics["train"]
    assert result.cross_validation is not None
    assert len(result.cross_validation) > 0
    assert any(entry["fold"] == 1 for entry in result.cross_validation)
    assert result.predictions["test"]
    assert (tmp_path / "points_baseline_metrics.json").exists()
    assert (tmp_path / "points_baseline_coefficients.json").exists()
    assert (tmp_path / "points_baseline_predictions.csv").exists()

    with (tmp_path / "points_baseline_predictions.csv").open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        saved_predictions = list(reader)

    assert any(row.get("split") == "train" for row in saved_predictions)
    assert any(row.get("split") == "test" for row in saved_predictions)

    interval_columns = [
        "predicted_points_lower_p80",
        "predicted_points_upper_p80",
        "predicted_points_lower_p95",
        "predicted_points_upper_p95",
        "predicted_points_expected_error_p95",
    ]
    for column in interval_columns + ["risk_band"]:
        assert all(column in row for row in result.predictions["test"])
        assert any(column in row and row[column] for row in saved_predictions)

    risk_values = {row.get("risk_band") for row in result.predictions["test"] if row.get("risk_band")}
    assert risk_values <= {"low", "medium", "high", "unknown"}
    assert result.diagnostics is not None
    assert "test" in result.diagnostics
    assert "baselines" in result.diagnostics["test"]
    assert "previous_match_points" in result.diagnostics["test"]["baselines"]
    assert "breakdowns" in result.diagnostics["test"]
    assert "position" in result.diagnostics["test"]["breakdowns"]
    assert "uncertainty" in result.diagnostics
    uncertainty = result.diagnostics["uncertainty"]
    assert set(uncertainty.get("confidence_levels", [])) == {0.8, 0.95}
    assert "global" in uncertainty
    assert 80 in uncertainty["global"]["levels"]


def test_pipeline_performs_alpha_tuning(tmp_path, monkeypatch):
    config = PipelineConfig(
        seasons=("2024-2025",),
        min_history_games=2,
        holdout_gameweeks=1,
        event_point_windows=(2,),
        expectation_windows=(2,),
        team_form_windows=(2,),
        cross_validation_folds=0,
        alpha_grid=(0.1, 5.0),
        tuning_metric="rmse",
        tuning_folds=2,
        tuning_min_train_gameweeks=2,
        output_dir=tmp_path,
    )

    dataset = _synthetic_dataset()

    monkeypatch.setattr(
        pipeline_module,
        "load_merged_gameweek_data",
        lambda _config: dataset,
    )

    def fake_evaluate(self, dataset, feature_columns, alpha):
        rmse = 0.2 if alpha < 1 else 1.0
        return [
            {"rmse": rmse, "fold": 1, "season": "2024-2025"},
            {"rmse": rmse, "fold": 2, "season": "2024-2025"},
        ]

    monkeypatch.setattr(PointsPredictionPipeline, "_evaluate_alpha", fake_evaluate)

    pipeline = PointsPredictionPipeline(config)
    result = pipeline.run()

    assert result.model.alpha == pytest.approx(0.1)
    assert result.hyperparameters is not None
    assert result.hyperparameters["alpha"] == pytest.approx(0.1)
    assert "tuning" in result.hyperparameters
    tuning_summary = result.hyperparameters["tuning"]
    assert tuning_summary["metric"] == "rmse"
    assert tuning_summary["selected_alpha"] == pytest.approx(0.1)
    assert any(record["alpha"] == pytest.approx(0.1) for record in tuning_summary["results"])

    payload = json.loads((tmp_path / "points_baseline_metrics.json").read_text())
    assert payload["hyperparameters"]["alpha"] == pytest.approx(0.1)
    assert payload["hyperparameters"]["tuning"]["selected_alpha"] == pytest.approx(0.1)
    assert "model_selection" in payload.get("diagnostics", {})
