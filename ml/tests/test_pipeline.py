import pandas as pd

from ml.config import PipelineConfig
from ml import pipeline as pipeline_module
from ml.pipeline import PointsPredictionPipeline


def _synthetic_dataset() -> pd.DataFrame:
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

    return pd.DataFrame(rows)


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
    assert not result.predictions["test"].empty
    assert (tmp_path / "points_baseline_metrics.json").exists()
    assert (tmp_path / "points_baseline_coefficients.json").exists()
    assert (tmp_path / "points_baseline_predictions.csv").exists()
    saved_predictions = pd.read_csv(tmp_path / "points_baseline_predictions.csv")
    assert set(saved_predictions["split"]) >= {"train", "test"}
    assert result.diagnostics is not None
    assert "test" in result.diagnostics
    assert "baselines" in result.diagnostics["test"]
    assert "previous_match_points" in result.diagnostics["test"]["baselines"]
    assert "breakdowns" in result.diagnostics["test"]
    assert "position" in result.diagnostics["test"]["breakdowns"]

