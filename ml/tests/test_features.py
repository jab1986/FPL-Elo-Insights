import pandas as pd

from ml.config import PipelineConfig
from ml.features import engineer_features


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "season": ["2024-2025"] * 4 + ["2024-2025"] * 4,
            "gameweek": [1, 2, 3, 4, 1, 2, 3, 4],
            "position": ["Midfielder", "Midfielder", "Midfielder", "Midfielder", "Defender", "Defender", "Defender", "Defender"],
            "team_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "opponent_team_id": [2, 2, 2, 2, 1, 1, 1, 1],
            "is_home": [True, False, True, False, True, False, True, False],
            "team_score": [2, 1, 3, 2, 1, 0, 1, 2],
            "opponent_score": [1, 1, 1, 2, 2, 1, 3, 2],
            "result": ["W", "D", "W", "L", "L", "L", "L", "D"],
            "team_elo": [1500, 1510, 1520, 1530, 1450, 1460, 1470, 1480],
            "opponent_elo": [1480, 1470, 1490, 1500, 1430, 1440, 1450, 1460],
            "selected_by_percent": [20.0, 21.0, 22.0, 23.0, 10.0, 10.5, 11.0, 11.5],
            "now_cost": [7.0, 7.0, 7.1, 7.1, 5.5, 5.6, 5.6, 5.7],
            "value_form": [0.5, 0.6, 0.7, 0.75, 0.3, 0.35, 0.4, 0.45],
            "value_season": [1.2, 1.3, 1.35, 1.4, 0.8, 0.85, 0.9, 0.95],
            "form": [3.0, 3.2, 3.5, 3.7, 2.0, 2.1, 2.2, 2.3],
            "ep_next": [4.5, 4.6, 4.8, 4.9, 3.0, 3.1, 3.2, 3.3],
            "transfers_in_event": [1000, 1100, 1200, 1300, 500, 550, 600, 650],
            "transfers_out_event": [400, 420, 430, 440, 200, 220, 230, 240],
            "expected_goal_involvements": [0.5, 0.6, 0.7, 0.8, 0.2, 0.25, 0.3, 0.35],
            "expected_goals_conceded": [1.2, 1.1, 1.0, 0.9, 1.5, 1.4, 1.3, 1.2],
            "ict_index": [10.0, 11.0, 12.0, 13.0, 7.0, 7.5, 8.0, 8.5],
            "event_points": [5, 7, 6, 8, 2, 3, 4, 5],
        }
    )


def test_engineer_features_creates_history_aware_columns():
    config = PipelineConfig(
        min_history_games=2,
        event_point_windows=(2,),
        expectation_windows=(2,),
        team_form_windows=(2,),
    )

    dataset, feature_columns = engineer_features(_sample_frame(), config)

    assert not dataset.empty
    assert "event_points_lag_1" in feature_columns
    assert "event_points_lag_2" in feature_columns
    assert "event_points_rolling_2" in feature_columns
    assert "expected_goal_involvements_lag_1" in feature_columns
    assert "position_code" in feature_columns
    assert "result_code" in feature_columns
    assert "score_diff" in feature_columns
    assert "team_event_points_mean_lag_1" in feature_columns
    assert "opponent_event_points_mean_lag_1" in feature_columns
    assert "team_event_points_mean_rolling_2" in feature_columns
    assert dataset["event_points_lag_1"].notna().all()
    assert dataset["expected_goal_involvements_lag_1"].notna().all()
    assert dataset["team_event_points_mean_lag_1"].notna().all()
    assert dataset["opponent_event_points_mean_lag_1"].notna().all()
    assert dataset[feature_columns].isna().sum().sum() == 0

