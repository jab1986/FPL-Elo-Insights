import math

from ml.config import PipelineConfig
from ml.features import engineer_features


def _sample_records():
    return [
        {
            "player_id": 1,
            "season": "2024-2025",
            "gameweek": gw,
            "position": "Midfielder",
            "team_id": 1,
            "opponent_team_id": 2,
            "is_home": gw % 2 == 0,
            "team_score": 2 + (gw % 2),
            "opponent_score": 1,
            "result": "W" if gw % 3 else "L",
            "team_elo": 1500 + gw,
            "opponent_elo": 1490 - gw,
            "selected_by_percent": 20.0 + gw,
            "now_cost": 7.0 + 0.1 * gw,
            "value_form": 0.5 + 0.05 * gw,
            "value_season": 1.2 + 0.05 * gw,
            "form": 3.0 + 0.2 * gw,
            "ep_next": 4.5 + 0.1 * gw,
            "transfers_in_event": 1000 + 50 * gw,
            "transfers_out_event": 400 + 20 * gw,
            "transfers_in": 5000 + 100 * gw,
            "transfers_out": 2000 + 50 * gw,
            "dreamteam_count": 3,
            "expected_goals": 0.5 + 0.05 * gw,
            "expected_assists": 0.2 + 0.03 * gw,
            "expected_goal_involvements": 0.7 + 0.05 * gw,
            "expected_goals_conceded": 1.2 - 0.05 * gw,
            "ict_index": 10.0 + 0.5 * gw,
            "event_points": 5 + gw % 4,
        }
        for gw in range(1, 7)
    ] + [
        {
            "player_id": 2,
            "season": "2024-2025",
            "gameweek": gw,
            "position": "Defender",
            "team_id": 2,
            "opponent_team_id": 1,
            "is_home": gw % 2 == 1,
            "team_score": 1 + (gw % 2),
            "opponent_score": 2,
            "result": "L" if gw % 2 else "D",
            "team_elo": 1480 + gw,
            "opponent_elo": 1500 - gw,
            "selected_by_percent": 10.0 + 0.5 * gw,
            "now_cost": 5.5 + 0.05 * gw,
            "value_form": 0.3 + 0.04 * gw,
            "value_season": 0.8 + 0.03 * gw,
            "form": 2.0 + 0.1 * gw,
            "ep_next": 3.5 + 0.08 * gw,
            "transfers_in_event": 600 + 30 * gw,
            "transfers_out_event": 300 + 15 * gw,
            "transfers_in": 3500 + 90 * gw,
            "transfers_out": 1700 + 45 * gw,
            "dreamteam_count": 1,
            "expected_goals": 0.2 + 0.04 * gw,
            "expected_assists": 0.1 + 0.02 * gw,
            "expected_goal_involvements": 0.3 + 0.03 * gw,
            "expected_goals_conceded": 1.5 - 0.04 * gw,
            "ict_index": 7.0 + 0.4 * gw,
            "event_points": 3 + (gw % 3),
        }
        for gw in range(1, 7)
    ]


def test_engineer_features_creates_history_aware_columns():
    config = PipelineConfig(
        min_history_games=2,
        event_point_windows=(2,),
        expectation_windows=(2,),
        team_form_windows=(2,),
    )

    dataset, feature_columns = engineer_features(_sample_records(), config)

    assert dataset
    column_set = set(feature_columns)
    for expected in [
        "event_points_lag_1",
        "event_points_lag_2",
        "event_points_rolling_2",
        "expected_goal_involvements_lag_1",
        "position_code",
        "result_code",
        "score_diff",
        "team_event_points_mean_lag_1",
        "opponent_event_points_mean_lag_1",
        "team_event_points_mean_rolling_2",
    ]:
        assert expected in column_set

    for row in dataset:
        for column in feature_columns:
            value = row[column]
            assert value is not None
            if isinstance(value, float):
                assert not math.isnan(value)
        assert row["event_points_lag_1"] is not None
        assert row["expected_goal_involvements_lag_1"] is not None
        assert row["team_event_points_mean_lag_1"] is not None
        assert row["opponent_event_points_mean_lag_1"] is not None
