"""Feature engineering utilities for the ML pipeline."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig

_NUMERIC_BASE_COLUMNS = [
    "gameweek",
    "team_id",
    "opponent_team_id",
    "team_score",
    "opponent_score",
    "team_elo",
    "opponent_elo",
    "now_cost",
    "selected_by_percent",
    "form",
    "value_form",
    "value_season",
    "ep_next",
    "ep_this",
    "transfers_in_event",
    "transfers_out_event",
    "transfers_in",
    "transfers_out",
    "dreamteam_count",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "influence",
    "creativity",
    "threat",
    "ict_index",
]

_POSITION_MAPPING = {
    "Goalkeeper": 0,
    "Defender": 1,
    "Midfielder": 2,
    "Forward": 3,
}

_RESULT_MAPPING = {"W": 1, "D": 0, "L": -1}


def _coerce_numeric_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Convert selected columns to numeric types in-place."""

    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _encode_position(frame: pd.DataFrame) -> None:
    """Add a numeric representation of the player's position."""

    if "position" not in frame.columns:
        return

    frame["position_code"] = frame["position"].map(_POSITION_MAPPING).fillna(-1).astype(int)


def _encode_match_result(frame: pd.DataFrame) -> None:
    """Represent match result as an ordinal feature."""

    if "result" not in frame.columns:
        return

    frame["result_code"] = frame["result"].map(_RESULT_MAPPING).fillna(0).astype(int)


def _add_boolean_feature(frame: pd.DataFrame, column: str, true_value: int = 1, false_value: int = 0) -> None:
    """Normalise boolean-like columns to integer features."""

    if column not in frame.columns:
        return

    mapping = {True: true_value, False: false_value, "True": true_value, "False": false_value}
    frame[column] = frame[column].map(mapping).fillna(frame[column]).fillna(false_value)
    frame[column] = frame[column].astype(int)


def _create_group(frame: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    """Helper to group by season and player."""

    return frame.groupby(["season", "player_id"], sort=False)


def _add_lag_features(frame: pd.DataFrame, columns: Dict[str, Sequence[int]]) -> None:
    """Add lagged features for each specified column."""

    grouped = _create_group(frame)
    for column, lags in columns.items():
        if column not in frame.columns:
            continue
        for lag in lags:
            frame[f"{column}_lag_{lag}"] = grouped[column].shift(lag)


def _add_rolling_features(frame: pd.DataFrame, columns: Dict[str, Sequence[int]]) -> None:
    """Add rolling mean features (using prior observations only)."""

    grouped = _create_group(frame)
    for column, windows in columns.items():
        if column not in frame.columns:
            continue
        series = grouped[column]
        for window in windows:
            frame[f"{column}_rolling_{window}"] = series.transform(
                lambda s, window=window: s.shift(1).rolling(window, min_periods=1).mean()
            )


def _add_differenced_features(frame: pd.DataFrame) -> None:
    """Add net transfer and ownership change features."""

    if {"transfers_in_event", "transfers_out_event"}.issubset(frame.columns):
        frame["transfers_net_event"] = frame["transfers_in_event"] - frame["transfers_out_event"]

    grouped = _create_group(frame)
    if "selected_by_percent" in frame.columns:
        frame["selected_by_percent_change"] = grouped["selected_by_percent"].diff()

    if "now_cost" in frame.columns:
        frame["now_cost_change"] = grouped["now_cost"].diff()


def _team_context_features(frame: pd.DataFrame, config: PipelineConfig) -> Tuple[pd.DataFrame, List[str]]:
    """Compute team and opponent context features for each appearance."""

    required = {"season", "gameweek", "team_id", "opponent_team_id"}
    if not required.issubset(frame.columns):
        return frame, []

    aggregations = {
        "event_points": "team_event_points_mean",
        "expected_goal_involvements": "team_expected_goal_involvements_mean",
        "expected_goals_conceded": "team_expected_goals_conceded_mean",
        "ict_index": "team_ict_index_mean",
    }
    available_aggs = {source: name for source, name in aggregations.items() if source in frame.columns}
    if not available_aggs:
        return frame, []

    team_grouped = frame.groupby(["season", "team_id", "gameweek"], sort=False)
    team_agg = team_grouped.agg({source: "mean" for source in available_aggs}).reset_index()
    team_agg.rename(columns=available_aggs, inplace=True)

    feature_columns: List[str] = []
    for column in available_aggs.values():
        history = team_agg.groupby(["season", "team_id"])[column]
        lag_name = f"{column}_lag_1"
        team_agg[lag_name] = history.shift(1)
        feature_columns.append(lag_name)
        for window in config.team_form_windows:
            rolling_name = f"{column}_rolling_{window}"
            team_agg[rolling_name] = history.shift(1).rolling(window, min_periods=1).mean()
            feature_columns.append(rolling_name)

    merge_columns = ["season", "team_id", "gameweek"] + feature_columns
    team_features = team_agg[merge_columns]

    frame = frame.merge(team_features, on=["season", "team_id", "gameweek"], how="left")

    opponent_columns = {col: col.replace("team_", "opponent_") for col in feature_columns}
    opponent_features = team_features.rename(
        columns={"team_id": "opponent_team_id", **opponent_columns}
    )
    frame = frame.merge(
        opponent_features,
        on=["season", "opponent_team_id", "gameweek"],
        how="left",
    )

    for column in feature_columns:
        frame[column] = frame[column].fillna(0.0)
    for column in opponent_columns.values():
        frame[column] = frame[column].fillna(0.0)

    return frame, feature_columns + list(opponent_columns.values())


def engineer_features(raw: pd.DataFrame, config: PipelineConfig) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare model features and return the dataset ready for training."""

    frame = raw.copy()
    _coerce_numeric_columns(frame, _NUMERIC_BASE_COLUMNS)

    frame["player_id"] = frame["player_id"].astype(int)
    if "team_id" in frame.columns:
        frame["team_id"] = frame["team_id"].astype(int)
    if "opponent_team_id" in frame.columns:
        frame["opponent_team_id"] = frame["opponent_team_id"].astype(int)
    frame.sort_values(["season", "player_id", "gameweek"], inplace=True)

    _encode_position(frame)
    _encode_match_result(frame)
    _add_boolean_feature(frame, "is_home")
    _add_differenced_features(frame)

    for column in ("selected_by_percent_change", "now_cost_change", "transfers_net_event"):
        if column in frame.columns:
            frame[column] = frame[column].fillna(0)

    if {"team_elo", "opponent_elo"}.issubset(frame.columns):
        frame["elo_diff"] = frame["team_elo"] - frame["opponent_elo"]

    if {"team_score", "opponent_score"}.issubset(frame.columns):
        frame["score_diff"] = frame["team_score"] - frame["opponent_score"]

    lag_config = {
        config.target: [1, 2],
        "expected_goal_involvements": [1],
        "expected_goals_conceded": [1],
        "ict_index": [1],
        "form": [1],
        "value_form": [1],
        "value_season": [1],
        "selected_by_percent": [1],
        "now_cost": [1],
        "transfers_net_event": [1],
        "ep_next": [1],
    }
    _add_lag_features(frame, lag_config)

    rolling_config = {
        config.target: config.event_point_windows,
        "expected_goal_involvements": config.expectation_windows,
        "expected_goals_conceded": config.expectation_windows,
        "ict_index": config.expectation_windows,
    }
    _add_rolling_features(frame, rolling_config)

    frame, team_context_features = _team_context_features(frame, config)

    frame["history_games"] = _create_group(frame).cumcount()
    frame = frame[frame["history_games"] >= config.min_history_games].copy()

    feature_candidates = [
        "is_home",
        "gameweek",
        "position_code",
        "result_code",
        "team_elo",
        "opponent_elo",
        "elo_diff",
        "team_score",
        "opponent_score",
        "score_diff",
        "selected_by_percent",
        "selected_by_percent_change",
        "now_cost",
        "now_cost_change",
        "value_form",
        "value_season",
        "form",
        "ep_next",
        "transfers_in_event",
        "transfers_out_event",
        "transfers_net_event",
        "transfers_in",
        "transfers_out",
        "dreamteam_count",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "ict_index",
    ]

    lagged_feature_sources = [
        config.target,
        "expected_goal_involvements",
        "expected_goals_conceded",
        "ict_index",
    ]
    for source in lagged_feature_sources:
        for lag in lag_config.get(source, []):
            candidate = f"{source}_lag_{lag}"
            if candidate not in feature_candidates:
                feature_candidates.append(candidate)

    for source, windows in rolling_config.items():
        for window in windows:
            candidate = f"{source}_rolling_{window}"
            if candidate not in feature_candidates:
                feature_candidates.append(candidate)

    feature_candidates.extend(team_context_features)

    feature_columns = [column for column in feature_candidates if column in frame.columns]

    frame = frame.dropna(subset=[config.target] + [col for col in feature_columns if "lag" in col or "rolling" in col])
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns + [config.target])

    return frame.reset_index(drop=True), feature_columns

