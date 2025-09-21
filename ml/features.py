"""Feature engineering utilities for the ML pipeline without third-party dependencies."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .config import PipelineConfig
from .utils import Dataset, is_nan, sort_records, to_float, to_int


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")

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
    "event_points",
]

_POSITION_MAPPING = {
    "Goalkeeper": 0,
    "Defender": 1,
    "Midfielder": 2,
    "Forward": 3,
}

_RESULT_MAPPING = {"W": 1, "D": 0, "L": -1}


def _coerce_numeric(record: MutableMapping[str, object], columns: Iterable[str]) -> None:
    for column in columns:
        if column in record:
            record[column] = to_float(record[column])


def _encode_position(record: MutableMapping[str, object]) -> None:
    position = record.get("position")
    record["position_code"] = _POSITION_MAPPING.get(position, -1)


def _encode_match_result(record: MutableMapping[str, object]) -> None:
    result = record.get("result")
    record["result_code"] = _RESULT_MAPPING.get(result, 0)


def _normalise_boolean(record: MutableMapping[str, object], column: str) -> None:
    value = record.get(column)
    if isinstance(value, bool):
        record[column] = int(value)
    elif isinstance(value, str):
        record[column] = 1 if value.lower() == "true" else 0 if value.lower() == "false" else 0
    else:
        record[column] = int(bool(value))


def _add_difference_features(records: Dataset) -> None:
    history: Dict[Tuple[object, object], Dict[str, float]] = {}
    for record in records:
        key = (record.get("season"), record.get("player_id"))
        state = history.setdefault(key, {})

        transfers_in_event = to_float(record.get("transfers_in_event"))
        transfers_out_event = to_float(record.get("transfers_out_event"))
        if not math.isnan(transfers_in_event) and not math.isnan(transfers_out_event):
            record["transfers_net_event"] = transfers_in_event - transfers_out_event
        else:
            record["transfers_net_event"] = 0.0

        selected = to_float(record.get("selected_by_percent"))
        if not math.isnan(selected) and not math.isnan(state.get("selected_by_percent", float("nan"))):
            record["selected_by_percent_change"] = selected - state["selected_by_percent"]
        else:
            record["selected_by_percent_change"] = 0.0

        now_cost = to_float(record.get("now_cost"))
        if not math.isnan(now_cost) and not math.isnan(state.get("now_cost", float("nan"))):
            record["now_cost_change"] = now_cost - state["now_cost"]
        else:
            record["now_cost_change"] = 0.0

        if not math.isnan(selected):
            state["selected_by_percent"] = selected
        if not math.isnan(now_cost):
            state["now_cost"] = now_cost


def _add_simple_diffs(record: MutableMapping[str, object]) -> None:
    team_elo = to_float(record.get("team_elo"))
    opponent_elo = to_float(record.get("opponent_elo"))
    if not math.isnan(team_elo) and not math.isnan(opponent_elo):
        record["elo_diff"] = team_elo - opponent_elo
    else:
        record["elo_diff"] = float("nan")

    team_score = to_float(record.get("team_score"))
    opponent_score = to_float(record.get("opponent_score"))
    if not math.isnan(team_score) and not math.isnan(opponent_score):
        record["score_diff"] = team_score - opponent_score
    else:
        record["score_diff"] = float("nan")


def _add_lag_features(records: Dataset, columns: Dict[str, Sequence[int]]) -> None:
    histories: Dict[Tuple[object, object], Dict[str, List[float]]] = {}
    for record in records:
        key = (record.get("season"), record.get("player_id"))
        player_history = histories.setdefault(key, {})
        for column, lags in columns.items():
            history = player_history.setdefault(column, [])
            for lag in lags:
                name = f"{column}_lag_{lag}"
                if len(history) >= lag:
                    value = history[-lag]
                else:
                    value = float("nan")
                record[name] = value
            value = to_float(record.get(column))
            history.append(value)


def _add_rolling_features(records: Dataset, columns: Dict[str, Sequence[int]]) -> None:
    histories: Dict[Tuple[object, object], Dict[str, List[float]]] = {}
    for record in records:
        key = (record.get("season"), record.get("player_id"))
        player_history = histories.setdefault(key, {})
        for column, windows in columns.items():
            history = player_history.setdefault(column, [])
            for window in windows:
                name = f"{column}_rolling_{window}"
                if history:
                    values = [value for value in history[-window:] if not is_nan(value)]
                    record[name] = _mean(values)
                else:
                    record[name] = float("nan")
            value = to_float(record.get(column))
            history.append(value)


def _team_context_features(records: Dataset, config: PipelineConfig) -> Tuple[List[str], List[str]]:
    aggregations = {
        "event_points": "team_event_points_mean",
        "expected_goal_involvements": "team_expected_goal_involvements_mean",
        "expected_goals_conceded": "team_expected_goals_conceded_mean",
        "ict_index": "team_ict_index_mean",
    }

    team_gameweek_values: Dict[Tuple[object, object, object], Dict[str, List[float]]] = {}
    for record in records:
        season = record.get("season")
        team_id = record.get("team_id")
        gameweek = record.get("gameweek")
        if team_id is None or gameweek is None:
            continue
        key = (season, team_id, gameweek)
        stats = team_gameweek_values.setdefault(key, {source: [] for source in aggregations})
        for source in aggregations:
            value = to_float(record.get(source))
            if not math.isnan(value):
                stats[source].append(value)

    team_features: Dict[Tuple[object, object, object], Dict[str, float]] = {}
    for key, stats in team_gameweek_values.items():
        feature_row: Dict[str, float] = {}
        for source, dest in aggregations.items():
            values = stats.get(source, [])
            feature_row[dest] = _mean(values)
        team_features[key] = feature_row

    team_feature_names: List[str] = []
    team_histories: Dict[Tuple[object, object], Dict[str, List[float]]] = {}
    for (season, team_id, gameweek) in sorted(team_features.keys()):
        key = (season, team_id)
        summary = team_features[(season, team_id, gameweek)]
        history = team_histories.setdefault(key, {name: [] for name in summary})
        enriched: Dict[str, float] = {}
        for name, current_value in summary.items():
            enriched[name] = current_value
            history_series = history.setdefault(name, [])
            lag_name = f"{name}_lag_1"
            if lag_name not in team_feature_names:
                team_feature_names.append(lag_name)
            enriched[lag_name] = history_series[-1] if history_series else float("nan")
            for window in config.team_form_windows:
                rolling_name = f"{name}_rolling_{window}"
                if rolling_name not in team_feature_names:
                    team_feature_names.append(rolling_name)
                if history_series:
                    values = [val for val in history_series[-window:] if not is_nan(val)]
                    enriched[rolling_name] = _mean(values)
                else:
                    enriched[rolling_name] = float("nan")
            history_series.append(current_value)
        summary.update(enriched)
        for candidate in summary.keys():
            if candidate not in team_feature_names:
                team_feature_names.append(candidate)

    opponent_feature_names = [name.replace("team_", "opponent_") for name in team_feature_names]

    for record in records:
        season = record.get("season")
        team_id = record.get("team_id")
        opponent_team_id = record.get("opponent_team_id")
        gameweek = record.get("gameweek")
        team_key = (season, team_id, gameweek)
        opponent_key = (season, opponent_team_id, gameweek)

        features = team_features.get(team_key, {})
        for name in team_feature_names:
            record[name] = features.get(name, 0.0)

        opponent_features = team_features.get(opponent_key, {})
        for source, target in zip(team_feature_names, opponent_feature_names):
            record[target] = opponent_features.get(source, 0.0)

    return team_feature_names, opponent_feature_names


def engineer_features(raw: Iterable[Mapping[str, object]], config: PipelineConfig) -> Tuple[Dataset, List[str]]:
    """Prepare model features and return the dataset ready for training."""

    records = sort_records(raw, ["season", "player_id", "gameweek"])

    for record in records:
        _coerce_numeric(record, _NUMERIC_BASE_COLUMNS)
        player_id = to_int(record.get("player_id"))
        if player_id is None:
            record["player_id"] = -1
        else:
            record["player_id"] = player_id
        if "team_id" in record:
            team_id = to_int(record.get("team_id"))
            record["team_id"] = team_id if team_id is not None else -1
        if "opponent_team_id" in record:
            opponent_team_id = to_int(record.get("opponent_team_id"))
            record["opponent_team_id"] = opponent_team_id if opponent_team_id is not None else -1
        _encode_position(record)
        _encode_match_result(record)
        _normalise_boolean(record, "is_home")
        _add_simple_diffs(record)

    _add_difference_features(records)

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
    _add_lag_features(records, lag_config)

    rolling_config = {
        config.target: config.event_point_windows,
        "expected_goal_involvements": config.expectation_windows,
        "expected_goals_conceded": config.expectation_windows,
        "ict_index": config.expectation_windows,
    }
    _add_rolling_features(records, rolling_config)

    team_features, opponent_features = _team_context_features(records, config)

    history_counts: Dict[Tuple[object, object], int] = {}
    for record in records:
        key = (record.get("season"), record.get("player_id"))
        count = history_counts.get(key, 0)
        record["history_games"] = count
        history_counts[key] = count + 1

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

    for source, lags in lag_config.items():
        for lag in lags:
            candidate = f"{source}_lag_{lag}"
            if candidate not in feature_candidates:
                feature_candidates.append(candidate)
    for source, windows in rolling_config.items():
        for window in windows:
            candidate = f"{source}_rolling_{window}"
            if candidate not in feature_candidates:
                feature_candidates.append(candidate)

    feature_candidates.extend(team_features)
    feature_candidates.extend(opponent_features)

    feature_columns = feature_candidates

    filtered: Dataset = []
    for record in records:
        if record.get("history_games", 0) < config.min_history_games:
            continue
        target_value = to_float(record.get(config.target))
        if math.isnan(target_value):
            continue
        invalid = False
        for column in feature_columns:
            value = record.get(column)
            if isinstance(value, (int, float)):
                if math.isnan(float(value)):
                    invalid = True
                    break
            elif value is None:
                invalid = True
                break
        if invalid:
            continue
        record[config.target] = target_value
        filtered.append(dict(record))

    return filtered, feature_columns
