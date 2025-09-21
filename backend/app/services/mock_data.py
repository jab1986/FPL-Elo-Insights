"""Offline-friendly sample data for the FPL Insights services."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

_PLAYERS_LIST: List[Dict[str, Any]] = [
    {
        "id": 1,
        "name": "Mohamed Salah",
        "team": "Liverpool",
        "position": "Midfielder",
        "now_cost": 130,
        "selected_by_percent": 25.5,
        "total_points": 180,
        "minutes": 2160,
        "goals_scored": 15,
        "assists": 8,
        "clean_sheets": 12,
        "goals_conceded": 25,
        "own_goals": 0,
        "penalties_saved": 0,
        "penalties_missed": 1,
        "yellow_cards": 3,
        "red_cards": 0,
        "saves": 0,
        "bonus": 20,
        "bps": 650,
        "influence": 850.5,
        "creativity": 720.3,
        "threat": 950.2,
        "ict_index": 252.0,
        "starts": 24,
        "expected_goals": 12.5,
        "expected_assists": 10.2,
        "expected_goal_involvements": 22.7,
        "expected_goals_conceded": 25.0,
        "value_season": 1.38,
        "value_form": 0.8,
        "points_per_game": 7.5,
        "transfers_in": 1_500_000,
        "transfers_out": 800_000,
        "transfers_in_event": 25_000,
        "transfers_out_event": 15_000,
    }
]

_PLAYER_DETAIL_TEMPLATE: Dict[str, Any] = {
    "id": 17,
    "name": "Kevin De Bruyne",
    "team": "Manchester City",
    "position": "Midfielder",
    "now_cost": 120,
    "selected_by_percent": 30.2,
    "total_points": 195,
    "minutes": 2340,
    "goals_scored": 8,
    "assists": 18,
    "clean_sheets": 15,
    "goals_conceded": 20,
    "own_goals": 0,
    "penalties_saved": 0,
    "penalties_missed": 0,
    "yellow_cards": 2,
    "red_cards": 0,
    "saves": 0,
    "bonus": 25,
    "bps": 720,
    "influence": 920.5,
    "creativity": 1100.3,
    "threat": 780.2,
    "ict_index": 280.0,
    "starts": 26,
    "expected_goals": 8.5,
    "expected_assists": 15.2,
    "expected_goal_involvements": 23.7,
    "expected_goals_conceded": 20.0,
    "value_season": 1.63,
    "value_form": 0.9,
    "points_per_game": 7.5,
    "transfers_in": 1_800_000,
    "transfers_out": 600_000,
    "transfers_in_event": 30_000,
    "transfers_out_event": 10_000,
}

_TOP_PLAYERS: List[Dict[str, Any]] = [
    {
        "id": 47,
        "name": "Erling Haaland",
        "team": "Manchester City",
        "position": "Forward",
        "now_cost": 143,
        "selected_by_percent": 88.5,
        "total_points": 264,
        "minutes": 1980,
        "goals_scored": 25,
        "assists": 5,
        "clean_sheets": 10,
        "goals_conceded": 30,
        "own_goals": 0,
        "penalties_saved": 0,
        "penalties_missed": 2,
        "yellow_cards": 4,
        "red_cards": 0,
        "saves": 0,
        "bonus": 30,
        "bps": 800,
        "influence": 950.5,
        "creativity": 450.3,
        "threat": 1200.2,
        "ict_index": 260.0,
        "starts": 22,
        "expected_goals": 22.5,
        "expected_assists": 6.2,
        "expected_goal_involvements": 28.7,
        "expected_goals_conceded": 30.0,
        "value_season": 1.83,
        "value_form": 1.2,
        "points_per_game": 10.0,
        "transfers_in": 2_500_000,
        "transfers_out": 400_000,
        "transfers_in_event": 50_000,
        "transfers_out_event": 8_000,
    }
]

_TEAMS_LIST: List[Dict[str, Any]] = [
    {
        "id": 1,
        "name": "Manchester City",
        "short_name": "MCI",
        "strength": 5,
        "played": 25,
        "win": 18,
        "draw": 4,
        "loss": 3,
        "points": 58,
        "position": 1,
        "form": "WWWDW",
        "strength_overall_home": 1350,
        "strength_overall_away": 1380,
        "strength_attack_home": 1350,
        "strength_attack_away": 1380,
        "strength_defence_home": 1350,
        "strength_defence_away": 1380,
    }
]

_TEAM_DETAIL_TEMPLATE: Dict[str, Any] = {
    "id": 2,
    "name": "Liverpool",
    "short_name": "LIV",
    "strength": 5,
    "played": 25,
    "win": 16,
    "draw": 6,
    "loss": 3,
    "points": 54,
    "position": 2,
    "form": "WDWWW",
    "strength_overall_home": 1320,
    "strength_overall_away": 1350,
    "strength_attack_home": 1320,
    "strength_attack_away": 1350,
    "strength_defence_home": 1320,
    "strength_defence_away": 1350,
}

_MATCHES_LIST: List[Dict[str, Any]] = [
    {
        "id": 1,
        "gameweek": 25,
        "home_team": "Manchester City",
        "away_team": "Liverpool",
        "home_score": 3,
        "away_score": 1,
        "kickoff_time": "2024-02-25T17:30:00Z",
        "finished": True,
        "minutes": 90,
        "provisional_start_time": False,
        "finished_provisional": True,
        "started": True,
    }
]

_MATCH_DETAIL_TEMPLATE: Dict[str, Any] = {
    "id": 2,
    "gameweek": 25,
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "home_score": 2,
    "away_score": 2,
    "kickoff_time": "2024-02-24T15:00:00Z",
    "finished": True,
    "minutes": 90,
    "provisional_start_time": False,
    "finished_provisional": True,
    "started": True,
}

_PLAYER_MATCH_STATS: List[Dict[str, Any]] = [
    {
        "id": 1001,
        "match_id": 1,
        "player_id": 47,
        "team_id": 1,
        "position": "FWD",
        "minutes": 90,
        "goals_scored": 2,
        "assists": 1,
        "clean_sheets": 0,
        "goals_conceded": 1,
        "own_goals": 0,
        "penalties_saved": 0,
        "penalties_missed": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "saves": 0,
        "bonus": 3,
        "bps": 45,
        "influence": 115.2,
        "creativity": 22.4,
        "threat": 150.3,
        "ict_index": 287.9,
        "total_points": 13,
        "in_dreamteam": True,
    },
    {
        "id": 1002,
        "match_id": 1,
        "player_id": 13,
        "team_id": 2,
        "position": "MID",
        "minutes": 90,
        "goals_scored": 1,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 3,
        "own_goals": 0,
        "penalties_saved": 0,
        "penalties_missed": 0,
        "yellow_cards": 1,
        "red_cards": 0,
        "saves": 0,
        "bonus": 2,
        "bps": 38,
        "influence": 98.4,
        "creativity": 34.1,
        "threat": 120.7,
        "ict_index": 253.2,
        "total_points": 8,
        "in_dreamteam": False,
    },
]

_GAMEWEEK_SUMMARIES: List[Dict[str, Any]] = [
    {
        "gameweek": 5,
        "average_entry_score": 56.2,
        "highest_score": 123,
        "deadline_time": "2024-09-14T10:30:00Z",
        "deadline_time_epoch": 1726300200,
        "finished": False,
        "data_checked": False,
        "highest_scoring_entry": 2_345_678,
        "most_selected": 47,
        "most_transferred_in": 13,
        "top_element": 47,
        "top_element_info": {"id": 47, "web_name": "Haaland", "points": 20},
        "most_captained": 47,
        "most_vice_captained": 17,
    },
    {
        "gameweek": 4,
        "average_entry_score": 48.1,
        "highest_score": 110,
        "deadline_time": "2024-09-01T10:30:00Z",
        "deadline_time_epoch": 1725186600,
        "finished": True,
        "data_checked": True,
        "highest_scoring_entry": 1_987_654,
        "most_selected": 13,
        "most_transferred_in": 47,
        "top_element": 13,
        "top_element_info": {"id": 13, "web_name": "Salah", "points": 18},
        "most_captained": 13,
        "most_vice_captained": 47,
    },
]


def _deepcopy(data: Any) -> Any:
    """Return a defensive copy of the provided sample payload."""

    return copy.deepcopy(data)


def _to_number(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        cleaned = str(value).strip().replace(",", "")
        if not cleaned:
            return None
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def _matches_position(player: Dict[str, Any], position_value: Optional[str]) -> bool:
    if not position_value:
        return True
    position_value = position_value.lower()
    return str(player.get("position", "")).lower() == position_value


def _matches_team(player: Dict[str, Any], team_value: Optional[str]) -> bool:
    if not team_value:
        return True
    team_value = team_value.lower()
    return team_value in str(player.get("team", "")).lower()


def _filter_players(
    players: List[Dict[str, Any]], filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    if not filters:
        return players

    min_price = _to_number(filters.get("min_price"))
    max_price = _to_number(filters.get("max_price"))
    min_points = _to_number(filters.get("min_points"))
    max_points = _to_number(filters.get("max_points"))
    team_filter = filters.get("team")
    position_filter = filters.get("position")

    filtered: List[Dict[str, Any]] = []
    for player in players:
        if not _matches_position(player, position_filter):
            continue
        if not _matches_team(player, team_filter):
            continue

        cost = _to_number(player.get("now_cost"))
        if min_price is not None and (cost is None or cost < min_price):
            continue
        if max_price is not None and (cost is None or cost > max_price):
            continue

        total_points = _to_number(player.get("total_points"))
        if min_points is not None and (total_points is None or total_points < min_points):
            continue
        if max_points is not None and (total_points is None or total_points > max_points):
            continue

        filtered.append(player)

    return filtered

def sample_players(filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return sample player data respecting simple filters."""

    players = [_deepcopy(player) for player in _PLAYERS_LIST]
    return _filter_players(players, filters)


def sample_player(player_id: int) -> Optional[Dict[str, Any]]:
    """Return a sample player payload for the requested identifier."""

    player = _deepcopy(_PLAYER_DETAIL_TEMPLATE)
    player["id"] = player_id
    return player


def sample_top_players(limit: int = 10) -> List[Dict[str, Any]]:
    """Return a list of the highest scoring sample players."""

    players = [_deepcopy(player) for player in _TOP_PLAYERS]
    players.sort(key=lambda p: p.get("total_points", 0), reverse=True)
    return players[:limit]


def _all_players_for_search() -> List[Dict[str, Any]]:
    seen: Dict[int, Dict[str, Any]] = {}
    for collection in (_PLAYERS_LIST, _TOP_PLAYERS, [_PLAYER_DETAIL_TEMPLATE]):
        for player in collection:
            seen[player["id"]] = _deepcopy(player)
    return list(seen.values())


def sample_search_players(query: str) -> List[Dict[str, Any]]:
    """Return players whose names match the supplied query."""

    query_lower = query.lower()
    return [
        player
        for player in _all_players_for_search()
        if query_lower in player.get("name", "").lower()
    ]


def sample_teams() -> List[Dict[str, Any]]:
    """Return sample teams data."""

    return [_deepcopy(team) for team in _TEAMS_LIST]


def sample_team(team_id: int) -> Optional[Dict[str, Any]]:
    """Return a sample team payload for the requested identifier."""

    template = _deepcopy(_TEAM_DETAIL_TEMPLATE)
    template["id"] = team_id
    return template


def sample_search_teams(query: str) -> List[Dict[str, Any]]:
    """Return teams whose name matches the supplied query."""

    query_lower = query.lower()
    teams = _TEAMS_LIST + [_TEAM_DETAIL_TEMPLATE]
    return [
        _deepcopy(team)
        for team in teams
        if query_lower in team.get("name", "").lower()
    ]


def sample_matches(gameweek: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return sample match data filtered by gameweek when supplied."""

    matches = [_deepcopy(match) for match in _MATCHES_LIST]
    if gameweek is None:
        return matches
    return [match for match in matches if match.get("gameweek") == gameweek]


def sample_match(match_id: int) -> Optional[Dict[str, Any]]:
    """Return a sample match payload for the requested identifier."""

    for match in _MATCHES_LIST:
        if match.get("id") == match_id:
            return _deepcopy(match)
    match = _deepcopy(_MATCH_DETAIL_TEMPLATE)
    match["id"] = match_id
    return match


def sample_player_match_stats(
    match_id: Optional[int] = None, player_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Return sample player match statistics with optional filtering."""

    stats = [_deepcopy(stat) for stat in _PLAYER_MATCH_STATS]
    if match_id is not None:
        stats = [stat for stat in stats if stat.get("match_id") == match_id]
    if player_id is not None:
        stats = [stat for stat in stats if stat.get("player_id") == player_id]
    return stats


def sample_gameweek_summaries() -> List[Dict[str, Any]]:
    """Return sample gameweek summary data."""

    return [_deepcopy(summary) for summary in _GAMEWEEK_SUMMARIES]


def sample_current_gameweek() -> Optional[Dict[str, Any]]:
    """Return the latest unfinished gameweek from the sample data."""

    summaries = sample_gameweek_summaries()
    if not summaries:
        return None

    summaries.sort(key=lambda item: item.get("gameweek", 0), reverse=True)
    for summary in summaries:
        if not summary.get("finished", False):
            return summary
    return summaries[0]


def sample_dashboard_stats() -> Dict[str, Any]:
    """Return sample dashboard metrics."""

    return {
        "totalPlayers": 623,
        "totalTeams": 20,
        "currentGameweek": 25,
        "averagePoints": 45.2,
        "topScorer": _deepcopy(_TOP_PLAYERS[0]),
        "mostValuable": _deepcopy(_PLAYERS_LIST[0]),
    }
