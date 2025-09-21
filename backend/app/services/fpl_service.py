"""Service helpers for interacting with the public Fantasy Premier League API."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx


def _convert_currency(value: Optional[int]) -> Optional[float]:
    """Convert FPL currency values that are stored as tenths into floats."""

    if value is None:
        return None
    try:
        return round(float(value) / 10.0, 1)
    except (TypeError, ValueError):
        return None


class FPLService:
    """Utility class for retrieving Fantasy Premier League entry data."""

    BASE_URL = "https://fantasy.premierleague.com/api"
    USER_AGENT = (
        "Mozilla/5.0 (compatible; FPLInsightsBot/1.0; +https://github.com/)"
    )

    def __init__(self) -> None:
        self._bootstrap_cache: Optional[Dict[str, Any]] = None
        self._bootstrap_cache_expiry: Optional[datetime] = None

        # Pre-computed example payload which can be returned when the live
        # service is unreachable. The request from the user specifically
        # references the entry id ``266343`` so the sample mirrors that id.
        self._sample_payload: Dict[str, Any] = {
            "team": {
                "id": 266343,
                "name": "Insights XI",
                "player_first_name": "Alex",
                "player_last_name": "Johnson",
                "player_region_name": "England",
                "summary_overall_points": 412,
                "summary_overall_rank": 158243,
                "summary_event_points": 82,
                "summary_event_rank": 85231,
                "summary_event_transfers": 1,
                "summary_event_transfers_cost": 4,
                "current_event": 5,
                "total_transfers": 12,
                "team_value": 101.7,
                "bank": 1.8,
                "favourite_team": 12,
                "favourite_team_name": "Liverpool",
                "joined_time": "2025-08-10T10:15:00Z",
            },
            "current_event": 5,
            "current_event_summary": {
                "event": 5,
                "points": 82,
                "total_points": 412,
                "rank": 85231,
                "overall_rank": 158243,
                "event_transfers": 1,
                "event_transfers_cost": 4,
                "bank": 1.8,
                "value": 101.7,
                "points_on_bench": 7,
            },
            "chips": [
                {
                    "name": "wildcard",
                    "status_for_entry": "played",
                    "played_by_entry": True,
                    "event": 3,
                },
                {
                    "name": "bench_boost",
                    "status_for_entry": "available",
                    "played_by_entry": False,
                    "event": None,
                },
            ],
            "picks": [
                {
                    "element": 207,
                    "position": 1,
                    "multiplier": 1,
                    "is_captain": False,
                    "is_vice_captain": False,
                    "player": {
                        "id": 207,
                        "web_name": "Areola",
                        "first_name": "Alphonse",
                        "second_name": "Areola",
                        "team": "West Ham United",
                        "team_short_name": "WHU",
                        "position": "GK",
                        "now_cost": 4.1,
                        "total_points": 135,
                        "selected_by_percent": 27.4,
                    },
                },
                {
                    "element": 180,
                    "position": 2,
                    "multiplier": 1,
                    "is_captain": False,
                    "is_vice_captain": False,
                    "player": {
                        "id": 180,
                        "web_name": "Alexander-Arnold",
                        "first_name": "Trent",
                        "second_name": "Alexander-Arnold",
                        "team": "Liverpool",
                        "team_short_name": "LIV",
                        "position": "DEF",
                        "now_cost": 7.5,
                        "total_points": 178,
                        "selected_by_percent": 28.9,
                    },
                },
                {
                    "element": 27,
                    "position": 3,
                    "multiplier": 1,
                    "is_captain": False,
                    "is_vice_captain": False,
                    "player": {
                        "id": 27,
                        "web_name": "Gabriel",
                        "first_name": "Gabriel",
                        "second_name": "Magalhães",
                        "team": "Arsenal",
                        "team_short_name": "ARS",
                        "position": "DEF",
                        "now_cost": 5.1,
                        "total_points": 152,
                        "selected_by_percent": 19.4,
                    },
                },
                {
                    "element": 270,
                    "position": 4,
                    "multiplier": 1,
                    "is_captain": False,
                    "is_vice_captain": False,
                    "player": {
                        "id": 270,
                        "web_name": "Estupiñán",
                        "first_name": "Pervis",
                        "second_name": "Estupiñán",
                        "team": "Brighton",
                        "team_short_name": "BHA",
                        "position": "DEF",
                        "now_cost": 5.2,
                        "total_points": 133,
                        "selected_by_percent": 15.2,
                    },
                },
                {
                    "element": 340,
                    "position": 5,
                    "multiplier": 1,
                    "is_captain": False,
                    "is_vice_captain": False,
                    "player": {
                        "id": 340,
                        "web_name": "Porro",
                        "first_name": "Pedro",
                        "second_name": "Porro",
                        "team": "Tottenham Hotspur",
                        "team_short_name": "TOT",
                        "position": "DEF",
                        "now_cost": 5.9,
                        "total_points": 164,
                        "selected_by_percent": 21.3,
                    },
                },
                {
                    "element": 13,
                    "position": 6,
                    "multiplier": 2,
                    "is_captain": True,
                    "is_vice_captain": False,
                    "player": {
                        "id": 13,
                        "web_name": "Salah",
                        "first_name": "Mohamed",
                        "second_name": "Salah",
                        "team": "Liverpool",
                        "team_short_name": "LIV",
                        "position": "MID",
                        "now_cost": 12.6,
                        "total_points": 228,
                        "selected_by_percent": 54.2,
                    },
                },
                {
                    "element": 36,
                    "position": 7,
                    "multiplier": 1,
                    "is_captain": False,
                    "is_vice_captain": True,
                    "player": {
                        "id": 36,
                        "web_name": "Saka",
                        "first_name": "Bukayo",
                        "second_name": "Saka",
                        "team": "Arsenal",
                        "team_short_name": "ARS",
                        "position": "MID",
                        "now_cost": 9.4,
                        "total_points": 214,
                        "selected_by_percent": 46.8,
                    },
                },
                {
                    "element": 420,
                    "position": 8,
                    "multiplier": 1,
                    "is_captain": False,
                    "is_vice_captain": False,
                    "player": {
                        "id": 420,
                        "web_name": "Son",
                        "first_name": "Heung-Min",
                        "second_name": "Son",
                        "team": "Tottenham Hotspur",
                        "team_short_name": "TOT",
                        "position": "MID",
                        "now_cost": 9.6,
                        "total_points": 216,
                        "selected_by_percent": 32.7,
                    },
                },
                {
                    "element": 640,
                    "position": 9,
                    "multiplier": 1,
                    "is_captain": False,
                    "is_vice_captain": False,
                    "player": {
                        "id": 640,
                        "web_name": "Palmer",
                        "first_name": "Cole",
                        "second_name": "Palmer",
                        "team": "Chelsea",
                        "team_short_name": "CHE",
                        "position": "MID",
                        "now_cost": 6.1,
                        "total_points": 196,
                        "selected_by_percent": 42.1,
                    },
                },
                {
                    "element": 47,
                    "position": 10,
                    "multiplier": 1,
                    "is_captain": False,
                    "is_vice_captain": False,
                    "player": {
                        "id": 47,
                        "web_name": "Haaland",
                        "first_name": "Erling",
                        "second_name": "Haaland",
                        "team": "Manchester City",
                        "team_short_name": "MCI",
                        "position": "FWD",
                        "now_cost": 14.3,
                        "total_points": 264,
                        "selected_by_percent": 88.5,
                    },
                },
                {
                    "element": 583,
                    "position": 11,
                    "multiplier": 1,
                    "is_captain": False,
                    "is_vice_captain": False,
                    "player": {
                        "id": 583,
                        "web_name": "Watkins",
                        "first_name": "Ollie",
                        "second_name": "Watkins",
                        "team": "Aston Villa",
                        "team_short_name": "AVL",
                        "position": "FWD",
                        "now_cost": 8.4,
                        "total_points": 208,
                        "selected_by_percent": 41.2,
                    },
                },
            ],
            "bench": [
                {
                    "element": 1,
                    "position": 12,
                    "multiplier": 0,
                    "is_captain": False,
                    "is_vice_captain": False,
                    "player": {
                        "id": 1,
                        "web_name": "Turner",
                        "first_name": "Matt",
                        "second_name": "Turner",
                        "team": "Nottingham Forest",
                        "team_short_name": "NFO",
                        "position": "GK",
                        "now_cost": 4.0,
                        "total_points": 76,
                        "selected_by_percent": 5.4,
                    },
                },
                {
                    "element": 437,
                    "position": 13,
                    "multiplier": 0,
                    "is_captain": False,
                    "is_vice_captain": False,
                    "player": {
                        "id": 437,
                        "web_name": "Udogie",
                        "first_name": "Destiny",
                        "second_name": "Udogie",
                        "team": "Tottenham Hotspur",
                        "team_short_name": "TOT",
                        "position": "DEF",
                        "now_cost": 4.9,
                        "total_points": 118,
                        "selected_by_percent": 12.3,
                    },
                },
                {
                    "element": 464,
                    "position": 14,
                    "multiplier": 0,
                    "is_captain": False,
                    "is_vice_captain": False,
                    "player": {
                        "id": 464,
                        "web_name": "Andreas",
                        "first_name": "Andreas",
                        "second_name": "Pereira",
                        "team": "Fulham",
                        "team_short_name": "FUL",
                        "position": "MID",
                        "now_cost": 5.5,
                        "total_points": 112,
                        "selected_by_percent": 6.1,
                    },
                },
                {
                    "element": 427,
                    "position": 15,
                    "multiplier": 0,
                    "is_captain": False,
                    "is_vice_captain": False,
                    "player": {
                        "id": 427,
                        "web_name": "Archer",
                        "first_name": "Cameron",
                        "second_name": "Archer",
                        "team": "Sheffield United",
                        "team_short_name": "SHU",
                        "position": "FWD",
                        "now_cost": 4.5,
                        "total_points": 84,
                        "selected_by_percent": 9.8,
                    },
                },
            ],
            "history": {
                "current": [
                    {
                        "event": 5,
                        "points": 82,
                        "total_points": 412,
                        "rank": 85231,
                        "overall_rank": 158243,
                        "event_transfers": 1,
                        "event_transfers_cost": 4,
                        "bank": 1.8,
                        "value": 101.7,
                        "points_on_bench": 7,
                    },
                    {
                        "event": 4,
                        "points": 79,
                        "total_points": 330,
                        "rank": 123114,
                        "overall_rank": 178452,
                        "event_transfers": 2,
                        "event_transfers_cost": 4,
                        "bank": 1.3,
                        "value": 101.2,
                        "points_on_bench": 5,
                    },
                    {
                        "event": 3,
                        "points": 68,
                        "total_points": 251,
                        "rank": 301223,
                        "overall_rank": 232540,
                        "event_transfers": 3,
                        "event_transfers_cost": 8,
                        "bank": 0.4,
                        "value": 100.3,
                        "points_on_bench": 4,
                    },
                    {
                        "event": 2,
                        "points": 58,
                        "total_points": 183,
                        "rank": 612334,
                        "overall_rank": 342117,
                        "event_transfers": 1,
                        "event_transfers_cost": 0,
                        "bank": 0.9,
                        "value": 100.7,
                        "points_on_bench": 6,
                    },
                    {
                        "event": 1,
                        "points": 101,
                        "total_points": 125,
                        "rank": 10234,
                        "overall_rank": 10234,
                        "event_transfers": 0,
                        "event_transfers_cost": 0,
                        "bank": 0.5,
                        "value": 100.0,
                        "points_on_bench": 3,
                    },
                ],
                "past": [
                    {
                        "season_name": "2024/25",
                        "total_points": 2398,
                        "rank": 48231,
                    },
                    {
                        "season_name": "2023/24",
                        "total_points": 2312,
                        "rank": 105234,
                    },
                ],
            },
            "source": "sample",
        }

    async def _get_json(self, endpoint: str) -> Dict[str, Any]:
        """Fetch a JSON document from the public FPL API."""

        headers = {
            "User-Agent": self.USER_AGENT,
            "Referer": "https://fantasy.premierleague.com/",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{self.BASE_URL}{endpoint}", headers=headers)
            response.raise_for_status()
            return response.json()

    async def _get_bootstrap(self) -> Dict[str, Any]:
        """Retrieve and cache the bootstrap-static dataset."""

        now = datetime.now(timezone.utc)
        if (
            self._bootstrap_cache is not None
            and self._bootstrap_cache_expiry is not None
            and now < self._bootstrap_cache_expiry
        ):
            return self._bootstrap_cache

        data = await self._get_json("/bootstrap-static/")
        # Cache for one hour to avoid repeated network calls
        self._bootstrap_cache = data
        self._bootstrap_cache_expiry = now + timedelta(hours=1)
        return data

    @staticmethod
    def _format_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise the entry payload into a UI friendly structure."""

        formatted = {
            "id": entry.get("id"),
            "name": entry.get("name"),
            "player_first_name": entry.get("player_first_name"),
            "player_last_name": entry.get("player_last_name"),
            "player_region_name": entry.get("player_region_name"),
            "summary_overall_points": entry.get("summary_overall_points"),
            "summary_overall_rank": entry.get("summary_overall_rank"),
            "summary_event_points": entry.get("summary_event_points"),
            "summary_event_rank": entry.get("summary_event_rank"),
            "summary_event_transfers": entry.get("summary_event_transfers"),
            "summary_event_transfers_cost": entry.get("summary_event_transfers_cost"),
            "current_event": entry.get("current_event"),
            "total_transfers": entry.get("summary_total_transfers")
            or entry.get("summary_overall_transfers"),
            "team_value": _convert_currency(entry.get("value")),
            "bank": _convert_currency(entry.get("bank")),
            "favourite_team": entry.get("favourite_team"),
            "favourite_team_name": entry.get("favourite_team_name"),
            "joined_time": entry.get("joined_time"),
        }

        # Attempt to look up the fan's favourite club name from the leagues data
        leagues = entry.get("leagues") or {}
        classic_leagues = leagues.get("classic") if isinstance(leagues, dict) else None
        if formatted.get("favourite_team") and classic_leagues:
            try:
                favourite = next(
                    (
                        league
                        for league in classic_leagues
                        if league.get("league_type") == "favourite"
                    ),
                    None,
                )
                if favourite and favourite.get("name"):
                    formatted["favourite_team_name"] = favourite.get("name")
            except (TypeError, StopIteration):
                pass

        return formatted

    @staticmethod
    def _format_pick(
        pick: Dict[str, Any],
        players_by_id: Dict[int, Dict[str, Any]],
        teams_by_id: Dict[int, Dict[str, Any]],
        positions_by_id: Dict[int, str],
    ) -> Dict[str, Any]:
        """Merge pick data with player metadata from bootstrap-static."""

        element_id = pick.get("element")
        player_info = players_by_id.get(element_id, {})
        team_info = teams_by_id.get(player_info.get("team")) if player_info else None

        player_payload = {
            "id": player_info.get("id"),
            "web_name": player_info.get("web_name"),
            "first_name": player_info.get("first_name"),
            "second_name": player_info.get("second_name"),
            "team": team_info.get("name") if team_info else None,
            "team_short_name": team_info.get("short_name") if team_info else None,
            "position": positions_by_id.get(player_info.get("element_type")),
            "now_cost": _convert_currency(player_info.get("now_cost")),
            "total_points": player_info.get("total_points"),
            "selected_by_percent": (
                float(player_info.get("selected_by_percent", 0.0))
                if player_info.get("selected_by_percent") is not None
                else None
            ),
            "event_points": player_info.get("event_points"),
            "status": player_info.get("status"),
        }

        return {
            "element": element_id,
            "position": pick.get("position"),
            "multiplier": pick.get("multiplier"),
            "is_captain": pick.get("is_captain", False),
            "is_vice_captain": pick.get("is_vice_captain", False),
            "player": player_payload,
        }

    @staticmethod
    def _format_history(history: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise the history payload by converting currency fields."""

        if not history:
            return {"current": [], "past": []}

        current_entries = []
        for event in history.get("current", []):
            current_entries.append(
                {
                    "event": event.get("event"),
                    "points": event.get("points"),
                    "total_points": event.get("total_points"),
                    "rank": event.get("rank"),
                    "overall_rank": event.get("overall_rank"),
                    "event_transfers": event.get("event_transfers"),
                    "event_transfers_cost": event.get("event_transfers_cost"),
                    "bank": _convert_currency(event.get("bank")),
                    "value": _convert_currency(event.get("value")),
                    "points_on_bench": event.get("points_on_bench"),
                }
            )

        past_entries = []
        for season in history.get("past", []):
            past_entries.append(
                {
                    "season_name": season.get("season_name"),
                    "total_points": season.get("total_points"),
                    "rank": season.get("rank"),
                }
            )

        return {"current": current_entries, "past": past_entries}

    async def get_user_team(
        self, team_id: int, event: Optional[int] = None
    ) -> Dict[str, Any]:
        """Retrieve a Fantasy Premier League entry with enriched pick details."""

        entry_data = await self._get_json(f"/entry/{team_id}/")
        history_data = await self._get_json(f"/entry/{team_id}/history/")

        current_event = event or entry_data.get("current_event")
        if current_event is None:
            current = history_data.get("current") or []
            if current:
                current_event = current[-1].get("event")

        if current_event is None:
            raise ValueError("Unable to determine the current event for the entry")

        picks_data = await self._get_json(
            f"/entry/{team_id}/event/{current_event}/picks/"
        )

        bootstrap = await self._get_bootstrap()
        players_by_id = {
            element.get("id"): element for element in bootstrap.get("elements", [])
        }
        teams_by_id = {
            team.get("id"): team for team in bootstrap.get("teams", [])
        }
        positions_by_id = {
            element_type.get("id"): element_type.get("singular_name_short")
            for element_type in bootstrap.get("element_types", [])
        }

        formatted_picks = [
            self._format_pick(pick, players_by_id, teams_by_id, positions_by_id)
            for pick in picks_data.get("picks", [])
        ]

        # Separate starting XI and bench
        starting = [pick for pick in formatted_picks if (pick.get("position") or 0) <= 11]
        bench = [pick for pick in formatted_picks if (pick.get("position") or 0) > 11]

        response = {
            "team": self._format_entry(entry_data),
            "current_event": current_event,
            "current_event_summary": {
                "event": current_event,
                "points": picks_data.get("entry_history", {}).get("points"),
                "total_points": picks_data.get("entry_history", {}).get("total_points"),
                "rank": picks_data.get("entry_history", {}).get("rank"),
                "overall_rank": picks_data.get("entry_history", {}).get("overall_rank"),
                "event_transfers": picks_data.get("entry_history", {}).get("event_transfers"),
                "event_transfers_cost": picks_data.get("entry_history", {}).get(
                    "event_transfers_cost"
                ),
                "bank": _convert_currency(picks_data.get("entry_history", {}).get("bank")),
                "value": _convert_currency(
                    picks_data.get("entry_history", {}).get("value")
                ),
                "points_on_bench": picks_data.get("entry_history", {}).get(
                    "points_on_bench"
                ),
            },
            "chips": picks_data.get("chips", []),
            "picks": starting,
            "bench": bench,
            "history": self._format_history(history_data),
            "source": "live",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        return response

    def get_sample_user_team(self) -> Dict[str, Any]:
        """Return an example payload for offline development."""

        sample = copy.deepcopy(self._sample_payload)
        sample["fetched_at"] = datetime.now(timezone.utc).isoformat()
        return sample


# Global service instance used by the API routers
fpl_service = FPLService()

