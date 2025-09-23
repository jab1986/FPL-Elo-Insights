import os
from pathlib import Path
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv, find_dotenv


class DataService:
    def __init__(self):
        # Load environment variables from the project root, robustly across processes
        # 1) Try explicit path: repo_root/.env (services -> app -> backend -> repo root)
        explicit_env_path = Path(__file__).resolve().parents[3] / ".env"
        # 2) Fallback to auto-discovery from current working dir
        discovered_env_path = find_dotenv(usecwd=True)

        loaded_any = False
        if explicit_env_path.exists():
            load_dotenv(dotenv_path=str(explicit_env_path), override=False)
            loaded_any = True
        if discovered_env_path and os.path.exists(discovered_env_path):
            load_dotenv(dotenv_path=discovered_env_path, override=False)
            loaded_any = True

        # Debug: show how .env was resolved (prints once on service init)
        try:
            print(
                "[DataService] .env debug => explicit:", str(explicit_env_path),
                "exists=", explicit_env_path.exists(), " discovered:", discovered_env_path,
                "exists=", os.path.exists(discovered_env_path) if discovered_env_path else False,
                " loaded_any=", loaded_any,
            )
        except Exception:
            # Avoid any noise if printing fails
            pass

        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")

        if not self.supabase_url or not self.supabase_key:
            print(
                "Warning: SUPABASE_URL and SUPABASE_KEY not found. "
                "Supabase client is not configured; data access will fail until set."
            )
            self.supabase = None
        else:
            print(f"✅ Supabase credentials found. Connecting to: {self.supabase_url}")
            self.supabase: Client = create_client(
                self.supabase_url, self.supabase_key
            )

    def _fetch_table_data(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch data from a Supabase table with optional filters"""
        if not self.supabase:
            raise RuntimeError("Supabase client is not configured")

        query = self.supabase.table(table_name).select("*")

        if filters:
            for key, value in filters.items():
                if value is not None:
                    query = query.eq(key, value)

        response = query.execute()
        return response.data

    def get_players(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get all players with optional filters"""
        filters = filters or {}

        players = self._fetch_table_data("players")

        if not filters:
            return players

        def _to_number(value: Any) -> Optional[float]:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return None
                value = value.replace(",", "")
                try:
                    return float(value)
                except ValueError:
                    return None
            return None

        def _matches_team(player: Dict[str, Any], team_value: Any) -> bool:
            if team_value in (None, ""):
                return True
            team_value = str(team_value).lower()
            potential_keys = (
                "team",
                "team_name",
                "team_short_name",
                "team_code",
                "team_id",
            )
            for key in potential_keys:
                candidate = player.get(key)
                if candidate is None:
                    continue
                if isinstance(candidate, str):
                    if team_value in candidate.lower():
                        return True
                else:
                    if str(candidate).lower() == team_value:
                        return True
            return False

        def _matches_position(player: Dict[str, Any], position_value: Any) -> bool:
            if position_value in (None, ""):
                return True
            position_value = str(position_value).lower()
            potential_keys = (
                "position",
                "position_short",
                "element_type",
                "element_type_name",
            )
            for key in potential_keys:
                candidate = player.get(key)
                if not candidate:
                    continue
                if str(candidate).lower() == position_value:
                    return True
            return False

        def _matches_gameweek(player: Dict[str, Any], gameweek_value: Optional[int]) -> bool:
            if gameweek_value is None:
                return True
            potential_keys = ("gameweek", "gw", "event")
            for key in potential_keys:
                candidate = player.get(key)
                if candidate is None:
                    continue
                try:
                    if int(candidate) == int(gameweek_value):
                        return True
                except (TypeError, ValueError):
                    if str(candidate).lower() == str(gameweek_value).lower():
                        return True
            return False

        min_price = _to_number(filters.get("min_price"))
        max_price = _to_number(filters.get("max_price"))
        min_points = _to_number(filters.get("min_points"))
        max_points = _to_number(filters.get("max_points"))
        team_filter = filters.get("team")
        position_filter = filters.get("position")
        gameweek_filter = filters.get("gameweek")

        filtered_players = []
        for player in players:
            if not _matches_position(player, position_filter):
                continue
            if not _matches_team(player, team_filter):
                continue
            if not _matches_gameweek(player, gameweek_filter):
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

            filtered_players.append(player)

        return filtered_players

    def get_player_by_id(self, player_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific player by ID"""
        players = self._fetch_table_data("players", {"id": player_id})
        return players[0] if players else None

    def get_teams(self) -> List[Dict[str, Any]]:
        """Get all teams"""
        return self._fetch_table_data("teams")

    def get_team_by_id(self, team_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific team by ID"""
        teams = self._fetch_table_data("teams", {"id": team_id})
        return teams[0] if teams else None

    def get_matches(
        self,
        gameweek: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get matches, optionally filtered by gameweek"""
        filters = {"gameweek": gameweek} if gameweek is not None else None
        return self._fetch_table_data("matches", filters)

    def get_match_by_id(self, match_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific match by ID"""
        matches = self._fetch_table_data("matches", {"id": match_id})
        return matches[0] if matches else None

    def get_player_match_stats(
        self,
        match_id: Optional[int] = None,
        player_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get player match statistics with optional filters"""
        filters = {}
        if match_id is not None:
            filters["match_id"] = match_id
        if player_id is not None:
            filters["player_id"] = player_id

        return self._fetch_table_data("playermatchstats", filters)

    def get_gameweek_summaries(self) -> List[Dict[str, Any]]:
        """Get all gameweek summaries"""
        return self._fetch_table_data("gameweek_summaries")

    def get_current_gameweek(self) -> Optional[Dict[str, Any]]:
        """Get the current active gameweek"""
        # Get the latest unfinished gameweek or the most recent finished one
        summaries = self.get_gameweek_summaries()
        if not summaries:
            return None

        # Sort by gameweek descending
        summaries.sort(key=lambda x: x["gameweek"], reverse=True)

        # Find the first unfinished gameweek, or return the latest
        for summary in summaries:
            if not summary.get("finished", False):
                return summary

        return summaries[0] if summaries else None

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics"""
        players = self.get_players()
        teams = self.get_teams()
        current_gw = self.get_current_gameweek()

        # Calculate stats
        total_players = len(players)
        total_teams = len(teams)
        current_gameweek = current_gw["gameweek"] if current_gw else 0

        if players:
            total_points = sum(p.get("total_points", 0) for p in players)
            average_points = (
                total_points / len(players) if total_points > 0 else 0
            )

            # Find top scorer
            top_scorer = max(
                players,
                key=lambda p: p.get("total_points", 0),
                default=None
            )

            most_valuable = max(
                (p for p in players if p.get("now_cost", 0) > 0),
                key=lambda p: (
                    p.get("total_points", 0) / (p.get("now_cost", 0) / 10)
                ),
                default=None
            )
        else:
            average_points = 0
            top_scorer = None
            most_valuable = None

        return {
            "totalPlayers": total_players,
            "totalTeams": total_teams,
            "currentGameweek": current_gameweek,
            "averagePoints": average_points,
            "topScorer": top_scorer,
            "mostValuable": most_valuable,
        }

    def get_top_players(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing players"""
        players = self.get_players()
        sorted_players = sorted(
            players,
            key=lambda p: p.get("total_points", 0),
            reverse=True
        )
        return sorted_players[:limit]

    def search_players(self, query: str) -> List[Dict[str, Any]]:
        """Search players by name"""
        players = self.get_players()
        query_lower = query.lower()
        return [
            p for p in players
            if query_lower in p.get("name", "").lower()
        ]

    def search_teams(self, query: str) -> List[Dict[str, Any]]:
        """Search teams by name"""
        teams = self.get_teams()
        query_lower = query.lower()
        return [
            t for t in teams
            if query_lower in t.get("name", "").lower()
        ]


# Global instance
data_service = DataService()
