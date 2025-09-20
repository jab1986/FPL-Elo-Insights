import os
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime


class DataService:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")

        if not self.supabase_url or not self.supabase_key:
            print(
                "Warning: SUPABASE_URL and SUPABASE_KEY not found. "
                "Using mock data for development."
            )
            self.supabase = None
        else:
            self.supabase: Client = create_client(
                self.supabase_url, self.supabase_key
            )

    def _fetch_table_data(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch data from a Supabase table with optional filters"""
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
        return self._fetch_table_data("players", filters)

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
        filters = {"gameweek": gameweek} if gameweek else None
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
        if match_id:
            filters["match_id"] = match_id
        if player_id:
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
