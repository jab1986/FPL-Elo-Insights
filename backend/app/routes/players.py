from fastapi import APIRouter, Query
from typing import List, Optional
from app.models import PlayerBase, PlayerFilters
from app.services.data_service import data_service

router = APIRouter()


@router.get("/players", response_model=List[PlayerBase])
async def get_players(
    position: Optional[str] = Query(None, description="Filter by player position"),
    team: Optional[str] = Query(None, description="Filter by team name"),
    min_price: Optional[int] = Query(None, description="Minimum price filter"),
    max_price: Optional[int] = Query(None, description="Maximum price filter"),
    min_points: Optional[int] = Query(None, description="Minimum total points filter"),
    max_points: Optional[int] = Query(None, description="Maximum total points filter"),
    gameweek: Optional[int] = Query(None, description="Filter by gameweek"),
):
    """Get all players with optional filters"""
    filters = {}
    if position:
        filters["position"] = position
    if team:
        filters["team"] = team
    if min_price is not None:
        filters["min_price"] = min_price
    if max_price is not None:
        filters["max_price"] = max_price
    if min_points is not None:
        filters["min_points"] = min_points
    if max_points is not None:
        filters["max_points"] = max_points
    if gameweek is not None:
        filters["gameweek"] = gameweek

    try:
        players = data_service.get_players(filters)
        return players
    except Exception as e:
        # Return mock data for development
        return [
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
                "transfers_in": 1500000,
                "transfers_out": 800000,
                "transfers_in_event": 25000,
                "transfers_out_event": 15000,
            }
        ]


@router.get("/players/{player_id}", response_model=PlayerBase)
async def get_player(player_id: int):
    """Get a specific player by ID"""
    try:
        player = data_service.get_player_by_id(player_id)
        if not player:
            return {"error": "Player not found"}
        return player
    except Exception:
        # Return mock data for development
        return {
            "id": player_id,
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
            "transfers_in": 1800000,
            "transfers_out": 600000,
            "transfers_in_event": 30000,
            "transfers_out_event": 10000,
        }


@router.get("/players/top/{limit}", response_model=List[PlayerBase])
async def get_top_players(limit: int = 10):
    """Get top performing players"""
    try:
        return data_service.get_top_players(limit)
    except Exception:
        # Return mock data for development
        return [
            {
                "id": 1,
                "name": "Erling Haaland",
                "team": "Manchester City",
                "position": "Forward",
                "now_cost": 120,
                "selected_by_percent": 45.2,
                "total_points": 220,
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
                "transfers_in": 2500000,
                "transfers_out": 400000,
                "transfers_in_event": 50000,
                "transfers_out_event": 8000,
            }
        ]


@router.get("/players/search", response_model=List[PlayerBase])
async def search_players(q: str = Query(..., description="Search query")):
    """Search players by name"""
    try:
        return data_service.search_players(q)
    except Exception:
        # Return mock data for development
        return []