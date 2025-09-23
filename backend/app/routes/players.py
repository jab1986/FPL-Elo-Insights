from fastapi import APIRouter, Query, HTTPException
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
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Player data unavailable (Supabase not configured or unreachable)") from exc


@router.get("/players/{player_id}", response_model=PlayerBase)
async def get_player(player_id: int):
    """Get a specific player by ID"""
    try:
        player = data_service.get_player_by_id(player_id)
        if not player:
            raise HTTPException(status_code=404, detail="Player not found")
        return player
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Player data unavailable (Supabase not configured or unreachable)") from exc


@router.get("/players/top/{limit}", response_model=List[PlayerBase])
async def get_top_players(limit: int = 10):
    """Get top performing players"""
    try:
        return data_service.get_top_players(limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Top players unavailable (Supabase not configured or unreachable)") from exc


@router.get("/players/search", response_model=List[PlayerBase])
async def search_players(q: str = Query(..., description="Search query")):
    """Search players by name"""
    try:
        return data_service.search_players(q)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Search unavailable (Supabase not configured or unreachable)") from exc
