from fastapi import APIRouter, Query
from typing import List, Optional
from app.models import MatchBase
from app.services.data_service import data_service

router = APIRouter()


@router.get("/matches", response_model=List[MatchBase])
async def get_matches(
    gameweek: Optional[int] = Query(None, description="Filter by gameweek")
):
    """Get matches, optionally filtered by gameweek"""
    try:
        return data_service.get_matches(gameweek)
    except Exception:
        # Return mock data for development
        return [
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


@router.get("/matches/{match_id}", response_model=MatchBase)
async def get_match(match_id: int):
    """Get a specific match by ID"""
    try:
        match = data_service.get_match_by_id(match_id)
        if not match:
            return {"error": "Match not found"}
        return match
    except Exception:
        # Return mock data for development
        return {
            "id": match_id,
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
