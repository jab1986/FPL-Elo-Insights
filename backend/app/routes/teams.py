from fastapi import APIRouter, Query
from typing import List, Optional
from app.models import TeamBase
from app.services.data_service import data_service

router = APIRouter()


@router.get("/teams", response_model=List[TeamBase])
async def get_teams():
    """Get all teams"""
    try:
        return data_service.get_teams()
    except Exception:
        # Return mock data for development
        return [
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


@router.get("/teams/{team_id}", response_model=TeamBase)
async def get_team(team_id: int):
    """Get a specific team by ID"""
    try:
        team = data_service.get_team_by_id(team_id)
        if not team:
            return {"error": "Team not found"}
        return team
    except Exception:
        # Return mock data for development
        return {
            "id": team_id,
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


@router.get("/teams/search", response_model=List[TeamBase])
async def search_teams(q: str = Query(..., description="Search query")):
    """Search teams by name"""
    try:
        return data_service.search_teams(q)
    except Exception:
        # Return mock data for development
        return []