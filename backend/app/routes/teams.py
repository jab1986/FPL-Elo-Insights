from fastapi import APIRouter, Query
from typing import List, Optional

from app.models import TeamBase
from app.services import mock_data
from app.services.data_service import data_service

router = APIRouter()


@router.get("/teams", response_model=List[TeamBase])
async def get_teams():
    """Get all teams"""
    try:
        return data_service.get_teams()
    except Exception:
        # Return mock data for development
        return mock_data.sample_teams()


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
        return mock_data.sample_team(team_id)


@router.get("/teams/search", response_model=List[TeamBase])
async def search_teams(q: str = Query(..., description="Search query")):
    """Search teams by name"""
    try:
        return data_service.search_teams(q)
    except Exception:
        # Return mock data for development
        return mock_data.sample_search_teams(q)