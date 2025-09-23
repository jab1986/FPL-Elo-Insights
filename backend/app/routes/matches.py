from fastapi import APIRouter, Query, HTTPException
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
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Match data unavailable (Supabase not configured or unreachable)") from exc


@router.get("/matches/{match_id}", response_model=MatchBase)
async def get_match(match_id: int):
    """Get a specific match by ID"""
    try:
        match = data_service.get_match_by_id(match_id)
        if not match:
            raise HTTPException(status_code=404, detail="Match not found")
        return match
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Match data unavailable (Supabase not configured or unreachable)") from exc
