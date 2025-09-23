from fastapi import APIRouter, HTTPException

from app.models import DashboardStats
from app.services.data_service import data_service

router = APIRouter()


@router.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        return data_service.get_dashboard_stats()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Dashboard stats unavailable (Supabase not configured or unreachable)") from exc
