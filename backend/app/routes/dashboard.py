from fastapi import APIRouter

from app.models import DashboardStats
from app.services import mock_data
from app.services.data_service import data_service

router = APIRouter()


@router.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        return data_service.get_dashboard_stats()
    except Exception:
        # Return mock data for development
        return mock_data.sample_dashboard_stats()
