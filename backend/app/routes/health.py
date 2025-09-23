from fastapi import APIRouter
from typing import Any, Dict

from app.services.data_service import data_service

router = APIRouter()


@router.get("/health/data")
async def data_health() -> Dict[str, Any]:
    """Report whether data source is configured and reachable.

    Returns example:
    {
      "source": "supabase" | "none",
      "configured": true/false,
      "reachable": true/false,
      "url": "https://..." | None
    }
    """
    configured = bool(getattr(data_service, "supabase_url", None) and getattr(data_service, "supabase_key", None))
    url = getattr(data_service, "supabase_url", None)

    # Default response when not configured
    status = {
        "source": "supabase" if configured else "none",
        "configured": configured,
        "reachable": False,
        "url": url,
    }

    if not configured or not getattr(data_service, "supabase", None):
        return status

    # Light probe: small select limited to 1 row
    try:
        _ = data_service.supabase.table("players").select("id").limit(1).execute()
        status["reachable"] = True
    except Exception:
        status["reachable"] = False

    return status
