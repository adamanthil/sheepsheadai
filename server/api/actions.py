from fastapi import APIRouter

from sheepshead import ACTION_LOOKUP

router = APIRouter()


@router.get("/api/actions")
def get_actions():
    """Return action id to string mapping for the UI."""
    return {"action_lookup": ACTION_LOOKUP}
