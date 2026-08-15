from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.push_fold_service import push_fold_service

router = APIRouter(prefix="/push-fold")


class PushFoldActionRequest(BaseModel):
    action_type: str  # "fold" | "call" | "all_in"


@router.post("/new-hand")
def new_hand():
    """Start a fresh push-fold hand: human resets to 100bb, agent stack redrawn 5-25bb."""
    return push_fold_service.new_hand()


@router.get("/state")
def get_state():
    """Fetch the current push-fold hand state (e.g. on reconnect/refresh)."""
    state = push_fold_service.get_state()
    if not state.get("active"):
        raise HTTPException(status_code=404, detail="No active push-fold hand.")
    return state


@router.post("/action")
def apply_action(req: PushFoldActionRequest):
    """Apply the human's fold/call/all_in decision, then let the agent respond if needed."""
    try:
        return push_fold_service.apply_hero_action(req.action_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
