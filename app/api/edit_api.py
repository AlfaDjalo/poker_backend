"""
edit_api.py — Hand Editor endpoints.

POST /game/edit/begin   — enter editing mode, clear current hand DB records
POST /game/edit/apply   — apply edited snapshot, resume live game
POST /game/edit/load    — load arbitrary snapshot (Replayer → Game Simulator)
POST /game/edit/cancel  — cancel editing, restore pre-edit snapshot
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.services.game_service import game_service

router = APIRouter(prefix="/game/edit")


# ── Request DTOs ──────────────────────────────────────────────────

class PlayerEditInput(BaseModel):
    seat: int                               # 1-based
    stack: int
    current_bet: int
    total_contribution: int
    has_folded: bool
    is_all_in: bool
    hole_cards: List[Optional[str]]


class EditStateRequest(BaseModel):
    game_name: str
    street_index: int
    pot: int
    dealer_position: int                    # 0-based
    current_player: int                     # 0-based engine index
    bet_to_call: int
    min_raise: int
    players: List[PlayerEditInput]
    node_cards: List[Optional[str]]         # indexed by node position
    discard_pile: List[str] = []


# ── Endpoints ─────────────────────────────────────────────────────

@router.post("/begin")
def begin_edit(db: Session = Depends(get_db)):
    """
    Enter editing mode.
    Snapshots the current in-memory state, clears DB records for the
    current hand, and suppresses future logging until edit is resolved.
    """
    try:
        game_service.begin_edit(db)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "editing"}


@router.post("/apply")
def apply_edit(req: EditStateRequest, db: Session = Depends(get_db)):
    """
    Apply the edited state to the live game and resume.
    Returns the new GameStateDTO or a 422 with validation errors.
    """
    try:
        dto = game_service.apply_edit(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return dto


@router.post("/load")
def load_edit(req: EditStateRequest):
    """
    Load an arbitrary snapshot from the Hand Replayer.
    Reconstructs PokerState mid-hand so the Game Simulator can resume.
    Does NOT log the resulting hand to the database.
    """    
    try:
        dto = game_service.load_edit(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return dto


@router.post("/cancel")
def cancel_edit():
    """
    Cancel editing. Restores the in-memory snapshot taken when /begin was called.
    The current hand is not saved regardless (DB records were cleared on /begin).
    """
    try:
        dto = game_service.cancel_edit()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return dto