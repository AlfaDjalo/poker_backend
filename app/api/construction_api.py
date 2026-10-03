"""
app/api/construction_api.py — REST endpoints for the graph-native Hand
Creator stepper (see app/services/hand_construction_service.py for the
full design). Replaces the OLD creation_phases-driven wizard contract.

POST   /creator/start              — begin a new construction session
GET    /creator/{session_id}/state — current pending step (reconnect/refresh)
POST   /creator/{session_id}/step  — submit the author's answer to the current step
POST   /creator/{session_id}/undo  — undo the last applied step
POST   /creator/{session_id}/finish — persist as a hypothetical hand (delegates to /tutorial/hands)
DELETE /creator/{session_id}       — abandon a session without saving
"""

from __future__ import annotations

import traceback

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from app.api.deps import get_db
from app.api.tutorial_api import (
    SaveHypotheticalHandRequest,
    save_hypothetical_hand,
)
from app.dto.construction_dto import (
    ConstructionStepDTO,
    ConstructionStepResponse,
    StartConstructionRequest,
)
from app.services.hand_construction_service import hand_construction_service

router = APIRouter(prefix="/creator")


@router.post("/start", response_model=ConstructionStepDTO)
def start_construction(req: StartConstructionRequest):
    """Begin a new construction session; returns the first pending step."""
    print("Starting hand construction")
    try:
        return hand_construction_service.start(req)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Unknown game variant: {req.game_name!r}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/resume/{hand_id}", response_model=ConstructionStepDTO)
def resume_construction(hand_id: int, db: DBSession = Depends(get_db)):
    """
    Load an existing saved hypothetical hand into a FRESH construction
    session by replaying its persisted hole cards / board cards /
    betting decisions through the real compiled graph — see
    hand_construction_service.resume_from_hand() for the full replay
    design. Once this returns, the session behaves exactly like one
    built via /start: the author can /undo back into the hand from
    wherever replay landed (normally the end) to re-edit it, then
    /finish to save over the original or as a new hand.

    KNOWN GAP: a hand containing a CARD_SELECT, CARD_PASS, or CHOICE
    decision cannot be resumed — SaveHypotheticalHandRequest never
    persisted enough to reconstruct those responses (see
    resume_from_hand()'s own docstring for the specifics per domain).
    Such a hand fails with 422 naming the unsupported domain rather
    than silently rebuilding a wrong hand.
    """
    try:
        return hand_construction_service.resume_from_hand(hand_id, db)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("/{session_id}/state", response_model=ConstructionStepDTO)
def get_construction_state(session_id: str):
    """Fetch the current pending step (e.g. on reconnect/refresh)."""
    try:
        return hand_construction_service.get_state(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{session_id}/step", response_model=ConstructionStepDTO)
def apply_construction_step(session_id: str, req: ConstructionStepResponse):
    """Submit the author's answer to the CURRENT pending step."""
    try:
        return hand_construction_service.apply_step(session_id, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/{session_id}/undo", response_model=ConstructionStepDTO)
def undo_construction_step(session_id: str):
    """Undo the last applied step (deal review/override or decision)."""
    try:
        return hand_construction_service.undo(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/{session_id}/finish")
def finish_construction(session_id: str, db: DBSession = Depends(get_db)):
    """
    Persist the constructed hand as a hypothetical hand by delegating
    to tutorial_api.save_hypothetical_hand() with a payload built from
    this session's recorded history — the same save/showdown-
    evaluation path a hand saved by the old creation_phases wizard
    already used (see hand_construction_service.finish()'s own
    docstring for a known CARD_SELECT/CARD_PASS ledger-fidelity gap).
    The session is discarded after a successful save either way, since
    a construction session has no further purpose once persisted.
    """
    try:
        payload = hand_construction_service.finish(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        save_req = SaveHypotheticalHandRequest(**payload)
        result = save_hypothetical_hand(save_req, db=db)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    hand_construction_service.discard(session_id)
    return result


@router.delete("/{session_id}", status_code=204)
def discard_construction_session(session_id: str):
    """Abandon a construction session without saving."""
    hand_construction_service.discard(session_id)