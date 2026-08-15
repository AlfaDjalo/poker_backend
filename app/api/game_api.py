from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.services.game_service import game_service

router = APIRouter(prefix="/game")


# --------------------------------------------------
# Request models
# --------------------------------------------------


class ActionRequest(BaseModel):
    type: str
    amount: int | None = None


class RestartRequest(BaseModel):
    game_name: str | None = None


class NewHandRequest(BaseModel):
    game_name: str | None = None


class SelectGameRequest(BaseModel):
    game_name: str


# --------------------------------------------------
# Routes
# --------------------------------------------------


@router.get("/variants")
def get_variants():
    """Return all available game variants and the currently active one."""
    return game_service.get_variants()


@router.get("/variants/{game_name}/config")
def get_variant_config(game_name: str):
    """
    Return the per-variant config block (board layout + creation phases)
    that the Hand Creation wizard needs to drive its phase machine.
    """
    config = game_service.get_variant_config(game_name)
    if config is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown game variant: {game_name!r}"
        )
    return config


@router.post("/select-game")
def select_game(req: SelectGameRequest):
    """
    Queue a game variant to be used on the next hand or restart.
    Safe to call between hands; rejected mid-hand by the frontend.
    """
    try:
        game_service.select_game(req.game_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "pending_game": req.game_name}


@router.post("/new-hand")
def new_hand(req: NewHandRequest = NewHandRequest()):
    print("New hand starting with game ", req.game_name)
    result = game_service.new_hand(game_name=req.game_name)
    return result


@router.post("/restart")
def restart(req: RestartRequest = RestartRequest(), db: Session = Depends(get_db)):
    print("game_name: ", req.game_name)
    try:
        dto_state = game_service.restart(db, game_name=req.game_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return dto_state


@router.get("/state")
def get_state():

    return game_service.get_state()


@router.post("/action")
def apply_action(req: ActionRequest):

    dto_state = game_service.apply_action(req)
    # print("dto_state: ", dto_state)
    return dto_state
