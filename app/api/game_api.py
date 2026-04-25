from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.api.deps import get_db
from app.services.session_logger import SessionLogger
from app.services.game_service import game_service


router = APIRouter(prefix="/game")


class ActionRequest(BaseModel):
    type: str
    amount: int | None = None


@router.post("/new-hand")
def new_hand():
    result = game_service.new_hand()
    return result

@router.post("/restart")
def restart(db: Session = Depends(get_db)):

    dto_state = game_service.restart(db)
    # print("DTO State: ", dto_state)
    return dto_state


@router.get("/state")
def get_state():

    return game_service.get_state()


@router.post("/action")
def apply_action(req: ActionRequest):

    dto_state = game_service.apply_action(req)
    return dto_state

@router.post("/start")
def start_game(config: dict, db: Session = Depends(get_db)):
    logger = GameLogger(db)
    logger.start_game(config)

    return {"status": "ok"}