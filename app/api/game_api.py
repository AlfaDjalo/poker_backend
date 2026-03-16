from fastapi import APIRouter
from pydantic import BaseModel

from app.services.game_service import game_service


router = APIRouter(prefix="/game")


class ActionRequest(BaseModel):
    type: str
    amount: int | None = None


@router.post("/new-hand")
def new_hand():

    return game_service.new_hand()


@router.get("/state")
def get_state():

    return game_service.get_state()


@router.post("/action")
def apply_action(req: ActionRequest):

    return game_service.apply_action(req)