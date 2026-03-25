from fastapi import APIRouter
from pydantic import BaseModel

from app.services.game_service import game_service


router = APIRouter(prefix="/game")


class ActionRequest(BaseModel):
    type: str
    amount: int | None = None


@router.post("/new-hand")
def new_hand():
    result = game_service.new_hand()
    # print("DEBUG:", result)
    return result

@router.post("/restart")
def restart():

    return game_service.restart()


@router.get("/state")
def get_state():

    return game_service.get_state()


@router.post("/action")
def apply_action(req: ActionRequest):

    return game_service.apply_action(req)

# @router.post("/deal_next_street")
# def deal_next_street():
#     if game_service.state is None:
#         return {"error": "No hand in progress"}
    
#     game_service.advance_street()
#     # game_service.state.advance_street()

#     return game_service.get_state()

# @router.post("/action")
# def apply_action(req: ActionRequest):
#     return game_service.apply_action(req)