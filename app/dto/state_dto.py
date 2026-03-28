from pydantic import BaseModel
from typing import List, Optional, Dict


class PlayerDTO(BaseModel):
    seat: int
    name: str
    stack: int
    bet: int
    folded: bool
    hand: List[Optional[str]]


class GameStateDTO(BaseModel):
    street: int
    pot: int
    board: List[Optional[str]]
    players: List[PlayerDTO]
    current_player: Optional[int]
    phase: str
    showdown: Optional[dict] = None
    winners: Optional[List[int]] = None
    # hand_strengths: Optional[Dict[int, str]]

    available_actions: Optional[List[str]]
    to_call: Optional[int]
    min_raise: Optional[int]
    max_raise: Optional[int]