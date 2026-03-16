from pydantic import BaseModel
from typing import List, Optional


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
    current_player: int