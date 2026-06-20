from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class PlayerDTO(BaseModel):
    seat: int
    name: str
    stack: int
    bet: int
    folded: bool
    hand: List[Optional[str]]

# Describes a single named point/board group for rendering
class PointDTO(BaseModel):
    name: str                   # e.g. "board1", "board2", "hand"
    score_type: str             # e.g. "HIGH", "LOW_27"
    node_sets: List[List[int]]  # each node_set is a list of node indices

class PlayerBoardResultDTO(BaseModel):
    player_index: int
    hand_category: Optional[str]
    hand_value: Optional[int]
    best_hand_cards: Optional[List[str]]    # ["Ah", "Kd", ...]
    hole_cards_used: Optional[List[str]]
    board_cards_used: Optional[List[str]]
    is_winner: bool

class PointResultDTO(BaseModel):
    name: str
    score_type: str
    board_winners: List[List[int]]                  # per board: list of 0-based player indices
    board_results: List[List[PlayerBoardResultDTO]]
    # board_scores: List[List[Optional[List[int]]]]   # per board: score tuple per player
    no_qualify: List[bool]                          # per board: True if no qualifier
    scoop: List[bool]                               # per board: True if scooped from paired high

class ShowdownDTO(BaseModel):
    payout_type: str                                # "points" | "split_pot"
    point_results: List[PointResultDTO]
    point_tallies: Optional[Dict[int, float]]       # player_index -> points (points game)
    payouts: Dict[int, int]                         # player_index -> chip amount
    pot_winners: List[int]                          # 0-based player indices
    
class GameStateDTO(BaseModel):
    street: int
    pot: int
    # board: List[Optional[str]]          # kept for backwards compatibility - flat list
    nodes: List[Optional[str]]          # all node cards indexed by node index
    layout_name: Optional[str]          # e.g. "double_board", "wheel"
    game_name: Optional[str]            # e.g. "double_board_plo_bomb_pot"
    street_names: Optional[List[str]]
    points: Optional[List[PointDTO]]    # point definitions with node_sets
    # board_layout: BoardLayoutDTO        # full layout descriptor
    
    players: List[PlayerDTO]
    current_player: Optional[int]
    phase: str
    showdown: Optional[ShowdownDTO] = None
    winners: Optional[List[int]] = None

    available_actions: Optional[List[str]]
    to_call: Optional[int]
    min_raise: Optional[int]
    max_raise: Optional[int]
    discard_pile: List[str] = []