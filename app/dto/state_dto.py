
from pydantic import BaseModel


class PlayerDTO(BaseModel):
    seat: int
    name: str
    stack: int
    bet: int
    folded: bool
    hand: list[str | None]


# Describes a single named point/board group for rendering
class PointDTO(BaseModel):
    name: str  # e.g. "board1", "board2", "hand"
    score_type: str  # e.g. "HIGH", "LOW_27"
    node_sets: list[list[int]]  # each node_set is a list of node indices


class PlayerBoardResultDTO(BaseModel):
    player_index: int
    hand_category: str | None
    hand_value: int | None
    best_hand_cards: list[str] | None  # ["Ah", "Kd", ...]
    hole_cards_used: list[str] | None
    board_cards_used: list[str] | None
    is_winner: bool


class PointResultDTO(BaseModel):
    name: str
    score_type: str
    board_winners: list[list[int]]  # per board: list of 0-based player indices
    board_results: list[list[PlayerBoardResultDTO]]
    # board_scores: List[List[Optional[List[int]]]]   # per board: score tuple per player
    no_qualify: list[bool]  # per board: True if no qualifier
    scoop: list[bool]  # per board: True if scooped from paired high


class ShowdownDTO(BaseModel):
    payout_type: str  # "points" | "split_pot"
    point_results: list[PointResultDTO]
    point_tallies: dict[int, float] | None  # player_index -> points (points game)
    payouts: dict[int, int]  # player_index -> chip amount
    pot_winners: list[int]  # 0-based player indices


class GameStateDTO(BaseModel):
    street: int
    pot: int
    # board: List[Optional[str]]          # kept for backwards compatibility - flat list
    nodes: list[str | None]  # all node cards indexed by node index
    layout_name: str | None  # e.g. "double_board", "wheel"
    game_name: str | None  # e.g. "double_board_plo_bomb_pot"
    street_names: list[str] | None
    points: list[PointDTO] | None  # point definitions with node_sets
    # board_layout: BoardLayoutDTO        # full layout descriptor

    players: list[PlayerDTO]
    current_player: int | None
    phase: str
    showdown: ShowdownDTO | None = None
    winners: list[int] | None = None

    available_actions: list[str] | None
    to_call: int | None
    min_raise: int | None
    max_raise: int | None
    discard_pile: list[str] = []
