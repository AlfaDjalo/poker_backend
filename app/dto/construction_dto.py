"""
app/dto/construction_dto.py — DTOs for the Hand Creator's graph-native
construction-mode stepper. Replaces creation_phases as the wizard's
DRIVING contract (see hand_construction_service.py's module docstring
for the full design rationale) — creation_phases may still exist as a
read-only preview (game_service._summarize_flow), but the wizard
itself steps through the session endpoints these DTOs describe.

ConstructionStepDTO generalizes DecisionRequestDTO (state_dto.py) to
also cover AUTO deal nodes, which live play never pauses at but the
Creator wizard must, so the author can choose exactly which cards are
dealt rather than have them drawn at random.
"""

from typing import Any

from pydantic import BaseModel

from app.dto.state_dto import DecisionOptionDTO, GameStateDTO


class ConstructionStepDTO(BaseModel):
    """
    One paused point in a construction-mode walk of the compiled
    GameGraph.

    domain:
      "DEAL_HOLE"  — an AUTO deal_hole_cards node. `count` cards must
                     be supplied for `seat`.
      "DEAL_BOARD" — an AUTO deal_board_cards node. `count` cards must
                     be supplied, filling `board_node_indices` in order.
      "BETTING" / "CARD_SELECT" / "CARD_PASS" / "BOOLEAN" / "CHOICE"
                   — a real DECISION node; shape matches
                     DecisionRequestDTO's own domain semantics exactly.
      "AUTO"       — a non-deal AUTO node with nothing for the author
                     to choose (e.g. posting blinds) — executed
                     automatically and surfaced here only so the
                     wizard's step log can show it happened. No
                     response is required for this domain; the next
                     /step call may be sent with an empty body.
      "COMPLETE"   — the walk reached the end of the graph. No further
                     /step calls are valid — call /finish instead.
    """

    session_id: str
    node_id: str
    # Best-effort human-readable label for this step, distinct from
    # `domain` (a fixed enum-like string) and `node_id` (a raw/synthetic
    # id not meant for display). For a real DECISION node, derived from
    # graph.node(current_node).metadata — the SAME metadata dict
    # game_service._summarize_flow already reads street_name/id from,
    # so this inherits that function's own unconfirmed-field-name
    # caveat (see game_service.py's ASSUMPTIONS block): if the real
    # graph loader uses different metadata keys than assumed here, this
    # falls back to the domain name rather than raising or guessing
    # further. For DEAL_HOLE/DEAL_BOARD steps there is NO reliable graph
    # node to label at all — by the time the diff-based design surfaces
    # these as a step, the AUTO node has already executed and
    # engine.state.current_node has typically already moved past it —
    # so these get a synthetic, non-graph-derived label instead
    # (e.g. "Deal hole cards — Seat 2"). Always non-null; never assume
    # it uniquely identifies a node the way node_id does.
    node_label: str | None = None
    domain: str
    seat: int | None = None  # 1-based; None for table-wide AUTO steps
    count: int | None = None  # DEAL_HOLE / DEAL_BOARD only
    board_node_indices: list[int] = []  # DEAL_BOARD only, fill order
    # DEAL_HOLE / DEAL_BOARD only — the ACTUAL cards the engine just
    # dealt for this step (real Deck draw, already applied to engine
    # state before this step is queued — see hand_construction_service
    # .py's module docstring, point 3). In dealt-card order: for
    # DEAL_BOARD this lines up 1:1 with board_node_indices; for
    # DEAL_HOLE it's just the seat's new cards, order not meaningful
    # beyond "as returned by mask_to_card_ids". Distinct from
    # eligible_cards, which lists the UNUSED pool (what COULD replace
    # these), not what WAS dealt. A caller that calls /step with
    # response.cards omitted/empty accepts exactly these cards as-is.
    dealt_cards: list[str] = []
    eligible_cards: list[str] | None = None  # not yet used anywhere this hand
    options: list[DecisionOptionDTO] = []  # DECISION domains only
    to_call: int | None = None
    min_raise: int | None = None
    max_raise: int | None = None
    node_metadata: dict[str, Any] = {}
    state: GameStateDTO  # current table snapshot — same shape live play uses


class ConstructionStepResponse(BaseModel):
    """
    Author's answer to the CURRENT ConstructionStepDTO. Populate only
    the field(s) matching the pending step's domain — mirrors
    ActionRequest's own "populate only what your domain needs"
    contract in game_api.py, extended with `cards` for the two new
    deal domains. Deliberately field-compatible with ActionRequest
    (type/amount/selected_cards/bool_value/choice) so
    game_service._build_decision_response()'s existing per-domain
    builders can be reused unmodified for the DECISION domains — see
    hand_construction_service.py.
    """

    cards: list[str] | None = None  # DEAL_HOLE / DEAL_BOARD
    type: str | None = None  # BETTING
    amount: int | None = None  # BETTING
    selected_cards: list[str] | None = None  # CARD_SELECT / CARD_PASS
    bool_value: bool | None = None  # BOOLEAN
    choice: str | None = None  # CHOICE


class StartConstructionRequest(BaseModel):
    game_name: str
    dealer_seat: int = 1
    seats: dict[int, str] = {}  # seat -> display name
    initial_stacks: dict[int, int] = {}  # seat -> starting stack