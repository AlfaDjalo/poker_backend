"""
app/dto/state_dto.py — response DTOs shared by both the legacy
PokerState path (app/engine_adapter.py) and the new GraphEngine path
(app/graph_engine_adapter.py). See DecisionRequestDTO's own docstring
below for the phase-string -> decision-object redesign rationale.
"""

from typing import Any

from pydantic import BaseModel, model_validator


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
    no_qualify: list[bool]  # per board: True if no qualifier
    scoop: list[bool]  # per board: True if scooped from paired high


class ShowdownDTO(BaseModel):
    payout_type: str  # "points" | "split_pot"
    point_results: list[PointResultDTO]
    point_tallies: dict[int, float] | None  # player_index -> points (points game)
    payouts: dict[int, int]  # player_index -> chip amount
    pot_winners: list[int]  # 0-based player indices


# --------------------------------------------------------------------
# Decision request — generic replacement for the old phase string +
# available_actions/to_call/min_raise/max_raise fields.
#
# Mirrors GraphEngine's DecisionRequest: `domain` names what KIND of
# decision is pending (today only "BETTING" exists in practice, but a
# future Draw/Discard/Pass-the-Trash variant produces "CARD_SELECT" /
# "CARD_PASS" / "BOOLEAN" / "CHOICE" with zero backend contract
# changes — the frontend just needs a renderer per domain). `decision`
# on GameStateDTO is None exactly when there's no pending decision
# (hand complete / between hands) — the direct analog of
# `engine.pending_request is None`.
#
# `options` is intentionally generic across domains:
#   - BETTING:      one DecisionOptionDTO per legal action
#                    ("fold"/"check"/"call"/"bet"/"raise"/"all_in"),
#                    min_amount/max_amount set on bet/raise-shaped ones.
#   - CARD_SELECT:   one option per selectable slot/constraint,
#                    min_count/max_count set as top-level fields;
#                    eligible-card info (if any) lives in `metadata`.
#   - CARD_PASS:     one option per legal pass target, min_count/
#                    max_count set the same way as CARD_SELECT;
#                    `metadata` may carry the target seat.
#   - BOOLEAN:       two options, action_name "yes"/"no".
#   - CHOICE:        one option per selectable choice, `label` is the
#                    human-readable text, action_name is the value to
#                    echo back in the DecisionResponse.
# --------------------------------------------------------------------


class DecisionOptionDTO(BaseModel):
    action_name: str  # value to send back in the response, e.g. "fold", "bet"
    min_amount: int | None = None  # bet/raise-shaped (BETTING) options only
    max_amount: int | None = None
    # CARD_SELECT / CARD_PASS-shaped options only — how many cards must
    # be chosen from this option's zone. Top-level fields (not nested
    # under `metadata`) to match poker_engine.graph.decision_dto's
    # CardSelectOption shape 1:1 — `metadata` is reserved for genuinely
    # free-form, domain-specific extras (e.g. eligible card ids), not
    # for structural constraints every CARD_SELECT/CARD_PASS renderer
    # needs to read.
    min_count: int | None = None
    max_count: int | None = None
    label: str | None = None  # human-readable text, mainly for CHOICE
    metadata: dict[str, Any] = {}  # domain-specific extras


class DecisionRequestDTO(BaseModel):
    domain: str  # "BETTING" | "CARD_SELECT" | "CARD_PASS" | "BOOLEAN" | "CHOICE" | ...
    seat: int  # 1-based acting player seat
    options: list[DecisionOptionDTO]
    # Convenience fields for the common BETTING case — null for every
    # other domain. Frontend's bet slider reads these directly instead
    # of hunting through `options` for the bet-shaped one.
    to_call: int | None = None
    min_raise: int | None = None
    max_raise: int | None = None
    # Pass-through of graph.node(current_node).metadata — replaces the
    # old creation_phases id-string lookups (street_index/street_name
    # etc. live here now instead of being inferred from a phase id).
    node_metadata: dict[str, Any] = {}


class GameStateDTO(BaseModel):
    street: int
    pot: int
    nodes: list[str | None]  # all node cards indexed by node index
    layout_name: str | None  # e.g. "double_board", "wheel"
    game_name: str | None  # e.g. "double_board_plo_bomb_pot"
    street_names: list[str] | None
    points: list[PointDTO] | None  # point definitions with node_sets

    players: list[PlayerDTO]

    # None = no decision pending (hand complete / between hands).
    # Non-None = someone needs to respond; see DecisionRequestDTO above.
    decision: DecisionRequestDTO | None = None
    hand_complete: bool = False

    # Convenience mirror of decision.seat (or None) — kept because
    # "whose turn is it" is extremely common UI, and forcing every
    # caller to null-check into `decision` just for that is friction
    # with no real payoff.
    current_player: int | None = None

    showdown: ShowdownDTO | None = None
    winners: list[int] | None = None

    discard_pile: list[str] = []

    # --------------------------------------------------------------
    # BACK-COMPAT — flat fields the Frontend still reads directly
    # (GameSimulator.jsx: `hand.phase === "BETTING"` gates
    # PlayerActionPanel; `hand.phase === "SHOWDOWN"|"HAND_COMPLETE"`
    # gates ShowdownSummary; PlayerActionPanel itself reads
    # to_call/min_raise/max_raise as top-level numbers — see the
    # documented Wire Format Reference, §4.8). These were dropped
    # from the DTO when `decision` replaced them during the
    # PokerState -> GraphEngine migration, which silently broke
    # PlayerActionPanel (it never sees phase === "BETTING" anymore,
    # since GameStateDTO simply has no `phase` field at all now).
    #
    # Rather than touch every adapter (engine_adapter.py AND
    # graph_engine_adapter.py) to populate these explicitly, they're
    # derived here once from `decision`/`hand_complete`/`showdown` —
    # the same three fields both adapters already populate correctly
    # — so this fix applies uniformly to the live GraphEngine game,
    # PushFoldService, and TrainerService without touching any of
    # them.
    #
    # `phase` intentionally only distinguishes the four states the
    # Frontend actually branches on (BETTING / SHOWDOWN /
    # HAND_COMPLETE / DEAL_BOARD-as-fallback) — GraphEngine has no
    # native "phase" concept beyond decision-pending-or-not, so this
    # is a presentation convenience, not a real engine state mirror.
    phase: str | None = None
    available_actions: list[str] | None = None
    to_call: int | None = None
    min_raise: int | None = None
    max_raise: int | None = None

    @model_validator(mode="after")
    def _derive_legacy_fields(self) -> "GameStateDTO":
        if self.decision is not None:
            # Only a real BETTING decision gets the legacy phase/
            # available_actions/to_call/min_raise/max_raise shim —
            # PlayerActionPanel's contract (fold/check/call/bet/raise/
            # all_in strings, numeric to_call/min_raise/max_raise) only
            # makes sense for that domain. This USED to fire for every
            # decision regardless of domain, so a CARD_SELECT/CARD_PASS/
            # BOOLEAN/CHOICE decision was reported as phase="BETTING"
            # with available_actions=["select"] (or similar synthetic
            # action_name) — any caller still reading the legacy fields
            # instead of decision.domain rendered the wrong panel, and a
            # normal betting-shaped submission against a pending
            # non-BETTING decision is correctly rejected by
            # game_service._build_decision_response as "not among the
            # legal options right now", surfacing as an unexplained 400
            # right when the player tries to act on what LOOKS like a
            # betting decision but the server is actually waiting on a
            # CARD_SELECT/CARD_PASS response.
            if self.decision.domain == "BETTING":
                self.phase = "BETTING"
                self.available_actions = [o.action_name for o in self.decision.options]
                self.to_call = self.decision.to_call
                self.min_raise = self.decision.min_raise
                self.max_raise = self.decision.max_raise
            else:
                # No legacy betting-shaped equivalent for these domains
                # — phase mirrors the real domain so any code still
                # checking `phase` (rather than decision.domain) fails
                # CLOSED (won't render a betting panel for it) instead
                # of failing open with a misleading available_actions
                # list.
                self.phase = self.decision.domain
        elif self.showdown is not None:
            self.phase = "SHOWDOWN"
        elif self.hand_complete:
            self.phase = "HAND_COMPLETE"
        else:
            # Transient — GraphEngine auto-advances through these with
            # no external pause, so this should rarely if ever be the
            # phase actually observed by a client.
            self.phase = "DEAL_BOARD"
        return self