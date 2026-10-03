"""
app/services/hand_construction_service.py — graph-native, node-by-node
stepper for the Hand Creator wizard. Replaces creation_phases as the
wizard's DRIVING contract (creation_phases / game_service._summarize_flow
may still exist as a read-only preview, since it's already graph-derived
and stateless — but nothing here reads or writes it).

Design summary:

  1. Each construction session owns a PRIVATE, ephemeral GraphEngine —
     never the shared live GameService.engine. This sidesteps Hand
     Editor's actual blocker (no snapshot/restore for the ONE live,
     shared, already-persisting engine) entirely: there is no live
     table to protect here, this is the same "build a throwaway engine
     from scratch" pattern tutorial_api.py's own showdown-evaluation
     path already uses.

  2. DECISION nodes (BETTING/CARD_SELECT/CARD_PASS/BOOLEAN/CHOICE) are
     driven with ZERO new Engine capability — submit_decision() /
     DecisionResponse reused exactly as live play uses them, via the
     same per-domain builders game_service.py already has (imported
     directly below rather than duplicated).

  3. AUTO deal nodes (deal_hole_cards / deal_board_cards) are NOT
     pre-seeded with author-chosen cards before they execute (that
     would require the engine to pause at arbitrary AUTO nodes, which
     it doesn't do, AND would require knowing which deal node comes
     next before a not-yet-submitted decision resolves the branch —
     unreliable in general). Instead: let the engine deal for real
     with its normal random Deck, DIFF what just got dealt against
     what this session has already presented (same diff-based idiom
     SessionLogger.log_board()/log_hole_cards() already use for extra-
     card mechanics), and expose each newly-dealt card as a REVIEWABLE
     step the author can either accept or override. Overriding is a
     direct in-place mutation of THIS session's own private, nothing-
     persisted-yet engine state — no CardMove/card_movement.py
     bookkeeping needed, since nothing is written to the DB until
     finish() hands the session off to tutorial_api's existing save
     path. This needs NO new Engine capability at all, not even a
     seedable Deck.

  4. Undo/back only ever needs to rebuild THIS private, ephemeral
     session, never a live shared object — implemented as event-
     sourced replay (re-run the recorded step history from a fresh
     engine), reusing the exact same "force a card" mutation primitive
     the interactive override path uses, just applied to every
     historically-recorded value instead of only the ones the author
     explicitly changed.

  5. finish() translates the recorded history into a payload shaped
     like tutorial_api.SaveHypotheticalHandRequest, so a constructed
     hand is persisted through the EXACT SAME save/showdown-evaluation
     path a hand from the old creation_phases wizard already used —
     see finish()'s own docstring for a known gap (CARD_SELECT/
     CARD_PASS card-movement ledger fidelity).

ASSUMPTIONS carried over from game_service.py / graph_engine_adapter.py
(same unconfirmed-against-real-source caveats already flagged there):
  - graph.node(node_id).kind / .auto_type / .config / .domain / .metadata
  - graph.outgoing(node_id)
  - engine.state.current_node, engine.pending_request, engine.is_complete
  - _game_obj(engine) — reused from graph_engine_adapter.py
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from poker_engine.cards.card import Card as CardObj
from poker_engine.cards.deck import Deck as GraphDeck
from poker_engine.cards.mask import mask_to_card_ids
from poker_engine.graph.callbacks import submit_decision
from poker_engine.graph.graph_engine import GraphEngine
from poker_engine.graph.hand_session import new_hand_state, start_new_hand
from poker_engine.rules.loader import load_game as load_game_graph
from poker_engine.scoring.scoring_engine import CppScoringEngine
from poker_engine.showdown.showdown_resolver import ShowdownResolver
from poker_engine.state.player_state import PlayerState

from app.dto.construction_dto import (
    ConstructionStepDTO,
    ConstructionStepResponse,
    StartConstructionRequest,
)
from app.graph_engine_adapter import (
    _domain_name,
    _game_obj,
    _options_to_dto,
    _betting_to_call_and_raise,
    graph_state_to_dto,
    node_metadata as _node_metadata,
)
from app.services.game_service import _build_decision_response


ALL_CARD_IDS = list(range(52))


def _card_str(cid: int) -> str:
    return str(CardObj(cid))


def _node_auto_deal_kind(node) -> Optional[str]:
    """
    "hole" | "board" | None — structural classification of an AUTO
    node, same technique confirmed correct for deal_hole_cards in the
    hole-card-count fix (poker_engine.graph.graph_hole_cards). Board's
    auto_type spelling ("deal_board_cards") mirrors that naming
    convention but is NOT independently confirmed — flagged here the
    same way graph_engine_adapter.py flags its own unconfirmed field
    names, so it's easy to grep-fix if the real spelling differs. Not
    currently called anywhere (the diff-based design below never needs
    to classify a node BEFORE it executes) — kept for any future
    caller that wants to preview upcoming node types without running
    the engine forward.
    """
    kind = getattr(node, "kind", None)
    kind_name = getattr(kind, "name", str(kind)) if kind is not None else None
    if kind_name != "AUTO":
        return None
    auto_type = getattr(node, "auto_type", None)
    if auto_type == "deal_hole_cards":
        return "hole"
    if auto_type == "deal_board_cards":
        return "board"
    return None


_DOMAIN_FALLBACK_LABELS = {
    "BETTING": "Betting decision",
    "CARD_SELECT": "Card selection",
    "CARD_PASS": "Card pass",
    "BOOLEAN": "Yes/no decision",
    "CHOICE": "Choice decision",
}


def _decision_node_label(engine, graph, domain_name: str) -> str:
    """
    Best-effort human-readable label for a real DECISION node, tried in
    this order:
      1. graph.node(current_node).metadata["label"]
      2. ...metadata["id"]
      3. ...metadata["street_name"]
      4. a fixed per-domain fallback string (_DOMAIN_FALLBACK_LABELS)
    Metadata key names here are the SAME ones game_service._summarize_flow
    reads (id/street_name) — unconfirmed against real graph-loader
    source, see that function's own ASSUMPTIONS block. Never raises;
    always returns a usable string.
    """
    metadata = _node_metadata(engine, graph)
    for key in ("label", "id", "street_name"):
        val = metadata.get(key)
        if val:
            return str(val)
    return _DOMAIN_FALLBACK_LABELS.get(domain_name, domain_name)


def _action_type_and_amount(domain: str, response: ConstructionStepResponse) -> tuple[str, int | None]:
    """
    Derive the (action_type, amount) pair to stamp on a saved Action
    row from the author's raw ConstructionStepResponse — mirrors
    graph_engine_callbacks.py's own BOOLEAN/CHOICE encoding convention
    (action_type as free text, amount repurposed for BOOLEAN's 1/0)
    since Action.action_type is already a free-text column, not a
    strict ActionType enum (see hands_api.py's own schema note on
    this).
    """
    if domain == "BETTING":
        return (response.type or "").upper(), response.amount
    if domain == "BOOLEAN":
        return "BOOLEAN", 1 if response.bool_value else 0
    if domain == "CHOICE":
        # response.choice's actual value isn't representable in the
        # int `amount` column — same acknowledged gap
        # graph_engine_callbacks._log_boolean_or_choice_decision notes
        # for CHOICE ("no shipped variant uses CHOICE yet").
        return "CHOICE", None
    # CARD_SELECT / CARD_PASS — see finish()'s own docstring for the
    # known card-movement-ledger gap this implies.
    return domain, None


@dataclass
class _ConstructionSession:
    session_id: str
    game_name: str
    game_def: Any
    rules: Any
    graph: Any
    engine: GraphEngine
    seats: list[int]
    dealer_seat: int
    initial_stacks: dict[int, int] = field(default_factory=dict)
    seat_names: dict[int, str] = field(default_factory=dict)

    # What this session has already presented to the author — a card
    # once diffed-in here is never re-diffed, whether the author
    # accepted or overrode it (the override itself already mutated
    # engine state to the final value before this is populated).
    logged_hole_cards: dict[int, set[int]] = field(default_factory=dict)  # seat_idx -> card ids
    logged_board_nodes: set[int] = field(default_factory=set)  # node indices

    # Mirrors SessionLogger._board_reveal_index exactly — incremented
    # once per NEW batch of board cards diffed-in, used to stamp
    # `street` on both BoardCard-equivalent and Action-equivalent
    # history entries, so a constructed hand's street numbering is
    # derived identically to a live hand's (see session_logger.py's
    # own docstring on why this single counter is the source of truth).
    board_reveal_index: int = 0

    # Steps awaiting author review/override, popped front-to-back.
    pending_queue: list[ConstructionStepDTO] = field(default_factory=list)

    # The step most recently HANDED to the caller — apply_step()
    # validates the incoming response against this, so a stale/
    # duplicate submission is rejected rather than silently misapplied
    # (mirrors GameService._busy's own rationale for the live path).
    current_step: Optional[ConstructionStepDTO] = None

    # Full recorded history, in order — for undo (event-sourced
    # replay) and finish() (translating into a saveable hand).
    #   {"kind": "deal", "domain": "DEAL_HOLE"|"DEAL_BOARD",
    #    "seat": int|None, "node": list[int]|None, "cards": [int,...],
    #    "street": int}
    #   {"kind": "decision", "domain": str, "response": dict,
    #    "street": int, "player_seat": int, "action_type": str,
    #    "amount": int|None, "pot_before": int, "stack_before": int}
    history: list[dict] = field(default_factory=list)

    # Reentrancy guard — apply_step()/undo() mutate this session's
    # engine in place with no locking. A double-submitted request
    # (double click, a retried request after a slow response) hitting
    # the same session concurrently could step the engine twice for
    # one authored answer, desyncing whatever the wizard still thinks
    # the current step is. Mirrors GameService._busy / TrainerService._busy
    # exactly — see either's own docstring for the same rationale.
    busy: bool = False


class HandConstructionService:
    """
    In-memory session registry — mirrors TrainerService/PushFoldService's
    own singleton pattern, generalized to multiple CONCURRENT sessions
    (keyed by session_id) since more than one Hand Creator tab/user may
    be building a hand at once, unlike those single-active-scenario
    services.
    """

    def __init__(self):
        self._sessions: dict[str, _ConstructionSession] = {}

    # ------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------

    def start(self, req: StartConstructionRequest) -> ConstructionStepDTO:
        game_def, rules, graph = load_game_graph(req.game_name)

        print("Starting hand construction function")

        seats = sorted(req.seats.keys()) if req.seats else sorted(req.initial_stacks.keys())
        if not seats:
            raise ValueError("At least one seat is required to start a construction session.")

        players = [
            PlayerState(stack=req.initial_stacks.get(seat, 0)) for seat in seats
        ]

        session_id = str(uuid.uuid4())
        hand_state = new_hand_state(players, game_def, dealer_position=req.dealer_seat - 1)
        deck = GraphDeck()
        showdown_resolver = ShowdownResolver(CppScoringEngine(), rules)
        engine = GraphEngine(graph, hand_state, deck, showdown_resolver=showdown_resolver)
        engine.game_def = game_def

        session = _ConstructionSession(
            session_id=session_id,
            game_name=req.game_name,
            game_def=game_def,
            rules=rules,
            graph=graph,
            engine=engine,
            seats=seats,
            dealer_seat=req.dealer_seat,
            initial_stacks=dict(req.initial_stacks),
            seat_names=dict(req.seats),
        )
        self._sessions[session_id] = session

        # No callbacks — construction sessions never write to
        # SessionLogger/the DB; the whole point is a scratch hand
        # that's only persisted once finish() hands it to tutorial_api.
        start_new_hand(engine, game_def, callbacks=None, advance_dealer=False)

        print("Starting new hand")

        return self._refresh_and_get_current_step(session)

    def resume_from_hand(self, hand_id: int, db) -> ConstructionStepDTO:
        """
        Load a previously saved hypothetical hand (tutorial_api.py's
        SaveHypotheticalHandRequest rows — Hand/HoleCard/BoardCard/
        Action) into a brand new construction session, by REPLAYING it
        forward through the real compiled graph via the exact same
        apply_step() path the wizard itself uses — never by hand-
        constructing session.history directly. This guarantees a
        resumed session is indistinguishable from one an author
        actually clicked through: same diff-based deal review, same
        DecisionResponse construction, same history entries, so /undo
        afterward works identically to any other in-progress session.

        Replay walks the graph from the very start, consuming persisted
        data in save order:
          - DEAL_HOLE / DEAL_BOARD steps: the persisted cards for that
            seat/those nodes are supplied as an explicit override (see
            _apply_deal_review — response.cards non-empty always takes
            the override path), so replay reproduces the ORIGINAL cards
            rather than whatever a fresh random Deck would deal.
          - BETTING steps: the next unconsumed Action row for the
            step's acting seat supplies type/amount.
          - BOOLEAN steps: the next unconsumed Action row supplies
            bool_value, decoded from the amount=1/0 encoding
            graph_engine_callbacks._log_boolean_or_choice_decision
            writes.

        KNOWN GAP — CARD_SELECT / CARD_PASS / CHOICE cannot be resumed:
          - CARD_SELECT/CARD_PASS: SaveHypotheticalHandRequest has no
            card-movement-ledger field at all (see
            HandConstructionService.finish()'s own docstring, and
            CAP_Technical_Quick_Reference.md §4.4's "Known gap") — the
            actual card ids a player selected were never persisted,
            only the FINAL hole cards after the fact. There is nothing
            to replay this decision with.
          - CHOICE: the chosen option's string value isn't
            representable in Action.amount (an int column) and was
            never stored anywhere else (see
            graph_engine_callbacks._log_boolean_or_choice_decision's
            own docstring).
        A hand containing any of these three domains fails loudly with
        ValueError naming the domain, rather than guessing a value and
        silently reconstructing a hand that doesn't match what was
        actually played. The half-built session is discarded before
        raising, so a failed resume never leaks a broken session into
        the registry.

        Raises
        ------
        LookupError
            No hypothetical hand with this id (mapped to 404 by the router).
        ValueError
            The hand's variant no longer loads, persisted data runs out
            before the graph reaches COMPLETE, or an unresumable
            CARD_SELECT/CARD_PASS/CHOICE decision is encountered
            (mapped to 422 by the router).
        """
        from app.db.models.actions import Action
        from app.db.models.board_cards import BoardCard
        from app.db.models.hands import Hand
        from app.db.models.hole_cards import HoleCard
        from app.db.models.players import Player

        hand = (
            db.query(Hand)
            .filter(Hand.hand_id == hand_id, Hand.is_hypothetical == True)
            .first()
        )
        if hand is None:
            raise LookupError(f"Hypothetical hand {hand_id} not found.")

        # ── Seats / names / starting stacks — same derivation tutorial_api's
        # own edit-state endpoint uses (negative player_id encodes seat). ──
        hc_rows = (
            db.query(HoleCard)
            .filter(HoleCard.hand_id == hand_id, HoleCard.status == "IN_HAND")
            .order_by(HoleCard.player_id, HoleCard.card)
            .all()
        )
        hole_cards_by_seat: dict[int, list[int]] = {}
        for hc in hc_rows:
            seat = -hc.player_id if hc.player_id < 0 else hc.player_id
            hole_cards_by_seat.setdefault(seat, []).append(hc.card)

        action_rows = (
            db.query(Action)
            .filter(Action.hand_id == hand_id)
            .order_by(Action.street, Action.action_index)
            .all()
        )
        initial_stacks: dict[int, int] = {}
        for a in action_rows:
            seat = -a.player_id if a.player_id < 0 else a.player_id
            if seat not in initial_stacks and a.stack_before is not None:
                initial_stacks[seat] = a.stack_before

        seats = sorted(set(hole_cards_by_seat) | set(initial_stacks))
        if not seats:
            raise ValueError(
                f"Hand {hand_id} has no seat data (no hole cards or actions) "
                f"— nothing to resume."
            )

        pids = [-s for s in seats]
        names_by_seat: dict[int, str] = {}
        if pids:
            for p in db.query(Player).filter(Player.player_id.in_(pids)).all():
                names_by_seat[-p.player_id] = (p.username or "").replace(
                    "[Hypothetical] ", ""
                )

        bc_rows = (
            db.query(BoardCard).filter(BoardCard.hand_id == hand_id).all()
        )
        board_cards_by_node: dict[int, int] = {bc.node: bc.card for bc in bc_rows}

        # Betting-shaped actions only — SEAT synthetic rows (action_index
        # == -1) and non-chip-moving domains are consumed separately by
        # domain below; excluded here so _pop_next_action never hands a
        # SEAT row to a real decision step.
        pending_actions = [a for a in action_rows if a.action_index != -1]

        try:
            game_def, _, _ = load_game_graph(hand.variant_name)
        except FileNotFoundError:
            raise ValueError(
                f"Hand {hand_id}'s variant {hand.variant_name!r} no longer "
                f"exists — cannot resume into a graph that isn't loadable."
            )

        start_req = StartConstructionRequest(
            game_name=hand.variant_name,
            dealer_seat=hand.dealer_seat or 1,
            seats={seat: names_by_seat.get(seat, f"Seat {seat}") for seat in seats},
            initial_stacks={seat: initial_stacks.get(seat, 0) for seat in seats},
        )
        step = self.start(start_req)
        session_id = step.session_id

        try:
            while step.domain != "COMPLETE":
                if step.domain == "DEAL_HOLE":
                    remaining = hole_cards_by_seat.get(step.seat, [])
                    if len(remaining) < (step.count or 0):
                        raise ValueError(
                            f"Hand {hand_id}: not enough persisted hole cards "
                            f"for seat {step.seat} to resume (need "
                            f"{step.count}, have {len(remaining)})."
                        )
                    chosen, rest = remaining[: step.count], remaining[step.count :]
                    hole_cards_by_seat[step.seat] = rest
                    response = ConstructionStepResponse(
                        cards=[_card_str(c) for c in chosen]
                    )

                elif step.domain == "DEAL_BOARD":
                    chosen = []
                    for n in step.board_node_indices:
                        cid = board_cards_by_node.get(n)
                        if cid is None:
                            raise ValueError(
                                f"Hand {hand_id}: no persisted board card for "
                                f"node {n} to resume this deal step."
                            )
                        chosen.append(cid)
                    response = ConstructionStepResponse(
                        cards=[_card_str(c) for c in chosen]
                    )

                elif step.domain == "BETTING":
                    action = self._pop_next_action(pending_actions, step.seat, hand_id)
                    response = ConstructionStepResponse(
                        type=(action.action_type or "").lower(), amount=action.amount
                    )

                elif step.domain == "BOOLEAN":
                    action = self._pop_next_action(pending_actions, step.seat, hand_id)
                    response = ConstructionStepResponse(bool_value=bool(action.amount))

                elif step.domain in ("CARD_SELECT", "CARD_PASS", "CHOICE"):
                    raise ValueError(
                        f"Hand {hand_id} cannot be resumed: it contains a "
                        f"{step.domain} decision at node {step.node_id!r}, and "
                        f"saved hypothetical hands don't retain enough data "
                        f"to replay that domain (see resume_from_hand()'s "
                        f"own docstring)."
                    )

                else:
                    raise ValueError(
                        f"Hand {hand_id}: unexpected step domain "
                        f"{step.domain!r} encountered during resume."
                    )

                step = self.apply_step(session_id, response)
        except Exception:
            self.discard(session_id)
            raise

        return step

    @staticmethod
    def _pop_next_action(queue: list, expected_seat: int, hand_id: int):
        """
        Pop the next persisted Action row off `queue` (FIFO, already
        ordered by street/action_index) for a BETTING or BOOLEAN replay
        step. Raises ValueError — rather than silently misattributing a
        different seat's action — if the queue is exhausted (persisted
        data ran out before the graph reached COMPLETE) or the popped
        row's seat doesn't match what the graph is actually waiting on
        (a real desync between the saved data and this variant's
        current graph shape, e.g. the variant's flow changed since the
        hand was saved).
        """
        if not queue:
            raise ValueError(
                f"Hand {hand_id}: ran out of persisted actions while the "
                f"graph still expects a decision from seat {expected_seat} "
                f"— the saved hand may be incomplete."
            )
        action = queue.pop(0)
        seat = -action.player_id if action.player_id < 0 else action.player_id
        if seat != expected_seat:
            raise ValueError(
                f"Hand {hand_id}: next persisted action belongs to seat "
                f"{seat}, but the graph is waiting on seat {expected_seat} "
                f"— saved data doesn't match this variant's current graph "
                f"shape."
            )
        return action

    def get_state(self, session_id: str) -> ConstructionStepDTO:
        session = self._require_session(session_id)
        if session.current_step is None:
            return self._refresh_and_get_current_step(session)
        return session.current_step

    def discard(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------

    def apply_step(
        self, session_id: str, response: ConstructionStepResponse
    ) -> ConstructionStepDTO:
        session = self._require_session(session_id)
        if session.busy:
            raise ValueError(
                "A previous step is still being processed for this session — "
                "please retry in a moment."
            )
        session.busy = True
        try:
            step = session.current_step
            if step is None:
                raise ValueError(
                    "No pending step for this session — call GET /creator/{id}/state first."
                )

            if step.domain in ("DEAL_HOLE", "DEAL_BOARD"):
                self._apply_deal_review(session, step, response)
            elif step.domain == "AUTO":
                # Non-deal AUTO nodes never actually reach apply_step()
                # in practice (they're executed and skipped over
                # silently by the engine's own forward progress, never
                # queued) — kept only for forward-compat visibility.
                pass
            elif step.domain == "COMPLETE":
                raise ValueError(
                    "This construction session has reached the end of its hand — "
                    "call /finish instead of /step."
                )
            else:
                self._apply_decision(session, step, response)

            return self._refresh_and_get_current_step(session)
        finally:
            session.busy = False

    def undo(self, session_id: str) -> ConstructionStepDTO:
        """
        Drop the last recorded step and rebuild the session from
        scratch by replaying everything before it — see module
        docstring point 4. Cheaper and far less risky than a true
        engine-level snapshot/restore mechanism (which doesn't exist),
        since this only ever rebuilds a private, uncommitted session.
        """
        session = self._require_session(session_id)
        if session.busy:
            raise ValueError(
                "A previous step is still being processed for this session — "
                "please retry in a moment."
            )
        if not session.history:
            raise ValueError("Nothing to undo — this session has no recorded steps yet.")

        session.busy = True
        try:
            history = session.history[:-1]
            rebuilt = self._rebuild(session, history)
        finally:
            session.busy = False
        self._sessions[session_id] = rebuilt
        return self._refresh_and_get_current_step(rebuilt)

    # ------------------------------------------------------------
    # Finish — persist via tutorial_api's existing save path
    # ------------------------------------------------------------

    def finish(self, session_id: str) -> dict[str, Any]:
        """
        Translate this session's recorded history into a payload
        shaped like tutorial_api.SaveHypotheticalHandRequest, ready to
        be handed to POST /tutorial/hands by the router (see
        construction_api.py) — reuses that existing save/showdown-
        evaluation path rather than writing DB rows directly here, so
        a constructed hand round-trips through exactly the same code a
        hand saved by the OLD creation_phases wizard already used.

        KNOWN GAP: CARD_SELECT / CARD_PASS decisions are recorded into
        `actions` with action_type="CARD_SELECT"/"CARD_PASS" and
        amount=None — the actual selected card ids are NOT separately
        represented, since SaveHypotheticalHandRequest has no card-
        movement-ledger concept beyond initial hole_cards/board_cards.
        A constructed drawmaha/pass-the-trash hand will save and
        replay its FINAL hole cards correctly (they're captured as
        plain hole_cards, same as any other variant), but the
        Replayer's HoleCardEventDTO timeline (real CardEvent rows) will
        be incomplete for those mid-hand movements specifically.
        Fixing this needs either extending SaveHypotheticalHandRequest
        with an explicit card-movement list, or this service writing
        Hand/CardEvent rows directly (via card_movement.py) instead of
        going through tutorial_api's save endpoint. Flagged rather
        than silently dropped — revisit once a real drawmaha/pass-the-
        trash hand needs saving from the Creator.
        """
        session = self._require_session(session_id)
        g = _game_obj(session.engine)

        players = []
        hole_cards = []
        for i, seat in enumerate(session.seats):
            cards = [_card_str(c) for c in mask_to_card_ids(g.players[i].hand_mask)]
            players.append(
                {
                    "seat": seat,
                    "name": session.seat_names.get(seat) or f"Seat {seat}",
                    "stack": session.initial_stacks.get(seat, 0),
                    "hole_cards": cards,
                }
            )
            hole_cards.append({"player_seat": seat, "cards": cards})

        node_cards = [
            (_card_str(c) if c is not None else None) for c in g.node_cards
        ]

        board_cards = [
            {"node": n, "card": _card_str(cid), "street": entry["street"]}
            for entry in session.history
            if entry["kind"] == "deal" and entry["domain"] == "DEAL_BOARD"
            for n, cid in zip(entry["node"], entry["cards"])
        ]

        actions = [
            {
                "street": entry["street"],
                "action_index": idx,
                "player_seat": entry["player_seat"],
                "action_type": entry["action_type"],
                "amount": entry["amount"],
                "stack_before": entry["stack_before"],
                "pot_before": entry["pot_before"],
            }
            for idx, entry in enumerate(
                e for e in session.history if e["kind"] == "decision"
            )
        ]

        return {
            "game_name": session.game_name,
            "variant_name": getattr(session.game_def, "game_name", session.game_name),
            "layout_name": getattr(session.game_def, "layout_name", "single_board"),
            "dealer_seat": session.dealer_seat,
            "pot": g.pot,
            "seats": {seat: session.seat_names.get(seat, f"Seat {seat}") for seat in session.seats},
            "initial_stacks": dict(session.initial_stacks),
            "players": players,
            "hole_cards": hole_cards,
            "node_cards": node_cards,
            "board_cards": board_cards,
            "discard_pile": [_card_str(c) for c in getattr(g, "discard_pile", [])],
            "actions": actions,
        }

    # ------------------------------------------------------------
    # Deal review / override
    # ------------------------------------------------------------

    def _apply_deal_review(
        self,
        session: _ConstructionSession,
        step: ConstructionStepDTO,
        response: ConstructionStepResponse,
    ) -> None:
        """
        Accept the already-dealt card(s) as-is (response.cards is None
        or empty) or override them with author-chosen replacements —
        either way, engine state ends up holding exactly the FINAL
        cards for this node/seat, and that final value (not the
        original random draw) is what gets recorded into history for
        replay/undo determinism.
        """
        g = _game_obj(session.engine)
        seat_idx = (step.seat - 1) if step.seat is not None else None

        if response.cards:
            if len(response.cards) != step.count:
                raise ValueError(
                    f"Expected {step.count} card(s) for this step, got "
                    f"{len(response.cards)}."
                )
            new_ids = [CardObj.from_str(c).id for c in response.cards]
            in_use = self._cards_in_use(session, exclude_seat=seat_idx)
            dupes = [c for c in new_ids if c in in_use]
            if dupes:
                raise ValueError(
                    f"Card(s) already in use elsewhere in this hand: "
                    f"{[_card_str(c) for c in dupes]}"
                )
        else:
            # Accept whatever the engine already dealt — read it back
            # off current state rather than trusting step.eligible_cards
            # (which lists what's NOT used, not what WAS just dealt).
            if step.domain == "DEAL_HOLE":
                new_ids = mask_to_card_ids(g.players[seat_idx].hand_mask)
            else:
                new_ids = [g.node_cards[n] for n in step.board_node_indices]

        if step.domain == "DEAL_HOLE":
            self._force_hole_cards(session, seat_idx, new_ids)
            session.logged_hole_cards.setdefault(seat_idx, set()).update(new_ids)
            session.history.append(
                {
                    "kind": "deal",
                    "domain": "DEAL_HOLE",
                    "seat": seat_idx,
                    "node": None,
                    "cards": new_ids,
                    "street": session.board_reveal_index,
                }
            )
        else:
            self._force_board_cards(session, step.board_node_indices, new_ids)
            session.logged_board_nodes.update(step.board_node_indices)
            session.history.append(
                {
                    "kind": "deal",
                    "domain": "DEAL_BOARD",
                    "seat": None,
                    "node": step.board_node_indices,
                    "cards": new_ids,
                    "street": session.board_reveal_index,
                }
            )

        session.pending_queue.pop(0)

    def _force_hole_cards(self, session: _ConstructionSession, seat_idx: int, card_ids: list[int]) -> None:
        """
        Direct in-place mutation — sets this player's hand_mask to
        exactly card_ids. Safe here (unlike a live/persisted hand)
        because nothing has been written to the DB yet; card_movement.py's
        CardEvent-ledger bookkeeping is applied only once, at
        finish()-time, via tutorial_api's own save path.
        """
        g = _game_obj(session.engine)
        mask = 0
        for cid in card_ids:
            mask |= 1 << cid
        g.players[seat_idx].hand_mask = mask

    def _force_board_cards(self, session: _ConstructionSession, node_indices: list[int], card_ids: list[int]) -> None:
        g = _game_obj(session.engine)
        for node, cid in zip(node_indices, card_ids):
            g.node_cards[node] = cid

    def _cards_in_use(self, session: _ConstructionSession, exclude_seat: Optional[int] = None) -> set[int]:
        g = _game_obj(session.engine)
        used: set[int] = set()
        for i, p in enumerate(g.players):
            if i == exclude_seat:
                continue
            used.update(mask_to_card_ids(p.hand_mask))
        used.update(c for c in g.node_cards if c is not None)
        return used

    # ------------------------------------------------------------
    # Decision stepping — zero new Engine capability, reuses
    # game_service._build_decision_response() and submit_decision()
    # exactly as live play does.
    # ------------------------------------------------------------

    def _apply_decision(
        self,
        session: _ConstructionSession,
        step: ConstructionStepDTO,
        response: ConstructionStepResponse,
    ) -> None:
        pending = session.engine.pending_request
        if pending is None:
            raise ValueError("No decision is currently pending for this session.")

        g = _game_obj(session.engine)
        player_index = getattr(pending, "player_index", None)
        pot_before = g.pot
        stack_before = (
            g.players[player_index].stack
            if player_index is not None and 0 <= player_index < len(g.players)
            else None
        )

        # ConstructionStepResponse is field-compatible with ActionRequest
        # (type/amount/selected_cards/bool_value/choice) by design — see
        # construction_dto.py's own docstring — so the existing builders
        # can be reused completely unmodified.
        decision_response, _player_index = _build_decision_response(session.engine, response)
        submit_decision(session.engine, decision_response, callbacks=None)

        action_type, amount = _action_type_and_amount(step.domain, response)

        session.history.append(
            {
                "kind": "decision",
                "domain": step.domain,
                "response": response.model_dump(),
                "street": session.board_reveal_index,
                "player_seat": (player_index + 1) if player_index is not None else None,
                "action_type": action_type,
                "amount": amount,
                "pot_before": pot_before,
                "stack_before": stack_before,
            }
        )

    # ------------------------------------------------------------
    # Diffing — turns "what changed in engine state" into the next
    # queued review step(s).
    # ------------------------------------------------------------

    def _refresh_and_get_current_step(self, session: _ConstructionSession) -> ConstructionStepDTO:
        if not session.pending_queue:
            self._diff_new_deals(session)

        if session.pending_queue:
            step = session.pending_queue[0]
        elif session.engine.pending_request is not None:
            step = self._decision_step(session)
        elif getattr(session.engine, "is_complete", False):
            step = self._complete_step(session)
        else:
            raise RuntimeError(
                "Construction session is in an unexpected state: no pending "
                "decision, no newly dealt cards, and the hand is not complete."
            )

        session.current_step = step
        return step

    def _diff_new_deals(self, session: _ConstructionSession) -> None:
        g = _game_obj(session.engine)

        for seat_idx in range(len(g.players)):
            current = set(mask_to_card_ids(g.players[seat_idx].hand_mask))
            logged = session.logged_hole_cards.get(seat_idx, set())
            new_cards = current - logged
            if new_cards:
                sorted_new = sorted(new_cards)
                session.pending_queue.append(
                    ConstructionStepDTO(
                        session_id=session.session_id,
                        node_id=f"hole:{seat_idx}",
                        node_label=f"Deal hole cards — Seat {seat_idx + 1}",
                        domain="DEAL_HOLE",
                        seat=seat_idx + 1,
                        count=len(new_cards),
                        dealt_cards=[_card_str(c) for c in sorted_new],
                        eligible_cards=self._eligible_cards_display(session, exclude_seat=seat_idx),
                        state=self._state_dto(session),
                    )
                )

        new_nodes = [
            n for n, c in enumerate(g.node_cards)
            if c is not None and n not in session.logged_board_nodes
        ]
        if new_nodes:
            # Increment the ONE reveal-order counter — mirrors
            # SessionLogger.log_board()'s own "increment exactly when
            # new board cards are first observed" rule, so a
            # constructed hand's street numbering matches a live
            # hand's numbering convention exactly.
            session.board_reveal_index += 1
            session.pending_queue.append(
                ConstructionStepDTO(
                    session_id=session.session_id,
                    node_id=f"board:{new_nodes[0]}-{new_nodes[-1]}",
                    node_label=f"Deal board cards — reveal {session.board_reveal_index}",
                    domain="DEAL_BOARD",
                    seat=None,
                    count=len(new_nodes),
                    board_node_indices=new_nodes,
                    dealt_cards=[_card_str(g.node_cards[n]) for n in new_nodes],
                    eligible_cards=self._eligible_cards_display(session),
                    state=self._state_dto(session),
                )
            )

    def _eligible_cards_display(self, session: _ConstructionSession, exclude_seat: Optional[int] = None) -> list[str]:
        used = self._cards_in_use(session, exclude_seat=exclude_seat)
        return [_card_str(c) for c in ALL_CARD_IDS if c not in used]

    def _decision_step(self, session: _ConstructionSession) -> ConstructionStepDTO:
        engine = session.engine
        req = engine.pending_request
        g = _game_obj(engine)
        domain_name = _domain_name(getattr(req, "domain", None)) or "UNKNOWN"
        player_index = getattr(req, "player_index", None)

        options = _options_to_dto(g, req)
        to_call = min_raise = max_raise = None
        if domain_name == "BETTING":
            to_call, min_raise, max_raise = _betting_to_call_and_raise(g, req)

        return ConstructionStepDTO(
            session_id=session.session_id,
            node_id=str(getattr(req, "node_id", "")),
            node_label=_decision_node_label(engine, session.graph, domain_name),
            domain=domain_name,
            seat=(player_index + 1) if player_index is not None else None,
            options=options,
            to_call=to_call,
            min_raise=min_raise,
            max_raise=max_raise,
            node_metadata=_node_metadata(engine, session.graph),
            state=self._state_dto(session),
        )

    def _complete_step(self, session: _ConstructionSession) -> ConstructionStepDTO:
        return ConstructionStepDTO(
            session_id=session.session_id,
            node_id="COMPLETE",
            node_label="Hand complete",
            domain="COMPLETE",
            state=self._state_dto(session),
        )

    def _state_dto(self, session: _ConstructionSession):
        return graph_state_to_dto(
            session.engine, session.game_def, session.rules, graph=session.graph
        )

    # ------------------------------------------------------------
    # Undo support
    # ------------------------------------------------------------

    def _rebuild(self, original: _ConstructionSession, history: list[dict]) -> _ConstructionSession:
        """
        Rebuild a session from scratch and replay `history` — the SAME
        _force_hole_cards/_force_board_cards mutation primitives the
        interactive override path uses, just applied unconditionally
        to every recorded deal rather than only author-changed ones.
        Determinism comes entirely from replaying recorded FINAL
        values, never from re-drawing randomly. board_reveal_index is
        re-derived identically (not copied) since _diff_new_deals is
        called the same way it was originally, against the same graph
        structure and seat count.
        """
        game_def, rules, graph = load_game_graph(original.game_name)
        players = [PlayerState(stack=p.stack) for p in _game_obj(original.engine).players]
        hand_state = new_hand_state(players, game_def, dealer_position=original.dealer_seat - 1)
        deck = GraphDeck()
        showdown_resolver = ShowdownResolver(CppScoringEngine(), rules)
        engine = GraphEngine(graph, hand_state, deck, showdown_resolver=showdown_resolver)
        engine.game_def = game_def

        session = _ConstructionSession(
            session_id=original.session_id,
            game_name=original.game_name,
            game_def=game_def,
            rules=rules,
            graph=graph,
            engine=engine,
            seats=original.seats,
            dealer_seat=original.dealer_seat,
            initial_stacks=dict(original.initial_stacks),
            seat_names=dict(original.seat_names),
        )

        start_new_hand(engine, game_def, callbacks=None, advance_dealer=False)

        for entry in history:
            if entry["kind"] == "deal":
                # Only diff when the queue is empty, and pop exactly one
                # matching item per entry — mirrors _refresh_and_get_current_step's
                # own guard and _apply_deal_review's single pop() exactly.
                # A previous version cleared the WHOLE queue per entry,
                # which double-incremented board_reveal_index whenever
                # hole and board cards were both dealt in the same
                # initial AUTO batch (some bomb-pot-style variants) —
                # entry N's diff would re-detect entry N+1's still-
                # unlogged domain and increment the counter again.
                if not session.pending_queue:
                    self._diff_new_deals(session)
                if entry["domain"] == "DEAL_HOLE":
                    self._force_hole_cards(session, entry["seat"], entry["cards"])
                    session.logged_hole_cards.setdefault(entry["seat"], set()).update(entry["cards"])
                else:
                    self._force_board_cards(session, entry["node"], entry["cards"])
                    session.logged_board_nodes.update(entry["node"])
                if session.pending_queue:
                    session.pending_queue.pop(0)
                session.history.append(entry)
            else:
                response = ConstructionStepResponse(**entry["response"])
                decision_response, _ = _build_decision_response(engine, response)
                submit_decision(engine, decision_response, callbacks=None)
                session.history.append(entry)

        return session

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _require_session(self, session_id: str) -> _ConstructionSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"No construction session with id {session_id!r}.")
        return session


hand_construction_service = HandConstructionService()