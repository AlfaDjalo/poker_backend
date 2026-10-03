"""
app/services/game_service.py — GameService, running entirely on
GraphEngine. Rewritten against the confirmed migration doc ("Backend
migration: PokerState -> GraphEngine").

Import paths (previous revision guessed several of these wrong —
corrected here per the doc's "Load"/"Run a hand" code blocks):
    from poker_engine.rules.loader import load_game as load_game_graph
    from poker_engine.graph.graph_engine import GraphEngine
    from poker_engine.graph.hand_session import new_hand_state, start_new_hand
    from poker_engine.graph.callbacks import submit_decision
    from poker_engine.graph.core_types import DecisionResponse
    from poker_engine.showdown.showdown_resolver import ShowdownResolver
    from poker_engine.scoring.scoring_engine import CppScoringEngine
    from poker_engine.cards.deck import Deck

Dealer rotation
------------------
new_hand_state(players, game_def, dealer_position=0) takes an explicit
starting dealer_position; start_new_hand(...) does the per-hand
rotation itself UNLESS advance_dealer=False is passed — the doc notes
"advance_dealer=False for hand 1 if dealer_position is already set"
(otherwise hand 1 would rotate away from the dealer_position we just
explicitly set). self._dealer_position tracks the last known position
(read back off engine state after every start_new_hand call) so a
variant switch mid-session (which rebuilds hand_state from scratch)
can hand the new hand_state a sensible starting position instead of
silently resetting to seat 0.

Decision construction
------------------------
Per the doc's "Run a hand" loop:
    req = engine.pending_request
    response = DecisionResponse(req.node_id, req.domain, req.player_index, value)
    submit_decision(engine, response, callbacks=cb)

`value`'s exact shape is still unconfirmed — _build_decision_response()
below constructs it as {"action_type": ..., "amount": ...} and matches
it against `req.options` by action_name, since that's the minimum any
DecisionResponse consumer would need to identify which legal option
was chosen. Flagged in that function's own docstring; update it there
if the real `value` shape turns out to be the option object itself (or
something else) rather than a plain dict.

Hand Editor is NOT carried forward — see the previous revision's
docstring for why (still true: no GraphEngine-native snapshot/restore
mechanism exists per the doc's "Behavioral differences" section).

get_variant_config()'s flow-walking (_summarize_flow) no longer trusts a
graph node's own metadata as the authoritative street number — it now
derives street/reveal-group numbering itself, the same way
SessionLogger derives it for live hands (see _summarize_flow's own
module-level note). The remaining graph.node(...) metadata KEY NAMES
(deals_hole, deals_board_street-as-presence-signal, street_name,
node_indices) are still pending confirmation against real graph-loader
source — see that function's own ASSUMPTIONS block.
"""

from importlib import resources
from pathlib import Path

from poker_engine.scoring.scoring_engine import CppScoringEngine
from poker_engine.state.player_state import PlayerState
from poker_engine.cards.deck import Deck as GraphDeck

from poker_engine.rules.loader import load_game as load_game_graph
from poker_engine.graph.graph_engine import GraphEngine
from poker_engine.graph.hand_session import new_hand_state, start_new_hand
from poker_engine.graph.callbacks import submit_decision
from poker_engine.graph.core_types import DecisionResponse
from poker_engine.showdown.showdown_resolver import ShowdownResolver
from poker_engine.graph.graph_hole_cards import first_deal_hole_cards_count

from app.db.models.players import Player
from app.db.models.poker_tables import PokerTable
from poker_engine.actions.action import Action as EngineAction

from app.graph_engine_adapter import (
    graph_state_to_dto,
    _game_obj,
    _option_action_name,
    _option_action_type_enum,
)
from app.services.graph_engine_callbacks import BackendGraphEngineCallbacks
from app.services.session_logger import SessionLogger

DEFAULT_GAME = "holdem"


def _apply_sitting_out(players) -> None:
    """
    Mark any zero-stack player as sitting out for the next hand, per
    the Engine team's wiring guidance: set PlayerState.sitting_out
    before constructing/reusing the player list, then call
    start_new_hand() as normal — no index shifting, no other engine
    call site needs to change (the Engine itself skips a sitting_out
    player for dealing/action/pass-routing).

    Works identically whether `players` is a freshly-constructed
    PlayerState list (see new_hand()'s variant-switch branch) or the
    Engine's OWN already-live player objects being reused in place
    (new_hand()'s same-variant branch, where no new PlayerState list
    is ever built at all — see that branch's own comment for why
    mutating the existing objects directly is required there).

    Only a stack<=0 check for now — there's no manual "sit this player
    out" control anywhere in this API yet; if/when one exists, it
    should set p.sitting_out = True itself and this function should
    stop clobbering an already-True value for a player who still has
    chips but chose to sit out (currently harmless since every caller
    of this function is about to (re)deal a brand new hand, so there's
    no prior manual sit-out state to preserve yet).
    """
    for p in players:
        p.sitting_out = getattr(p, "stack", 0) <= 0


class GameService:

    def __init__(self):
        self.engine = None
        self.graph = None
        self.game_def = None
        self.rules = None
        self.logger = None
        self.graph_callbacks = None

        self.current_game = DEFAULT_GAME
        self.pending_game = None

        # Last known dealer_position, read back off engine state after
        # every start_new_hand() call — see module docstring.
        self._dealer_position = 0

        # Hand Editor state — kept as attributes so edit_api.py's
        # existing shape doesn't need to change even though the
        # methods below just raise NotImplementedError.
        self.editing_mode = False
        self.pre_edit_snapshot = None

        # Reentrancy guard — apply_action() mutates self.engine in place
        # with no locking (submit_decision runs synchronously against
        # the shared engine object). A second request landing while the
        # first is still being processed — a double-click, a retried
        # request after a slow response, or (per the ESG crash report)
        # the frontend firing twice off the extra-card AUTO node's rapid
        # state update — would otherwise find engine.pending_request
        # already None/changed and surface as the confusing "No
        # decision is currently pending." ValueError instead of a
        # clear "still processing, try again" signal. Mirrors
        # TrainerService._busy exactly — see that class's own docstring
        # for the same rationale.
        self._busy = False

    # --------------------------------------------------
    # Variant discovery
    # --------------------------------------------------

    def get_variants(self):
        """Return all .yaml game definitions found in the engine's games directory."""
        games_dir = Path(str(resources.files("poker_engine") / "games"))

        if not games_dir.exists():
            return {"variants": [], "current": self.current_game}

        names = sorted(p.stem for p in games_dir.glob("*.yaml"))
        return {"variants": names, "current": self.current_game}

    def get_variant_config(self, game_name: str):
        """
        Replaces the old board_layout/betting/creation_phases YAML
        projection with an equivalent derived from the real graph —
        see _summarize_flow()'s ASSUMPTIONS. Returns None if the
        variant doesn't exist / fails to load.
        """
        try:
            game_def, rules, graph = load_game_graph(game_name)
        except FileNotFoundError:
            return None

        return _summarize_flow(game_def, rules, graph)

    # --------------------------------------------------
    # Game selection (queued; applied between hands only)
    # --------------------------------------------------

    def select_game(self, game_name: str):
        games_dir = Path(str(resources.files("poker_engine") / "games"))
        if not (games_dir / f"{game_name}.yaml").exists():
            raise ValueError(f"Unknown game variant: '{game_name}'")
        self.pending_game = game_name

    def _apply_pending_game(self, game_name: str | None = None):
        if game_name:
            self.select_game(game_name)
        if self.pending_game:
            self.current_game = self.pending_game
            self.pending_game = None
        return self.current_game

    # --------------------------------------------------
    # Session restart
    # --------------------------------------------------

    def restart(self, db, game_name: str | None = None):
        self._apply_pending_game(game_name)

        active_table = db.query(PokerTable).first()
        if not active_table:
            raise Exception("No poker tables found in database.")

        db_players = (
            db.query(Player)
            .filter(Player.username.in_([f"Player {i}" for i in range(1, 7)]))
            .order_by(Player.username)
            .all()
        )
        if len(db_players) < 6:
            raise Exception(f"Found only {len(db_players)} players.")

        players = [PlayerState(stack=100) for _ in db_players]
        _apply_sitting_out(players)

        game_def, rules, graph = load_game_graph(self.current_game)

        self.logger = SessionLogger(db)
        self.graph_callbacks = BackendGraphEngineCallbacks(
            self.logger, game_service_ref=self
        )

        self._dealer_position = 0
        hand_state = new_hand_state(players, game_def, dealer_position=self._dealer_position)
        deck = GraphDeck()
        showdown_resolver = ShowdownResolver(CppScoringEngine(), rules)

        self.engine = GraphEngine(graph, hand_state, deck, showdown_resolver=showdown_resolver)
        self.engine.game_def = game_def   # <-- add this
        self.graph = graph
        self.game_def = game_def
        self.rules = rules

        # self.engine = GraphEngine(graph, hand_state, deck, showdown_resolver=showdown_resolver)
        # self.graph = graph
        # self.game_def = game_def
        # self.rules = rules

        self.logger.start_game(
            {
                "table_id": active_table.table_id,
                "player_ids": [p.player_id for p in db_players],
            }
        )

        # First hand: dealer_position was just explicitly set above,
        # so don't let start_new_hand rotate it again — see module
        # docstring.
        start_new_hand(
            self.engine, game_def, callbacks=self.graph_callbacks, advance_dealer=False
        )
        self._sync_dealer_position()

        return graph_state_to_dto(self.engine, game_def, rules, graph=self.graph)

    def _sync_dealer_position(self):
        try:
            self._dealer_position = _game_obj(self.engine).dealer_position
        except Exception:
            pass

    # --------------------------------------------------
    # State
    # --------------------------------------------------

    def get_state(self):
        if self.engine is None:
            return None
        return graph_state_to_dto(self.engine, self.game_def, self.rules, graph=self.graph)

    # --------------------------------------------------
    # Player actions
    # --------------------------------------------------

    def apply_action(self, req):
        if self.engine is None:
            return None

        if self._busy:
            raise ValueError(
                "A previous action is still being processed for this hand — "
                "please retry in a moment."
            )
        self._busy = True
        try:
            return self._apply_action_locked(req)
        finally:
            self._busy = False

    def _apply_action_locked(self, req):
        print("Req: ", req)

        # Diagnostic: confirm exactly what the engine thinks is pending
        # BEFORE we try to build a response against it — if this ever
        # prints "pending=None" right before the ValueError below, that
        # proves the engine had already resolved/advanced past the
        # decision by the time this request was processed (a stale/
        # duplicate submission), rather than something wrong in how the
        # response itself is built.
        pending = self.engine.pending_request
        is_complete = getattr(self.engine, "is_complete", None)
        current_node = getattr(getattr(self.engine, "state", None), "current_node", None)
        print(
            f"[apply_action] pending_request={pending!r} "
            f"domain={getattr(pending, 'domain', None)!r} "
            f"player_index={getattr(pending, 'player_index', None)!r} "
            f"is_complete={is_complete!r} current_node={current_node!r} "
            f"last_showdown={getattr(self.engine, 'last_showdown', None)!r} "
            f"last_winners={getattr(self.engine, 'last_winners', None)!r}"
        )

        response, player_index = _build_decision_response(self.engine, req)
        print(f"Response: {req} player_index: {player_index} graph_callbacks: {self.graph_callbacks}")

        if self.graph_callbacks is not None and player_index is not None:
            self.graph_callbacks.note_pre_decision(self.engine, player_index)

        try:
            submit_decision(self.engine, response, callbacks=self.graph_callbacks)
        except ValueError:
            raise

        return graph_state_to_dto(self.engine, self.game_def, self.rules, graph=self.graph)

    def advance_street(self):
        """GraphEngine advances streets internally via the graph itself
        — no caller-driven pump loop needed. Kept as a no-op
        passthrough only so game_api.py's existing route doesn't need
        touching."""
        return self.get_state()

    # --------------------------------------------------
    # New hand (within the same session)
    # --------------------------------------------------

    def new_hand(self, game_name: str | None = None):
        """
        Start a new hand. If game_name is supplied (or a pending_game is
        queued) the variant is switched before dealing.
        """
        self._apply_pending_game(game_name)

        if self.engine is None:
            raise RuntimeError("No active game session. Call /game/restart first.")

        game_def, rules, graph = load_game_graph(self.current_game)

        if self.game_def is None or self.game_def.game_name != game_def.game_name:
            players = [PlayerState(stack=p.stack) for p in _game_obj(self.engine).players]
            _apply_sitting_out(players)
            hand_state = new_hand_state(
                players, game_def, dealer_position=self._dealer_position
            )
            deck = GraphDeck()
            showdown_resolver = ShowdownResolver(CppScoringEngine(), rules)
            self.engine = GraphEngine(
                graph, hand_state, deck, showdown_resolver=showdown_resolver
            )
            self.engine.game_def = game_def   # <-- add this
            self.graph = graph
            # self.engine = GraphEngine(
            #     graph, hand_state, deck, showdown_resolver=showdown_resolver
            # )
            # self.graph = graph
        else:
            # Same variant, continuing session: no new PlayerState list
            # is ever constructed on this path — start_new_hand() deals
            # directly into the Engine's existing live player objects.
            # sitting_out has to be set on those SAME objects, in
            # place, or a player who busted on the previous hand would
            # never be marked out for this one.
            _apply_sitting_out(_game_obj(self.engine).players)

        self.game_def = game_def
        self.rules = rules

        # Continuing hand (same or switched variant, same session) —
        # let the button rotate normally.
        start_new_hand(self.engine, game_def, callbacks=self.graph_callbacks)
        self._sync_dealer_position()

        return graph_state_to_dto(self.engine, game_def, rules, graph=self.graph)

    # --------------------------------------------------
    # Hand Editor — not carried forward, see module docstring.
    # --------------------------------------------------

    # def begin_edit(self, db):
    #     raise NotImplementedError(
    #         "Hand editing has no GraphEngine-native snapshot/restore "
    #         "mechanism yet — see game_service.py's module docstring."
    #     )

    # def apply_edit(self, req):
    #     raise NotImplementedError(
    #         "Hand editing has no GraphEngine-native snapshot/restore "
    #         "mechanism yet — see game_service.py's module docstring."
    #     )

    # def load_edit(self, req):
    #     raise NotImplementedError(
    #         "Hand editing has no GraphEngine-native snapshot/restore "
    #         "mechanism yet — see game_service.py's module docstring."
    #     )

    # def cancel_edit(self):
    #     raise NotImplementedError(
    #         "Hand editing has no GraphEngine-native snapshot/restore "
    #         "mechanism yet — see game_service.py's module docstring."
    #     )


# --------------------------------------------------------------------
# Decision construction — see module docstring's "Decision
# construction" section.
# --------------------------------------------------------------------


def _build_decision_response(engine, req_body):
    """
    Build a DecisionResponse(node_id, domain, player_index, value) for
    the pending decision, dispatching on pending.domain — see each
    per-domain builder below for its own value-shape docstring. Only
    BETTING was previously supported (see module docstring's original
    note on this function); CARD_SELECT / CARD_PASS / BOOLEAN / CHOICE
    are new — added so non-betting variants (Drawmaha-style
    discard/redraw, pass-the-trash, Grinch-style yes/no decisions) can
    actually reach the engine, now that ActionRequest (game_api.py)
    carries the extra fields each of those needs.

    Returns (response, player_index) in every case — player_index is
    handed back separately so the caller can pass it to
    BackendGraphEngineCallbacks.note_pre_decision() without re-reading
    pending_request itself.
    """
    pending = engine.pending_request
    if pending is None:
        raise ValueError("No decision is currently pending.")

    domain_name = getattr(pending.domain, "name", str(pending.domain))

    if domain_name == "BETTING":
        return _build_betting_decision_response(engine, pending, req_body)
    if domain_name in ("CARD_SELECT", "CARD_PASS"):
        return _build_card_select_decision_response(engine, pending, req_body)
    if domain_name == "BOOLEAN":
        return _build_boolean_decision_response(engine, pending, req_body)
    if domain_name == "CHOICE":
        return _build_choice_decision_response(engine, pending, req_body)

    raise ValueError(
        f"Unsupported decision domain: {domain_name!r} — no response "
        f"builder wired up for it in game_service.py."
    )


def _build_betting_decision_response(engine, pending, req_body):
    if not req_body.type:
        raise ValueError(
            "BETTING decision requires 'type' (fold/check/call/bet/raise/all_in)."
        )
    action_type = req_body.type.lower()
    amount = req_body.amount

    # The engine names the SAME bet-sizing button "bet" when opening the
    # street (nothing owed yet) and "raise" when facing a bet already
    # (e.g. SB opening over a posted BB is legally a raise, not a bet) —
    # see betting_rules.py's boundary-action semantics (§2.3). The
    # frontend's sizing control has no reliable way to know which one
    # the engine will call it this decision (PlayerActionPanel.handleBet
    # picks the DISPLAY label off player.bet, a client-side heuristic —
    # see the Frontend doc's own note on that quirk) and always submits
    # type: "bet" regardless. Treat "bet"/"raise" as synonyms here so a
    # legal bet-like action is matched by whichever name the engine
    # actually gave it, instead of raising "not among the legal options"
    # any time the two labels disagree.
    _ACTION_SYNONYMS = {"bet": {"bet", "raise"}, "raise": {"bet", "raise"}}
    candidate_names = _ACTION_SYNONYMS.get(action_type, {action_type})

    options = getattr(pending, "options", ()) or ()
    matching = None
    for opt in options:
        name = _option_action_name(opt)
        if name is not None and name in candidate_names:
            matching = opt
            break

    if matching is None:
        legal = [_option_action_name(o) for o in options]
        raise ValueError(
            f"Action {action_type!r} is not among the legal options right now "
            f"(legal: {legal})"
        )

    # BettingResolver.validate() requires response.value to be a real
    # poker_engine.actions.action.Action — a plain {"action_type":...,
    # "amount":...} dict (the previous shape here) raises "response.value
    # must be an Action, got <class 'dict'>" the instant a hero submits
    # anything. Build the real thing, using the SAME enum the matched
    # option itself carries (via _option_action_type_enum) rather than
    # re-deriving it from the lowercase action_type string, so this can
    # never disagree with what _options_to_dto told the frontend was
    # legal in the first place.
    engine_action_type = _option_action_type_enum(matching)
    if engine_action_type is None:
        raise ValueError(
            f"Could not resolve an engine ActionType for option "
            f"{action_type!r} — matched option has no recognizable "
            f"action-identity field (see _option_raw_action_value)."
        )

    # amount semantics mirror Action's own docstring (§2.2): CALL's
    # amount is the amount to call — and BettingResolver.validate()
    # enforces it exactly (legal range collapses to [to_call, to_call],
    # confirmed by "Action amount 0 outside legal range [6, 6] for
    # ActionType.CALL" once a client omits it); BET/RAISE's amount is
    # the TOTAL target bet for the street; FOLD/CHECK ignore amount
    # entirely. The frontend only ever sends an explicit amount for
    # bet-like actions (see PlayerActionPanel.handleBet) — fold/check/
    # call/all_in submit none. Previously this defaulted a missing
    # amount straight to 0, which is correct for fold/check but wrong
    # for call/all_in, whose legal amount is a real, engine-computed
    # value that happens to live on the matched option itself
    # (min_amount == max_amount for these). Fall back to that instead
    # of a bare 0 whenever the client didn't supply one.
    option_min = getattr(matching, "min_amount", None)
    if amount is None:
        amount = option_min if option_min is not None else 0

    value = EngineAction(
        type=engine_action_type,
        amount=amount,
    )
    player_index = getattr(pending, "player_index", None)

    response = DecisionResponse(pending.node_id, pending.domain, player_index, value)
    return response, player_index


def _build_card_select_decision_response(engine, pending, req_body):
    """
    CARD_SELECT (discard/draw, e.g. Drawmaha) and CARD_PASS
    (pass-the-trash) both take `value: Tuple[int, ...]` — the card ids
    the player is choosing from their own hand (§2.9.1 core_types.py:
    "CARD_SELECT/CARD_PASS -> Tuple[int,...]"). CARD_PASS's target
    seat is NOT part of the response — it's resolved automatically by
    CardPassResolver from a fixed config rule ("left"/"right"), so the
    frontend never chooses one; only which cards to send.

    req_body.selected_cards carries the player's chosen cards as
    strings ("Ah", "2c", ...) — the same string format used
    everywhere else in this API (hole_cards, node_cards, etc.).
    Converted to engine card ids via Card.from_str().

    Count/eligibility (min_count/max_count, which zone the cards must
    come from) is NOT re-validated here — CardSelectResolver.apply()
    re-validates the response against a freshly rebuilt DecisionRequest
    before mutating any state (per its own contract, shared by every
    resolver — see graph docs §2.9.6), so an illegal selection is
    rejected by the engine itself with a real error rather than
    silently accepted or re-checked twice.
    """
    from poker_engine.cards.card import Card as CardObj

    if req_body.selected_cards is None:
        raise ValueError(
            "CARD_SELECT/CARD_PASS decision requires 'selected_cards' "
            "(a list of card strings, e.g. ['Ah', '2c'] — pass an empty "
            "list [] to stand pat / select zero cards when the pending "
            "option's min_count is 0)."
        )

    try:
        card_ids = tuple(
            CardObj.from_str(c).id for c in req_body.selected_cards
        )
    except Exception as exc:
        raise ValueError(f"Invalid card string in selected_cards: {exc}")

    player_index = getattr(pending, "player_index", None)
    response = DecisionResponse(pending.node_id, pending.domain, player_index, card_ids)
    return response, player_index


def _build_boolean_decision_response(engine, pending, req_body):
    """BOOLEAN (e.g. Grinch's "Christmas next street?") takes a plain
    bool as value — see core_types.py's DecisionResponse.value table."""
    if req_body.bool_value is None:
        raise ValueError("BOOLEAN decision requires 'bool_value' (true/false).")

    player_index = getattr(pending, "player_index", None)
    response = DecisionResponse(
        pending.node_id, pending.domain, player_index, bool(req_body.bool_value)
    )
    return response, player_index


def _build_choice_decision_response(engine, pending, req_body):
    """
    CHOICE takes the chosen option's action_name as a plain string
    value (see core_types.py's DecisionResponse.value table). Matched
    against pending.options by action_name first, same legality check
    the BETTING builder does, so an illegal/unknown choice is rejected
    here with a clear error rather than reaching ChoiceResolver.apply()
    and failing there with less context.
    """
    if not req_body.choice:
        raise ValueError("CHOICE decision requires 'choice'.")

    options = getattr(pending, "options", ()) or ()
    valid_names = {_option_action_name(o) for o in options}
    if req_body.choice not in valid_names:
        raise ValueError(
            f"Choice {req_body.choice!r} is not among the legal options "
            f"right now (legal: {sorted(n for n in valid_names if n)})"
        )

    player_index = getattr(pending, "player_index", None)
    response = DecisionResponse(
        pending.node_id, pending.domain, player_index, req_body.choice
    )
    return response, player_index


# --------------------------------------------------------------------
# Flow summarization for get_variant_config() — GRAPH-NATIVE.
#
# Previously this trusted a node's own metadata["deals_board_street"]
# VALUE as the authoritative "which street is this" number (a direct
# port of the old fixed flop/turn/river creation_phases YAML shape),
# and callers (tutorial_api.py's Hand Creator) built their own street
# stamps around that same assumption. Two problems with that, both the
# same class of bug session_logger.py already had to fix once for live
# hands (see that module's own "ONE STREET COUNTER" docstring):
#
#   1. It assumes the compiled graph node's metadata dict preserves
#      some flow-YAML-authored street index verbatim — never actually
#      confirmed, and exactly the kind of assumption that turned out
#      wrong for live hands (graph_engine_callbacks.py's now-removed
#      node-street-map helpers had the identical bug).
#   2. Even if present and correct, it's a variant-authored number
#      with flop/turn/river-shaped assumptions baked in — meaningless
#      for a bomb pot, a multi-board/hopscotch layout, or any custom
#      flow with a different number/shape of board reveals.
#
# Fixed the same way live hands were fixed: `deals_board_street` is
# now read only as a PRESENCE signal ("does this node deal board
# cards at all?"), never as a trusted index. The actual street number
# every phase gets — `reveal_group` — is OUR OWN running counter,
# incremented once per board-dealing phase encountered during the
# walk, in walk order. This is identical in spirit to
# SessionLogger._board_reveal_index / current_street(): "which street
# is this" is defined as "how many board reveals have happened so far
# in this walk", nothing else. tutorial_api.py's Hand Creator now
# stamps BoardCard.street / Action.street with this same reveal_group
# value, so a hypothetical hand's street numbering is derived exactly
# the same way a live hand's is, and both replay identically for any
# board layout — standard, bomb pot, multi-board, or custom.
#
# `decision_domain` (the DECISION node's real domain name — BETTING,
# CARD_SELECT, CARD_PASS, BOOLEAN, CHOICE, or None for an AUTO node)
# replaces the old BETTING-only `allows_betting` bool as the primary
# field; `allows_betting` is kept, derived from it, purely for
# backward compatibility with any caller still reading the old name
# (the Hand Creator wizard itself is still BETTING-only today — see
# the CAP_Technical_Quick_Reference's "Frontend needs generic decision
# UI" gap — but callers can now at least see a non-BETTING decision
# is pending instead of it silently reading as allows_betting=False
# with no explanation).
# --------------------------------------------------------------------


from poker_engine.graph.graph_hole_cards import first_deal_hole_cards_count


def _node_deals_hole_cards(node) -> tuple[bool, int]:
    """
    Structural (not metadata-based) check for whether a compiled graph
    node deals hole cards, and how many. Per the Engine team's
    confirmation: metadata never carries deals_hole/card_count — the
    real signal is node.kind == NodeKind.AUTO, node.auto_type ==
    "deal_hole_cards", and node.config["count"]. Isolated here as the
    one place this structural check lives, since the per-node walk
    below (for creation_phases) needs a per-node answer, distinct from
    graph_hole_cards.first_deal_hole_cards_count()'s single
    first-match-wins answer used for the summary's aggregate field.
    """
    kind = getattr(node, "kind", None)
    kind_name = getattr(kind, "name", str(kind)) if kind is not None else None
    if kind_name != "AUTO":
        return False, 0
    if getattr(node, "auto_type", None) != "deal_hole_cards":
        return False, 0
    config = getattr(node, "config", {}) or {}
    return True, int(config.get("count", 1))


def _summarize_flow(game_def, rules, graph):
    """
    ASSUMPTIONS (unconfirmed against real graph-module source — same
    remaining ones as before; the deals_hole/card_count assumption is
    now CONFIRMED and fixed, see _node_deals_hole_cards above and the
    Engine team's note):
      - graph has a `.start_node` attribute (or `.root`/`.entry` —
        tried in that order) giving the first node id.
      - graph.node(node_id) returns an object with `.domain` (for
        DECISION nodes, e.g. "BETTING") and `.metadata` (dict) — per
        the migration doc's confirmed mapping. `.metadata` is STILL
        used for `deals_board_street` (presence-only) and
        `street_name`/`node_indices` — only the hole-card signal
        moved off metadata entirely.
      - graph.outgoing(node_id) returns an iterable of next node ids
        (or (condition, next_id) pairs for the transitions: override
        path) — this walk only follows the FIRST outgoing edge.

    Cycle-safe: stops walking a branch the moment it revisits a node
    id (betting's own loop branch would otherwise spin forever).

    SETUP phase
    ------------
    creation_phases[0] is ALWAYS a synthetic player-setup step
    (id="SETUP", no cards, no betting, decision_domain=None) —
    required by the Hand Creator wizard so users can add players to
    empty seats before any hole-card or board-dealing phase exists.
    """
    SETUP_PHASE = {
        "id": "SETUP",
        "label": "Setup",
        "deals_hole": False,
        "deals_board": False,
        "reveal_group": 0,
        "board_node_indices": [],
        "decision_domain": None,
        "allows_betting": False,
    }

    start = (
        getattr(graph, "start_node", None)
        or getattr(graph, "root", None)
        or getattr(graph, "entry", None)
    )

    # Aggregate hole-card count — canonical helper, soft-fails to None
    # (omitted from the response) rather than raising, since this
    # endpoint (GET /game/variants/{game_name}/config) is read by the
    # Frontend wizard for display and shouldn't 500 a whole variant's
    # config over a hole-card-count derivation gap.
    hole_cards_total = first_deal_hole_cards_count(graph)

    if start is None:
        result = {
            "game_name": getattr(game_def, "game_name", None),
            "layout_name": getattr(game_def, "layout_name", None),
            "hole_cards": hole_cards_total,
            "board_layout": {
                "nodes": getattr(game_def, "node_count", None),
                "streets": {},
                "street_names": {},
            },
            "creation_phases": [SETUP_PHASE],
        }
        return result

    creation_phases = [SETUP_PHASE]
    streets: dict[int, list[int]] = {}
    street_names: dict[int, str] = {}

    reveal_group = 0

    visited = set()
    current = start
    idx = 0

    while current is not None and current not in visited:
        visited.add(current)
        node = graph.node(current)
        domain = getattr(node, "domain", None)
        domain_name = getattr(domain, "name", str(domain)) if domain is not None else None
        metadata = dict(getattr(node, "metadata", {}) or {})

        deals_hole, _hole_card_count = _node_deals_hole_cards(node)
        # Presence-only signal — the VALUE (if any) is never trusted
        # as a street number.
        deals_board = metadata.get("deals_board_street") is not None
        street_name = metadata.get("street_name")
        node_indices = metadata.get("node_indices", []) if deals_board else []
        decision_domain = domain_name

        if deals_board:
            reveal_group += 1
            streets.setdefault(reveal_group, list(node_indices))
            if street_name:
                street_names[reveal_group] = street_name

        creation_phases.append(
            {
                "id": metadata.get("id", f"NODE_{idx}"),
                "deals_hole": deals_hole,
                "deals_board": deals_board,
                "reveal_group": reveal_group,
                "board_node_indices": node_indices,
                "decision_domain": decision_domain,
                "allows_betting": decision_domain == "BETTING",  # back-compat
            }
        )
        idx += 1

        outgoing = list(getattr(graph, "outgoing", lambda _n: [])(current))
        if not outgoing:
            break
        nxt = outgoing[0]
        if isinstance(nxt, tuple):
            nxt = nxt[-1]
        current = nxt

    return {
        "game_name": getattr(game_def, "game_name", None),
        "layout_name": getattr(game_def, "layout_name", None),
        "hole_cards": hole_cards_total,
        "board_layout": {
            "nodes": getattr(game_def, "node_count", None),
            "streets": streets,
            "street_names": street_names,
        },
        "creation_phases": creation_phases,
    }

game_service = GameService()