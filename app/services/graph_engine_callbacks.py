"""
app/services/graph_engine_callbacks.py — BackendGraphEngineCallbacks,
parallel to app/services/engine_callbacks.py's BackendEngineCallbacks
for the (now retired) PokerState path.

Named BackendGraphEngineCallbacks (not GraphEngineCallbacks) on
purpose: poker_engine.graph.callbacks already exports a real
GraphEngineCallbacks name per the migration doc's API mapping table
("EngineCallbacks.on_hand_start/on_action/on_showdown" ->
"GraphEngineCallbacks.on_hand_start/on_decision/on_showdown"). Reusing
that exact name here for our own duck-typed hook implementation would
shadow/confuse imports the moment both are needed side by side (e.g.
if the real GraphEngineCallbacks turns out to be a base class this
should subclass rather than just duck-type against — TODO once that's
confirmed).

Per the migration doc's API mapping:
    on_action -> on_decision. Old signature was (state, action,
    player_index, pot_before, stack_before). New is
    (engine, response: DecisionResponse). "If you log pot/stack
    deltas, snapshot engine.state yourself immediately before calling
    submit_decision(...)."

Because on_decision no longer receives pot_before/stack_before as
arguments, this class exposes note_pre_decision(engine, player_index),
which the CALLER (game_service.py, right before
poker_engine.graph.callbacks.submit_decision(...)) must invoke first —
it stashes the pre-action pot/stack so on_decision can still log the
same pot_before/stack_before fields SessionLogger.log_action() has
always taken. If a caller forgets to call note_pre_decision() first,
on_decision() falls back to POST-action pot/stack (degraded logging,
not a crash — see the inline comment there).

DecisionResponse's real shape, per the migration doc's "Run a hand"
example:
    response = DecisionResponse(req.node_id, req.domain, req.player_index, value)
i.e. positional (node_id, domain, player_index, value) — NOT the
(action_type, amount) kwargs an earlier revision of this file assumed.
`value` is CONFIRMED for the BETTING domain: a real
poker_engine.actions.action.Action(type=ActionType, amount=int) — see
game_service.py's _build_decision_response, which was fixed after
BettingResolver.validate() raised "response.value must be an Action"
at runtime against an earlier plain-dict shape. on_decision() below
reads `.type` (unwrapped via `.name` to a string, e.g. "FOLD") off
`value` as the primary path, with dict / bare-`.action_type`-attribute
handling kept only as a defensive fallback for any other domain whose
value isn't shaped like a BETTING Action.

CARD_SELECT / CARD_PASS dispatch (drawmaha / pass-the-trash)
-----------------------------------------------------------------
`response.value` for these two domains is `Tuple[int, ...]` — the
engine card ids the player selected from their OWN hand (see
core_types.py: "CARD_SELECT/CARD_PASS -> Tuple[int,...]", and
game_service.py's _build_card_select_decision_response, which builds
exactly this shape). Neither domain is BETTING-shaped, so on_decision()
checks the response's domain FIRST and routes to _log_card_decision()
before falling into the Action-shaped BETTING logic below — see that
method's own docstring for the CARD_SELECT vs. CARD_PASS split.

Street resolution — REWRITTEN, no more graph walking
-----------------------------------------------------------------
Every street stamped anywhere (Action.street, hole_card_events[].street,
BoardCard.street) now comes from ONE counter:
SessionLogger.current_street() — see that class's own docstring on
self._board_reveal_index for the full rationale. This file previously
tried to derive "which street is this node on" by walking the compiled
GameGraph and reading a `street_index` value off each node's metadata.
That was fragile in two independent ways that both turned out to bite:
  (1) it assumed the compiled graph node's metadata dict preserves the
      flow YAML's `street_index` key verbatim — never actually
      confirmed against the real graph loader, and apparently wrong
      (board cards still weren't showing up on the right frames after
      that "fix").
  (2) even when correct, it computed street from a DIFFERENT signal
      than BoardCard.street (reveal order), which are only guaranteed
      to agree for the simplest single-board standard-street variants
      — a bomb pot, multi-board/hopscotch layout, or any custom flow
      could silently disagree between the two.
Deriving every street-stamped write from SessionLogger's own reveal
counter removes the graph from the question entirely: "which street is
this on" is now identically defined as "how many board reveals have
happened so far", for every table, for every layout, with zero graph
introspection and zero chance of the two numbering schemes drifting
apart. All the graph-walking helper functions that used to live in
this file (_street_index_from_metadata, _node_street_map_from_graph,
_betting_node_street_map, _card_node_street_map, _all_nodes_street_map,
_street_index_for_node, _card_street_index_for_node,
_current_node_street) are gone — callers just call
self.logger.current_street() directly, always AFTER log_board() has
had a chance to observe any board cards revealed as part of the same
AUTO-walk (on_decision()/on_cards_distributed() already call log_board()
first, so this ordering is preserved).

Remaining ASSUMPTIONS not yet confirmed against real source:
  - callbacks.on_hand_start(engine, game_def) fires once per new hand
    (game_def passed explicitly, since GraphEngine's `state` may not
    carry it) — if it's actually on_hand_start(engine) only, only the
    signature below needs adjusting.
  - callbacks.on_decision(engine, response) fires AFTER the decision
    is applied to engine.state (mirrors old on_action firing
    post-application) — consistent with why pot_before/stack_before
    had to move to the caller-snapshotted note_pre_decision() path.
  - callbacks.on_showdown(engine, result) mirrors old
    on_showdown(state, result) 1:1 with `engine` replacing `state`.
  - CARD_PASS's target seat: the engine resolves it internally
    (CardPassResolver, from a fixed "left"/"right" config rule — see
    game_api.py's ActionRequest docstring) and it is NOT part of
    `response` — see _log_card_decision()'s own docstring for why this
    file no longer tries to guess it at selection time at all.
"""

from app.graph_engine_adapter import _domain_name, _game_obj


class _GraphHandStateShim:
    """
    Minimal duck-type adapter so SessionLogger.finish_hand()/log_board()
    — written against PokerState's shape, where `.game` is a NESTED
    GameState and `.last_showdown` lives on PokerState itself (see
    engine_callbacks.py's own on_showdown(state, result), which passes
    `state` straight through) — can be reused unchanged for a
    GraphEngine-driven hand without touching session_logger.py, which
    is shared by both the legacy and graph paths.

    Two facts this bridges:
      - GraphGameState IS-A GameState (§2.9.4) — it has no nested
        `.game` attribute at all; the board/pot/player fields ARE the
        object. `_game_obj(engine)` already handles this (tries
        `engine.state.game`, falls back to `engine.state` itself), so
        `.game` here is just that.
      - `last_showdown`/`last_winners` are set on the GraphEngine
        itself by the `showdown`/`award_last_player` AUTO handlers
        (§2.9.5), not on `engine.state` — mirrored here from the
        `result` callbacks.py already resolved and passed into
        on_showdown, rather than re-reading engine.last_showdown
        redundantly.

    AttributeError: 'GraphGameState' object has no attribute 'game' was
    the exact symptom of not having this shim — log_board(state) did
    `g = state.game` straight against a raw GraphGameState.
    """

    __slots__ = ("game", "last_showdown")

    def __init__(self, engine, result=None):
        self.game = _game_obj(engine)
        self.last_showdown = result if result is not None else getattr(
            engine, "last_showdown", None
        )


class BackendGraphEngineCallbacks:

    def __init__(self, logger, game_service_ref=None):
        self.logger = logger
        self._game_service = game_service_ref

        # Set by note_pre_decision(), consumed (and cleared) by the
        # next on_decision() call — see module docstring.
        self._pending_player_index = None
        self._pending_pot_before = None
        self._pending_stack_before = None

    def _is_editing(self):
        return self._game_service and self._game_service.editing_mode

    # ------------------------------------------------------------
    # Pre-decision snapshot — call this immediately before
    # poker_engine.graph.callbacks.submit_decision(engine, response,
    # callbacks=cb).
    # ------------------------------------------------------------

    def note_pre_decision(self, engine, player_index: int):
        if self._is_editing():
            return

        g = _game_obj(engine)
        self._pending_player_index = player_index
        self._pending_pot_before = g.pot
        self._pending_stack_before = (
            g.players[player_index].stack
            if 0 <= player_index < len(g.players)
            else None
        )

    # ------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------

    def on_hand_start(self, engine, game_def=None):
        if self._is_editing():
            return

        g = _game_obj(engine)
        gd = game_def or getattr(engine, "game_def", None) or getattr(
            engine.state, "game_def", None
        )
        rules = getattr(engine, "rules", None) or getattr(engine.state, "rules", None)

        self.logger.start_hand(
            {
                "variant_name": getattr(gd, "game_name", None),
                "layout_name": getattr(gd, "layout_name", None),
                "split_pot": (getattr(rules, "payout_type", None) == "split_pot"),
                "betting_config_id": 1,
                "dealer_seat": g.dealer_position,
                "pot": g.pot,
                "ended_at": None,
                "players": g.players,
                "game_def": gd,
                # node_street_map dropped — street is no longer derived
                # from graph metadata anywhere (see module docstring),
                # and SessionLogger.start_hand() never consumed this
                # key in the first place.
            }
        )

    def on_decision(self, engine, response):
        if self._is_editing():
            return

        g = _game_obj(engine)

        # Persist any board cards revealed, AND any extra hole cards
        # dealt (ESG / Catchup ESG / Christmas / Grinch's
        # deal_extra_hole_card, or a CARD_SELECT redraw's replacement
        # cards) since the last decision. Both are AUTO-node side
        # effects the engine runs with no pause in between — by the
        # time a player submits THIS decision, anything dealt in
        # between is already sitting in g.node_cards / player
        # hand_masks. log_board() MUST run first: it's the only place
        # that advances SessionLogger's street counter, and everything
        # logged below (hole cards from this same AUTO-walk, and the
        # decision itself) needs to be stamped with whatever that
        # counter reads AFTER this potential advance.
        self.logger.log_board(_GraphHandStateShim(engine))

        current_street = self.logger.current_street()
        self.logger.log_hole_cards(_GraphHandStateShim(engine), street=current_street)

        player_index = self._pending_player_index
        if player_index is None:
            player_index = getattr(response, "player_index", None)

        domain_name = _domain_name(getattr(response, "domain", None))

        if domain_name in ("CARD_SELECT", "CARD_PASS"):
            self._log_card_decision(
                engine, response, player_index, domain_name, current_street
            )
            self._pending_player_index = None
            self._pending_pot_before = None
            self._pending_stack_before = None
            return

        # BOOLEAN / CHOICE — neither domain's `value` is an Action
        # (BOOLEAN's is a plain bool; CHOICE's is a plain str — see
        # core_types.py's DecisionResponse.value table), so route both
        # to a dedicated logger method BEFORE falling into the
        # Action-shaped BETTING logic below. Without this branch, a
        # Grinch "Christmas next street?" decision fell straight
        # through into the BETTING path, which reads `.type`/`.amount`
        # off `response.value` — a bool/str has neither.
        if domain_name in ("BOOLEAN", "CHOICE"):
            self._log_boolean_or_choice_decision(
                engine, response, player_index, domain_name, current_street
            )
            self._pending_player_index = None
            self._pending_pot_before = None
            self._pending_stack_before = None
            return

        pot_before = (
            self._pending_pot_before if self._pending_pot_before is not None else g.pot
        )
        stack_before = self._pending_stack_before
        if (
            stack_before is None
            and player_index is not None
            and 0 <= player_index < len(g.players)
        ):
            stack_before = g.players[player_index].stack

        value = getattr(response, "value", None)
        if isinstance(value, dict):
            action_type = value.get("action_type")
            amount = value.get("amount")
        else:
            action_type = getattr(value, "type", None)
            if action_type is None:
                action_type = getattr(value, "action_type", None)
            action_type = getattr(action_type, "name", action_type)
            amount = getattr(value, "amount", None)

        self.logger.log_action(
            street=current_street,
            player_index=player_index,
            action=action_type,
            amount=amount,
            pot_before=pot_before,
            stack_before=stack_before,
        )

        self._pending_player_index = None
        self._pending_pot_before = None
        self._pending_stack_before = None

    def _log_boolean_or_choice_decision(
        self, engine, response, player_index, domain_name, current_street
    ):
        """
        Persist a BOOLEAN (e.g. Grinch's "Christmas next street?") or
        CHOICE decision via SessionLogger.log_action(), reusing the
        Action table rather than adding a new one — action_type is
        stamped as "BOOLEAN"/"CHOICE" (not a real ActionType member,
        but Action.action_type is a free-text column — see hands.py's
        schema note that action_type already isn't a strict ActionType
        enum on the DB side, e.g. "SEAT" synthetic rows) and `amount`
        is repurposed to encode the answer: 1/0 for BOOLEAN's
        true/false, left NULL for CHOICE (its answer is a string, not
        representable in an int column — the actual chosen label is
        only recoverable from a richer log if this ever needs to be
        queried directly; acceptable for now since no shipped variant
        uses CHOICE yet).

        If player_index is unresolvable, skip logging (mirrors
        _log_card_decision's own guard) rather than write a row with
        no attributable player.

        `current_street` is passed in by on_decision() (already
        resolved once via self.logger.current_street() after
        log_board() ran) rather than re-resolved here, so a BOOLEAN/
        CHOICE decision is guaranteed to log the same street value any
        board-card reveal earlier in this same call would have used.
        """
        if player_index is None:
            return

        value = getattr(response, "value", None)

        if domain_name == "BOOLEAN":
            action_label = "BOOLEAN"
            amount = 1 if bool(value) else 0
        else:  # CHOICE
            action_label = "CHOICE"
            amount = None

        pot_before = (
            self._pending_pot_before
            if self._pending_pot_before is not None
            else _game_obj(engine).pot
        )
        stack_before = self._pending_stack_before

        self.logger.log_action(
            street=current_street,
            player_index=player_index,
            action=action_label,
            amount=amount,
            pot_before=pot_before,
            stack_before=stack_before,
        )

    def _log_card_decision(
        self, engine, response, player_index, domain_name, current_street
    ):
        """
        Persist a CARD_SELECT (drawmaha-style discard) or CARD_PASS
        (pass-the-trash-style) decision via card_movement.py (through
        SessionLogger.log_card_select()), so HoleCard/CardEvent stay
        in sync the same way BETTING actions already do through
        log_action().

        response.value is Tuple[int, ...] — the engine card ids the
        player selected from their OWN hand (see core_types.py and
        game_service.py's _build_card_select_decision_response, which
        builds exactly this shape for both domains).

        CARD_SELECT is unambiguous: every selected card is discarded
        to the muck — no destination player to resolve.

        CARD_PASS's target seat is resolved automatically by the
        engine (CardPassResolver — real routing, including which
        seats to skip, e.g. folded/eliminated players) and is NOT
        part of `response`, so it can't be logged here at all. Every
        CARD_PASS selection is logged as a plain discard (give-side
        only) — always correct regardless of routing, since the
        giver's own hand definitely loses these cards no matter where
        they end up. on_cards_distributed() is SOLELY responsible for
        the receiving side, via its existing diff-based
        log_hole_cards() call, which reads the Engine's real
        post-distribution state rather than guessing.

        `current_street` — same passed-in value on_decision() already
        resolved, see _log_boolean_or_choice_decision()'s own note.
        """
        # NOTE: card_ids=() is a legal "stood pat / selected zero cards"
        # response (min_count can be 0 — see drawmaha's discard config),
        # not a missing one — only player_index being unresolvable is a
        # real reason to skip logging.
        card_ids = tuple(getattr(response, "value", None) or ())

        if player_index is None:
            return

        self.logger.log_card_select(current_street, player_index, card_ids)

    def on_cards_distributed(self, engine, distributed_passes=None):
        """
        Fires once every eligible CARD_PASS player has submitted and
        the engine has moved the passed cards into their targets'
        hand_masks (poker_engine/graph/callbacks.py:145 —
        `callbacks.on_cards_distributed(engine,
        engine.last_distributed_passes)`).

        Deliberately does NOT try to replay `distributed_passes` card
        -by-card — its exact shape isn't confirmed against real Engine
        source (unlike DecisionResponse.value, which the migration doc
        pinned down explicitly per domain). Instead, reuses the same
        diff-based mechanism SessionLogger.log_hole_cards() already
        provides for extra-card mechanics and CARD_SELECT redraws:
        compare each player's CURRENT hand_mask (now updated by the
        engine's real distribution) against what's already been
        logged, and log whatever's new — under whatever street
        self.logger.current_street() reads AFTER log_board() has run
        for this snapshot, same ordering on_decision() uses, so a
        pass-the-trash card lands on the street it was actually passed
        on rather than one early.

        This is now the ONLY place CARD_PASS's receiving side is ever
        logged — _log_card_decision() (above) only ever logs the
        GIVING side (as a discard) at selection time, specifically so
        the receiving side always comes from the Engine's real
        post-distribution state rather than a guess.
        """
        if self._is_editing():
            return

        self.logger.log_board(_GraphHandStateShim(engine))
        current_street = self.logger.current_street()
        self.logger.log_hole_cards(_GraphHandStateShim(engine), street=current_street)

    def on_showdown(self, engine, result=None):
        if self._is_editing():
            return

        self.logger.finish_hand(_GraphHandStateShim(engine, result))