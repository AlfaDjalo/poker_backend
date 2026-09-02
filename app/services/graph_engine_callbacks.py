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
(action_type, amount) kwargs the previous revision of this file
assumed. `value` is now CONFIRMED for the BETTING domain: a real
poker_engine.actions.action.Action(type=ActionType, amount=int) — see
game_service.py's _build_decision_response, which was fixed after
BettingResolver.validate() raised "response.value must be an Action"
at runtime against the previous plain-dict shape. on_decision() below
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
now checks the response's domain FIRST and routes to
_log_card_decision() before falling into the Action-shaped BETTING
logic below — see that method's own docstring for the CARD_SELECT vs.
CARD_PASS split and the pass_direction resolution it depends on.

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
    `response`. _log_card_decision() below assumes the current graph
    node's `metadata` carries a `pass_direction` key ("left"|"right")
    describing that same rule — same mechanism
    _betting_node_street_map()/_node_street_map_from_graph() already
    lean on for other per-node metadata. If that key turns out to
    live somewhere else, only _log_card_decision()'s direction lookup
    needs updating.
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


def _node_street_map_from_graph(graph) -> dict[int, int]:
    """
    Best-effort node_index -> 1-based street number map, built by
    walking the graph's dealing-node metadata (`deals_board_street` +
    `node_indices`) — replaces game_def.street_nodes, which no longer
    exists on the slimmed-down GameDefinition (see the migration
    doc). Falls back to an empty dict on any failure — SessionLogger.
    log_board() already defaults any un-mapped node to street=1, so an
    empty map here degrades to the OLD (wrong-but-non-crashing)
    behavior rather than raising.

    Unlike game_service.py's _summarize_flow() (which only follows the
    FIRST outgoing edge per node, since it just wants one
    representative pass for wizard display), this does a full
    visited-guarded traversal of every reachable node — logging needs
    every dealing node's street, not just the ones on one branch.

    ASSUMPTIONS: same as _summarize_flow() in game_service.py —
    graph.node(id).metadata may contain `deals_board_street` (int) and
    `node_indices` (list[int]); graph exposes `.start_node` (or
    `.root`/`.entry`) and `.outgoing(id)`.
    """
    if graph is None:
        return {}
    try:
        start = (
            getattr(graph, "start_node", None)
            or getattr(graph, "root", None)
            or getattr(graph, "entry", None)
        )
        if start is None:
            return {}

        node_map: dict[int, int] = {}
        visited = set()
        frontier = [start]
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)

            node = graph.node(current)
            metadata = dict(getattr(node, "metadata", {}) or {})
            deals_board_street = metadata.get("deals_board_street")
            if deals_board_street is not None:
                for node_idx in metadata.get("node_indices", []):
                    node_map[int(node_idx)] = int(deals_board_street)

            outgoing = list(getattr(graph, "outgoing", lambda _n: [])(current))
            for nxt in outgoing:
                if isinstance(nxt, tuple):
                    nxt = nxt[-1]
                if nxt not in visited:
                    frontier.append(nxt)
        return node_map
    except Exception:
        return {}


def _betting_node_street_map(graph) -> dict:
    """
    Map every BETTING-domain DECISION node id -> the 1-based board
    street it actually belongs to, in the SAME numbering
    board_cards.street already uses (1=flop, 2=turn, 3=river; 0 =
    nothing dealt yet, i.e. a genuine preflop betting round).

    Superseded approach, and why it was wrong
    ------------------------------------------
    An earlier version of this file read a betting node's OWN
    `metadata.get("street_index")` directly and used it as-is. That
    field is real, but it's a SEQUENTIAL BETTING-ROUND COUNTER
    (0=first betting round in the hand, 1=second, ...) — a completely
    different numbering from a `deal` node's own `street_index`
    metadata (1=flop, 2=turn, 3=river — 1-based, and the one that
    board_cards.street / _node_street_map_from_graph actually use).

    For a STANDARD variant (real preflop betting round first) the two
    conventions happen to agree after preflop: betting round 1 (flop)
    == board street 1 (flop), round 2 (turn) == board street 2, etc. —
    only preflop (betting round 0) has no board-street equivalent,
    which is harmless since no board cards exist at street 0 anyway.
    But confirmed empirically (a live flop-betting node's own metadata
    read street_index=0, not 1): the two are NOT simply off by a
    constant +1 either, and for a BOMB POT variant — which deals the
    flop before any betting happens at all — the FIRST betting round
    (also sequentially numbered 0 by this same convention) is the
    FLOP round, needing board street=1, not 0. A blanket +1 offset
    would be correct for the standard case and wrong for bomb pots, or
    vice versa for a blanket "treat 0 as preflop, don't shift" rule.

    The only convention-independent way to get this right for every
    variant is to derive a betting node's street from what the FLOW
    ITSELF most recently dealt on the way to that node, not from
    either node's own street_index counter. This walks the graph from
    its entry node, tracking a running "last dealt board street" value
    (starts at 0 = nothing dealt) that gets overwritten every time a
    `deals_board_street` node is passed, and records that value
    against every BETTING node reached along the way — a standard
    variant's opening betting node is reached before any deal node, so
    it correctly resolves to 0; a bomb pot's opening betting node is
    reached AFTER the flop's deal node, so it correctly resolves to 1.

    This walk also drives _log_card_decision()'s street label for
    CARD_SELECT/CARD_PASS nodes via _street_index_for_node() below —
    those aren't BETTING-domain, so they're recorded separately by
    _card_node_street_map() rather than folded into this map.

    Falls back to an empty dict on any failure — on_decision() already
    treats a missing entry as street=0, the same degraded "everything
    on the first street" behavior _node_street_map_from_graph's own
    empty-dict fallback accepts for board-card street stamping.

    ASSUMPTIONS: same as _node_street_map_from_graph — graph.node(id)
    exposes `.domain` (compared by `.name`, e.g. "BETTING") and
    `.metadata` (dict, may contain `deals_board_street`); graph exposes
    `.start_node`/`.root`/`.entry` and `.outgoing(id)`.
    """
    if graph is None:
        return {}
    try:
        start = (
            getattr(graph, "start_node", None)
            or getattr(graph, "root", None)
            or getattr(graph, "entry", None)
        )
        if start is None:
            return {}

        betting_street: dict = {}
        visited = set()
        # (node_id, street_so_far) — the running street value travels
        # WITH the walk rather than living in one shared variable, so
        # two branches at different points in the flow can never
        # clobber each other's in-progress value. Overwritten (not
        # incremented) on every deals_board_street node, so it's
        # self-correcting regardless of what a branch's value was
        # before reaching that node.
        frontier = [(start, 0)]

        while frontier:
            node_id, current_street = frontier.pop()
            if node_id in visited:
                continue
            visited.add(node_id)

            node = graph.node(node_id)
            domain = getattr(node, "domain", None)
            domain_name = (
                getattr(domain, "name", str(domain)) if domain is not None else None
            )
            metadata = dict(getattr(node, "metadata", {}) or {})

            deals_board_street = metadata.get("deals_board_street")
            if deals_board_street is not None:
                current_street = int(deals_board_street)

            if domain_name == "BETTING":
                betting_street[node_id] = current_street

            outgoing = list(getattr(graph, "outgoing", lambda _n: [])(node_id))
            for nxt in outgoing:
                if isinstance(nxt, tuple):
                    nxt = nxt[-1]
                if nxt not in visited:
                    frontier.append((nxt, current_street))

        return betting_street
    except Exception:
        return {}


def _card_node_street_map(graph) -> dict:
    """
    Same walk as _betting_node_street_map(), but records the running
    "last dealt board street" against every CARD_SELECT/CARD_PASS
    DECISION node instead of BETTING ones — used by
    _log_card_decision() to stamp a discard/pass with the same
    street-as-label convention every other table already uses (Action,
    BoardCard, CardEvent). A discard/pass node reached before any
    board card is dealt (e.g. drawmaha's pre-flop draw) correctly
    resolves to street=0, matching a genuine preflop BETTING node's
    own street=0 under _betting_node_street_map().

    Kept as a separate walk (rather than merging into
    _betting_node_street_map() and returning a combined dict) so a
    future caller that only cares about one domain doesn't have to
    filter the other one back out — small duplication, clearer
    call sites.
    """
    if graph is None:
        return {}
    try:
        start = (
            getattr(graph, "start_node", None)
            or getattr(graph, "root", None)
            or getattr(graph, "entry", None)
        )
        if start is None:
            return {}

        card_street: dict = {}
        visited = set()
        frontier = [(start, 0)]

        while frontier:
            node_id, current_street = frontier.pop()
            if node_id in visited:
                continue
            visited.add(node_id)

            node = graph.node(node_id)
            domain = getattr(node, "domain", None)
            domain_name = (
                getattr(domain, "name", str(domain)) if domain is not None else None
            )
            metadata = dict(getattr(node, "metadata", {}) or {})

            deals_board_street = metadata.get("deals_board_street")
            if deals_board_street is not None:
                current_street = int(deals_board_street)

            if domain_name in ("CARD_SELECT", "CARD_PASS"):
                card_street[node_id] = current_street

            outgoing = list(getattr(graph, "outgoing", lambda _n: [])(node_id))
            for nxt in outgoing:
                if isinstance(nxt, tuple):
                    nxt = nxt[-1]
                if nxt not in visited:
                    frontier.append((nxt, current_street))

        return card_street
    except Exception:
        return {}


def _street_index_for_node(engine, node_id) -> int:
    """
    Resolve the board-street number to log on Action.street for the
    BETTING decision node `node_id` — see _betting_node_street_map()'s
    own docstring for why this can't just read a node's own
    street_index metadata directly.

    THIS FUNCTION WAS PREVIOUSLY CALLED BUT NEVER DEFINED — every
    on_decision() call (i.e. every single player action, of any kind)
    raised NameError here, uncaught, which propagated up through
    submit_decision() and surfaced as an unhandled 500 from
    POST /game/action. Defined now as a thin wrapper around the
    already-implemented _betting_node_street_map(graph) walk.

    Recomputes the graph walk fresh on every call rather than caching
    it on the engine/callbacks instance — actions are infrequent
    relative to other state churn, so correctness (never serving a
    stale map after some future graph-mutation feature) is worth more
    here than the walk's small cost.
    """
    if node_id is None:
        return 0
    graph = getattr(engine, "graph", None)
    return _betting_node_street_map(graph).get(node_id, 0)


def _card_street_index_for_node(engine, node_id) -> int:
    """CARD_SELECT/CARD_PASS sibling of _street_index_for_node() — see
    _card_node_street_map()'s own docstring."""
    if node_id is None:
        return 0
    graph = getattr(engine, "graph", None)
    return _card_node_street_map(graph).get(node_id, 0)


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
                # See _node_street_map_from_graph()'s own docstring —
                # replaces game_def.street_nodes, which no longer exists.
                "node_street_map": _node_street_map_from_graph(
                    getattr(engine, "graph", None)
                ),
            }
        )

    def on_decision(self, engine, response):
        if self._is_editing():
            return

        g = _game_obj(engine)

        # Persist any board cards revealed since the last decision (or
        # since hand start) BEFORE logging this action. GraphEngine
        # auto-runs every AUTO node — including `deal` — up to the next
        # DECISION node with no external pause, so by the time a player
        # submits THIS decision, whatever street was dealt in between is
        # already sitting in g.node_cards. Calling log_board() here,
        # once per decision, is what gives each street its own reveal
        # group (see SessionLogger.log_board / log_board's own
        # docstring) — previously this only happened once, at showdown
        # (on_showdown -> finish_hand -> log_board), which is why every
        # board card ended up batched into a single reveal regardless of
        # street numbering.
        self.logger.log_board(_GraphHandStateShim(engine))

        player_index = self._pending_player_index
        if player_index is None:
            # note_pre_decision() wasn't called first — fall back to
            # whatever the response itself claims.
            player_index = getattr(response, "player_index", None)

        # CARD_SELECT / CARD_PASS dispatch — neither domain's `value`
        # is an Action, so route to the card-movement write path
        # BEFORE any of the Action-shaped BETTING logic below runs.
        # See _log_card_decision()'s own docstring for the two
        # domains' write shapes and the CARD_PASS target-seat lookup.
        domain_name = _domain_name(getattr(response, "domain", None))
        if domain_name in ("CARD_SELECT", "CARD_PASS"):
            self._log_card_decision(engine, response, player_index, domain_name)
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

        # `value` is CONFIRMED now (see game_service.py's
        # _build_decision_response, fixed after "response.value must be
        # an Action" surfaced at runtime): a real
        # poker_engine.actions.action.Action, whose action-identity
        # field is `.type` (an ActionType enum member), NOT
        # `.action_type` — reading `.action_type` here always returned
        # None, which silently wrote NULL into actions.action_type on
        # every single logged action (no crash on write; DB allowed the
        # NULL) until it blew up downstream as a pydantic
        # ValidationError the first time that hand was read back via
        # GET /replay/hands/{id} ("Input should be a valid string
        # [type=string_type], input_value=None"). `.type` is unwrapped
        # to its `.name` string exactly the way engine_callbacks.py's
        # own on_action does for the PokerState path
        # (`action.type.name`), so both callback implementations write
        # the same action_type string shape into the DB.
        #
        # The dict / bare-"action_type"-attribute branches are kept
        # only as defensive fallbacks for a value shape that isn't the
        # real Action (e.g. editing-mode replay tooling, or a future
        # non-BETTING domain whose value isn't an Action at all) —
        # they are no longer the expected path for a live BETTING
        # decision.
        value = getattr(response, "value", None)
        if isinstance(value, dict):
            action_type = value.get("action_type")
            amount = value.get("amount")
        else:
            action_type = getattr(value, "type", None)
            if action_type is None:
                action_type = getattr(value, "action_type", None)
            action_type = getattr(action_type, "name", action_type)  # enum -> "FOLD"
            amount = getattr(value, "amount", None)

        street = _street_index_for_node(engine, getattr(response, "node_id", None))

        self.logger.log_action(
            street=street,
            player_index=player_index,
            action=action_type,
            amount=amount,
            pot_before=pot_before,
            stack_before=stack_before,
        )

        self._pending_player_index = None
        self._pending_pot_before = None
        self._pending_stack_before = None

    def _log_card_decision(self, engine, response, player_index, domain_name):
        """
        Persist a CARD_SELECT (drawmaha-style discard) or CARD_PASS
        (pass-the-trash-style) decision via card_movement.py (through
        SessionLogger.log_card_select()/log_card_pass()), so
        HoleCard/CardEvent stay in sync the same way BETTING actions
        already do through log_action().

        response.value is Tuple[int, ...] — the engine card ids the
        player selected from their OWN hand (see core_types.py and
        game_service.py's _build_card_select_decision_response, which
        builds exactly this shape for both domains).

        CARD_SELECT is unambiguous: every selected card is discarded
        to the muck — no destination player to resolve.

        CARD_PASS's target seat is resolved automatically by the
        engine (CardPassResolver, from a fixed "left"/"right" config
        rule — see game_api.py's ActionRequest docstring) and is NOT
        part of `response`. This reads a `pass_direction` key
        ("left" | "right") off the current graph node's metadata —
        see module docstring's ASSUMPTIONS — and derives the target
        seat as (player_index ± 1) % n. If that key is missing, or
        engine.graph/g.players aren't available for any reason, this
        falls back to logging the cards as a plain discard instead of
        a pass: it's the safer degradation (the giver's own hand state
        stays correct — cards leave it either way) versus guessing a
        target seat and silently attributing the pass to the wrong
        player. A warning is printed either way so it's diagnosable
        rather than a silently wrong replay.
        """
        # NOTE: card_ids=() is a legal "stood pat / selected zero cards"
        # response (min_count can be 0 — see drawmaha's discard config),
        # not a missing one — only player_index being unresolvable is a
        # real reason to skip logging. An earlier version of this check
        # was `if not card_ids or player_index is None`, which silently
        # dropped standing-pat responses from the log entirely (no
        # correctness impact on gameplay — CardSelectResolver.apply()
        # already marks the player as acted on the engine side
        # independent of this logging call — but it meant a player who
        # stood pat left no history/replay trace of having acted at all).
        card_ids = tuple(getattr(response, "value", None) or ())
        street = _card_street_index_for_node(
            engine, getattr(response, "node_id", None)
        )

        if player_index is None:
            return

        if domain_name == "CARD_SELECT":
            self.logger.log_card_select(street, player_index, card_ids)
            return

        # CARD_PASS — resolve target seat from node metadata.
        g = _game_obj(engine)
        n = len(getattr(g, "players", []) or [])
        graph = getattr(engine, "graph", None)
        node_id = getattr(response, "node_id", None)

        direction = None
        if graph is not None and node_id is not None:
            try:
                node = graph.node(node_id)
                metadata = dict(getattr(node, "metadata", {}) or {})
                direction = metadata.get("pass_direction")
            except Exception:
                direction = None

        target_index = None
        if n and direction == "left":
            target_index = (player_index + 1) % n
        elif n and direction == "right":
            target_index = (player_index - 1) % n

        if target_index is None:
            print(
                f"[graph_engine_callbacks] CARD_PASS at node {node_id!r} has "
                f"no resolvable pass_direction in node metadata — logging "
                f"as a discard instead of a pass (target player unknown)."
            )
            self.logger.log_card_select(street, player_index, card_ids)
            return

        self.logger.log_card_pass(street, player_index, target_index, card_ids)

    def on_showdown(self, engine, result=None):
        if self._is_editing():
            return

        self.logger.finish_hand(_GraphHandStateShim(engine, result))