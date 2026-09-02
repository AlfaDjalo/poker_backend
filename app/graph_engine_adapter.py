"""
app/graph_engine_adapter.py — GraphEngine -> GameStateDTO.

Rewritten against the confirmed migration doc ("Backend migration:
PokerState -> GraphEngine", API mapping section):

    poker_state.legal_actions()   -> engine.pending_request.options
    poker_state.phase             -> graph.node(state.current_node)
                                      (.kind, .domain, .metadata)
    poker_state.last_showdown /
    poker_state.last_winners      -> engine.last_showdown /
                                      engine.last_winners (same shape)

and the "Run a hand" loop:

    while not engine.is_complete:
        req = engine.pending_request
        response = DecisionResponse(req.node_id, req.domain, req.player_index, value)
        submit_decision(engine, response, callbacks=cb)

which confirms DecisionRequest's real field names: `.node_id`,
`.domain`, `.player_index`, `.options` — the previous revision of this
file guessed `.seat`; that's fixed here throughout.

Remaining ASSUMPTIONS (not yet confirmed against real source):

  - engine.state exposes the same per-hand fields the old
    `poker_state.game` object did (players, node_cards, pot,
    street_index, dealer_position, bet_to_call, min_raise,
    raises_this_street, discard_pile). `_game_obj()` isolates this to
    one call site — tries `engine.state.game`, falls back to
    `engine.state` itself.
  - `engine.pending_request.options` entries are BettingOption-shaped
    with at least `.action_name` (str) and optionally `.min_amount`/
    `.max_amount` for bet/raise-shaped options, OR CardSelectOption-
    shaped with `.min_count`/`.max_count` and no action-identity field
    at all — the doc confirms `.options` replaces `legal_actions()`
    but not every option object's own field names; state_dto.py's own
    DecisionOptionDTO docstring is the source of truth for which
    fields matter per domain.
  - `graph.node(node_id).domain` is compared by `.name` (e.g.
    "BETTING") — the doc confirms `.domain`/`.kind`/`.metadata` exist
    on a graph node but not whether domain is a plain enum, a str, or
    something else with a `.name` attribute; `_domain_name()` below
    isolates that assumption.
"""

from poker_engine.cards.card import Card
from poker_engine.actions.action_type import ActionType as EngineActionType

from app.dto.state_dto import (
    DecisionOptionDTO,
    DecisionRequestDTO,
    GameStateDTO,
    PlayerDTO,
    PointDTO,
)

# Reuse the showdown-DTO builder unchanged — engine.last_showdown is
# the same ShowdownResult shape PokerState produced (doc confirms
# "same ShowdownResult").
from app.engine_adapter import build_showdown_dto, card_to_str


# --------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------


def _game_obj(engine):
    """See module docstring's ASSUMPTIONS."""
    inner = getattr(engine.state, "game", None)
    return inner if inner is not None else engine.state


def _domain_name(domain) -> str | None:
    if domain is None:
        return None
    return getattr(domain, "name", str(domain))


def _options_to_dto(g, req) -> list[DecisionOptionDTO]:
    """
    Translate engine.pending_request.options into DecisionOptionDTO
    entries.

    BUG FIXED HERE: CARD_SELECT/CARD_PASS-shaped options carry
    min_count/max_count but have NO action-identity field at all —
    they describe a SELECTABLE ZONE ("discard 0-3 cards from your
    hand"), not a named action like "fold"/"bet". _option_action_name()
    correctly returns None for these, but this function used to treat
    that None as "not a real option" and DROP it entirely. That
    silently emptied `options` for every CARD_SELECT/CARD_PASS
    decision — decision.options[0] didn't exist at all on the wire,
    which is exactly why the frontend reported "the server didn't
    report how many cards to select" with Confirm permanently
    disabled, even though min_count/max_count were present and correct
    on the underlying engine option the entire time.

    Only BETTING/CHOICE-shaped options are actually IDENTIFIED by
    their action_name (game_service.py's response builders match a
    submitted action/choice against it by name) — an unnamed option of
    that shape really would be unusable, so those are still dropped.
    A CARD_SELECT/CARD_PASS option is identified by its
    min_count/max_count (and position/metadata), never a name, so it's
    now always kept — action_name is set to a synthetic "select"
    purely so DecisionOptionDTO (which requires action_name: str)
    still validates; the frontend for these two domains never reads
    action_name, only decision.options[0].min_count/.max_count, per
    ActionRequest's own docstring in game_api.py.
    """
    options = getattr(req, "options", ()) or ()
    out = []
    for opt in options:
        name = _option_action_name(opt)
        min_count = getattr(opt, "min_count", None)
        max_count = getattr(opt, "max_count", None)
        is_card_select_shaped = min_count is not None or max_count is not None

        if name is None and not is_card_select_shaped:
            continue

        out.append(
            DecisionOptionDTO(
                action_name=name or "select",
                min_amount=getattr(opt, "min_amount", None),
                max_amount=getattr(opt, "max_amount", None),
                # CARD_SELECT/CARD_PASS-shaped options only — read as
                # plain top-level attributes to match
                # poker_engine.graph.decision_dto's CardSelectOption
                # shape 1:1 (NOT nested under opt.metadata — see
                # DecisionOptionDTO's own docstring in state_dto.py for
                # why that's the wrong place for a structural
                # constraint every CARD_SELECT/CARD_PASS renderer
                # needs).
                min_count=min_count,
                max_count=max_count,
                label=getattr(opt, "label", None),
                metadata=dict(getattr(opt, "metadata", {}) or {}),
            )
        )
    return out


def _betting_to_call_and_raise(g, req):
    """
    Convenience to_call/min_raise/max_raise for the common BETTING
    domain — read from the acting player's position in `g` (mirrors
    the old flat computation in engine_adapter.state_to_dto), not
    from the options themselves, since not every BETTING option is
    bet/raise-shaped (fold/call/check carry no amount).
    """
    player_index = getattr(req, "player_index", None)
    if player_index is None or player_index >= len(g.players):
        return None, None, None

    player = g.players[player_index]
    to_call = max(getattr(g, "bet_to_call", 0) - player.current_bet, 0)
    min_raise = getattr(g, "min_raise", None)

    max_raise = None
    for opt in getattr(req, "options", ()) or ():
        opt_max = getattr(opt, "max_amount", None)
        if opt_max is not None:
            max_raise = opt_max if max_raise is None else max(max_raise, opt_max)

    return to_call, min_raise, max_raise


def node_metadata(engine, graph) -> dict:
    """Pass-through of the current graph node's metadata — replaces
    creation_phases id-string lookups (street_index/street_name etc.
    live here per the migration doc)."""
    if graph is None:
        return {}
    node = graph.node(engine.state.current_node)
    return dict(getattr(node, "metadata", {}) or {})


# --------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------


def graph_state_to_dto(engine, game_def, rules, graph=None) -> GameStateDTO:
    g = _game_obj(engine)

    nodes = [card_to_str(c) if c is not None else None for c in g.node_cards]

    points = [
        PointDTO(
            name=point.name,
            score_type=str(point.score_type.name),
            node_sets=[list(ns) for ns in point.node_sets],
        )
        for point in rules.points
    ]

    players = []
    for i, p in enumerate(g.players):
        hand = []
        mask = p.hand_mask
        while mask:
            lsb = mask & -mask
            cid = lsb.bit_length() - 1
            hand.append(card_to_str(cid))
            mask ^= lsb

        players.append(
            PlayerDTO(
                seat=i + 1,
                name=f"Player {i + 1}",
                stack=p.stack,
                bet=p.current_bet,
                folded=p.has_folded,
                hand=hand,
            )
        )

    req = engine.pending_request
    decision = None
    current_player = None

    if req is not None:
        domain_name = _domain_name(getattr(req, "domain", None)) or "UNKNOWN"
        player_index = getattr(req, "player_index", None)
        seat = player_index + 1 if player_index is not None else None

        options = _options_to_dto(g, req)
        to_call = min_raise = max_raise = None
        if domain_name == "BETTING":
            to_call, min_raise, max_raise = _betting_to_call_and_raise(g, req)

        decision = DecisionRequestDTO(
            domain=domain_name,
            seat=seat,
            options=options,
            to_call=to_call,
            min_raise=min_raise,
            max_raise=max_raise,
            node_metadata=node_metadata(engine, graph),
        )
        current_player = seat

    hand_complete = bool(getattr(engine, "is_complete", False))

    showdown = None
    winners = None
    last_showdown = getattr(engine, "last_showdown", None)
    if last_showdown:
        active_players = [i for i, p in enumerate(g.players) if not p.has_folded]
        showdown = build_showdown_dto(
            last_showdown,
            rules,
            active_players,
            node_cards=g.node_cards,
            player_hand_masks={i: p.hand_mask for i, p in enumerate(g.players)},
        )
        winners = [p + 1 for p, amt in last_showdown.payouts.items() if amt > 0]
    else:
        last_winners = getattr(engine, "last_winners", None)
        if last_winners:
            winners = [w + 1 for w in last_winners]

    discard_pile = [card_to_str(cid) for cid in getattr(g, "discard_pile", [])]

    return GameStateDTO(
        street=getattr(g, "street_index", 0),
        pot=g.pot,
        nodes=nodes,
        layout_name=game_def.layout_name,
        game_name=game_def.game_name,
        street_names=None,  # GameDefinition.street_names removed — see module docstring
        points=points,
        players=players,
        decision=decision,
        hand_complete=hand_complete,
        current_player=current_player,
        showdown=showdown,
        winners=winners,
        discard_pile=discard_pile,
    )


# --------------------------------------------------------------------
# Option identity helpers — shared by _options_to_dto (frontend DTO)
# and game_service._build_decision_response (matching a submitted
# ActionRequest against the pending decision's legal options).
# --------------------------------------------------------------------


def _option_raw_action_value(opt):
    """
    Best-effort raw action-identity value off an option object of
    unconfirmed exact shape — tries the string-ish fields first
    (action_name/name), then falls back to an enum-ish field
    (type/action_type) that a BettingOption might carry instead.
    Returns None if nothing recognizable is found (expected and
    normal for a CARD_SELECT/CARD_PASS-shaped option — see
    _options_to_dto's own docstring; callers must not treat None here
    as "invalid option").
    """
    for attr in ("action_name", "name"):
        val = getattr(opt, attr, None)
        if val is not None:
            return val
    for attr in ("type", "action_type"):
        val = getattr(opt, attr, None)
        if val is not None:
            return val
    return None


def _option_action_name(opt) -> str | None:
    """
    Lowercase action-name string for an option, e.g. "fold", "bet" —
    unwraps an enum member (via .name) if that's what
    _option_raw_action_value returned instead of a plain string.
    """
    raw = _option_raw_action_value(opt)
    if raw is None:
        return None
    name = getattr(raw, "name", raw)
    return str(name).lower()


def _option_action_type_enum(opt) -> EngineActionType | None:
    """
    Resolve the real poker_engine.actions.action_type.ActionType enum
    member for an option, by matching its action name against
    ActionType's own member names (FOLD/CHECK/CALL/BET/RAISE/ALL_IN).
    Returns None if the option's name doesn't correspond to a known
    ActionType member — game_service._build_decision_response raises
    a clear ValueError in that case rather than silently guessing.
    """
    name = _option_action_name(opt)
    if name is None:
        return None
    try:
        return EngineActionType[name.upper()]
    except KeyError:
        return None