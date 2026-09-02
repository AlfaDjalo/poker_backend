from poker_engine.cards.card import Card

from app.dto.state_dto import (
    DecisionOptionDTO,
    DecisionRequestDTO,
    GameStateDTO,
    PlayerBoardResultDTO,
    PlayerDTO,
    PointDTO,
    PointResultDTO,
    ShowdownDTO,
)

# Actions whose legality implies a chip amount range — the frontend's
# bet slider reads min_amount/max_amount off these options instead of
# the old top-level min_raise/max_raise pair.
_BET_LIKE_ACTIONS = {"bet", "raise"}


def card_to_str(card_id):

    if card_id is None:
        return None

    return str(Card(card_id))


def _mask_to_card_strs(mask: int) -> list:
    """LSB-iterate a 52-bit card bitmask into card strings — same idiom
    used elsewhere in this module's own hand-decoding loop and in
    poker_engine's mask_to_card_ids."""
    cards = []
    m = mask or 0
    while m:
        lsb = m & -m
        cid = lsb.bit_length() - 1
        cards.append(card_to_str(cid))
        m ^= lsb
    return cards


def _board_card_mask(node_cards, node_mask: int) -> int:
    """
    Convert a PointResult's node_mask — a bitmask of NODE POSITIONS
    ("bitmask of node indices (union across node_sets)", per the
    hand_points.node_set DB column's own doc comment), not card ids —
    into the actual 52-bit CARD mask for that board, by looking up
    which card is dealt at each set node position. Needed because
    best_hand_mask is a card mask; intersecting it against a
    node-POSITION mask directly would be comparing the wrong units.
    """
    if not node_cards or not node_mask:
        return 0
    mask = 0
    for node_idx, card_id in enumerate(node_cards):
        if card_id is None:
            continue
        if (node_mask >> node_idx) & 1:
            mask |= 1 << card_id
    return mask


def state_to_dto(poker_state):

    g = poker_state.game
    game_def = poker_state.game_def
    rules = poker_state.rules

    street_names = getattr(game_def, "street_names", None)

    # print("Game def: ", game_def)

    # --------------------------------------------------
    # Legacy flat board: first 5 nodes (holdem/omaha compat)
    # --------------------------------------------------
    board = []
    for c in g.node_cards[:5]:

        if c is None:
            board.append(None)
        else:
            board.append(card_to_str(c))

    # --------------------------------------------------
    # full node array — all nodes indexed by position
    # --------------------------------------------------
    nodes = [card_to_str(c) if c is not None else None for c in g.node_cards]

    # --------------------------------------------------
    # NEW: point definitions for frontend rendering
    # --------------------------------------------------
    points = [
        PointDTO(
            name=point.name,
            score_type=str(point.score_type.name),
            node_sets=[list(ns) for ns in point.node_sets],
        )
        for point in rules.points
    ]

    # --------------------------------------------------
    # Players
    # --------------------------------------------------
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
                name=f"Player {i+1}",
                stack=p.stack,
                bet=p.current_bet,
                folded=p.has_folded,
                hand=hand,
            )
        )

    # --------------------------------------------------
    # Decision request — populated only while a player action is
    # pending (phase == BETTING). None otherwise: DEAL_BOARD is
    # transient (auto-progressed before this is ever externally
    # visible — see GameService._progress_engine), and SHOWDOWN /
    # HAND_COMPLETE need no response from anyone.
    # --------------------------------------------------
    decision = None
    current_player = None

    if poker_state.phase.name == "BETTING":
        current = g.current_player
        player = g.players[current]
        to_call = max(g.bet_to_call - player.current_bet, 0)
        min_raise = g.min_raise

        betting_type = getattr(poker_state.game_def, "betting_type", "no_limit")
        if betting_type == "pot_limit":
            pot_raise_max = g.pot + 2 * to_call
            max_raise = min(player.stack, pot_raise_max)
        else:
            max_raise = player.stack

        action_names = [a.name.lower() for a in g.legal_actions()]
        options = [
            DecisionOptionDTO(
                action_name=name,
                min_amount=min_raise if name in _BET_LIKE_ACTIONS else None,
                max_amount=max_raise if name in _BET_LIKE_ACTIONS else None,
            )
            for name in action_names
        ]

        decision = DecisionRequestDTO(
            domain="BETTING",
            seat=current + 1,
            options=options,
            to_call=to_call,
            min_raise=min_raise,
            max_raise=max_raise,
            # PokerState has no graph node to pull metadata from —
            # street index/name are already top-level GameStateDTO
            # fields (street / street_names), so this is left empty on
            # the legacy path. The graph-engine adapter populates it
            # from graph.node(current_node).metadata instead.
            node_metadata={},
        )
        current_player = current + 1

    hand_complete = poker_state.phase.name == "HAND_COMPLETE"

    # --------------------------------------------------
    # Showdown
    # --------------------------------------------------
    showdown = None
    winners = None

    if hasattr(poker_state, "last_showdown") and poker_state.last_showdown:

        result = poker_state.last_showdown

        active_players = [i for i, p in enumerate(g.players) if not p.has_folded]

        showdown = build_showdown_dto(
            result,
            rules,
            active_players,
            node_cards=g.node_cards,
            player_hand_masks={i: p.hand_mask for i, p in enumerate(g.players)},
        )

        winners = [p + 1 for p, amt in result.payouts.items() if amt > 0]

    # --------------------------------------------------
    # Discard pile
    # --------------------------------------------------
    discard_pile = []
    raw_discard = getattr(g, "discard_pile", [])
    for cid in raw_discard:
        discard_pile.append(card_to_str(cid))

    return GameStateDTO(
        street=g.street_index,
        pot=g.pot,
        nodes=nodes,
        layout_name=game_def.layout_name,
        game_name=game_def.game_name,
        street_names=street_names,
        points=points,
        players=players,
        decision=decision,
        hand_complete=hand_complete,
        current_player=current_player,
        showdown=showdown,
        winners=winners,
        discard_pile=discard_pile,
    )


def build_showdown_dto(result, rules, active_players, node_cards=None, player_hand_masks=None):
    """
    Build a rich showdown payload for the frontend.

    active_players: list of player indices in the order
                    the scoring engine evaluated them.
    node_cards: g.node_cards (list[int|None], indexed by node position)
                — used with a PointResult's node_mask to derive the
                real card mask for that board (see _board_card_mask).
    player_hand_masks: {player_index: hand_mask} — each active
                player's own hole-card bitmask, for splitting
                best_hand_mask into its hole vs board components.

    NOTE — best_hand_cards/hole_cards_used/board_cards_used are now
    derived entirely from PlayerPointResult.best_hand_mask (a
    DB-confirmed real field — point_results.best_hand_mask is written
    verbatim from pr.best_hand_mask by both session_logger.py and
    tutorial_api.py) rather than read off possibly-nonexistent
    best_hand_cards/hole_cards_used/board_cards_used attributes
    directly. Those three previously came back via
    getattr(r, "...", []) with a silent [] default, and EVERY result
    — winners included — showed all three as [] regardless of
    hand_category/hand_value being correct (confirmed via a live
    showdown payload during testing), meaning at least one of those
    attribute names doesn't actually exist on the real
    PlayerPointResult and the getattr default was masking it rather
    than raising. Deriving from best_hand_mask sidesteps needing to
    know which of the pre-split attributes are real:
      - best_hand_cards      = decode(best_hand_mask)
      - hole_cards_used      = decode(best_hand_mask & player's own hand_mask)
      - board_cards_used     = decode(best_hand_mask & this board's card mask)

    node_cards/player_hand_masks default to None for backward
    compatibility with any other caller that doesn't have them handy
    — best_hand_cards still decodes fine in that case, but
    hole_cards_used/board_cards_used fall back to empty (same
    degraded behavior as before) since there's nothing to split
    against.
    """
    if result is None:
        return None

    player_hand_masks = player_hand_masks or {}

    # point_results = []

    grouped = {}

    for i, p in enumerate(result.points):
        # board_results = []
        key = p.name

        if key not in grouped:
            grouped[key] = {
                "score_type": p.score_type,
                "showdown_type": p.showdown_type,
                "boards": [],
            }

        grouped[key]["boards"].append(p)

    point_results_dto = []

    for point_idx, (name, data) in enumerate(grouped.items()):

        # boards = data["boards"]

        board_results = []
        board_winners = []
        no_qualify = []
        scoop = []

        for board_idx, board_obj in enumerate(data["boards"]):

            results = board_obj.results
            board_mask = _board_card_mask(
                node_cards, getattr(board_obj, "node_mask", 0)
            )

            players = []
            winners = []

            for r in results:

                is_winner = getattr(r, "is_winner", False)
                p_index = getattr(r, "player_index", None)

                if is_winner:
                    winners.append(p_index)

                best_hand_mask = getattr(r, "best_hand_mask", 0) or 0
                hole_mask = player_hand_masks.get(p_index, 0)

                players.append(
                    PlayerBoardResultDTO(
                        player_index=p_index,
                        hand_category=getattr(r, "category", None),
                        hand_value=getattr(r, "value", 0),
                        best_hand_cards=_mask_to_card_strs(best_hand_mask),
                        hole_cards_used=_mask_to_card_strs(
                            best_hand_mask & hole_mask
                        ),
                        board_cards_used=_mask_to_card_strs(
                            best_hand_mask & board_mask
                        ),
                        is_winner=is_winner,
                    )
                )

            board_results.append(players)
            board_winners.append(winners)
            no_qualify.append(len(winners) == 0)

            scoop_flag = False
            if result.scoop_flags:
                try:
                    scoop_flag = result.scoop_flags[point_idx][board_idx]
                except (IndexError, TypeError):
                    pass

            scoop.append(scoop_flag)

        point_results_dto.append(
            PointResultDTO(
                name=name,
                score_type=str(data["score_type"]),
                board_winners=board_winners,
                board_results=board_results,
                no_qualify=no_qualify,
                scoop=scoop,
            )
        )

    pot_winners = [p for p, amt in result.payouts.items() if amt > 0]

    return ShowdownDTO(
        payout_type=result.payout_type,
        point_results=point_results_dto,
        point_tallies=result.point_tallies,
        payouts=result.payouts,
        pot_winners=pot_winners,
    )