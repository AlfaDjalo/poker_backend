from app.dto.state_dto import GameStateDTO, PlayerDTO, PointDTO, PointResultDTO, PlayerBoardResultDTO, ShowdownDTO
from cards.card import Card
from poker_eval import ScoreType


def card_to_str(card_id):

    if card_id is None:
        return None
    
    return str(Card(card_id))


def state_to_dto(poker_state):

    g = poker_state.game
    game_def = poker_state.game_def
    rules = poker_state.rules

    street_names = getattr(game_def, "street_names", None)

    print("Game def: ", game_def)

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
    nodes = [
        card_to_str(c) if c is not None else None
        for c in g.node_cards
    ]
    
    # --------------------------------------------------
    # NEW: point definitions for frontend rendering
    # --------------------------------------------------    
    points = [
        PointDTO(
            name=point.name,
            score_type=str(point.score_type.name),
            node_sets=[list(ns) for ns in point.node_sets]
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
                seat = i + 1,
                name = f"Player {i+1}",
                stack = p.stack,
                bet = p.current_bet,
                folded = p.has_folded,
                hand = hand
            )
        )

    # --------------------------------------------------
    # Betting state
    # --------------------------------------------------
    # if g.current_player is not None:
    if poker_state.phase.name == "BETTING":
        current = g.current_player    
        player = g.players[current]
        to_call = g.bet_to_call - player.current_bet
        max_raise = player.stack
        actions = [a.name.lower() for a in g.legal_actions()]
    else:
        current = None
        to_call = 0
        max_raise = 0
        actions = []

    print("Actions (backend): ", actions)

    # Temporary - need to change to reflect betting type
    min_raise = g.min_raise
    # min_raise = max(g.min_raise, g.last_raise_size)
    # max_raise = player.stack

    current_player = (
        g.current_player + 1
        if poker_state.phase.name == "BETTING"
        else None
    )

    print("Phase: ", poker_state.phase)
    print("Current player: ", current_player)

    # --------------------------------------------------
    # Showdown
    # --------------------------------------------------
    showdown = None
    winners = None

    if hasattr(poker_state, "last_showdown") and poker_state.last_showdown:

        result = poker_state.last_showdown

        active_players = [
            i for i, p in enumerate(g.players)
            if not p.has_folded
        ]

        showdown = build_showdown_dto(result, rules, active_players)

        winners = [
            p + 1 for p, amt in result.payouts.items()
            if amt > 0
        ]

    return GameStateDTO(
        street = g.street_index,
        pot = g.pot,
        # board = board,
        nodes = nodes,
        layout_name = game_def.layout_name,
        game_name = game_def.game_name,
        street_names = street_names,
        points = points,
        players = players,
        current_player = current_player,
        phase = poker_state.phase.name,
        # hand_strengths = hand_strengths,

        showdown = showdown,
        winners = winners,

        available_actions = actions,
        to_call = max(0, to_call),
        min_raise = min_raise,
        max_raise = max_raise
    )

def build_showdown_dto(result, rules, active_players):
    """
    Build a rich showdown payload for the frontend.
    
    active_players: list of player indices in the order
                    the scoring engine evaluated them.
    """
    if result is None:
        return None
    
    # point_results = []

    grouped = {}

    for i, p in enumerate(result.points):
        # board_results = []
        key = p.name

        if key not in grouped:
            grouped[key] = {
                "score_type": p.score_type,
                "showdown_type": p.showdown_type,
                "boards": []
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

            players = []
            winners = []

            for r in results:
                
                is_winner = getattr(r, 'is_winner', False)
                p_index = getattr(r, 'player_index', None)

                if is_winner:
                    winners.append(p_index)

                players.append(PlayerBoardResultDTO(
                    player_index=p_index,
                    hand_category=getattr(r, 'category', None),
                    hand_value=getattr(r, 'value', 0),
                    best_hand_cards=[card_to_str(c) for c in getattr(r, 'best_hand_cards', [])],
                    hole_cards_used=[card_to_str(c) for c in getattr(r, 'hole_cards_used', [])],
                    board_cards_used=[card_to_str(c) for c in getattr(r, 'board_cards_used', [])],
                    is_winner=is_winner
                ))

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

        point_results_dto.append(PointResultDTO(
            name=name,
            score_type=str(data["score_type"]),
            board_winners=board_winners,
            board_results=board_results,
            no_qualify=no_qualify,
            scoop=scoop,
        ))

    pot_winners = [p for p, amt in result.payouts.items() if amt > 0]

    return ShowdownDTO(
        payout_type=result.payout_type,
        point_results=point_results_dto,
        point_tallies=result.point_tallies,
        payouts=result.payouts,
        pot_winners=pot_winners
    )
