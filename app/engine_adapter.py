from app.dto.state_dto import GameStateDTO, PlayerDTO
from cards.card import Card


def card_to_str(card_id):

    if card_id is None:
        return None
    
    return str(Card(card_id))


def state_to_dto(poker_state):

    g = poker_state.game

    board = []

    for c in g.node_cards[:5]:

        if c is None:
            board.append(None)
        else:
            board.append(card_to_str(c))

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

    # if g.current_player is not None:
    if poker_state.phase.name == "BETTING":
        current = g.current_player    
        player = g.players[current]
        to_call = g.bet_to_call - player.current_bet
        max_raise = player.stack
    else:
        current = None
        to_call = 0
        max_raise = 0

    # current = g.current_player
    # player = g.players[current]

    # to_call = g.bet_to_call - player.current_bet

    if poker_state.phase.name == "BETTING":
        actions = [a.name.lower() for a in g.legal_actions()]
    else:
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

    showdown = None
    winners = None

    if hasattr(poker_state, "last_showdown") and poker_state.last_showdown:

        result = poker_state.last_showdown

        showdown = {
            "payouts": result.payouts,
            "winners_by_pot": result.winners_by_pot,
            "scores": [
                {
                    "score_type": str(p["score_type"]),
                    "boards": p["boards"],
                    "scores": p["scores"],
                }
                for p in result.scores
            ],
            "boards": result.boards,
        }

        winners = [
            p + 1 for p, amt in result.payouts.items()
            if amt > 0
        ]

    return GameStateDTO(
        street = g.street_index,
        pot = g.pot,
        board = board,
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