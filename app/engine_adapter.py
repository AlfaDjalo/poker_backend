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

    current = g.current_player
    player = g.players[current]

    to_call = g.bet_to_call - player.current_bet

    actions = [a.name.lower() for a in g.legal_actions()]

    print("Actions (backend): ", actions)
    # actions = []

    # if not player.has_folded:

    #     actions.append("fold")

    #     if to_call > 0:
    #         actions.append("call")
    #     else:
    #         actions.append("check")

    #     if player.stack > to_call:

    #         if g.bet_to_call == 0:
    #             actions.append("bet")
    #         else:
    #             actions.append("raise")

    # Temporary - need to change to reflect betting type
    min_raise = g.min_raise
    # min_raise = max(g.min_raise, g.last_raise_size)
    max_raise = player.stack

    current_player = (
        g.current_player + 1
        if poker_state.phase.name == "BETTING"
        else None
    )

    print("Phase: ", poker_state.phase)
    print("Current player: ", current_player)

    # hand_strengths = (
    #     poker_state.last_showdown.hand_ranks
    #     if poker_state.lastshowdown else None
    # )

    return GameStateDTO(
        street = g.street_index,
        pot = g.pot,
        board = board,
        players = players,
        current_player = current_player,
        phase = poker_state.phase.name,
        # hand_strengths = hand_strengths,

        available_actions = actions,
        to_call = max(0, to_call),
        min_raise = min_raise,
        max_raise = max_raise
    )