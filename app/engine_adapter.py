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

        mask = p.mask

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

    return GameStateDTO(
        street = g.street_index,
        pot = g.pot,
        board = board,
        players = players,
        current_player = g.current_player + 1
    )