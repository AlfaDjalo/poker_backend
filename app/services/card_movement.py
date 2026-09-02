"""
app/services/card_movement.py — shared write path for HoleCard +
CardEvent, used by both SessionLogger (live games) and
tutorial_api.py (hypothetical hands), so the two tables can never
drift out of sync no matter which caller wrote them.

Every card movement in a hand — the initial deal, a drawmaha-style
discard/draw, or a pass-the-trash transfer — goes through one of the
three functions here. Each writes exactly one CardEvent row (the
source-of-truth ledger — see card_events.py) and mutates HoleCard (the
derived current-state table — see hole_cards.py) to match, so a caller
never has to remember to touch both tables itself.

None of these commit — callers batch within their own transaction,
same convention SessionLogger / tutorial_api._write_hand_payload
already follow elsewhere.
"""

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models.card_events import CardEvent
from app.db.models.hole_cards import HoleCard


def _next_sequence(db: Session, hand_id: int) -> int:
    """
    Next sequence number for this hand's CardEvent ledger. Queried
    fresh rather than cached on a stateful logger — card movements are
    infrequent relative to actions/board deals, and this keeps the
    write path usable from both SessionLogger's per-process instance
    and tutorial_api's stateless per-request handlers without needing
    to share counter state between them.
    """
    current_max = (
        db.query(func.max(CardEvent.sequence))
        .filter(CardEvent.hand_id == hand_id)
        .scalar()
    )
    return (current_max or -1) + 1


def deal_card(
    db: Session,
    hand_id: int,
    player_id: int,
    card: int,
    street: int = 0,
    visible: bool = False,
    drawn: bool = False,
) -> HoleCard:
    """
    Record a card entering a player's hand from the deck — the
    initial deal (drawn=False) or a drawmaha-style replacement
    (drawn=True, logged as event_type DRAWN instead of DEALT so the
    replayer can label it correctly without inferring it from
    position in the sequence).
    """
    event = CardEvent(
        hand_id=hand_id,
        card=card,
        event_type="DRAWN" if drawn else "DEALT",
        from_player_id=None,
        to_player_id=player_id,
        street=street,
        sequence=_next_sequence(db, hand_id),
    )
    db.add(event)

    hole_card = HoleCard(
        hand_id=hand_id,
        player_id=player_id,
        street=street,
        card=card,
        visible=visible,
        status="IN_HAND",
    )
    db.add(hole_card)
    db.flush()
    return hole_card


def discard_card(
    db: Session,
    hand_id: int,
    player_id: int,
    card: int,
    street: int = 0,
) -> HoleCard:
    """
    Record a card leaving a player's hand to the muck. Flips the
    existing IN_HAND HoleCard row for (hand_id, player_id, card) to
    DISCARDED — raises if no such row exists, since discarding a card
    the player was never dealt (or already discarded/passed away) is
    a real bug, not something to silently no-op through.
    """
    hole_card = (
        db.query(HoleCard)
        .filter(
            HoleCard.hand_id == hand_id,
            HoleCard.player_id == player_id,
            HoleCard.card == card,
            HoleCard.status == "IN_HAND",
        )
        .first()
    )
    if hole_card is None:
        raise ValueError(
            f"Cannot discard card {card!r} for player {player_id} in hand "
            f"{hand_id} — no IN_HAND HoleCard row found (already "
            f"discarded/passed, or never dealt)."
        )

    event = CardEvent(
        hand_id=hand_id,
        card=card,
        event_type="DISCARDED",
        from_player_id=player_id,
        to_player_id=None,
        street=street,
        sequence=_next_sequence(db, hand_id),
    )
    db.add(event)

    hole_card.status = "DISCARDED"
    db.flush()
    return hole_card


def pass_card(
    db: Session,
    hand_id: int,
    from_player_id: int,
    to_player_id: int,
    card: int,
    street: int = 0,
    visible: bool = False,
) -> HoleCard:
    """
    Record a card moving directly from one player's hand to another's
    (pass-the-trash / Anaconda-style) — ONE CardEvent row with both
    from_player_id and to_player_id set, not a DISCARDED+DEALT pair,
    so the replayer can render it as a single transfer rather than
    stitching two unrelated-looking events back together.

    Flips the giver's HoleCard row to PASSED_OUT and inserts a NEW
    HoleCard row for the receiver (status=IN_HAND) — never mutates the
    giver's row into the receiver's, so "who held this card before it
    moved" stays queryable straight off HoleCard without consulting
    CardEvent.
    """
    hole_card = (
        db.query(HoleCard)
        .filter(
            HoleCard.hand_id == hand_id,
            HoleCard.player_id == from_player_id,
            HoleCard.card == card,
            HoleCard.status == "IN_HAND",
        )
        .first()
    )
    if hole_card is None:
        raise ValueError(
            f"Cannot pass card {card!r} from player {from_player_id} in "
            f"hand {hand_id} — no IN_HAND HoleCard row found."
        )

    event = CardEvent(
        hand_id=hand_id,
        card=card,
        event_type="PASSED",
        from_player_id=from_player_id,
        to_player_id=to_player_id,
        street=street,
        sequence=_next_sequence(db, hand_id),
    )
    db.add(event)

    hole_card.status = "PASSED_OUT"

    received = HoleCard(
        hand_id=hand_id,
        player_id=to_player_id,
        street=street,
        card=card,
        visible=visible,
        status="IN_HAND",
    )
    db.add(received)
    db.flush()
    return received