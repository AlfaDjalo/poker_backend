from sqlalchemy import Column, ForeignKey, Integer, String

from app.db.base import Base


class CardEvent(Base):
    """
    Append-only ledger of every hole-card movement in a hand — the
    source of truth for discard/draw/pass-the-trash-style CAP variants.
    HoleCard (see hole_cards.py) is a DERIVED current-state table,
    written alongside this by the same service call; CardEvent is what
    the replayer walks to show "what actually happened" in order.

    event_type:
      DEALT      — card enters a player's hand from the deck.
                   from_player_id=None, to_player_id=recipient.
      DISCARDED  — card leaves a player's hand to the muck.
                   from_player_id=discarder, to_player_id=None.
      DRAWN      — replacement card enters a player's hand from the
                   deck (drawmaha-style). Same shape as DEALT
                   (from_player_id=None, to_player_id=recipient) —
                   kept as a distinct event_type rather than reusing
                   DEALT so the replayer can label it "drew a card"
                   vs. the initial deal without inferring it from
                   street/sequence position.
      PASSED     — card moves from one player's hand directly to
                   another's (pass-the-trash / Anaconda-style).
                   from_player_id=giver, to_player_id=receiver — ONE
                   row per card movement, not a pair; the replayer
                   reconstructs both sides from the two fields, so
                   there's no separate PASSED_OUT/PASSED_IN pairing
                   key to keep in sync.

    `sequence` orders events within a hand — same role
    Action.action_index plays for betting actions — since multiple
    events (e.g. every player's simultaneous discard) can share a
    `street` label but still need a stable replay order.

    `street` is a display label only (same graph-derived convention as
    Action.street / BoardCard.street), never used to drive flow logic.
    """

    __tablename__ = "card_events"

    card_event_id = Column(Integer, primary_key=True)
    hand_id = Column(Integer, ForeignKey("hands.hand_id"), nullable=False)
    card = Column(Integer, nullable=False)
    event_type = Column(String, nullable=False)  # DEALT | DISCARDED | DRAWN | PASSED
    from_player_id = Column(Integer, ForeignKey("players.player_id"), nullable=True)
    to_player_id = Column(Integer, ForeignKey("players.player_id"), nullable=True)
    street = Column(Integer)
    sequence = Column(Integer, nullable=False)