from sqlalchemy import Boolean, Column, ForeignKey, Integer, String

from app.db.base import Base


class HoleCard(Base):
    """
    DERIVED current-state view of a player's hole cards — one row per
    card a player has ever held during a hand. The source of truth for
    HOW a card arrived/left is card_events.py's CardEvent ledger; this
    table exists so the many existing readers (replay_api.get_hand,
    tutorial_api's edit-state reconstruction, equity/showdown scoring)
    can keep querying "what does this player's hand look like" without
    replaying the event log on every read.

    Written by the SAME service call that inserts the corresponding
    CardEvent row (see session_logger.py / a future
    _record_card_movement() helper) — never mutated independently of
    it, so the two tables can't drift.

    `status` replaces the old single `discarded` boolean: DISCARDED
    (went to the muck) and PASSED_OUT (given to another player) are
    different outcomes for the replayer to render, and only one of the
    two "this card is no longer in {player}'s hand" reasons — a bool
    can't distinguish them.

      IN_HAND     — currently part of this player's hand.
      DISCARDED   — discarded to the muck (drawmaha-style).
      PASSED_OUT  — passed to another player (pass-the-trash-style);
                    the receiving player gets their OWN new HoleCard
                    row (status=IN_HAND) for the same `card` value,
                    not a mutation of this row — this row stays as the
                    historical record of "the card X held before
                    passing it away."

    A drawn replacement card gets its own new row (status=IN_HAND),
    same reasoning as a passed-in card — never an in-place mutation of
    a DISCARDED row's `card` value, so "what was in hand at each
    point" stays reconstructable from this table alone.

    Every reader that means "this player's current/showdown hand" MUST
    filter `status == "IN_HAND"` — earlier code that read every
    HoleCard row unconditionally was only correct because nothing
    could leave a hand mid-hand yet.

    `street` remains a display label only (graph-derived), same
    convention as Action.street / BoardCard.street / CardEvent.street.
    """

    __tablename__ = "hole_cards"

    hole_card_id = Column(Integer, primary_key=True)
    hand_id = Column(Integer, ForeignKey("hands.hand_id"))
    player_id = Column(Integer, ForeignKey("players.player_id"))
    street = Column(Integer)
    card = Column(Integer)
    visible = Column(Boolean, default=False)
    status = Column(String, nullable=False, default="IN_HAND")