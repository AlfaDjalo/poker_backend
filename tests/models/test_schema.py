"""
§9.1 Database Schema Integrity Tests — DB-01 to DB-08

Tests run against SQLite in-memory via the shared `db` fixture in conftest.py.
"""

import pytest
from sqlalchemy.exc import IntegrityError

from app.db.base import Base
from app.db.models.players import Player
from app.db.models.poker_tables import PokerTable
from app.db.models.poker_sessions import PokerSession
from app.db.models.table_seating import TableSeat
from app.db.models.hands import Hand
from app.db.models.actions import Action
from app.db.models.hole_cards import HoleCard
from app.db.models.annotations import Annotation
from app.db.models.betting_config import BettingConfig


class TestSchemaCreation:
    def test_db01_create_all_succeeds(self, db):
        """DB-01: Base.metadata.create_all() succeeds on a fresh SQLite DB.
        The `db` fixture already called create_all; if we got here it worked."""
        # Verify at least one known table exists by querying it
        result = db.execute(
            __import__("sqlalchemy").text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
        table_names = {r[0] for r in result}
        assert "players" in table_names
        assert "hands" in table_names
        assert "actions" in table_names
        assert "poker_sessions" in table_names


class TestForeignKeyConstraints:
    def test_db03_hand_unknown_session_id_raises(self, db):
        """DB-03: Hand with unknown session_id raises FK / integrity violation."""
        hand = Hand(
            session_id=99999,
            variant_name="holdem",
            layout_name="single_board",
            split_pot=False,
        )
        db.add(hand)
        with pytest.raises(IntegrityError):
            db.flush()
        db.rollback()

    def test_db04_action_unknown_hand_id_raises(self, db):
        """DB-04: Action with unknown hand_id raises FK / integrity violation."""
        action = Action(
            hand_id=99999,
            street=0,
            action_index=0,
            action_type="call",
        )
        db.add(action)
        with pytest.raises(IntegrityError):
            db.flush()
        db.rollback()


class TestNullableConstraints:
    def test_db05_annotation_comment_not_null(self, db, seeded_db):
        """DB-05: Annotation.comment cannot be null."""
        hand = Hand(
            session_id=seeded_db["session"].session_id,
            variant_name="holdem",
            layout_name="single_board",
            split_pot=False,
        )
        db.add(hand)
        db.flush()

        ann = Annotation(
            hand_id=hand.hand_id,
            user_id=1,
            comment=None,  # should violate NOT NULL
        )
        db.add(ann)
        with pytest.raises((IntegrityError, Exception)):
            db.flush()
        db.rollback()

    def test_db07_hole_card_visible_defaults_false(self, db, seeded_db):
        """DB-07: HoleCard.visible defaults to False."""
        hand = Hand(
            session_id=seeded_db["session"].session_id,
            variant_name="holdem",
            layout_name="single_board",
            split_pot=False,
        )
        db.add(hand)
        db.flush()

        hc = HoleCard(
            hand_id=hand.hand_id,
            player_id=seeded_db["players"][0].player_id,
            street=0,
            card=0,
            # visible not set — should default to False
        )
        db.add(hc)
        db.flush()
        db.refresh(hc)
        assert hc.visible is False or hc.visible == 0


class TestUniqueConstraints:
    def test_db06_player_username_unique(self, db):
        """DB-06: Player.username has a unique constraint."""
        p1 = Player(username="UniqueUser", is_bot=False)
        p2 = Player(username="UniqueUser", is_bot=False)
        db.add(p1)
        db.flush()
        db.add(p2)
        with pytest.raises(IntegrityError):
            db.flush()
        db.rollback()

    def test_db08_table_seat_composite_pk(self, db, seeded_db):
        """DB-08: TableSeat PK is (session_id, seat_number) — duplicate raises IntegrityError."""
        session_id = seeded_db["session"].session_id
        player_id = seeded_db["players"][0].player_id

        seat1 = TableSeat(session_id=session_id, seat_number=99, player_id=player_id)
        db.add(seat1)
        db.flush()

        seat2 = TableSeat(session_id=session_id, seat_number=99, player_id=player_id)
        db.add(seat2)
        with pytest.raises(IntegrityError):
            db.flush()
        db.rollback()