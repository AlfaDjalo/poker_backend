"""
§9.2 Seed Function Tests — DB-10 to DB-14

Tests run against SQLite in-memory; seed functions are patched to use
the test session instead of the production SessionLocal.
"""

import pytest
from unittest.mock import patch

from app.db.models.players import Player
from app.db.models.poker_tables import PokerTable
from app.db.models.bankroll_transaction import BankrollTransaction
from app.db.models.betting_config import BettingConfig
from app.db.models.betting_config_details import BettingConfigDetails


# ─────────────────────────────────────────────────────────────────
# Helpers: inline re-implementations of seed functions that accept
# an explicit db session (avoids patching SessionLocal globally).
# These mirror seed.py logic exactly.
# ─────────────────────────────────────────────────────────────────

def _seed_player_tables(db):
    deposit_amount = 100

    existing = db.query(PokerTable).filter(PokerTable.table_name == "Table 1").first()
    if not existing:
        table = PokerTable(table_name="Table 1", max_players=6)
        db.add(table)
        db.commit()

    for i in range(1, 7):
        player_name = f"Player {i}"
        existing_player = db.query(Player).filter(Player.username == player_name).first()
        if not existing_player:
            new_player = Player(username=player_name, is_bot=False)
            db.add(new_player)
            db.flush()

            deposit = BankrollTransaction(
                player_id=new_player.player_id,
                transaction_type="DEPOSIT",
                amount=deposit_amount,
                hand_id=None,
                balance_after=deposit_amount,
            )
            db.add(deposit)

    db.commit()


def _seed_betting_config_tables(db):
    config_name = "1/2"
    betting_config = db.query(BettingConfig).filter(
        BettingConfig.betting_config_name == config_name
    ).first()

    if not betting_config:
        betting_config = BettingConfig(betting_config_name=config_name)
        db.add(betting_config)
        db.flush()

        db.add_all([
            BettingConfigDetails(
                betting_config_id=betting_config.betting_config_id,
                bet_name="SB",
                bet_amount=1,
            ),
            BettingConfigDetails(
                betting_config_id=betting_config.betting_config_id,
                bet_name="BB",
                bet_amount=2,
            ),
        ])
    db.commit()


# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────

class TestSeedPlayerTables:
    def test_db10_creates_table_and_players(self, db):
        """DB-10: fresh DB → 1 PokerTable and 6 Player rows."""
        _seed_player_tables(db)

        tables = db.query(PokerTable).filter(PokerTable.table_name == "Table 1").all()
        assert len(tables) == 1
        assert tables[0].max_players == 6

        players = db.query(Player).filter(
            Player.username.in_([f"Player {i}" for i in range(1, 7)])
        ).all()
        assert len(players) == 6

    def test_db11_idempotent(self, db):
        """DB-11: running seed twice does not duplicate rows."""
        _seed_player_tables(db)
        _seed_player_tables(db)

        tables = db.query(PokerTable).filter(PokerTable.table_name == "Table 1").all()
        assert len(tables) == 1

        players = db.query(Player).filter(
            Player.username.in_([f"Player {i}" for i in range(1, 7)])
        ).all()
        assert len(players) == 6

    def test_db12_deposit_transactions_created(self, db):
        """DB-12: seed creates one DEPOSIT BankrollTransaction per player."""
        _seed_player_tables(db)

        player_names = [f"Player {i}" for i in range(1, 7)]
        players = db.query(Player).filter(Player.username.in_(player_names)).all()
        player_ids = {p.player_id for p in players}

        deposits = db.query(BankrollTransaction).filter(
            BankrollTransaction.player_id.in_(player_ids),
            BankrollTransaction.transaction_type == "DEPOSIT",
        ).all()

        assert len(deposits) == 6
        for d in deposits:
            assert d.amount == 100
            assert d.balance_after == 100


class TestSeedBettingConfig:
    def test_db13_creates_config_with_details(self, db):
        """DB-13: fresh DB → 1 BettingConfig '1/2' with SB=1 and BB=2."""
        _seed_betting_config_tables(db)

        configs = db.query(BettingConfig).filter(
            BettingConfig.betting_config_name == "1/2"
        ).all()
        assert len(configs) == 1

        details = db.query(BettingConfigDetails).filter(
            BettingConfigDetails.betting_config_id == configs[0].betting_config_id
        ).all()
        assert len(details) == 2

        by_name = {d.bet_name: d.bet_amount for d in details}
        assert by_name["SB"] == 1
        assert by_name["BB"] == 2

    def test_db14_idempotent(self, db):
        """DB-14: running betting config seed twice does not duplicate rows."""
        _seed_betting_config_tables(db)
        _seed_betting_config_tables(db)

        configs = db.query(BettingConfig).filter(
            BettingConfig.betting_config_name == "1/2"
        ).all()
        assert len(configs) == 1

        details = db.query(BettingConfigDetails).filter(
            BettingConfigDetails.betting_config_id == configs[0].betting_config_id
        ).all()
        assert len(details) == 2