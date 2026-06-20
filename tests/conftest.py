"""
conftest.py — shared fixtures for the test suite.
Place this file in the same directory as your test_*.py files.
"""
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.db.base import Base
import app.db.models
from app.db.models.players import Player
from app.db.models.poker_tables import PokerTable
from app.db.models.poker_sessions import PokerSession

from types import SimpleNamespace

@pytest.fixture()
def db():
    """Fresh SQLite in-memory session per test, all tables created."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # print("TABLES:")
    # for name in sorted(Base.metadata.tables.keys()):
    #     print(name)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture()
def seeded_db(db):
    """
    1 PokerTable, 6 Players (Player 1..6), 1 PokerSession.

    TableSeat rows are NOT inserted here. Tests that need the seat map
    (RAPI-12, RAPI-13) insert them themselves so there are no
    UNIQUE(session_id, seat_number) collisions across fixtures.
    """
    from app.db.models.betting_config import BettingConfig
    from app.db.models.betting_config_details import BettingConfigDetails

    table = PokerTable(table_name="Test Table", max_players=6)
    db.add(table)
    db.flush()

    players = [Player(username=f"Player {i}", is_bot=False) for i in range(1, 7)]
    db.add_all(players)
    db.flush()

    bc = BettingConfig(betting_config_id=1, betting_config_name="default")
    db.add(bc)
    db.flush()    
    db.add_all([
        BettingConfigDetails(betting_config_id=1, bet_name="SB", bet_amount=1),
        BettingConfigDetails(betting_config_id=1, bet_name="BB", bet_amount=2),
    ])

    session = PokerSession(table_id=table.table_id)
    db.add(session)
    db.commit()

    return {"table": table, "players": players, "session": session, "betting_config": bc}


@pytest.fixture()
def seeded_client(db, seeded_db):
    """TestClient with get_db overridden to use the in-memory SQLite DB."""
    from app.main import app
    from app.api.deps import get_db

    # Use a concrete inner function instead of a broken lambda generator
    def _override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db

    # app.dependency_overrides[get_db] = lambda: (yield db)

    with TestClient(app) as client:
        yield client, seeded_db

    app.dependency_overrides.clear()

class FakePlayer:
    """Minimal stand-in for engine PlayerState — only attrs game_service.py touches."""
    def __init__(self, stack=100, hand_mask=0, current_bet=0,
                 total_contribution=0, has_folded=False, is_all_in=False):
        self.stack = stack
        self.hand_mask = hand_mask
        self.current_bet = current_bet
        self.total_contribution = total_contribution
        self.has_folded = has_folded
        self.is_all_in = is_all_in


class FakeGame:
    """Minimal stand-in for PokerState.game."""
    def __init__(self, players=None, street_index=0, pot=0, dealer_position=0,
                 current_player=0, bet_to_call=0, min_raise=2,
                 node_cards=None, discard_pile=None):
        self.players = players or [FakePlayer() for _ in range(2)]
        self.street_index = street_index
        self.pot = pot
        self.dealer_position = dealer_position
        self.current_player = current_player
        self.bet_to_call = bet_to_call
        self.min_raise = min_raise
        self.node_cards = node_cards if node_cards is not None else [None] * 5
        self.discard_pile = discard_pile if discard_pile is not None else []

    def legal_actions(self):
        """Stub — real engine returns ActionType enum members; an empty
        list is sufficient since state_to_dto only does `a.name.lower()`
        over whatever this returns."""
        return []        


class FakePokerState:
    """
    Minimal stand-in for PokerState — exposes only what game_service.py's
    edit methods and state_to_dto touch. Tests that need real DTO shape
    assembly should use the real engine (integration tier); this fixture
    is for GSE-* unit tests that only exercise GameService logic.
    """
    def __init__(self, game=None, game_def=None, rules=None, phase=None):
        self.game = game or FakeGame()
        self.game_def = game_def or SimpleNamespace(
            game_name="holdem", hole_cards=2, node_count=5,
            layout_name="single_board", street_names=None            
        )
        self.rules = rules or SimpleNamespace(points=[], showdown_type=0)
        self.phase = phase or SimpleNamespace(name="BETTING")
        self.last_showdown = None


class FakeLogger:
    """Stand-in for SessionLogger — records calls without touching the DB."""
    def __init__(self, hand_id=None):
        self.hand_id = hand_id
        self.calls = []

    def start_game(self, config):
        self.calls.append(("start_game", config))

    def start_hand(self, config):
        self.calls.append(("start_hand", config))
        self.hand_id = self.hand_id or 1

    def log_action(self, **kwargs):
        self.calls.append(("log_action", kwargs))

    def finish_hand(self, state):
        self.calls.append(("finish_hand", state))


@pytest.fixture()
def fake_player():
    """Factory for FakePlayer so tests can override fields per-case."""
    return FakePlayer


@pytest.fixture()
def fake_poker_state():
    """
    A GameService-ready fake PokerState with 2 players, holdem-shaped
    game_def, BETTING phase. Use .game.players[i] to mutate per test.
    """
    return FakePokerState()


@pytest.fixture()
def game_service_with_state(fake_poker_state):
    """
    A GameService instance wired to a fake PokerState + fake logger,
    bypassing restart()/the real engine entirely. For GSE-* tests that
    exercise begin_edit/apply_edit/load_edit/cancel_edit/_snapshot_state/etc.
    """
    from app.services.game_service import GameService

    svc = GameService()
    svc.state = fake_poker_state
    svc.current_game = fake_poker_state.game_def.game_name
    svc.logger = FakeLogger()
    svc.callbacks = None
    return svc


def make_edit_state_request(**overrides):
    """
    Builds a valid EditStateRequest payload (as a plain dict, for posting
    via TestClient, or import EditStateRequest directly to construct the
    pydantic model in service-level tests).

    Defaults: 2 players, holdem-shaped, no node cards dealt, empty discard.
    """
    base = {
        "game_name": "holdem",
        "street_index": 0,
        "pot": 0,
        "dealer_position": 0,
        "current_player": 0,
        "bet_to_call": 0,
        "min_raise": 2,
        "players": [
            {
                "seat": 1, "stack": 100, "current_bet": 0,
                "total_contribution": 0, "has_folded": False,
                "is_all_in": False, "hole_cards": ["Ah", "Kd"],
            },
            {
                "seat": 2, "stack": 100, "current_bet": 0,
                "total_contribution": 0, "has_folded": False,
                "is_all_in": False, "hole_cards": ["Qh", "Jc"],
            },
        ],
        "node_cards": [None, None, None, None, None],
        "discard_pile": [],
    }
    base.update(overrides)
    return base


@pytest.fixture()
def edit_state_request_factory():
    """Returns the make_edit_state_request callable so tests can override fields."""
    return make_edit_state_request