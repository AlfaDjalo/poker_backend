"""
tests/benchmarks/test_api_throughput.py
BENCH-BE-01 … BENCH-BE-07

Requires: pytest-benchmark
Install:  pip install pytest-benchmark

Run benchmarks only:
    pytest tests/benchmarks/ --benchmark-only -v

Run with comparison against previous baseline:
    pytest tests/benchmarks/ --benchmark-compare

Note: These are informational — do not gate CI on thresholds unless a
regression budget is explicitly defined (per §11 of the Backend Test Plan).
"""

import pytest
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────

MOCK_STATE = {
    "street": 0,
    "pot": 100,
    "nodes": [None] * 5,
    "layout_name": "single_board",
    "game_name": "holdem",
    "street_names": None,
    "points": [],
    "players": [
        {
            "seat": i,
            "name": f"Player {i}",
            "stack": 100,
            "bet": 0,
            "folded": False,
            "hand": [],
        }
        for i in range(1, 7)
    ],
    "current_player": 1,
    "phase": "BETTING",
    "showdown": None,
    "winners": None,
    "available_actions": ["call", "fold", "raise"],
    "to_call": 2,
    "min_raise": 4,
    "max_raise": 100,
}

MOCK_EQUITY_RESPONSE = {
    "equity": {1: {"board1": 0.55}, 2: {"board1": 0.45}},
    "method": "exact",
    "iterations": 1326,
    "elapsed_ms": 10.0,
}

MOCK_EQUITY_RESPONSE_PLO = {
    "equity": {
        1: {"board1": 0.30},
        2: {"board1": 0.25},
        3: {"board1": 0.25},
        4: {"board1": 0.20},
    },
    "method": "monte_carlo",
    "iterations": 20000,
    "elapsed_ms": 800.0,
}


@pytest.fixture(scope="module")
def db_module():
    """
    Module-scoped SQLite in-memory DB — created once, shared across all
    benchmarks in this module for speed.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db.base import Base

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="module")
def app_client(db_module):
    """
    Module-scoped TestClient with game_service and equity_service mocked.
    Uses a module-scoped DB so the schema is created once for the whole
    benchmark session rather than per-test.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.deps import get_db

    app.dependency_overrides[get_db] = lambda: (yield db_module)

    with patch("app.api.game_api.game_service") as mock_gs, \
         patch("app.api.equity_api.equity_service") as mock_es:

        mock_gs.get_state.return_value = MOCK_STATE
        mock_gs.apply_action.return_value = MOCK_STATE
        mock_gs.restart.return_value = MOCK_STATE
        mock_es.calculate.return_value = MOCK_EQUITY_RESPONSE

        with TestClient(app) as client:
            yield client, mock_gs, mock_es

    app.dependency_overrides.clear()


# ─────────────────────────────────────────────────────────────────
# BENCH-BE-01  GET /game/state  (no DB, cached state)
# Target: < 5 ms
# ─────────────────────────────────────────────────────────────────

def test_bench_be_01_get_state(benchmark, app_client):
    """BENCH-BE-01: GET /game/state — target < 5 ms"""
    client, _, _ = app_client

    result = benchmark(client.get, "/game/state")
    assert result.status_code == 200


# ─────────────────────────────────────────────────────────────────
# BENCH-BE-02  POST /game/action  (single betting action)
# Target: < 20 ms
# ─────────────────────────────────────────────────────────────────

def test_bench_be_02_apply_action(benchmark, app_client):
    """BENCH-BE-02: POST /game/action — target < 20 ms"""
    client, _, _ = app_client

    def _call():
        return client.post("/game/action", json={"type": "call"})

    result = benchmark(_call)
    assert result.status_code == 200


# ─────────────────────────────────────────────────────────────────
# BENCH-BE-03  POST /equity/calculate  — 2 players, holdem, exact
# Target: < 500 ms
# ─────────────────────────────────────────────────────────────────

EQUITY_REQUEST_2P = {
    "variant_name": "holdem",
    "players": [
        {"seat": 1, "hole_cards": ["Ah", "Kd"]},
        {"seat": 2, "hole_cards": ["Qh", "Jc"]},
    ],
    "board_nodes": [],
}


def test_bench_be_03_equity_2p_holdem(benchmark, app_client):
    """BENCH-BE-03: equity/calculate 2-player holdem exact — target < 500 ms"""
    client, _, mock_es = app_client
    mock_es.calculate.return_value = MOCK_EQUITY_RESPONSE

    def _call():
        return client.post("/equity/calculate", json=EQUITY_REQUEST_2P)

    result = benchmark(_call)
    assert result.status_code == 200


# ─────────────────────────────────────────────────────────────────
# BENCH-BE-04  POST /equity/calculate  — 4 players, PLO, Monte Carlo
# Target: < 2 s
# ─────────────────────────────────────────────────────────────────

EQUITY_REQUEST_4P_PLO = {
    "variant_name": "plo",
    "players": [
        {"seat": 1, "hole_cards": ["Ah", "Kd", "Qh", "Jc"]},
        {"seat": 2, "hole_cards": ["Ts", "9h", "8d", "7c"]},
        {"seat": 3, "hole_cards": ["6s", "5h", "4d", "3c"]},
        {"seat": 4, "hole_cards": ["2s", "Kh", "Qd", "Jh"]},
    ],
    "board_nodes": [],
    "mc_iterations": 20000,
}


def test_bench_be_04_equity_4p_plo_mc(benchmark, app_client):
    """BENCH-BE-04: equity/calculate 4-player PLO Monte Carlo — target < 2 s"""
    client, _, mock_es = app_client
    mock_es.calculate.return_value = MOCK_EQUITY_RESPONSE_PLO

    def _call():
        return client.post("/equity/calculate", json=EQUITY_REQUEST_4P_PLO)

    result = benchmark(_call)
    assert result.status_code == 200


# ─────────────────────────────────────────────────────────────────
# BENCH-BE-05  GET /replay/hands/{id}  — hand with 50 actions
# Target: < 100 ms
# ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def seeded_replay_hand(db_module):
    """
    Insert a single hand with 50 actions into the module-scoped DB.
    Returns the hand_id for use in the benchmark.
    """
    from app.db.models.poker_tables import PokerTable
    from app.db.models.players import Player
    from app.db.models.poker_sessions import PokerSession
    from app.db.models.hands import Hand
    from app.db.models.actions import Action

    db = db_module

    table = PokerTable(table_name="BenchTable", max_players=6)
    db.add(table)
    db.flush()

    players = [Player(username=f"BenchPlayer{i}", is_bot=False) for i in range(1, 7)]
    db.add_all(players)
    db.flush()

    session = PokerSession(table_id=table.table_id)
    db.add(session)
    db.flush()

    hand = Hand(
        session_id=session.session_id,
        variant_name="holdem",
        layout_name="single_board",
        split_pot=False,
        betting_config_id=None,
        dealer_seat=1,
        pot=300,
    )
    db.add(hand)
    db.flush()

    action_types = ["call", "raise", "fold", "check", "call"]
    for i in range(50):
        p = players[i % 6]
        db.add(Action(
            hand_id=hand.hand_id,
            street=i // 10,
            action_index=i,
            player_id=p.player_id,
            action_type=action_types[i % len(action_types)],
            amount=None,
            stack_before=100 - i,
            pot_before=i * 2,
        ))
    db.commit()

    return hand.hand_id


def test_bench_be_05_replay_hand_50_actions(benchmark, app_client, seeded_replay_hand):
    """BENCH-BE-05: GET /replay/hands/{id} with 50 actions — target < 100 ms"""
    client, _, _ = app_client
    hand_id = seeded_replay_hand

    def _call():
        return client.get(f"/replay/hands/{hand_id}")

    result = benchmark(_call)
    assert result.status_code == 200


# ─────────────────────────────────────────────────────────────────
# BENCH-BE-06  SessionLogger.finish_hand()  — 6-player PLO8
# Target: < 50 ms  (DB write path only)
# ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def logger_with_active_hand(db_module):
    """
    Set up a SessionLogger with an active hand so finish_hand() can be
    called repeatedly by the benchmark.
    """
    from app.db.models.poker_tables import PokerTable
    from app.db.models.players import Player
    from app.db.models.poker_sessions import PokerSession
    from app.services.session_logger import SessionLogger

    db = db_module

    table = PokerTable(table_name="LoggerBenchTable", max_players=6)
    db.add(table)
    db.flush()

    players = [Player(username=f"LBPlayer{i}", is_bot=False) for i in range(1, 7)]
    db.add_all(players)
    db.flush()

    session = PokerSession(table_id=table.table_id)
    db.add(session)
    db.commit()

    lg = SessionLogger(db)
    lg.start_game({
        "table_id": table.table_id,
        "player_ids": [p.player_id for p in players],
    })

    return lg, players


def _make_plo8_showdown_result(num_players):
    """Build a mock showdown result for a PLO8 hand."""
    hi_results = [
        MagicMock(
            player_index=i,
            is_winner=(i == 0),
            category="FullHouse" if i == 0 else "Pair",
            value=6_000_001 if i == 0 else 1_000_001,
            best_hand_mask=0,        # ← add this
            rank=1,                  # ← add this
            best_hand_cards=[],
            hole_cards_used=[],
            board_cards_used=[],
            share=1.0 if i == 0 else 0.0,
        )
        for i in range(num_players)
    ]
    lo_results = [
        MagicMock(
            player_index=i,
            is_winner=(i == 1),
            category="Low",
            value=1,
            best_hand_mask=0,        # ← add this
            rank=1,                  # ← add this
            best_hand_cards=[],
            hole_cards_used=[],
            board_cards_used=[],
            share=1.0 if i == 1 else 0.0,
        )
        for i in range(num_players)
    ]

    hi_point = MagicMock()
    hi_point.name = "high"
    hi_point.showdown_type = "HOLDEM"
    hi_point.score_type = "HIGH"
    hi_point.node_mask = 0b11111
    hi_point.results = hi_results

    lo_point = MagicMock()
    lo_point.name = "low"
    lo_point.showdown_type = "HOLDEM"
    lo_point.score_type = "LOW_A5"
    lo_point.node_mask = 0b11111
    lo_point.results = lo_results

    showdown = MagicMock()
    showdown.points = [hi_point, lo_point]
    showdown.payouts = {0: 150, 1: 150, **{i: 0 for i in range(2, num_players)}}

    return showdown


def test_bench_be_06_session_logger_finish_hand(benchmark, logger_with_active_hand):
    """BENCH-BE-06: SessionLogger.finish_hand() 6-player PLO8 — target < 50 ms"""
    lg, players = logger_with_active_hand

    def _call():
        # Re-start the hand each iteration so finish_hand() has a valid hand_id
        lg.start_hand({
            "variant_name": "plo8",
            "layout_name": "single_board",
            "split_pot": True,
            "betting_config_id": 1,
            "dealer_seat": 1,
            "pot": 0,
            "ended_at": None,
            "players": [MagicMock(hand_mask=0b11110000) for _ in players],
            "game_def": MagicMock(street_nodes=[[0, 1, 2], [3], [4]]),
        })
        state = MagicMock()
        state.game = MagicMock()
        state.game.node_cards = [10, 20, 30, 40, 50]
        state.game.players = [MagicMock(hand_mask=0b1111) for _ in players]
        state.last_showdown = _make_plo8_showdown_result(len(players))
        lg.finish_hand(state)

    benchmark(_call)


# ─────────────────────────────────────────────────────────────────
# BENCH-BE-07  Full hand lifecycle
# restart → actions (fold all but winner) → showdown
# Target: < 500 ms
# ─────────────────────────────────────────────────────────────────

def test_bench_be_07_full_hand_lifecycle(benchmark, app_client):
    """
    BENCH-BE-07: Full lifecycle — restart → actions → showdown.
    Uses mocked game_service so this measures HTTP + serialisation overhead,
    not engine performance.  Target: < 500 ms total.
    """
    client, mock_gs, _ = app_client

    showdown_state = {
        **MOCK_STATE,
        "phase": "HAND_COMPLETE",
        "showdown": {
            "payout_type": "split_pot",
            "point_results": [],
            "point_tallies": None,
            "payouts": {0: 300},
            "pot_winners": [0],
        },
        "winners": [1],
    }

    def _lifecycle():
        mock_gs.restart.return_value = MOCK_STATE
        mock_gs.apply_action.side_effect = [
            MOCK_STATE, MOCK_STATE, MOCK_STATE, MOCK_STATE, MOCK_STATE, showdown_state,
        ]

        r = client.post("/game/restart", json={})
        assert r.status_code == 200

        for _ in range(5):
            r = client.post("/game/action", json={"type": "fold"})
            assert r.status_code == 200

        r = client.post("/game/action", json={"type": "fold"})
        assert r.status_code == 200

    benchmark(_lifecycle)