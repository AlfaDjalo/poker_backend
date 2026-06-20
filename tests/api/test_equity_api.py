"""
tests/api/test_equity_api.py
EAPI-01 ... EAPI-15, EAPI-P-01 ... EAPI-P-03
"""
import pytest
from unittest.mock import patch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


# VALID_CARDS must be defined before any class that references it
VALID_CARDS = [
    f"{rank}{suit}"
    for rank in "23456789TJQKA"
    for suit in "cdhs"
]

MOCK_EQUITY_RESPONSE = {
    "equity": {"1": {"board1": 0.6}, "2": {"board1": 0.4}},
    "method": "exact",
    "iterations": 1326,
    "elapsed_ms": 12.5,
}


@pytest.fixture()
def mock_equity_service():
    with patch("app.api.equity_api.equity_service") as mock:
        mock.calculate.return_value = {
            "equity": {1: {"board1": 0.6}, 2: {"board1": 0.4}},
            "method": "exact",
            "iterations": 1326,
            "elapsed_ms": 12.5,
        }
        yield mock


@pytest.fixture()
def equity_client(db, mock_equity_service):
    from app.main import app
    from app.api.deps import get_db
    app.dependency_overrides[get_db] = lambda: (yield db)
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


VALID_REQUEST = {
    "variant_name": "holdem",
    "players": [
        {"seat": 1, "hole_cards": ["Ah", "Kd"]},
        {"seat": 2, "hole_cards": ["Qh", "Jc"]},
    ],
    "board_nodes": [],
}


# ── Input Validation ──────────────────────────────────────────────────────────

class TestEquityInputValidation:
    def test_eapi_01_all_null_cards(self, equity_client):
        """EAPI-01 — all hole cards null -> 422"""
        resp = equity_client.post("/equity/calculate", json={
            "variant_name": "holdem",
            "players": [
                {"seat": 1, "hole_cards": [None, None]},
                {"seat": 2, "hole_cards": [None, None]},
            ],
        })
        assert resp.status_code == 422
        assert "known hole card" in resp.json()["detail"][0]["msg"].lower() \
               or "known hole card" in str(resp.json()).lower()

    def test_eapi_02_duplicate_card_same_player(self, equity_client):
        """EAPI-02"""
        resp = equity_client.post("/equity/calculate", json={
            "variant_name": "holdem",
            "players": [
                {"seat": 1, "hole_cards": ["Ah", "Ah"]},
                {"seat": 2, "hole_cards": ["Kd", None]},
            ],
        })
        assert resp.status_code == 422

    def test_eapi_03_duplicate_card_across_players(self, equity_client):
        """EAPI-03"""
        resp = equity_client.post("/equity/calculate", json={
            "variant_name": "holdem",
            "players": [
                {"seat": 1, "hole_cards": ["Ah", "Kd"]},
                {"seat": 2, "hole_cards": ["Ah", "Jc"]},
            ],
        })
        assert resp.status_code == 422

    def test_eapi_04_duplicate_card_player_and_board(self, equity_client):
        """EAPI-04"""
        resp = equity_client.post("/equity/calculate", json={
            "variant_name": "holdem",
            "players": [
                {"seat": 1, "hole_cards": ["Ah", "Kd"]},
                {"seat": 2, "hole_cards": ["Qh", "Jc"]},
            ],
            "board_nodes": [{"node": 0, "card": "Ah"}],
        })
        assert resp.status_code == 422

    def test_eapi_05_valid_request_reaches_service(self, equity_client, mock_equity_service):
        """EAPI-05"""
        resp = equity_client.post("/equity/calculate", json=VALID_REQUEST)
        assert resp.status_code == 200
        mock_equity_service.calculate.assert_called_once()

    def test_eapi_06_empty_players_list(self, equity_client):
        """EAPI-06"""
        resp = equity_client.post("/equity/calculate", json={
            "variant_name": "holdem",
            "players": [],
        })
        assert resp.status_code == 422

    def test_eapi_07_unknown_variant(self, equity_client, mock_equity_service):
        """EAPI-07"""
        mock_equity_service.calculate.side_effect = FileNotFoundError("Unknown variant")
        resp = equity_client.post("/equity/calculate", json={
            **VALID_REQUEST,
            "variant_name": "nonexistent_game",
        })
        assert resp.status_code == 404
        assert "nonexistent_game" in resp.json()["detail"]


# ── Response Shape ────────────────────────────────────────────────────────────

class TestEquityResponseShape:
    def test_eapi_10_equity_keyed_by_seat_strings(self, equity_client):
        """EAPI-10"""
        resp = equity_client.post("/equity/calculate", json=VALID_REQUEST)
        equity = resp.json()["equity"]
        for key in equity:
            assert isinstance(key, str)

    def test_eapi_11_each_seat_keyed_by_point_name(self, equity_client):
        """EAPI-11"""
        resp = equity_client.post("/equity/calculate", json=VALID_REQUEST)
        equity = resp.json()["equity"]
        for seat_val in equity.values():
            assert isinstance(seat_val, dict)
            for point_name in seat_val:
                assert isinstance(point_name, str)

    def test_eapi_12_equity_fractions_in_range(self, equity_client):
        """EAPI-12"""
        resp = equity_client.post("/equity/calculate", json=VALID_REQUEST)
        equity = resp.json()["equity"]
        for seat_val in equity.values():
            for frac in seat_val.values():
                assert 0.0 <= frac <= 1.0

    def test_eapi_13_method_is_valid_string(self, equity_client):
        """EAPI-13"""
        resp = equity_client.post("/equity/calculate", json=VALID_REQUEST)
        assert resp.json()["method"] in ("exact", "monte_carlo")

    def test_eapi_14_elapsed_ms_non_negative(self, equity_client):
        """EAPI-14"""
        resp = equity_client.post("/equity/calculate", json=VALID_REQUEST)
        assert resp.json()["elapsed_ms"] >= 0.0

    def test_eapi_15_equity_fractions_sum_to_one(self, equity_client):
        """EAPI-15 — per point, all seats should sum to ~1.0"""
        resp = equity_client.post("/equity/calculate", json=VALID_REQUEST)
        equity = resp.json()["equity"]
        point_totals: dict[str, float] = {}
        for seat_val in equity.values():
            for point_name, frac in seat_val.items():
                point_totals[point_name] = point_totals.get(point_name, 0.0) + frac
        for point_name, total in point_totals.items():
            assert abs(total - 1.0) < 1e-6, f"Point {point_name} sums to {total}"


# ── Property Tests (Hypothesis) ───────────────────────────────────────────────

class TestEquityPropertyTests:
    @given(
        cards=st.lists(
            st.sampled_from(VALID_CARDS), min_size=4, max_size=4, unique=True
        )
    )
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_eapi_p01_equity_sums_to_one(self, equity_client, cards):
        """EAPI-P-01 — for any valid 2-player input, equity fractions sum to 1.0"""
        resp = equity_client.post("/equity/calculate", json={
            "variant_name": "holdem",
            "players": [
                {"seat": 1, "hole_cards": [cards[0], cards[1]]},
                {"seat": 2, "hole_cards": [cards[2], cards[3]]},
            ],
        })
        assert resp.status_code == 200
        equity = resp.json()["equity"]
        point_totals: dict = {}
        for seat_val in equity.values():
            for pname, frac in seat_val.items():
                point_totals[pname] = point_totals.get(pname, 0.0) + frac
        for total in point_totals.values():
            assert abs(total - 1.0) < 1e-6
