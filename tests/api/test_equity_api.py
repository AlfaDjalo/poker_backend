"""
tests/api/test_equity_api.py
EAPI-01 ... EAPI-15, EAPI-P-01 ... EAPI-P-03
"""

from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Define a profile that completely removes execution time limitations
settings.register_profile("api_testing", deadline=None, max_examples=20)
settings.load_profile("api_testing")

# VALID_CARDS must be defined before any class that references it
VALID_CARDS = [f"{rank}{suit}" for rank in "23456789TJQKA" for suit in "cdhs"]

MOCK_RAW_RESPONSE = {
    "players": {
        0: {
            "overall_equity_fraction": 0.6,
            "overall_equity_currency": 0.0,
            "scoop_probability": 0.6,
            "split_probability": 0.0,
            "points": {
                "high": {
                    "win_share": 0.6,
                    "win_probability": 0.6,
                    "tie_probability": 0.0,
                    "equity_currency": 0.0,
                    "equity_percent": 0.6,
                }
            },
        },
        1: {
            "overall_equity_fraction": 0.4,
            "overall_equity_currency": 0.0,
            "scoop_probability": 0.4,
            "split_probability": 0.0,
            "points": {
                "high": {
                    "win_share": 0.4,
                    "win_probability": 0.4,
                    "tie_probability": 0.0,
                    "equity_currency": 0.0,
                    "equity_percent": 0.4,
                }
            },
        },
    },
    "method": "exact",
    "iterations": 1326,
    "elapsed_ms": 12.5,
}


@pytest.fixture()
def mock_equity_service():
    # Patch the method directly on the class class/instance to catch it everywhere
    with patch("app.api.equity_api.equity_service.calculate") as mock_calc:
        mock_calc.return_value = MOCK_RAW_RESPONSE
        yield mock_calc


@pytest.fixture()
def equity_client(db, mock_equity_service):
    from app.api.deps import get_db
    from app.main import app

    app.dependency_overrides[get_db] = lambda: (yield db)
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


VALID_REQUEST = {
    "variant_name": "holdem",
    "players": [
        {"seat": 0, "hole_cards": ["Ah", "As"]},
        {"seat": 1, "hole_cards": ["Kh", "Ks"]},
    ],
    "board_nodes": [],
    "pot_size": 0.0,
}


# ── Input Validation ──────────────────────────────────────────────────────────


class TestEquityInputValidation:

    @given(
        cards=st.lists(
            st.sampled_from(VALID_CARDS), min_size=4, max_size=4, unique=True
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_eapi_p01_equity_sums_to_one(self, equity_client, cards):
        """EAPI-01 — all hole cards null -> 422"""
        resp = equity_client.post(
            "/equity/calculate",
            json={
                "variant_name": "holdem",
                "players": [
                    {"seat": 0, "hole_cards": [None, None]},
                    {"seat": 1, "hole_cards": [None, None]},
                ],
                "board_nodes": [],
            },
        )
        assert resp.status_code == 422
        assert "known hole card" in str(resp.json()).lower()

    def test_eapi_02_duplicate_card_same_player(self, equity_client):
        """EAPI-02"""
        resp = equity_client.post(
            "/equity/calculate",
            json={
                "variant_name": "holdem",
                "players": [
                    {"seat": 0, "hole_cards": ["Ah", "Ah"]},
                    {"seat": 1, "hole_cards": ["Kd", None]},
                ],
                "board_nodes": [],
            },
        )
        assert resp.status_code == 422
        assert "duplicate card" in str(resp.json()).lower()

    def test_eapi_03_duplicate_card_across_players(self, equity_client):
        """EAPI-03"""
        resp = equity_client.post(
            "/equity/calculate",
            json={
                "variant_name": "holdem",
                "players": [
                    {"seat": 0, "hole_cards": ["Ah", "Kd"]},
                    {"seat": 1, "hole_cards": ["Ah", "Jc"]},
                ],
                "board_nodes": [],
            },
        )
        assert resp.status_code == 422
        assert "duplicate card" in str(resp.json()).lower()

    def test_eapi_04_duplicate_card_player_and_board(self, equity_client):
        """EAPI-04"""
        resp = equity_client.post(
            "/equity/calculate",
            json={
                "variant_name": "holdem",
                "players": [
                    {"seat": 0, "hole_cards": ["Ah", "Kd"]},
                    {"seat": 1, "hole_cards": ["Qh", "Jc"]},
                ],
                "board_nodes": [{"node": 0, "card": "Ah"}],
            },
        )
        assert resp.status_code == 422
        assert "duplicate card" in str(resp.json()).lower()

    def test_eapi_05_valid_request_reaches_service(
        self, equity_client, mock_equity_service
    ):
        """EAPI-05"""
        resp = equity_client.post("/equity/calculate", json=VALID_REQUEST)
        assert resp.status_code == 200
        # mock_equity_service.calculate.assert_called_once()
        mock_equity_service.assert_called_once()

    def test_eapi_06_empty_players_list(self, equity_client):
        """EAPI-06"""
        resp = equity_client.post(
            "/equity/calculate",
            json={"variant_name": "holdem", "players": [], "board_nodes": []},
        )
        assert resp.status_code == 422

    def test_eapi_07_unknown_variant(self, equity_client, mock_equity_service):
        """EAPI-07"""
        # mock_equity_service.calculate.side_effect = FileNotFoundError("Unknown variant")
        mock_equity_service.side_effect = FileNotFoundError("Unknown variant")
        resp = equity_client.post(
            "/equity/calculate",
            json={
                **VALID_REQUEST,
                "variant_name": "nonexistent_game",
            },
        )
        assert resp.status_code == 404
        assert "nonexistent_game" in resp.json()["detail"]


# ── Response Shape ────────────────────────────────────────────────────────────


class TestEquityResponseShape:
    def test_eapi_10_equity_keyed_by_seat_strings(self, equity_client):
        """EAPI-10"""
        resp = equity_client.post("/equity/calculate", json=VALID_REQUEST)
        players = resp.json()["players"]
        for key in players:
            assert isinstance(key, str)

    def test_eapi_11_each_seat_contains_points_breakdown(self, equity_client):
        """EAPI-11"""
        resp = equity_client.post("/equity/calculate", json=VALID_REQUEST)
        players = resp.json()["players"]
        for player_dto in players.values():
            assert "overall_equity_fraction" in player_dto
            assert isinstance(player_dto["points"], dict)
            for point_name, point_dto in player_dto["points"].items():
                assert isinstance(point_name, str)
                assert "win_share" in point_dto

    def test_eapi_12_equity_fractions_in_range(self, equity_client):
        """EAPI-12"""
        resp = equity_client.post("/equity/calculate", json=VALID_REQUEST)
        players = resp.json()["players"]
        for player_dto in players.values():
            assert 0.0 <= player_dto["overall_equity_fraction"] <= 1.0
            for point_dto in player_dto["points"].values():
                assert 0.0 <= point_dto["win_share"] <= 1.0
                assert 0.0 <= point_dto["win_probability"] <= 1.0

    def test_eapi_13_method_is_valid_string(self, equity_client):
        """EAPI-13"""
        resp = equity_client.post("/equity/calculate", json=VALID_REQUEST)
        assert resp.json()["method"] in ("exact", "monte_carlo")

    def test_eapi_14_elapsed_ms_non_negative(self, equity_client):
        """EAPI-14"""
        resp = equity_client.post("/equity/calculate", json=VALID_REQUEST)
        assert resp.json()["elapsed_ms"] >= 0.0

    def test_eapi_15_equity_fractions_sum_to_one(self, equity_client):
        """EAPI-15 — overall fractions should sum to ~1.0"""
        resp = equity_client.post("/equity/calculate", json=VALID_REQUEST)
        players = resp.json()["players"]
        total_fraction = sum(p["overall_equity_fraction"] for p in players.values())
        assert abs(total_fraction - 1.0) < 1e-6


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
        resp = equity_client.post(
            "/equity/calculate",
            json={
                "variant_name": "holdem",
                "players": [
                    {"seat": 0, "hole_cards": [cards[0], cards[1]]},
                    {"seat": 1, "hole_cards": [cards[2], cards[3]]},
                ],
                "board_nodes": [],
            },
        )
        assert resp.status_code == 200
        players = resp.json()["players"]
        total_fraction = sum(p["overall_equity_fraction"] for p in players.values())
        assert abs(total_fraction - 1.0) < 1e-6
