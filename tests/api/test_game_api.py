"""
tests/api/test_game_api.py
GAPI-01 … GAPI-12
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Helper: minimal GameStateDTO-shaped dict ──────────────────────────────────

def assert_game_state_shape(data):
    assert "street" in data
    assert "pot" in data
    assert "players" in data
    assert isinstance(data["players"], list)
    assert "phase" in data


MOCK_STATE = {
    "street": 0, "pot": 0, "nodes": [None] * 5,
    "layout_name": "single_board", "game_name": "holdem",
    "street_names": None, "points": [],
    "players": [], "current_player": None, "phase": "BETTING",
    "showdown": None, "winners": None,
    "available_actions": [], "to_call": 0, "min_raise": 2, "max_raise": 100,
}


@pytest.fixture()
def mock_game_service():
    """Patch game_service inside game_api with a MagicMock."""
    with patch("app.api.game_api.game_service") as mock:
        mock.get_variants.return_value = {"variants": ["holdem", "plo"], "current": "holdem"}
        mock.select_game.return_value = None
        mock.new_hand.return_value = MOCK_STATE
        mock.restart.return_value = MOCK_STATE
        mock.get_state.return_value = MOCK_STATE
        mock.apply_action.return_value = MOCK_STATE
        yield mock


@pytest.fixture()
def api_client(db, mock_game_service):
    from app.main import app
    from app.api.deps import get_db
    app.dependency_overrides[get_db] = lambda: (yield db)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGetVariants:
    def test_gapi_01_returns_variants_list(self, api_client, mock_game_service):
        """GAPI-01"""
        resp = api_client.get("/game/variants")
        assert resp.status_code == 200
        body = resp.json()
        assert "variants" in body
        assert isinstance(body["variants"], list)
        assert "current" in body


class TestSelectGame:
    def test_gapi_02_valid_game(self, api_client, mock_game_service):
        """GAPI-02"""
        resp = api_client.post("/game/select-game", json={"game_name": "holdem"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["pending_game"] == "holdem"

    def test_gapi_03_unknown_game(self, api_client, mock_game_service):
        """GAPI-03"""
        mock_game_service.select_game.side_effect = ValueError("Unknown game variant: 'nonexistent'")
        resp = api_client.post("/game/select-game", json={"game_name": "nonexistent"})
        assert resp.status_code == 400
        assert "nonexistent" in resp.json()["detail"]


class TestNewHand:
    def test_gapi_04_no_body(self, api_client, mock_game_service):
        """GAPI-04"""
        resp = api_client.post("/game/new-hand")
        assert resp.status_code == 200
        assert_game_state_shape(resp.json())

    def test_gapi_05_with_game_name(self, api_client, mock_game_service):
        """GAPI-05"""
        resp = api_client.post("/game/new-hand", json={"game_name": "plo"})
        assert resp.status_code == 200
        mock_game_service.new_hand.assert_called_once_with(game_name="plo")


class TestRestart:
    def test_gapi_06_valid_db_state(self, api_client, mock_game_service):
        """GAPI-06"""
        resp = api_client.post("/game/restart")
        assert resp.status_code == 200
        assert_game_state_shape(resp.json())

    def test_gapi_07_no_table_in_db(self, api_client, mock_game_service):
        """GAPI-07"""
        mock_game_service.restart.side_effect = ValueError("No poker tables found")
        resp = api_client.post("/game/restart")
        assert resp.status_code == 400


class TestGetState:
    def test_gapi_08_state_is_none(self, api_client, mock_game_service):
        """GAPI-08"""
        mock_game_service.get_state.return_value = None
        resp = api_client.get("/game/state")
        assert resp.status_code == 200
        assert resp.json() is None

    def test_gapi_09_active_game(self, api_client, mock_game_service):
        """GAPI-09"""
        resp = api_client.get("/game/state")
        assert resp.status_code == 200
        assert_game_state_shape(resp.json())


class TestApplyAction:
    def test_gapi_10_valid_action(self, api_client, mock_game_service):
        """GAPI-10"""
        resp = api_client.post("/game/action", json={"type": "call"})
        assert resp.status_code == 200
        assert_game_state_shape(resp.json())

    def test_gapi_11_invalid_action_type(self, db, mock_game_service):
        """GAPI-11"""
        from app.main import app
        from app.api.deps import get_db
        mock_game_service.apply_action.side_effect = Exception("Unknown action")
        app.dependency_overrides[get_db] = lambda: (yield db)
        from fastapi.testclient import TestClient
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/game/action", json={"type": "invalid_action"})
        app.dependency_overrides.clear()
        assert resp.status_code == 500
        

class TestStartRoute:
    def test_gapi_12_start_route_broken(self, db):
        """GAPI-12 — /game/start has broken GameLogger import → NameError → 500"""
        from app.main import app
        from app.api.deps import get_db
        app.dependency_overrides[get_db] = lambda: (yield db)
        # raise_server_exceptions=False so NameError becomes a 500 response
        from fastapi.testclient import TestClient
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/game/start", json={})
        app.dependency_overrides.clear()
        assert resp.status_code in (422, 500)