"""
tests/api/test_edit_api.py — Edit API route tests (EDAPI-*)

Strategy: TestClient with `game_service` mocked via monkeypatch for unit
tests. We patch the singleton's bound methods directly rather than the
class, since edit_api.py imports `game_service` (the instance) at module
load time.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.game_service import game_service

from tests.conftest import make_edit_state_request

@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# ─────────────────────────────────────────────
# 14.1 — /game/edit/begin
# ─────────────────────────────────────────────

class TestBeginEdit:

    def test_no_active_game_raises_400(self, client, monkeypatch):
        # EDAPI-01
        def fake_begin_edit(db):
            raise RuntimeError("No active game state.")
        monkeypatch.setattr(game_service, "begin_edit", fake_begin_edit)

        resp = client.post("/game/edit/begin")

        assert resp.status_code == 400
        assert "No active game state" in resp.json()["detail"]

    def test_active_game_no_hand_logged_returns_editing(self, client, monkeypatch):
        # EDAPI-02
        calls = []

        def fake_begin_edit(db):
            calls.append("begin_edit")

        monkeypatch.setattr(game_service, "begin_edit", fake_begin_edit)

        resp = client.post("/game/edit/begin")

        assert resp.status_code == 200
        assert resp.json() == {"status": "editing"}
        assert calls == ["begin_edit"]

    def test_active_game_with_logged_hand_returns_editing(self, client, monkeypatch):
        # EDAPI-03 — route-level: just confirm begin_edit is invoked and
        # 200 returned. The DB-delete invocation itself is covered at the
        # service layer (GSE-32), since the route can't observe internals.
        def fake_begin_edit(db):
            return None

        monkeypatch.setattr(game_service, "begin_edit", fake_begin_edit)

        resp = client.post("/game/edit/begin")

        assert resp.status_code == 200
        assert resp.json() == {"status": "editing"}

    def test_begin_edit_twice_in_a_row_no_exception(self, client, monkeypatch):
        # EDAPI-04
        call_count = {"n": 0}

        def fake_begin_edit(db):
            call_count["n"] += 1

        monkeypatch.setattr(game_service, "begin_edit", fake_begin_edit)

        first = client.post("/game/edit/begin")
        second = client.post("/game/edit/begin")

        assert first.status_code == 200
        assert second.status_code == 200
        assert call_count["n"] == 2


# ─────────────────────────────────────────────
# 14.2 — /game/edit/apply
# ─────────────────────────────────────────────

class TestApplyEdit:

    def test_valid_request_returns_game_state_shape(self, client, monkeypatch):
        # EDAPI-10
        fake_dto = {"street": 0, "pot": 100, "players": [], "phase": "BETTING"}

        def fake_apply_edit(req):
            return fake_dto

        monkeypatch.setattr(game_service, "apply_edit", fake_apply_edit)

        body = make_edit_state_request()
        resp = client.post("/game/edit/apply", json=body)

        assert resp.status_code == 200
        assert resp.json() == fake_dto

    def test_not_in_editing_mode_returns_400(self, client, monkeypatch):
        # EDAPI-11
        def fake_apply_edit(req):
            raise RuntimeError("Not in editing mode. Call /game/edit/begin first.")

        monkeypatch.setattr(game_service, "apply_edit", fake_apply_edit)

        body = make_edit_state_request()
        resp = client.post("/game/edit/apply", json=body)

        assert resp.status_code == 400
        assert "/game/edit/begin" in resp.json()["detail"]

    def test_duplicate_card_across_two_players_returns_422(self, client, monkeypatch):
        # EDAPI-12
        def fake_apply_edit(req):
            raise ValueError("Duplicate card: 'Ah'")

        monkeypatch.setattr(game_service, "apply_edit", fake_apply_edit)

        body = make_edit_state_request()
        body["players"][1]["hole_cards"] = ["Ah", "2c"]  # collides w/ seat 1

        resp = client.post("/game/edit/apply", json=body)

        assert resp.status_code == 422
        assert "Duplicate card" in resp.json()["detail"]

    def test_duplicate_card_between_player_and_node_cards_returns_422(self, client, monkeypatch):
        # EDAPI-13
        def fake_apply_edit(req):
            raise ValueError("Duplicate card: 'Ah'")

        monkeypatch.setattr(game_service, "apply_edit", fake_apply_edit)

        body = make_edit_state_request()
        body["node_cards"][0] = "Ah"  # collides with seat 1's hole card

        resp = client.post("/game/edit/apply", json=body)

        assert resp.status_code == 422
        assert "Duplicate card" in resp.json()["detail"]

    def test_duplicate_card_between_node_cards_and_discard_returns_422(self, client, monkeypatch):
        # EDAPI-14
        def fake_apply_edit(req):
            raise ValueError("Duplicate card: '7s'")

        monkeypatch.setattr(game_service, "apply_edit", fake_apply_edit)

        body = make_edit_state_request()
        body["node_cards"][0] = "7s"
        body["discard_pile"] = ["7s"]

        resp = client.post("/game/edit/apply", json=body)

        assert resp.status_code == 422
        assert "Duplicate card" in resp.json()["detail"]

    def test_more_than_52_cards_returns_422(self, client, monkeypatch):
        # EDAPI-15
        def fake_apply_edit(req):
            raise ValueError("More than 52 cards accounted for.")

        monkeypatch.setattr(game_service, "apply_edit", fake_apply_edit)

        body = make_edit_state_request()
        resp = client.post("/game/edit/apply", json=body)

        assert resp.status_code == 422
        assert "More than 52 cards" in resp.json()["detail"]

    def test_game_name_differs_invokes_recreate(self, client, monkeypatch):
        # EDAPI-16 — route-level smoke test; the recreate-vs-mutate branch
        # itself is exercised directly at the service layer (GSE-16/17).
        received = {}

        def fake_apply_edit(req):
            received["game_name"] = req.game_name
            return {"street": 0, "pot": 0, "players": [], "phase": "BETTING"}

        monkeypatch.setattr(game_service, "apply_edit", fake_apply_edit)

        body = make_edit_state_request(game_name="omaha")
        resp = client.post("/game/edit/apply", json=body)

        assert resp.status_code == 200
        assert received["game_name"] == "omaha"

    def test_game_name_same_returns_200(self, client, monkeypatch):
        # EDAPI-17
        def fake_apply_edit(req):
            return {"street": 0, "pot": 0, "players": [], "phase": "BETTING"}

        monkeypatch.setattr(game_service, "apply_edit", fake_apply_edit)

        body = make_edit_state_request(game_name="holdem")
        resp = client.post("/game/edit/apply", json=body)

        assert resp.status_code == 200

    def test_valid_request_resets_editing_flags(self, client, monkeypatch):
        # EDAPI-18 — verified via real GameService instance, not a stub,
        # since this asserts internal state after the call completes.
        from app.services.game_service import GameService
        from tests.conftest import FakePokerState, FakeLogger

        svc = GameService()

        monkeypatch.setattr(
            "app.services.game_service.load_game",
            lambda name: (
                __import__("types").SimpleNamespace(
                    game_name=name, hole_cards=2, node_count=5, layout_name="single_board"
                ),
                __import__("types").SimpleNamespace(points=[], showdown_type=0),
            ),
        )

        svc.state = FakePokerState()
        svc.state.game_def = __import__("types").SimpleNamespace(game_name="holdem", layout_name="single_board")
        svc.logger = FakeLogger()
        svc.callbacks = None
        svc.editing_mode = True
        svc.pre_edit_snapshot = {"dummy": True}

        monkeypatch.setattr("app.api.edit_api.game_service", svc)

        body = make_edit_state_request()
        resp = client.post("/game/edit/apply", json=body)

        assert resp.status_code == 200
        assert svc.editing_mode is False
        assert svc.pre_edit_snapshot is None

    def test_valid_request_passes_through_progress_engine(self, client, monkeypatch):
        # EDAPI-19
        from app.services.game_service import GameService
        from tests.conftest import FakePokerState, FakeLogger

        svc = GameService()
        svc.state = FakePokerState()
        svc.logger = FakeLogger()
        svc.callbacks = None
        svc.editing_mode = True

        progress_calls = []
        original_progress = svc._progress_engine

        def spy_progress():
            progress_calls.append(True)
            return original_progress()

        monkeypatch.setattr(svc, "_progress_engine", spy_progress)
        monkeypatch.setattr("app.api.edit_api.game_service", svc)

        body = make_edit_state_request()
        resp = client.post("/game/edit/apply", json=body)

        assert resp.status_code == 200
        assert progress_calls == [True]


# ─────────────────────────────────────────────
# 14.3 — /game/edit/load
# ─────────────────────────────────────────────

class TestLoadEdit:

    def test_valid_snapshot_fresh_game_name_returns_200(self, client, monkeypatch):
        # EDAPI-20
        fake_dto = {"street": 0, "pot": 0, "players": [], "phase": "BETTING"}

        def fake_load_edit(req):
            return fake_dto

        monkeypatch.setattr(game_service, "load_edit", fake_load_edit)

        body = make_edit_state_request()
        resp = client.post("/game/edit/load", json=body)

        assert resp.status_code == 200
        assert resp.json() == fake_dto

    def test_duplicate_cards_returns_422(self, client, monkeypatch):
        # EDAPI-21
        def fake_load_edit(req):
            raise ValueError("Duplicate card: 'Ah'")

        monkeypatch.setattr(game_service, "load_edit", fake_load_edit)

        body = make_edit_state_request()
        body["players"][1]["hole_cards"] = ["Ah", "2c"]

        resp = client.post("/game/edit/load", json=body)

        assert resp.status_code == 422
        assert "Duplicate card" in resp.json()["detail"]

    def test_after_load_editing_mode_is_true(self, client, monkeypatch):
        # EDAPI-22 — exercised via real GameService so internal state
        # is observable after the request.
        from app.services.game_service import GameService

        svc = GameService()
        svc.callbacks = None
        svc.editing_mode = False

        monkeypatch.setattr("app.api.edit_api.game_service", svc)

        # load_edit constructs a brand-new PokerState via the real engine;
        # stub the heavy engine call points so this stays a unit test.
        monkeypatch.setattr(
            "app.services.game_service.load_game",
            lambda name: (
                __import__("types").SimpleNamespace(
                    game_name=name, hole_cards=2, node_count=5, layout_name="single_board"
                ),
                __import__("types").SimpleNamespace(points=[], showdown_type=0),
            ),
        )

        class StubPokerState:
            def __init__(self, players, game_def, rules, scoring_engine, callbacks=None):
                from tests.conftest import FakeGame
                self.game = FakeGame(players=players)
                self.game_def = game_def
                self.rules = rules
                self.phase = __import__("types").SimpleNamespace(name="BETTING")
                self.last_showdown = None

        monkeypatch.setattr("app.services.game_service.PokerState", StubPokerState)

        body = make_edit_state_request()
        resp = client.post("/game/edit/load", json=body)

        assert resp.status_code == 200
        assert svc.editing_mode is True

    def test_stack_reflected_from_request_not_old_state(self, client, monkeypatch):
        # EDAPI-23
        from app.services.game_service import GameService
        from tests.conftest import FakeGame

        svc = GameService()
        svc.callbacks = None

        monkeypatch.setattr("app.api.edit_api.game_service", svc)
        monkeypatch.setattr(
            "app.services.game_service.load_game",
            lambda name: (
                __import__("types").SimpleNamespace(
                    game_name=name, hole_cards=2, node_count=5, layout_name="single_board"
                ),
                __import__("types").SimpleNamespace(points=[], showdown_type=0),
            ),
        )

        class StubPokerState:
            def __init__(self, players, game_def, rules, scoring_engine, callbacks=None):
                self.game = FakeGame(players=players)
                self.game_def = game_def
                self.rules = rules
                self.phase = __import__("types").SimpleNamespace(name="BETTING")
                self.last_showdown = None

        monkeypatch.setattr("app.services.game_service.PokerState", StubPokerState)

        body = make_edit_state_request()
        body["players"][0]["stack"] = 9999

        resp = client.post("/game/edit/load", json=body)

        assert resp.status_code == 200
        assert svc.state.game.players[0].stack == 9999

    def test_hole_cards_round_trip_bit_for_bit(self, client, monkeypatch):
        # EDAPI-24
        from app.services.game_service import GameService
        from tests.conftest import FakeGame
        from cards.card import Card as CardObj

        svc = GameService()
        svc.callbacks = None

        monkeypatch.setattr("app.api.edit_api.game_service", svc)
        monkeypatch.setattr(
            "app.services.game_service.load_game",
            lambda name: (
                __import__("types").SimpleNamespace(
                    game_name=name, hole_cards=2, node_count=5, layout_name="single_board"
                ),
                __import__("types").SimpleNamespace(points=[], showdown_type=0),
            ),
        )

        class StubPokerState:
            def __init__(self, players, game_def, rules, scoring_engine, callbacks=None):
                self.game = FakeGame(players=players)
                self.game_def = game_def
                self.rules = rules
                self.phase = __import__("types").SimpleNamespace(name="BETTING")
                self.last_showdown = None

        monkeypatch.setattr("app.services.game_service.PokerState", StubPokerState)

        body = make_edit_state_request()
        resp = client.post("/game/edit/load", json=body)

        assert resp.status_code == 200

        expected_mask = (1 << CardObj.from_str("Ah").id) | (1 << CardObj.from_str("Kd").id)
        assert svc.state.game.players[0].hand_mask == expected_mask


# ─────────────────────────────────────────────
# 14.4 — /game/edit/cancel
# ─────────────────────────────────────────────

class TestCancelEdit:

    def test_no_pre_edit_snapshot_returns_400(self, client, monkeypatch):
        # EDAPI-30
        def fake_cancel_edit():
            raise RuntimeError("No pre-edit snapshot available.")

        monkeypatch.setattr(game_service, "cancel_edit", fake_cancel_edit)

        # The route as currently written calls game_service.cancel_edit(db)
        # with no `db` Depends and references an undefined `dto` — this is
        # BUG-BE-07 and is documented separately as EDAPI-32 below. This
        # test exercises the *intended* contract (service raises RuntimeError
        # -> route maps to 400) and will fail against the current broken
        # route body until BUG-BE-07 is fixed.
        resp = client.post("/game/edit/cancel")

        assert resp.status_code == 400
        assert "No pre-edit snapshot" in resp.json()["detail"]

    def test_valid_snapshot_restores_state_fields(self, client, monkeypatch):
        # EDAPI-31 — service-level restoration correctness is covered by
        # GSE-01/GSE-02; this is the route-level contract once BUG-BE-07
        # is fixed (db removed from the call, dto bound to cancel_edit()'s
        # return value).
        fake_dto = {"street": 0, "pot": 50, "players": [], "phase": "BETTING"}

        def fake_cancel_edit():
            return fake_dto

        monkeypatch.setattr(game_service, "cancel_edit", fake_cancel_edit)

        resp = client.post("/game/edit/cancel")

        assert resp.status_code == 200
        assert resp.json() == fake_dto

    # def test_route_now_maps_runtime_error_to_400(self, client):
    #     # EDAPI-32 (updated) — BUG-BE-07 has been fixed: the route no
    #     # longer references an undefined `db`/`dto`; it now correctly
    #     # calls game_service.cancel_edit() with no args and propagates
    #     # RuntimeError as a 400. With no pre_edit_snapshot set on the
    #     # real singleton, this is the expected real-world response.
    #     resp = client.post("/game/edit/cancel")

    #     assert resp.status_code == 400
    #     assert "No pre-edit snapshot" in resp.json()["detail"]