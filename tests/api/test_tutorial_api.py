"""
test_tutorial_api.py — API layer tests for tutorial_api.py (§15 of BACKEND_TEST_PLAN.md)

Strategy: TestClient with seeded SQLite `db` fixture (see conftest.py).

NOTE on BUG-BE-15 (discovered while writing this suite):
`save_hypothetical_hand` constructs `Hand(..., pot=req.pot, ...)` but
`SaveHypotheticalHandRequest` has no `pot` field. Every call raises
`AttributeError` regardless of the seat_to_pid typo fix. TAPI-01 is
written against current (pre-fix) behavior — expected 500. TAPI-02
onward assume this is also fixed (a `pot: int = 0` field added to the
request model) and are marked to fail until then, mirroring how the
BUG-BE-08 (seat_to_pit) tests were structured against the original plan.
If your local fix added `pot` to the request model already, TAPI-01
will fail (no longer 500) — that's expected and is itself confirmation
the fix landed; update/remove TAPI-01 at that point.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.deps import get_db
from app.db.models.hands import Hand
from app.db.models.players import Player
from app.db.models.hole_cards import HoleCard
from app.db.models.board_cards import BoardCard
from app.db.models.actions import Action


# ── Helpers ─────────────────────────────────────────────────────────

def make_save_request(**overrides):
    base = {
        "game_name": "holdem",
        "variant_name": "holdem",
        "layout_name": "single_board",
        "dealer_seat": 1,
        "players": [
            {"seat": 1, "name": "Alice", "stack": 100, "hole_cards": ["Ah", "Kd"]},
            {"seat": 2, "name": "Bob", "stack": 100, "hole_cards": ["Qh", "Jc"]},
        ],
        "node_cards": [None, None, None, None, None],
        "actions": [],
    }
    base.update(overrides)
    return base


@pytest.fixture()
def client(db):
    app.dependency_overrides[get_db] = lambda: (yield db)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _insert_hand(db, *, is_hypothetical, variant_name="holdem", layout_name="single_board",
                  pot=10, dealer_seat=1, session_id=None):
    hand = Hand(
        session_id=session_id,
        variant_name=variant_name,
        layout_name=layout_name,
        split_pot=False,
        dealer_seat=dealer_seat,
        pot=pot,
        is_hypothetical=is_hypothetical,
    )
    db.add(hand)
    db.commit()
    db.refresh(hand)
    return hand


# ── 15.1 Save Hypothetical Hand ──────────────────────────────────────

class TestSaveHypotheticalHand:

    # def test_TAPI_01_pot_attribute_bug(self, client, db):
    #     """
    #     BUG-BE-15: SaveHypotheticalHandRequest has no `pot` field, but the
    #     route reads `req.pot`. Expect 500 until a `pot` field is added to
    #     the request model (or the route stops referencing it).
    #     """
    #     from app.db.models.players import Player # Adjust import path to your Player model
    
    #     if not db.query(Player).filter(Player.player_id == -1).first():
    #         db.add(Player(player_id=-1, username="Tutorial Dummy"))
    #         db.commit()

    #     resp = client.post("/tutorial/hands", json=make_save_request())
    #     assert resp.status_code == 500

    # @pytest.mark.xfail(
    #     reason="BUG-BE-15: route reads req.pot which doesn't exist on the "
    #            "request model yet; un-xfail once a pot field is added",
    #     strict=False,
    # )
    def test_TAPI_02_valid_request_creates_hand(self, client, db):
        resp = client.post("/tutorial/hands", json=make_save_request())
        assert resp.status_code == 200
        body = resp.json()

        hand = db.query(Hand).filter(Hand.hand_id == body["hand_id"]).first()
        assert hand is not None
        assert hand.is_hypothetical is True
        assert hand.session_id is None

    # @pytest.mark.xfail(reason="BUG-BE-15 blocks all writes", strict=False)
    def test_TAPI_03_negative_synthetic_player_ids(self, client, db):
        resp = client.post("/tutorial/hands", json=make_save_request())
        assert resp.status_code == 200
        hand_id = resp.json()["hand_id"]

        hole_cards = db.query(HoleCard).filter(HoleCard.hand_id == hand_id).all()
        assert hole_cards, "expected hole card rows to be written"
        seats_seen = {1, 2}
        for hc in hole_cards:
            assert hc.player_id < 0
            assert -hc.player_id in seats_seen

    # @pytest.mark.xfail(reason="BUG-BE-15 blocks all writes", strict=False)
    def test_TAPI_04_none_node_cards_skipped(self, client, db):
        req = make_save_request(node_cards=["7s", None, "2c", None, None])
        resp = client.post("/tutorial/hands", json=req)
        assert resp.status_code == 200
        hand_id = resp.json()["hand_id"]

        board_rows = db.query(BoardCard).filter(BoardCard.hand_id == hand_id).all()
        assert len(board_rows) == 2
        nodes_written = {bc.node for bc in board_rows}
        assert nodes_written == {0, 2}

    # @pytest.mark.xfail(reason="BUG-BE-15 blocks all writes", strict=False)
    def test_TAPI_05_node_cards_hardcoded_street_1(self, client, db):
        """
        Documents BUG-BE-13: all node cards get street=1 regardless of
        which street they actually belong to (no _node_to_street_map
        equivalent for hypothetical hands).
        """
        req = make_save_request(node_cards=["7s", "8s", "9s", "Th", "Jh"])
        resp = client.post("/tutorial/hands", json=req)
        assert resp.status_code == 200
        hand_id = resp.json()["hand_id"]

        board_rows = db.query(BoardCard).filter(BoardCard.hand_id == hand_id).all()
        assert len(board_rows) == 5
        assert all(bc.street == 1 for bc in board_rows)

    # @pytest.mark.xfail(reason="BUG-BE-15 blocks all writes", strict=False)
    def test_TAPI_06_actions_resolved_via_seat_to_pid(self, client, db):
        req = make_save_request(actions=[
            {
                "street": 0, "action_index": 0, "player_seat": 1,
                "action_type": "CALL", "amount": 2,
                "stack_before": 100, "pot_before": 0,
            },
            {
                "street": 0, "action_index": 1, "player_seat": 2,
                "action_type": "CHECK", "amount": None,
                "stack_before": 98, "pot_before": 4,
            },
        ])
        resp = client.post("/tutorial/hands", json=req)
        assert resp.status_code == 200
        hand_id = resp.json()["hand_id"]

        actions = (
            db.query(Action)
            .filter(Action.hand_id == hand_id)
            .order_by(Action.action_index)
            .all()
        )
        assert len(actions) == 2
        assert actions[0].player_id == -1
        assert actions[1].player_id == -2

    # @pytest.mark.xfail(reason="BUG-BE-15 blocks all writes", strict=False)
    def test_TAPI_06b_action_unknown_seat_falls_back(self, client, db):
        """player_seat not present in seat_to_pid falls back to -seat."""
        req = make_save_request(actions=[
            {
                "street": 0, "action_index": 0, "player_seat": 9,
                "action_type": "FOLD", "amount": None,
                "stack_before": None, "pot_before": None,
            },
        ])
        resp = client.post("/tutorial/hands", json=req)
        assert resp.status_code == 200
        hand_id = resp.json()["hand_id"]

        actions = db.query(Action).filter(Action.hand_id == hand_id).all()
        assert len(actions) == 1
        assert actions[0].player_id == -9

    # @pytest.mark.xfail(reason="BUG-BE-15 blocks all writes", strict=False)
    def test_TAPI_07_response_player_names_from_request(self, client):
        """
        player_names in the response come from req.players[i].name,
        NOT from a real Player DB row (no such row exists for synthetic
        hypothetical-hand seats).
        """
        req = make_save_request(players=[
            {"seat": 1, "name": "Zara", "stack": 50, "hole_cards": ["Ah", None]},
            {"seat": 2, "name": "Milo", "stack": 75, "hole_cards": [None, None]},
        ])
        resp = client.post("/tutorial/hands", json=req)
        assert resp.status_code == 200
        assert resp.json()["player_names"] == ["Zara", "Milo"]


# ── 15.2 List Hypothetical Hands ─────────────────────────────────────

class TestListHypotheticalHands:

    def test_TAPI_10_only_hypothetical_returned(self, client, db):
        _insert_hand(db, is_hypothetical=True, variant_name="holdem")
        _insert_hand(db, is_hypothetical=False, variant_name="holdem")
        _insert_hand(db, is_hypothetical=True, variant_name="omaha")

        resp = client.get("/tutorial/hands")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        assert all(item["variant_name"] in ("holdem", "omaha") for item in body)

    def test_TAPI_11_filtered_by_variant(self, client, db):
        _insert_hand(db, is_hypothetical=True, variant_name="holdem")
        _insert_hand(db, is_hypothetical=True, variant_name="omaha")

        resp = client.get("/tutorial/hands", params={"variant": "omaha"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["variant_name"] == "omaha"

    def test_TAPI_12_player_names_always_empty(self, client, db):
        _insert_hand(db, is_hypothetical=True)

        resp = client.get("/tutorial/hands")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["player_names"] == []

    def test_TAPI_pagination_limit_offset(self, client, db):
        for i in range(5):
            _insert_hand(db, is_hypothetical=True, pot=i)

        resp = client.get("/tutorial/hands", params={"limit": 2, "offset": 1})
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_TAPI_empty_list(self, client, db):
        resp = client.get("/tutorial/hands")
        assert resp.status_code == 200
        assert resp.json() == []


# ── 15.3 Fetch Single Hypothetical Hand ──────────────────────────────

class TestGetHypotheticalHand:

    def test_TAPI_20_existing_hypothetical_hand_returns_full_dto(self, client, db):
        hand = _insert_hand(db, is_hypothetical=True, session_id=None)

        resp = client.get(f"/tutorial/hands/{hand.hand_id}")
        assert resp.status_code == 200
        body = resp.json()

        # Shape matches HandReplayDTO (delegated to replay_api.get_hand)
        for key in (
            "hand_id", "variant_name", "layout_name", "split_pot", "pot",
            "dealer_seat", "started_at", "seats", "initial_stacks",
            "actions", "hole_cards", "board_cards", "point_results", "payouts",
        ):
            assert key in body
        assert body["hand_id"] == hand.hand_id

    def test_TAPI_21_real_hand_not_flagged_hypothetical_404s(self, client, db):
        hand = _insert_hand(db, is_hypothetical=False)

        resp = client.get(f"/tutorial/hands/{hand.hand_id}")
        assert resp.status_code == 404

    def test_TAPI_22_unknown_hand_id_404s(self, client, db):
        resp = client.get("/tutorial/hands/999999")
        assert resp.status_code == 404

    def test_TAPI_23_no_session_or_table_seat_rows_does_not_crash(self, client, db):
        """
        Hypothetical hands have session_id=None and no TableSeat rows.
        replay_api.get_hand's seat_of()/name_of() helpers must fall back
        gracefully (seat_names.get(seat, f"Player {seat}")) rather than
        raising when there's no seat map to join against.
        """
        hand = _insert_hand(db, is_hypothetical=True, session_id=None)

        resp = client.get(f"/tutorial/hands/{hand.hand_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["seats"] == {}
        assert body["hole_cards"] == []
        assert body["board_cards"] == []