"""
tests/api/test_replay_api.py
RAPI-01 … RAPI-29
"""
import pytest
from datetime import datetime


# ── DB row factories ──────────────────────────────────────────────────────────

def make_hand(db, seeded, variant="holdem", pot=100):
    from app.db.models.hands import Hand
    h = Hand(
        session_id=seeded["session"].session_id,
        variant_name=variant,
        layout_name="single_board",
        split_pot=False,
        betting_config_id=1,
        dealer_seat=1,
        pot=pot,
    )
    db.add(h); db.commit(); db.refresh(h)
    return h


def make_action(db, hand_id, player_id, street=0, idx=0,
                action_type="call", amount=None, stack_before=100, pot_before=0):
    from app.db.models.actions import Action
    a = Action(
        hand_id=hand_id, street=street, action_index=idx,
        player_id=player_id, action_type=action_type,
        amount=amount, stack_before=stack_before, pot_before=pot_before,
    )
    db.add(a); db.commit(); db.refresh(a)
    return a


def make_hole_card(db, hand_id, player_id, card=0, street=0):
    from app.db.models.hole_cards import HoleCard
    hc = HoleCard(hand_id=hand_id, player_id=player_id, street=street, card=card, visible=True)
    db.add(hc); db.commit(); db.refresh(hc)
    return hc


def make_board_card(db, hand_id, street=1, node=0, card=10):
    from app.db.models.board_cards import BoardCard
    bc = BoardCard(hand_id=hand_id, street=street, node=node, card=card)
    db.add(bc); db.commit(); db.refresh(bc)
    return bc


def make_annotation(db, hand_id, user_id=1, comment="test note", action_id=None):
    from app.db.models.annotations import Annotation
    ann = Annotation(
        hand_id=hand_id, action_id=action_id,
        user_id=user_id, comment=comment, selected_cards=[],
    )
    db.add(ann); db.commit(); db.refresh(ann)
    return ann


# ── Hand Listing ──────────────────────────────────────────────────────────────

class TestHandListing:
    def test_rapi_01_no_hands(self, seeded_client):
        """RAPI-01"""
        client, _ = seeded_client
        resp = client.get("/replay/hands")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_rapi_02_three_hands(self, seeded_client, db):
        """RAPI-02"""
        client, seeded = seeded_client
        for _ in range(3):
            make_hand(db, seeded)
        resp = client.get("/replay/hands")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_rapi_03_variant_filter(self, seeded_client, db):
        """RAPI-03"""
        client, seeded = seeded_client
        make_hand(db, seeded, variant="holdem")
        make_hand(db, seeded, variant="plo")
        make_hand(db, seeded, variant="holdem")
        resp = client.get("/replay/hands?variant=holdem")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert all(h["variant_name"] == "holdem" for h in data)

    def test_rapi_04_pagination(self, seeded_client, db):
        """RAPI-04"""
        client, seeded = seeded_client
        for _ in range(5):
            make_hand(db, seeded)
        resp = client.get("/replay/hands?limit=2&offset=1")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_rapi_05_variants_endpoint(self, seeded_client, db):
        """RAPI-05"""
        client, seeded = seeded_client
        make_hand(db, seeded, variant="holdem")
        make_hand(db, seeded, variant="plo")
        resp = client.get("/replay/variants")
        assert resp.status_code == 200
        variants = resp.json()
        assert "holdem" in variants
        assert "plo" in variants


# ── Full Hand Replay ──────────────────────────────────────────────────────────

class TestHandReplay:
    def _add_seats(self, db, seeded):
        """Add TableSeat rows so the replay endpoint can build the seat map."""
        from app.db.models.table_seating import TableSeat
        for i, p in enumerate(seeded["players"]):
            db.add(TableSeat(
                session_id=seeded["session"].session_id,
                seat_number=i + 1,
                player_id=p.player_id,
            ))
        db.commit()

    def test_rapi_10_valid_hand(self, seeded_client, db):
        client, seeded = seeded_client          # unpack FIRST
        hand = make_hand(db, seeded)
        resp = client.get(f"/replay/hands/{hand.hand_id}")
        assert resp.status_code == 200
        data = resp.json()
        for field in ["hand_id", "variant_name", "actions", "hole_cards",
                      "board_cards", "point_results", "payouts", "seats"]:
            assert field in data, f"Missing field: {field}"

    def test_rapi_11_unknown_hand(self, seeded_client):
        """RAPI-11"""
        client, _ = seeded_client
        resp = client.get("/replay/hands/99999")
        assert resp.status_code == 404

    def test_rapi_12_seat_map(self, seeded_client, db):
        """RAPI-12 — seats dict maps seat numbers to usernames"""
        from app.db.models.table_seating import TableSeat
        client, seeded = seeded_client
        # Add TableSeat rows (replay endpoint needs these to build the seat map)
        for i, p in enumerate(seeded["players"]):
            db.add(TableSeat(
                session_id=seeded["session"].session_id,
                seat_number=i + 1,
                player_id=p.player_id,
            ))
        db.commit()
        hand = make_hand(db, seeded)
        resp = client.get(f"/replay/hands/{hand.hand_id}")
        seats = resp.json()["seats"]
        assert len(seats) == 6
        assert seats["1"] == "Player 1"

    def test_rapi_13_initial_stacks(self, seeded_client, db):
        """RAPI-13"""
        from app.db.models.table_seating import TableSeat
        client, seeded = seeded_client
        for i, p in enumerate(seeded["players"]):
            db.add(TableSeat(
                session_id=seeded["session"].session_id,
                seat_number=i + 1,
                player_id=p.player_id,
            ))
        db.commit()
        hand = make_hand(db, seeded)
        p = seeded["players"][0]
        make_action(db, hand.hand_id, p.player_id, stack_before=150)
        resp = client.get(f"/replay/hands/{hand.hand_id}")
        stacks = resp.json()["initial_stacks"]
        assert stacks.get("1") == 150

    def test_rapi_14_hole_cards_formatted(self, seeded_client, db):
        """RAPI-14 — hole card strings like 'Ah'"""
        client, seeded = seeded_client
        hand = make_hand(db, seeded)
        p = seeded["players"][0]
        make_hole_card(db, hand.hand_id, p.player_id, card=51)  # Ace of spades
        resp = client.get(f"/replay/hands/{hand.hand_id}")
        hole_cards = resp.json()["hole_cards"]
        assert len(hole_cards) >= 1
        # All cards should be 2-char strings
        for hcs in hole_cards:
            for c in hcs["cards"]:
                assert isinstance(c, str)
                assert len(c) == 2

    def test_rapi_15_board_cards_street_node(self, seeded_client, db):
        """RAPI-15"""
        client, seeded = seeded_client
        hand = make_hand(db, seeded)
        make_board_card(db, hand.hand_id, street=1, node=0, card=10)
        resp = client.get(f"/replay/hands/{hand.hand_id}")
        board_cards = resp.json()["board_cards"]
        assert len(board_cards) == 1
        assert board_cards[0]["street"] == 1
        assert board_cards[0]["node"] == 0

    def test_rapi_17_payouts_sum_equals_pot(self, seeded_client, db):
        """RAPI-17"""
        from app.db.models.payouts import Payout
        client, seeded = seeded_client
        hand = make_hand(db, seeded, pot=100)
        p = seeded["players"][0]
        payout = Payout(hand_id=hand.hand_id, player_id=p.player_id, amount=100, point_id=None)
        db.add(payout); db.commit()
        resp = client.get(f"/replay/hands/{hand.hand_id}")
        payouts = resp.json()["payouts"]
        total = sum(p["amount"] for p in payouts)
        assert total == hand.pot


# ── Annotation CRUD ───────────────────────────────────────────────────────────

class TestAnnotations:
    def test_rapi_20_no_annotations(self, seeded_client, db):
        """RAPI-20"""
        client, seeded = seeded_client
        hand = make_hand(db, seeded)
        resp = client.get(f"/replay/hands/{hand.hand_id}/annotations")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_rapi_21_create_annotation(self, seeded_client, db):
        """RAPI-21"""
        client, seeded = seeded_client
        hand = make_hand(db, seeded)
        resp = client.post(
            f"/replay/hands/{hand.hand_id}/annotations",
            json={"comment": "Great bluff here"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "annotation_id" in data
        assert data["comment"] == "Great bluff here"

    def test_rapi_22_create_annotation_unknown_hand(self, seeded_client):
        """RAPI-22"""
        client, _ = seeded_client
        resp = client.post(
            "/replay/hands/99999/annotations",
            json={"comment": "test"},
        )
        assert resp.status_code == 404

    def test_rapi_23_create_then_list(self, seeded_client, db):
        """RAPI-23"""
        client, seeded = seeded_client
        hand = make_hand(db, seeded)
        client.post(f"/replay/hands/{hand.hand_id}/annotations", json={"comment": "note1"})
        resp = client.get(f"/replay/hands/{hand.hand_id}/annotations")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["comment"] == "note1"

    def test_rapi_24_patch_comment(self, seeded_client, db):
        """RAPI-24"""
        client, seeded = seeded_client
        hand = make_hand(db, seeded)
        create_resp = client.post(
            f"/replay/hands/{hand.hand_id}/annotations",
            json={"comment": "original"},
        )
        ann_id = create_resp.json()["annotation_id"]
        patch_resp = client.patch(f"/replay/annotations/{ann_id}", json={"comment": "updated"})
        assert patch_resp.status_code == 200
        assert patch_resp.json()["comment"] == "updated"

    def test_rapi_25_patch_selected_cards(self, seeded_client, db):
        """RAPI-25"""
        client, seeded = seeded_client
        hand = make_hand(db, seeded)
        create_resp = client.post(
            f"/replay/hands/{hand.hand_id}/annotations",
            json={"comment": "cards note"},
        )
        ann_id = create_resp.json()["annotation_id"]
        patch_resp = client.patch(
            f"/replay/annotations/{ann_id}",
            json={"selected_cards": ["Ah", "Kd"]},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["selected_cards"] == ["Ah", "Kd"]

    def test_rapi_26_patch_wrong_user(self, seeded_client, db):
        """RAPI-26 — annotation belongs to user 1 (stub); wrong user → 404"""
        client, seeded = seeded_client
        hand = make_hand(db, seeded)
        # Create annotation for user_id=1 (STUB_USER_ID)
        ann = make_annotation(db, hand.hand_id, user_id=999, comment="other user")
        resp = client.patch(f"/replay/annotations/{ann.annotation_id}", json={"comment": "hack"})
        # STUB_USER_ID=1 ≠ 999 → 404
        assert resp.status_code == 404

    def test_rapi_27_delete_annotation(self, seeded_client, db):
        """RAPI-27"""
        client, seeded = seeded_client
        hand = make_hand(db, seeded)
        create_resp = client.post(
            f"/replay/hands/{hand.hand_id}/annotations",
            json={"comment": "to delete"},
        )
        ann_id = create_resp.json()["annotation_id"]
        del_resp = client.delete(f"/replay/annotations/{ann_id}")
        assert del_resp.status_code == 204

    def test_rapi_28_delete_unknown(self, seeded_client):
        """RAPI-28"""
        client, _ = seeded_client
        resp = client.delete("/replay/annotations/99999")
        assert resp.status_code == 404

    def test_rapi_29_delete_then_list(self, seeded_client, db):
        """RAPI-29"""
        client, seeded = seeded_client
        hand = make_hand(db, seeded)
        create_resp = client.post(
            f"/replay/hands/{hand.hand_id}/annotations",
            json={"comment": "gone"},
        )
        ann_id = create_resp.json()["annotation_id"]
        client.delete(f"/replay/annotations/{ann_id}")
        resp = client.get(f"/replay/hands/{hand.hand_id}/annotations")
        assert resp.json() == []