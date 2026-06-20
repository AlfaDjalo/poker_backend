"""
test_game_service_edit.py — GameService edit/snapshot/validation methods.
Test plan §16 (GSE-01 .. GSE-43).

Strategy: direct instantiation via the `game_service_with_state` fixture
(fake PokerState/game/players — bypasses restart() and the real engine
entirely, per conftest.py). DB-touching tests (_delete_hand_records) use
the real `db` SQLite fixture.
"""
import pytest
from types import SimpleNamespace

from app.services.game_service import GameService
from app.api.edit_api import EditStateRequest, PlayerEditInput

from tests.conftest import FakePlayer, FakeGame, FakePokerState, FakeLogger


# ──────────────────────────────────────────────────────────────────
# 16.1 — _snapshot_state / _restore_snapshot
# ──────────────────────────────────────────────────────────────────

class TestSnapshotRestore:

    def test_GSE_01_snapshot_then_restore_no_mutation(self, game_service_with_state):
        svc = game_service_with_state
        snap = svc._snapshot_state()
        svc._restore_snapshot(snap)

        g = svc.state.game
        assert g.pot == snap["pot"]
        assert g.street_index == snap["street_index"]
        assert g.node_cards == snap["node_cards"]
        for i, p in enumerate(g.players):
            assert p.stack == snap["players"][i]["stack"]
            assert p.hand_mask == snap["players"][i]["hand_mask"]

    def test_GSE_02_snapshot_mutate_then_restore(self, game_service_with_state):
        svc = game_service_with_state
        snap = svc._snapshot_state()

        svc.state.game.pot = 9999
        svc.state.game.players[0].stack = 1

        svc._restore_snapshot(snap)

        assert svc.state.game.pot == snap["pot"]
        assert svc.state.game.players[0].stack == snap["players"][0]["stack"]

    def test_GSE_03_discard_pile_absent_on_game(self, game_service_with_state):
        svc = game_service_with_state
        g = svc.state.game
        # Remove the attribute entirely to simulate an engine build without it
        del g.discard_pile

        snap = svc._snapshot_state()
        assert snap["discard_pile"] == []

        # restore should not raise even though hasattr(g, "discard_pile") is False
        svc._restore_snapshot(snap)
        assert not hasattr(g, "discard_pile")

    def test_GSE_04_player_attrs_absent_uses_getattr_defaults(self, game_service_with_state):
        svc = game_service_with_state
        p = svc.state.game.players[0]
        del p.total_contribution
        del p.is_all_in

        snap = svc._snapshot_state()
        assert snap["players"][0]["total_contribution"] == 0
        assert snap["players"][0]["has_folded"] == p.has_folded
        assert snap["players"][0]["is_all_in"] is False

        # restore should not raise — guarded by hasattr
        svc._restore_snapshot(snap)
        assert not hasattr(p, "total_contribution")
        assert not hasattr(p, "is_all_in")

    def test_GSE_05_snapshot_copies_node_cards_not_aliases(self, game_service_with_state):
        svc = game_service_with_state
        snap = svc._snapshot_state()

        svc.state.game.node_cards[0] = 99
        svc.state.game.node_cards.append(7)

        assert snap["node_cards"] != svc.state.game.node_cards
        assert 99 not in snap["node_cards"] or snap["node_cards"][0] != 99


# ──────────────────────────────────────────────────────────────────
# 16.2 — _apply_snapshot_to_state
# ──────────────────────────────────────────────────────────────────

class TestApplySnapshotToState:

    def _make_req(self, edit_state_request_factory, **overrides):
        return EditStateRequest(**edit_state_request_factory(**overrides))

    def test_GSE_10_none_node_cards_stay_none(self, game_service_with_state, edit_state_request_factory, mocker):
        svc = game_service_with_state
        fake_card = mocker.patch("app.services.game_service.CardObj")
        fake_card.from_str.side_effect = lambda s: SimpleNamespace(id=hash(s) % 52)

        req = self._make_req(
            edit_state_request_factory,
            node_cards=[None, "Ah", None, None, None],
        )
        svc._apply_snapshot_to_state(req)

        assert svc.state.game.node_cards[0] is None
        assert svc.state.game.node_cards[2] is None
        assert svc.state.game.node_cards[1] is not None

    def test_GSE_11_mixed_known_unknown_hole_cards(self, game_service_with_state, edit_state_request_factory, mocker):
        svc = game_service_with_state
        fake_card = mocker.patch("app.services.game_service.CardObj")
        # deterministic small ids so bit math is easy to assert on
        id_map = {"Ah": 0, "Kd": 1, "Qh": 2, "Jc": 3}
        fake_card.from_str.side_effect = lambda s: SimpleNamespace(id=id_map[s])

        req = self._make_req(
            edit_state_request_factory,
            players=[
                {
                    "seat": 1, "stack": 100, "current_bet": 0,
                    "total_contribution": 0, "has_folded": False,
                    "is_all_in": False, "hole_cards": ["Ah", None],
                },
                {
                    "seat": 2, "stack": 100, "current_bet": 0,
                    "total_contribution": 0, "has_folded": False,
                    "is_all_in": False, "hole_cards": [None, None],
                },
            ],
        )
        svc._apply_snapshot_to_state(req)

        p0 = svc.state.game.players[0]
        assert p0.hand_mask == (1 << 0)  # only "Ah" bit set
        p1 = svc.state.game.players[1]
        assert p1.hand_mask == 0

    def test_GSE_12_seat_out_of_range_skipped(self, game_service_with_state, edit_state_request_factory, mocker):
        svc = game_service_with_state
        fake_card = mocker.patch("app.services.game_service.CardObj")
        fake_card.from_str.side_effect = lambda s: SimpleNamespace(id=0)

        req = self._make_req(
            edit_state_request_factory,
            players=[
                {
                    "seat": 99, "stack": 1, "current_bet": 0,
                    "total_contribution": 0, "has_folded": False,
                    "is_all_in": False, "hole_cards": [None, None],
                },
                {
                    "seat": 0, "stack": 1, "current_bet": 0,
                    "total_contribution": 0, "has_folded": False,
                    "is_all_in": False, "hole_cards": [None, None],
                },
            ],
        )
        # should not raise IndexError despite out-of-range seats
        svc._apply_snapshot_to_state(req)
        # original 2 fake players untouched
        assert svc.state.game.players[0].stack != 1
        assert svc.state.game.players[1].stack != 1

    def test_GSE_13_discard_pile_rebuilt_when_present(self, game_service_with_state, edit_state_request_factory, mocker):
        svc = game_service_with_state
        fake_card = mocker.patch("app.services.game_service.CardObj")
        fake_card.from_str.side_effect = lambda s: SimpleNamespace(id=5)

        req = self._make_req(edit_state_request_factory, discard_pile=["2c", "3d"])
        svc._apply_snapshot_to_state(req)

        assert svc.state.game.discard_pile == [5, 5]

    def test_GSE_14_discard_pile_absent_skipped(self, game_service_with_state, edit_state_request_factory, mocker):
        svc = game_service_with_state
        del svc.state.game.discard_pile
        fake_card = mocker.patch("app.services.game_service.CardObj")
        fake_card.from_str.side_effect = lambda s: SimpleNamespace(id=5)

        req = self._make_req(edit_state_request_factory, discard_pile=["2c"])
        # should not raise even though g has no discard_pile attribute
        svc._apply_snapshot_to_state(req)
        assert not hasattr(svc.state.game, "discard_pile")


# ──────────────────────────────────────────────────────────────────
# 16.3 — _validate_edit_request
# ──────────────────────────────────────────────────────────────────

class TestValidateEditRequest:

    def _req(self, edit_state_request_factory, **overrides):
        return EditStateRequest(**edit_state_request_factory(**overrides))

    def test_GSE_20_all_unique_no_exception(self, game_service_with_state, edit_state_request_factory):
        svc = game_service_with_state
        req = self._req(edit_state_request_factory)
        svc._validate_edit_request(req)  # should not raise

    def test_GSE_21_duplicate_within_one_players_hole_cards(self, game_service_with_state, edit_state_request_factory):
        svc = game_service_with_state
        req = self._req(
            edit_state_request_factory,
            players=[
                {
                    "seat": 1, "stack": 100, "current_bet": 0,
                    "total_contribution": 0, "has_folded": False,
                    "is_all_in": False, "hole_cards": ["Ah", "Ah"],
                },
            ],
        )
        with pytest.raises(ValueError, match="Duplicate card"):
            svc._validate_edit_request(req)

    def test_GSE_22_duplicate_between_node_and_discard(self, game_service_with_state, edit_state_request_factory):
        svc = game_service_with_state
        req = self._req(
            edit_state_request_factory,
            node_cards=["2c", None, None, None, None],
            discard_pile=["2c"],
        )
        with pytest.raises(ValueError, match="Duplicate card"):
            svc._validate_edit_request(req)

    def test_GSE_23_more_than_52_cards(self, game_service_with_state, edit_state_request_factory):
        svc = game_service_with_state
        ranks = "23456789TJQKA"
        suits = "shdc"
        all_cards = [r + s for r in ranks for s in suits]  # 52 unique
        extra_cards = ["2s2", "3s2"]  # two more bogus-but-unique strings

        players = [
            {
                "seat": 1, "stack": 100, "current_bet": 0,
                "total_contribution": 0, "has_folded": False,
                "is_all_in": False, "hole_cards": all_cards[:2],
            },
        ]
        node_cards = all_cards[2:7]
        discard_pile = all_cards[7:] + extra_cards

        req = self._req(
            edit_state_request_factory,
            players=players,
            node_cards=node_cards,
            discard_pile=discard_pile,
        )
        with pytest.raises(ValueError, match="More than 52 cards"):
            svc._validate_edit_request(req)

    def test_GSE_24_multiple_violations_joined(self, game_service_with_state, edit_state_request_factory):
        svc = game_service_with_state
        ranks = "23456789TJQKA"
        suits = "shdc"
        all_cards = [r + s for r in ranks for s in suits]  # 52 unique
        extra_cards = ["2s2", "3s2"]

        players = [
            {
                "seat": 1, "stack": 100, "current_bet": 0,
                "total_contribution": 0, "has_folded": False,
                "is_all_in": False, "hole_cards": [all_cards[0], all_cards[0]],
            },
        ]
        node_cards = all_cards[1:6]
        discard_pile = all_cards[6:] + extra_cards

        req = self._req(
            edit_state_request_factory,
            players=players,
            node_cards=node_cards,
            discard_pile=discard_pile,
        )
        with pytest.raises(ValueError) as exc_info:
            svc._validate_edit_request(req)

        msg = str(exc_info.value)
        assert "Duplicate card" in msg
        assert "More than 52 cards" in msg
        assert "; " in msg


# ──────────────────────────────────────────────────────────────────
# 16.4 — begin_edit / apply_edit / load_edit / cancel_edit
# ──────────────────────────────────────────────────────────────────

class TestEditLifecycle:

    def test_GSE_30_begin_edit_no_state_raises(self):
        svc = GameService()
        assert svc.state is None
        with pytest.raises(RuntimeError):
            svc.begin_edit(db=None)

    def test_GSE_31_begin_edit_no_logger_skips_delete(self, game_service_with_state, mocker):
        svc = game_service_with_state
        svc.logger = None
        delete_spy = mocker.patch.object(svc, "_delete_hand_records")

        svc.begin_edit(db="fake_db")

        assert svc.pre_edit_snapshot is not None
        assert svc.editing_mode is True
        delete_spy.assert_not_called()

    def test_GSE_32_begin_edit_with_logger_hand_id_calls_delete(self, game_service_with_state, mocker):
        svc = game_service_with_state
        svc.logger = FakeLogger(hand_id=42)
        delete_spy = mocker.patch.object(svc, "_delete_hand_records")

        svc.begin_edit(db="fake_db")

        delete_spy.assert_called_once_with("fake_db", 42)

    def test_GSE_33_apply_edit_not_editing_raises(self, game_service_with_state, edit_state_request_factory):
        svc = game_service_with_state
        svc.editing_mode = False
        req = EditStateRequest(**edit_state_request_factory())

        with pytest.raises(RuntimeError, match="/game/edit/begin"):
            svc.apply_edit(req)

    def test_GSE_34_apply_edit_invalid_request_keeps_editing_mode_true(
        self, game_service_with_state, edit_state_request_factory, mocker
    ):
        svc = game_service_with_state
        svc.editing_mode = True
        mocker.patch.object(svc, "_load_snapshot_from_request")
        mocker.patch.object(svc, "_progress_engine", return_value="dto")

        req = EditStateRequest(**edit_state_request_factory(
            node_cards=["Ah", "Ah", None, None, None],
        ))

        with pytest.raises(ValueError, match="Duplicate card"):
            svc.apply_edit(req)

        # editing_mode untouched on failure — verify this is intentional per BUG-BE-11
        assert svc.editing_mode is True
        svc._load_snapshot_from_request.assert_not_called()

    def test_GSE_35_load_edit_always_sets_editing_mode_true(
        self, game_service_with_state, edit_state_request_factory, mocker
    ):
        svc = game_service_with_state
        svc.editing_mode = False

        mocker.patch("app.services.game_service.load_game", return_value=(
            SimpleNamespace(game_name="holdem", hole_cards=2, node_count=5), SimpleNamespace()
        ))
        mocker.patch("app.services.game_service.PokerState", return_value=svc.state)
        mocker.patch("app.services.game_service.CppScoringEngine")
        mocker.patch.object(svc, "_apply_snapshot_to_state")
        mocker.patch("app.services.game_service.state_to_dto", return_value="dto")

        req = EditStateRequest(**edit_state_request_factory())
        result = svc.load_edit(req)

        assert svc.editing_mode is True
        assert result == "dto"

    def test_GSE_36_load_edit_builds_fresh_players_from_request_only(
        self, game_service_with_state, edit_state_request_factory, mocker
    ):
        svc = game_service_with_state

        mocker.patch("app.services.game_service.load_game", return_value=(
            SimpleNamespace(game_name="omaha", hole_cards=4, node_count=5), SimpleNamespace()
        ))
        captured = {}

        def fake_poker_state_ctor(players, game_def, rules, scoring_engine, callbacks=None):
            captured["players"] = players
            return svc.state  # reuse existing fake state object as the "new" one

        mocker.patch("app.services.game_service.PokerState", side_effect=fake_poker_state_ctor)
        mocker.patch("app.services.game_service.CppScoringEngine")
        mocker.patch.object(svc, "_apply_snapshot_to_state")
        mocker.patch("app.services.game_service.state_to_dto", return_value="dto")
        build_continuation_spy = mocker.patch.object(svc, "_build_continuation_players")

        req = EditStateRequest(**edit_state_request_factory(
            players=[
                {
                    "seat": 1, "stack": 777, "current_bet": 0,
                    "total_contribution": 0, "has_folded": False,
                    "is_all_in": False, "hole_cards": ["Ah", "Kd"],
                },
                {
                    "seat": 2, "stack": 888, "current_bet": 0,
                    "total_contribution": 0, "has_folded": False,
                    "is_all_in": False, "hole_cards": ["Qh", "Jc"],
                },
            ],
        ))
        svc.load_edit(req)

        # fresh PlayerState objects built only from req.players[i].stack
        stacks = [p.stack for p in captured["players"]]
        assert stacks == [777, 888]
        # does NOT reuse _build_continuation_players (that's the in-place apply_edit path)
        build_continuation_spy.assert_not_called()

    def test_GSE_37_cancel_edit_no_snapshot_raises(self, game_service_with_state):
        svc = game_service_with_state
        svc.pre_edit_snapshot = None

        with pytest.raises(RuntimeError, match="No pre-edit snapshot"):
            svc.cancel_edit()

    def test_GSE_38_cancel_edit_restores_and_resets_flags(self, game_service_with_state, mocker):
        svc = game_service_with_state
        snap = svc._snapshot_state()
        svc.pre_edit_snapshot = snap
        svc.editing_mode = True

        # mutate live state so restore has something to undo
        svc.state.game.pot = 12345

        mocker.patch("app.services.game_service.state_to_dto", return_value="dto")
        result = svc.cancel_edit()

        assert svc.state.game.pot == snap["pot"]
        assert svc.editing_mode is False
        assert svc.pre_edit_snapshot is None
        assert result == "dto"

    def test_GSE_39_cancel_edit_signature_takes_no_args(self, game_service_with_state):
        """
        Confirms GameService.cancel_edit() takes no db/req argument — matching
        the method as written, NOT the broken edit_api.py route (BUG-BE-07),
        which incorrectly calls game_service.cancel_edit(db).
        """
        svc = game_service_with_state
        svc.pre_edit_snapshot = svc._snapshot_state()

        import inspect
        sig = inspect.signature(svc.cancel_edit)
        assert list(sig.parameters.keys()) == []

        # calling with an extra positional arg should fail
        with pytest.raises(TypeError):
            svc.cancel_edit("unexpected_db_arg")


# ──────────────────────────────────────────────────────────────────
# 16.5 — _delete_hand_records
# ──────────────────────────────────────────────────────────────────

class TestDeleteHandRecords:

    def test_GSE_40_full_fk_chain_deleted(self, db, seeded_db, game_service_with_state):
        from app.db.models.hands import Hand
        from app.db.models.actions import Action
        from app.db.models.hole_cards import HoleCard
        from app.db.models.board_cards import BoardCard
        from app.db.models.hand_points import HandPoint
        from app.db.models.point_results import PointResult
        from app.db.models.point_cards import PointCard
        from app.db.models.payouts import Payout

        hand = Hand(session_id=None, variant_name="holdem", layout_name="single_board",
                    split_pot=False, dealer_seat=0, pot=10, is_hypothetical=False)
        db.add(hand)
        db.flush()

        hid = hand.hand_id
        pid = seeded_db["players"][0].player_id

        hp = HandPoint(hand_id=hid, name="hand", showdown_type="SHOWDOWN",
                        score_type="HIGH", node_set=0)
        db.add(hp)
        db.flush()

        pr = PointResult(point_id=hp.point_id, player_id=pid, best_hand_mask=0,
                          rank=1, hand_value=1, hand_category="pair", point_share=1.0)
        db.add(pr)
        db.flush()

        db.add(PointCard(point_result_id=pr.point_result_id, card=0, source="hole"))
        db.add(Payout(hand_id=hid, point_id=hp.point_id, player_id=pid, amount=10))

        db.add(Action(hand_id=hid, street=0, action_index=0, player_id=pid,
                      action_type="CALL", amount=2, pot_before=0, stack_before=100))
        db.add(HoleCard(hand_id=hid, player_id=pid, street=0, card=0, visible=True))
        db.add(BoardCard(hand_id=hid, street=1, node=0, card=1))
        db.commit()

        point_id = hp.point_id
        point_result_id = pr.point_result_id

        svc = game_service_with_state
        svc._delete_hand_records(db, hid)

        assert db.query(Hand).filter(Hand.hand_id == hid).first() is None
        assert db.query(Action).filter(Action.hand_id == hid).count() == 0
        assert db.query(HoleCard).filter(HoleCard.hand_id == hid).count() == 0
        assert db.query(BoardCard).filter(BoardCard.hand_id == hid).count() == 0
        assert db.query(HandPoint).filter(HandPoint.hand_id == hid).count() == 0
        assert db.query(PointResult).filter(PointResult.point_id == point_id).count() == 0
        assert db.query(PointCard).filter(PointCard.point_result_id == point_result_id).count() == 0
        assert db.query(Payout).filter(Payout.hand_id == hid).count() == 0

    def test_GSE_41_no_hand_point_rows_skips_cleanly(self, db, seeded_db, game_service_with_state):
        from app.db.models.hands import Hand
        from app.db.models.actions import Action

        hand = Hand(session_id=None, variant_name="holdem", layout_name="single_board",
                    split_pot=False, dealer_seat=0, pot=0, is_hypothetical=False)
        db.add(hand)
        db.flush()
        hid = hand.hand_id

        pid = seeded_db["players"][0].player_id

        db.add(Action(hand_id=hid, street=0, action_index=0, player_id=pid,
                       action_type="FOLD", amount=None, pot_before=0, stack_before=100))
        db.commit()

        svc = game_service_with_state
        svc._delete_hand_records(db, hid)  # should not raise

        assert db.query(Hand).filter(Hand.hand_id == hid).first() is None
        assert db.query(Action).filter(Action.hand_id == hid).count() == 0

    def test_GSE_42_hand_points_but_no_point_results_skips_cleanly(self, db, game_service_with_state):
        from app.db.models.hands import Hand
        from app.db.models.hand_points import HandPoint

        hand = Hand(session_id=None, variant_name="holdem", layout_name="single_board",
                    split_pot=False, dealer_seat=0, pot=0, is_hypothetical=False)
        db.add(hand)
        db.flush()
        hid = hand.hand_id

        hp = HandPoint(hand_id=hid, name="hand", showdown_type="SHOWDOWN",
                        score_type="HIGH", node_set=0)
        db.add(hp)
        db.commit()

        svc = game_service_with_state
        svc._delete_hand_records(db, hid)  # pr_ids empty list — no PointCard delete attempted

        assert db.query(Hand).filter(Hand.hand_id == hid).first() is None
        assert db.query(HandPoint).filter(HandPoint.hand_id == hid).count() == 0

    def test_GSE_43_unknown_hand_id_is_noop(self, db, game_service_with_state):
        svc = game_service_with_state
        # no exception, commit still happens, for a hand_id that doesn't exist anywhere
        svc._delete_hand_records(db, 999999)