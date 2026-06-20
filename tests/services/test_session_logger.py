"""
tests/services/test_session_logger.py
SL-01 … SL-46
"""
import pytest
from unittest.mock import MagicMock


def make_engine_player(hand_mask=0b11):
    """Minimal mock of engine PlayerState."""
    p = MagicMock()
    p.hand_mask = hand_mask
    return p


def make_game_def(street_nodes=None):
    gd = MagicMock()
    gd.street_nodes = street_nodes or [[0, 1, 2], [3], [4]]
    return gd


def make_state(node_cards=None, players=None, last_showdown=None):
    state = MagicMock()
    state.game = MagicMock()
    state.game.node_cards = node_cards or [None] * 5
    state.game.players = players or []
    state.last_showdown = last_showdown
    return state


@pytest.fixture()
def logger(db, seeded_db):
    from app.services.session_logger import SessionLogger
    lg = SessionLogger(db)
    lg.start_game({
        "table_id": seeded_db["table"].table_id,
        "player_ids": [p.player_id for p in seeded_db["players"]],
    })
    return lg, seeded_db


# ── start_game ────────────────────────────────────────────────────────────────

class TestStartGame:
    def test_sl_01_session_created(self, db, seeded_db):
        """SL-01"""
        from app.services.session_logger import SessionLogger
        from app.db.models.poker_sessions import PokerSession

        lg = SessionLogger(db)
        lg.start_game({
            "table_id": seeded_db["table"].table_id,
            "player_ids": [p.player_id for p in seeded_db["players"]],
        })

        sessions = db.query(PokerSession).all()
        assert len(sessions) >= 1
        assert lg.session_id is not None

    def test_sl_02_table_seats_created(self, db, seeded_db):
        """SL-02"""
        from app.services.session_logger import SessionLogger
        from app.db.models.table_seating import TableSeat

        player_ids = [p.player_id for p in seeded_db["players"]]
        lg = SessionLogger(db)
        lg.start_game({"table_id": seeded_db["table"].table_id, "player_ids": player_ids})

        seats = db.query(TableSeat).filter(TableSeat.session_id == lg.session_id).all()
        assert len(seats) == len(player_ids)

    def test_sl_03_seat_numbers_one_indexed(self, db, seeded_db):
        """SL-03"""
        from app.services.session_logger import SessionLogger
        from app.db.models.table_seating import TableSeat

        player_ids = [p.player_id for p in seeded_db["players"]]
        lg = SessionLogger(db)
        lg.start_game({"table_id": seeded_db["table"].table_id, "player_ids": player_ids})

        seats = db.query(TableSeat).filter(
            TableSeat.session_id == lg.session_id
        ).order_by(TableSeat.seat_number).all()

        seat_numbers = [s.seat_number for s in seats]
        assert seat_numbers[0] == 1
        assert seat_numbers == list(range(1, len(player_ids) + 1))


# ── start_hand ────────────────────────────────────────────────────────────────

class TestStartHand:
    def _start_hand(self, lg, seeded_db, variant="holdem"):
        players = [make_engine_player(hand_mask=0b11) for _ in seeded_db["players"]]
        gd = make_game_def()
        lg.start_hand({
            "variant_name": variant,
            "layout_name": "single_board",
            "split_pot": False,
            "betting_config_id": 1,
            "dealer_seat": 1,
            "pot": 0,
            "ended_at": None,
            "players": players,
            "game_def": gd,
        })

    def test_sl_10_hand_row_created(self, logger, db):
        """SL-10"""
        from app.db.models.hands import Hand
        lg, seeded = logger
        self._start_hand(lg, seeded)
        hands = db.query(Hand).filter(Hand.session_id == lg.session_id).all()
        assert len(hands) == 1
        assert hands[0].variant_name == "holdem"
        assert hands[0].layout_name == "single_board"
        assert hands[0].split_pot is False

    def test_sl_11_hole_card_rows_created(self, logger, db):
        """SL-11"""
        from app.db.models.hole_cards import HoleCard
        lg, seeded = logger
        # hand_mask=0b11 → 2 set bits → 2 cards
        self._start_hand(lg, seeded)
        cards = db.query(HoleCard).filter(HoleCard.hand_id == lg.hand_id).all()
        assert len(cards) == 6 * 2  # 6 players × 2 cards

    def test_sl_12_hand_id_stored(self, logger):
        """SL-12"""
        lg, seeded = logger
        players = [make_engine_player() for _ in seeded["players"]]
        lg.start_hand({
            "variant_name": "holdem", "layout_name": "single_board",
            "split_pot": False, "betting_config_id": 1, "dealer_seat": 1,
            "pot": 0, "ended_at": None, "players": players,
            "game_def": make_game_def(),
        })
        assert lg.hand_id is not None

    def test_sl_13_logged_nodes_reset(self, logger):
        """SL-13"""
        lg, seeded = logger
        lg._logged_nodes = {(99, 0), (99, 1)}
        players = [make_engine_player() for _ in seeded["players"]]
        lg.start_hand({
            "variant_name": "holdem", "layout_name": "single_board",
            "split_pot": False, "betting_config_id": 1, "dealer_seat": 1,
            "pot": 0, "ended_at": None, "players": players,
            "game_def": make_game_def(),
        })
        assert lg._logged_nodes == set()

    def test_sl_14_node_street_map_built(self, logger):
        """SL-14 — node 0 → street 1 for holdem layout"""
        lg, seeded = logger
        players = [make_engine_player() for _ in seeded["players"]]
        lg.start_hand({
            "variant_name": "holdem", "layout_name": "single_board",
            "split_pot": False, "betting_config_id": 1, "dealer_seat": 1,
            "pot": 0, "ended_at": None, "players": players,
            "game_def": make_game_def(street_nodes=[[0, 1, 2], [3], [4]]),
        })
        assert lg._node_to_street_map[0] == 1
        assert lg._node_to_street_map[3] == 2
        assert lg._node_to_street_map[4] == 3

    def test_sl_15_none_game_def(self, logger):
        """SL-15 — no exception when game_def is None"""
        lg, seeded = logger
        players = [make_engine_player() for _ in seeded["players"]]
        lg.start_hand({
            "variant_name": "holdem", "layout_name": "single_board",
            "split_pot": False, "betting_config_id": 1, "dealer_seat": 1,
            "pot": 0, "ended_at": None, "players": players,
            "game_def": None,
        })
        assert lg._node_to_street_map == {}


# ── log_action ────────────────────────────────────────────────────────────────

class TestLogAction:
    def _setup_hand(self, lg, seeded):
        players = [make_engine_player() for _ in seeded["players"]]
        lg.start_hand({
            "variant_name": "holdem", "layout_name": "single_board",
            "split_pot": False, "betting_config_id": 1, "dealer_seat": 1,
            "pot": 0, "ended_at": None, "players": players,
            "game_def": make_game_def(),
        })

    def test_sl_20_action_row_created(self, logger, db):
        """SL-20"""
        from app.db.models.actions import Action
        lg, seeded = logger
        self._setup_hand(lg, seeded)
        lg.log_action(street=0, player_index=0, action="CALL",
                      amount=10, pot_before=5, stack_before=100)
        actions = db.query(Action).filter(Action.hand_id == lg.hand_id).all()
        assert len(actions) == 1
        assert actions[0].action_type == "CALL"
        assert actions[0].amount == 10

    def test_sl_21_action_index_increments(self, logger, db):
        """SL-21"""
        from app.db.models.actions import Action
        lg, seeded = logger
        self._setup_hand(lg, seeded)
        lg.log_action(street=0, player_index=0, action="CALL",
                      amount=10, pot_before=5, stack_before=100)
        lg.log_action(street=0, player_index=1, action="RAISE",
                      amount=20, pot_before=15, stack_before=90)
        actions = db.query(Action).filter(Action.hand_id == lg.hand_id).order_by(
            Action.action_index
        ).all()
        assert actions[0].action_index == 0
        assert actions[1].action_index == 1

    def test_sl_22_player_index_maps_to_player_id(self, logger, db):
        """SL-22"""
        from app.db.models.actions import Action
        lg, seeded = logger
        self._setup_hand(lg, seeded)
        lg.log_action(street=0, player_index=2, action="FOLD",
                      amount=None, pot_before=0, stack_before=100)
        action = db.query(Action).filter(Action.hand_id == lg.hand_id).first()
        expected_pid = seeded["players"][2].player_id
        assert action.player_id == expected_pid


# ── log_board ─────────────────────────────────────────────────────────────────

class TestLogBoard:
    def _setup(self, logger, db):
        lg, seeded = logger
        players = [make_engine_player() for _ in seeded["players"]]
        lg.start_hand({
            "variant_name": "holdem", "layout_name": "single_board",
            "split_pot": False, "betting_config_id": 1, "dealer_seat": 1,
            "pot": 0, "ended_at": None, "players": players,
            "game_def": make_game_def(),
        })
        return lg, seeded

    def test_sl_30_board_cards_logged(self, logger, db):
        """SL-30"""
        from app.db.models.board_cards import BoardCard
        lg, seeded = self._setup(logger, db)
        state = make_state(node_cards=[10, 20, 30, None, None])
        lg.log_board(state)
        cards = db.query(BoardCard).filter(BoardCard.hand_id == lg.hand_id).all()
        assert len(cards) == 3

    def test_sl_31_deduplication(self, logger, db):
        """SL-31 — calling log_board twice logs each node only once"""
        from app.db.models.board_cards import BoardCard
        lg, seeded = self._setup(logger, db)
        state = make_state(node_cards=[10, 20, None, None, None])
        lg.log_board(state)
        lg.log_board(state)
        cards = db.query(BoardCard).filter(BoardCard.hand_id == lg.hand_id).all()
        assert len(cards) == 2

    def test_sl_32_street_from_map(self, logger, db):
        """SL-32 — node 3 gets street=2 (turn) for holdem"""
        from app.db.models.board_cards import BoardCard
        lg, seeded = self._setup(logger, db)
        state = make_state(node_cards=[None, None, None, 42, None])
        lg.log_board(state)
        card = db.query(BoardCard).filter(BoardCard.hand_id == lg.hand_id).first()
        assert card.street == 2


# ── finish_hand ───────────────────────────────────────────────────────────────

class TestFinishHand:
    def _setup(self, logger, db):
        lg, seeded = logger
        players = [make_engine_player() for _ in seeded["players"]]
        lg.start_hand({
            "variant_name": "holdem", "layout_name": "single_board",
            "split_pot": False, "betting_config_id": 1, "dealer_seat": 1,
            "pot": 0, "ended_at": None, "players": players,
            "game_def": make_game_def(),
        })
        return lg, seeded

    def test_sl_40_no_showdown_no_exception(self, logger, db):
        """SL-40"""
        lg, seeded = self._setup(logger, db)
        state = make_state(last_showdown=None)
        lg.finish_hand(state)  # must not raise

    def test_sl_41_hand_points_created(self, logger, db):
        """SL-41"""
        from app.db.models.hand_points import HandPoint
        lg, seeded = self._setup(logger, db)

        point = MagicMock()
        point.name = "board1"
        point.showdown_type = "HOLDEM"
        point.score_type = "HIGH"
        point.node_mask = 0b11111
        point.results = []

        showdown = MagicMock()
        showdown.points = [point]
        showdown.payouts = {}
        state = make_state(last_showdown=showdown)
        lg.finish_hand(state)

        hp = db.query(HandPoint).filter(HandPoint.hand_id == lg.hand_id).all()
        assert len(hp) == 1
        assert hp[0].name == "board1"

    def test_sl_44_payout_rows_created(self, logger, db):
        """SL-44"""
        from app.db.models.payouts import Payout
        lg, seeded = self._setup(logger, db)

        showdown = MagicMock()
        showdown.points = []
        showdown.payouts = {0: 100, 1: 0}
        state = make_state(last_showdown=showdown)
        lg.finish_hand(state)

        payouts = db.query(Payout).filter(Payout.hand_id == lg.hand_id).all()
        assert len(payouts) == 2

    def test_sl_45_hand_pot_updated(self, logger, db):
        """SL-45"""
        from app.db.models.hands import Hand
        lg, seeded = self._setup(logger, db)

        showdown = MagicMock()
        showdown.points = []
        showdown.payouts = {0: 60, 1: 40}
        state = make_state(last_showdown=showdown)
        lg.finish_hand(state)

        hand = db.query(Hand).filter(Hand.hand_id == lg.hand_id).first()
        assert hand.pot == 100