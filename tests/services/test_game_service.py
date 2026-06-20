"""
tests/services/test_game_service.py
GS-01 ... GS-18
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


@pytest.fixture()
def svc():
    """Fresh GameService instance (not the singleton) for each test."""
    from app.services.game_service import GameService
    return GameService()


@pytest.fixture()
def mock_load_game():
    """Patch games.loader.load_game used inside game_service."""
    game_def = MagicMock()
    game_def.game_name = "holdem"
    game_def.layout_name = "single_board"
    game_def.hole_cards = 2
    game_def.node_count = 5
    game_def.street_nodes = [[0, 1, 2], [3], [4]]
    game_def.street_names = None

    rules = MagicMock()
    rules.payout_type = "split_pot"
    rules.points = []
    rules.showdown_type = MagicMock()

    with patch("app.services.game_service.load_game", return_value=(game_def, rules)) as m:
        yield m, game_def, rules


@pytest.fixture()
def mock_poker_state():
    with patch("app.services.game_service.PokerState") as MockState:
        instance = MagicMock()
        instance.phase = MagicMock()
        instance.phase.name = "BETTING"
        instance.game = MagicMock()
        instance.game.players = [MagicMock(stack=100) for _ in range(6)]
        instance.game.street_index = 0
        instance.game.pot = 0
        instance.game.node_cards = [None] * 5
        instance.game.current_player = 0
        instance.game.bet_to_call = 0
        instance.game.min_raise = 2
        instance.game.legal_actions.return_value = []
        instance.game_def = MagicMock()
        instance.game_def.game_name = "holdem"
        instance.game_def.layout_name = "single_board"
        instance.game_def.street_names = None
        instance.rules = MagicMock()
        instance.rules.points = []
        instance.last_showdown = None
        instance.last_winners = []
        MockState.return_value = instance
        yield MockState, instance


class TestGetVariants:
    def test_gs_01_returns_variants(self, svc, tmp_path):
        """GS-01"""
        games_dir = tmp_path / "games"
        games_dir.mkdir()
        (games_dir / "holdem.yaml").touch()
        (games_dir / "plo.yaml").touch()
        (games_dir / "plo8.yaml").touch()

        with patch("app.services.game_service.engine_root", tmp_path):
            result = svc.get_variants()

        assert "variants" in result
        assert len(result["variants"]) == 3
        assert "current" in result


class TestSelectGame:
    def test_gs_02_valid_game(self, svc, tmp_path):
        """GS-02"""
        games_dir = tmp_path / "games"
        games_dir.mkdir()
        (games_dir / "holdem.yaml").touch()

        with patch("app.services.game_service.engine_root", tmp_path):
            svc.select_game("holdem")

        assert svc.pending_game == "holdem"

    def test_gs_03_unknown_game_raises(self, svc, tmp_path):
        """GS-03"""
        games_dir = tmp_path / "games"
        games_dir.mkdir()

        with patch("app.services.game_service.engine_root", tmp_path):
            with pytest.raises(ValueError, match="Unknown game variant"):
                svc.select_game("nonexistent")


class TestApplyPendingGame:
    def test_gs_04_pending_game_consumed(self, svc, tmp_path):
        """GS-04"""
        games_dir = tmp_path / "games"
        games_dir.mkdir()
        (games_dir / "plo.yaml").touch()

        with patch("app.services.game_service.engine_root", tmp_path):
            svc.pending_game = "plo"
            result = svc._apply_pending_game()

        assert result == "plo"
        assert svc.pending_game is None
        assert svc.current_game == "plo"

    def test_gs_05_explicit_overrides_pending(self, svc, tmp_path):
        """GS-05"""
        games_dir = tmp_path / "games"
        games_dir.mkdir()
        (games_dir / "plo.yaml").touch()
        (games_dir / "holdem.yaml").touch()

        with patch("app.services.game_service.engine_root", tmp_path):
            svc.pending_game = "plo"
            result = svc._apply_pending_game(game_name="holdem")

        assert result == "holdem"
        assert svc.current_game == "holdem"


class TestRestart:
    def test_gs_06_no_table_raises(self, svc, db):
        """GS-06"""
        with pytest.raises(Exception, match="No poker tables"):
            svc.restart(db)

    def test_gs_07_too_few_players_raises(self, svc, db):
        """GS-07 — only 3 players, no seats"""
        from app.db.models.poker_tables import PokerTable
        from app.db.models.players import Player
        table = PokerTable(table_name="T", max_players=6)
        db.add(table); db.flush()
        for i in range(3):
            db.add(Player(username=f"P{i}", is_bot=False))
        db.commit()
        with pytest.raises(Exception):
            svc.restart(db)

    def test_gs_08_valid_restart_returns_dto(self, svc, db, seeded_db,
                                               mock_load_game, mock_poker_state):
        """GS-08"""
        _, mock_state = mock_poker_state

        with patch("app.services.game_service.SessionLogger"), \
             patch("app.services.game_service.BackendEngineCallbacks"), \
             patch("app.services.game_service.CppScoringEngine"), \
             patch("app.services.game_service.state_to_dto") as mock_dto:
            mock_dto.return_value = {"phase": "BETTING", "street": 0, "pot": 0,
                                      "players": [], "nodes": [], "layout_name": "single_board",
                                      "game_name": "holdem", "street_names": None, "points": [],
                                      "current_player": None, "showdown": None, "winners": None,
                                      "available_actions": [], "to_call": 0,
                                      "min_raise": 2, "max_raise": 100}
            result = svc.restart(db)

        assert svc.state is not None
        assert result is not None


class TestNewHand:
    def test_gs_09_new_hand_without_state_raises(self, svc):
        """GS-09"""
        with pytest.raises(RuntimeError, match="No active game"):
            svc.new_hand()

    def test_gs_10_same_variant_no_recreate(self, svc, mock_load_game, mock_poker_state):
        """GS-10"""
        _, mock_state = mock_poker_state
        _, game_def, rules = mock_load_game
        game_def.game_name = "holdem"
        svc.state = mock_state
        svc.state.game_def.game_name = "holdem"
        svc.current_game = "holdem"

        with patch("app.services.game_service.state_to_dto") as mock_dto:
            mock_dto.return_value = {}
            svc.new_hand()

        mock_state.start_hand.assert_called_once()

    def test_gs_11_different_variant_recreates_state(self, svc, mock_load_game, mock_poker_state, tmp_path):
        """GS-11"""
        MockState, mock_state = mock_poker_state
        _, game_def, _ = mock_load_game
        game_def.game_name = "plo"

        games_dir = tmp_path / "games"
        games_dir.mkdir()
        (games_dir / "plo.yaml").touch()

        svc.state = mock_state
        svc.state.game_def.game_name = "holdem"
        svc.current_game = "holdem"
        svc.callbacks = MagicMock()

        with patch("app.services.game_service.engine_root", tmp_path), \
            patch("app.services.game_service.state_to_dto") as mock_dto, \
            patch("app.services.game_service.CppScoringEngine"):
            mock_dto.return_value = {}
            svc.new_hand(game_name="plo")

        assert MockState.call_count >= 1


class TestGetState:
    def test_gs_12_none_state(self, svc):
        """GS-12"""
        assert svc.get_state() is None

    def test_gs_13_active_state(self, svc):
        """GS-13"""
        with patch("app.services.game_service.state_to_dto", return_value={"phase": "BETTING"}):
            svc.state = MagicMock()
            result = svc.get_state()
        assert result == {"phase": "BETTING"}


class TestApplyAction:
    def test_gs_14_valid_action(self, svc):
        """GS-14"""
        svc.state = MagicMock()
        svc.state.phase.name = "BETTING"

        req = MagicMock()
        req.type = "call"
        req.amount = None

        with patch("app.services.game_service.state_to_dto", return_value={"phase": "BETTING"}), \
             patch("app.services.game_service.to_engine_action", return_value=MagicMock()):
            result = svc.apply_action(req)

        svc.state.step.assert_called_once()
        assert result is not None

    def test_gs_15_none_state_returns_none(self, svc):
        """GS-15"""
        req = MagicMock()
        assert svc.apply_action(req) is None


class TestProgressEngine:
    def test_gs_16_loops_through_deal_board(self, svc):
        """GS-16 — _progress_engine loops while phase == Phase.DEAL_BOARD."""
        from app.services.game_service import Phase

        # Use a spec'd SimpleNamespace so we can assign .phase directly
        # without touching MagicMock's shared metaclass.
        from types import SimpleNamespace

        idx = [0]
        phases = [Phase.DEAL_BOARD, Phase.DEAL_BOARD, Phase.BETTING]

        class _FakeState:
            last_showdown = None
            last_winners = []

            @property
            def phase(self):
                return phases[min(idx[0], len(phases) - 1)]

            def step(self, _action):
                if idx[0] < len(phases) - 1:
                    idx[0] += 1

            def __getattr__(self, name):
                # satisfy any other attribute access (e.g. game) with a MagicMock
                return MagicMock()

        fake_state = _FakeState()
        svc.state = fake_state

        with patch("app.services.game_service.state_to_dto") as mock_dto:
            mock_dto.return_value = MagicMock(winners=None)
            svc._progress_engine()

        # step() was called at least twice (once per DEAL_BOARD phase)
        assert idx[0] >= 2

    def test_gs_17_showdown_populates_winners(self, svc):
        """GS-17 — when phase is SHOWDOWN, winners are populated in the DTO."""
        from app.services.game_service import Phase

        class _FakeState:
            phase = Phase.SHOWDOWN
            last_showdown = None
            last_winners = [0, 2]

            def step(self, _action):
                pass

            def __getattr__(self, name):
                return MagicMock()

        svc.state = _FakeState()

        dto = MagicMock()
        dto.winners = None

        with patch("app.services.game_service.state_to_dto", return_value=dto):
            result = svc._progress_engine()

        assert result.winners == [1, 3]


class TestGetWinners:
    def test_gs_18_one_indexed(self, svc):
        """GS-18"""
        svc.state = MagicMock()
        svc.state.last_winners = [0, 2]
        assert svc._get_winners() == [1, 3]
