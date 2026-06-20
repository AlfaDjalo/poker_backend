"""
§8 Adapter and DTO Tests — ADP-*

Tests for engine_adapter.state_to_dto and build_showdown_dto.
Uses minimal mock PokerState objects (SimpleNamespace) — no real engine needed.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.engine_adapter import state_to_dto, build_showdown_dto, card_to_str


# ─────────────────────────────────────────────────────────────────
# Helpers / factories
# ─────────────────────────────────────────────────────────────────

def make_player(stack=100, current_bet=0, has_folded=False, hand_mask=0):
    return SimpleNamespace(
        stack=stack,
        current_bet=current_bet,
        has_folded=has_folded,
        hand_mask=hand_mask,
    )


def make_point_def(name="board1", score_type_name="HIGH", node_sets=None):
    score_type = SimpleNamespace(name=score_type_name)
    return SimpleNamespace(
        name=name,
        score_type=score_type,
        node_sets=node_sets or [[0, 1, 2]],
    )


def make_game(
    node_cards=None,
    players=None,
    street_index=0,
    pot=0,
    current_player=0,
    bet_to_call=0,
    min_raise=0,
    legal_actions_list=None,
):
    g = SimpleNamespace(
        node_cards=node_cards or [None] * 5,
        players=players or [make_player()],
        street_index=street_index,
        pot=pot,
        current_player=current_player,
        bet_to_call=bet_to_call,
        min_raise=min_raise,
    )
    if legal_actions_list is not None:
        action_objs = [SimpleNamespace(name=a) for a in legal_actions_list]
        g.legal_actions = lambda: action_objs
    else:
        g.legal_actions = lambda: []
    return g


def make_game_def(
    layout_name="single_board",
    game_name="holdem",
    street_names=None,
    node_count=5,
):
    return SimpleNamespace(
        layout_name=layout_name,
        game_name=game_name,
        street_names=street_names or ["preflop", "flop", "turn", "river"],
        node_count=node_count,
    )


def make_rules(points=None, payout_type="split_pot"):
    return SimpleNamespace(
        points=points or [make_point_def()],
        payout_type=payout_type,
    )


def make_poker_state(
    phase_name="BETTING",
    node_cards=None,
    players=None,
    current_player=0,
    points=None,
    last_showdown=None,
    last_winners=None,
):
    game = make_game(
        node_cards=node_cards or [None] * 5,
        players=players or [make_player(hand_mask=0b11)],
        current_player=current_player,
        legal_actions_list=["call", "fold", "raise"],
    )
    game_def = make_game_def()
    rules = make_rules(points=points)

    state = SimpleNamespace(
        phase=SimpleNamespace(name=phase_name),
        game=game,
        game_def=game_def,
        rules=rules,
        last_showdown=last_showdown,
        last_winners=last_winners or [],
    )
    return state


# ─────────────────────────────────────────────────────────────────
# ADP-30 / ADP-31 — card_to_str
# ─────────────────────────────────────────────────────────────────

class TestCardToStr:
    def test_adp30_none_returns_none(self):
        """ADP-30: card_id=None → None"""
        assert card_to_str(None) is None

    def test_adp31_valid_card_returns_string(self):
        """ADP-31: valid card id → 2-char string like 'Ah'"""
        result = card_to_str(0)
        assert isinstance(result, str)
        assert len(result) == 2


# ─────────────────────────────────────────────────────────────────
# ADP-01 – ADP-10 — state_to_dto
# ─────────────────────────────────────────────────────────────────

class TestStateToDtoNodes:
    def test_adp01_all_none_node_cards(self):
        """ADP-01: 5 None node_cards → nodes list of 5 Nones"""
        state = make_poker_state(node_cards=[None] * 5)
        dto = state_to_dto(state)
        assert dto.nodes == [None] * 5

    def test_adp02_one_card_set(self):
        """ADP-02: node_cards[2] = 0 (card id) → nodes[2] is a card string"""
        node_cards = [None, None, 0, None, None]
        state = make_poker_state(node_cards=node_cards)
        dto = state_to_dto(state)
        assert dto.nodes[2] is not None
        assert isinstance(dto.nodes[2], str)
        assert dto.nodes[0] is None


class TestStateToDtoPlayers:
    def test_adp03_players_with_hand_mask(self):
        """ADP-03: players with non-zero hand_mask → non-empty hand lists"""
        # hand_mask with bits 0 and 1 set = cards 0 and 1
        players = [make_player(hand_mask=0b11), make_player(hand_mask=0b11)]
        state = make_poker_state(players=players)
        dto = state_to_dto(state)
        for p in dto.players:
            assert len(p.hand) > 0

    def test_adp10_folded_player(self):
        """ADP-10: player with has_folded=True → PlayerDTO.folded == True"""
        players = [make_player(has_folded=True)]
        state = make_poker_state(players=players)
        dto = state_to_dto(state)
        assert dto.players[0].folded is True

    def test_players_are_one_indexed(self):
        """Seats are 1-indexed."""
        players = [make_player(), make_player(), make_player()]
        state = make_poker_state(players=players)
        dto = state_to_dto(state)
        seats = [p.seat for p in dto.players]
        assert seats == [1, 2, 3]


class TestStateToDtoBettingPhase:
    def test_adp04_non_betting_phase(self):
        """ADP-04: phase != BETTING → current_player=None, available_actions=[]"""
        state = make_poker_state(phase_name="DEAL_BOARD")
        dto = state_to_dto(state)
        assert dto.current_player is None
        assert dto.available_actions == []

    def test_adp05_betting_phase_current_player(self):
        """ADP-05: phase=BETTING, engine current_player=2 → DTO current_player=3"""
        players = [make_player(), make_player(), make_player()]
        game = make_game(
            players=players,
            current_player=2,
            legal_actions_list=["call", "fold"],
        )
        state = make_poker_state(phase_name="BETTING", players=players)
        state.game = game
        state.game.current_player = 2
        dto = state_to_dto(state)
        assert dto.current_player == 3

    def test_adp06_to_call_clamped_to_zero(self):
        """ADP-06: if computed to_call < 0, clamped to 0"""
        # bet_to_call=0, player.current_bet=5 → to_call = -5 → clamped to 0
        player = make_player(current_bet=5)
        game = make_game(players=[player], current_player=0, bet_to_call=0, legal_actions_list=["check"])
        state = make_poker_state(phase_name="BETTING", players=[player])
        state.game = game
        dto = state_to_dto(state)
        assert dto.to_call >= 0


class TestStateToDtoShowdown:
    def test_adp07_showdown_not_none(self):
        """ADP-07: last_showdown is not None → showdown is ShowdownDTO"""
        from app.dto.state_dto import ShowdownDTO
        showdown_result = _make_showdown_result()
        state = make_poker_state(phase_name="SHOWDOWN", last_showdown=showdown_result)
        dto = state_to_dto(state)
        assert dto.showdown is not None

    def test_adp08_showdown_is_none(self):
        """ADP-08: last_showdown is None → showdown is None"""
        state = make_poker_state(phase_name="BETTING", last_showdown=None)
        dto = state_to_dto(state)
        assert dto.showdown is None


class TestStateToDtoPoints:
    def test_adp09_points_mapped_correctly(self):
        """ADP-09: rules.points → PointDTO list with correct name and score_type strings"""
        points = [
            make_point_def("board1", "HIGH", [[0, 1, 2]]),
            make_point_def("board2", "LOW_A5", [[3]]),
        ]
        state = make_poker_state(points=points)
        dto = state_to_dto(state)
        assert len(dto.points) == 2
        assert dto.points[0].name == "board1"
        assert dto.points[0].score_type == "HIGH"
        assert dto.points[1].name == "board2"
        assert dto.points[1].score_type == "LOW_A5"
        assert dto.points[0].node_sets == [[0, 1, 2]]


# ─────────────────────────────────────────────────────────────────
# ADP-20 – ADP-27 — build_showdown_dto
# ─────────────────────────────────────────────────────────────────

def _make_player_result(player_index=0, is_winner=True, category="Pair", value=1500000, best_hand=None, hole_used=None, board_used=None):
    return SimpleNamespace(
        player_index=player_index,
        is_winner=is_winner,
        category=category,
        value=value,
        best_hand_cards=best_hand or [],
        hole_cards_used=hole_used or [],
        board_cards_used=board_used or [],
    )


def _make_point_obj(name="board1", score_type_name="HIGH", showdown_type_val=0, results=None, node_mask=0b111):
    return SimpleNamespace(
        name=name,
        score_type=SimpleNamespace(name=score_type_name),
        showdown_type=SimpleNamespace(name="HOLDEM"),
        results=results or [_make_player_result(0, True), _make_player_result(1, False)],
        node_mask=node_mask,
    )


def _make_showdown_result(points=None, payouts=None, payout_type="split_pot", point_tallies=None, scoop_flags=None):
    return SimpleNamespace(
        points=points or [_make_point_obj()],
        payouts=payouts or {0: 100, 1: 0},
        payout_type=payout_type,
        point_tallies=point_tallies,
        scoop_flags=scoop_flags,
    )


class TestBuildShowdownDto:
    def test_adp20_none_result_returns_none(self):
        """ADP-20: result=None → None"""
        assert build_showdown_dto(None, None, []) is None

    def test_adp21_single_point_single_board_one_winner(self):
        """ADP-21: single point, single board, player 0 wins"""
        result = _make_showdown_result(
            points=[_make_point_obj(results=[
                _make_player_result(0, True),
                _make_player_result(1, False),
            ])]
        )
        rules = make_rules()
        dto = build_showdown_dto(result, rules, [0, 1])
        assert dto is not None
        assert dto.point_results[0].board_winners == [[0]]

    def test_adp22_two_boards_under_one_point_name(self):
        """ADP-22: two boards grouped under same point name → board_results length 2"""
        board1 = _make_point_obj(name="main", results=[_make_player_result(0, True)])
        board2 = _make_point_obj(name="main", results=[_make_player_result(1, True)])
        result = _make_showdown_result(points=[board1, board2])
        rules = make_rules()
        dto = build_showdown_dto(result, rules, [0, 1])
        assert len(dto.point_results[0].board_results) == 2

    def test_adp23_no_qualify_when_no_winners(self):
        """ADP-23: no winners → no_qualify[0] == True"""
        point = _make_point_obj(results=[
            _make_player_result(0, False),
            _make_player_result(1, False),
        ])
        result = _make_showdown_result(points=[point])
        rules = make_rules()
        dto = build_showdown_dto(result, rules, [0, 1])
        assert dto.point_results[0].no_qualify[0] is True

    def test_adp24_scoop_flags_present(self):
        """ADP-24: scoop_flags present → scoop[i] matches result.scoop_flags[point][board]"""
        result = _make_showdown_result(scoop_flags=[[True]])
        rules = make_rules()
        dto = build_showdown_dto(result, rules, [0, 1])
        assert dto.point_results[0].scoop[0] is True

    def test_adp25_scoop_flags_absent(self):
        """ADP-25: scoop_flags=None → no exception, scoop[i] == False"""
        result = _make_showdown_result(scoop_flags=None)
        rules = make_rules()
        dto = build_showdown_dto(result, rules, [0, 1])
        assert dto.point_results[0].scoop[0] is False

    def test_adp26_pot_winners_excludes_zero_payout(self):
        """ADP-26: pot_winners only contains players with amount > 0"""
        result = _make_showdown_result(payouts={0: 100, 1: 0, 2: 50})
        rules = make_rules()
        dto = build_showdown_dto(result, rules, [0, 1, 2])
        assert set(dto.pot_winners) == {0, 2}
        assert 1 not in dto.pot_winners

    def test_adp27_best_hand_cards_are_strings(self):
        """ADP-27: best_hand_cards mapped via card_to_str, not raw ints"""
        result = _make_showdown_result(
            points=[_make_point_obj(results=[
                _make_player_result(0, True, best_hand=[0, 1, 2, 3, 4]),
            ])]
        )
        rules = make_rules()
        dto = build_showdown_dto(result, rules, [0])
        cards = dto.point_results[0].board_results[0][0].best_hand_cards
        assert all(isinstance(c, str) for c in cards)