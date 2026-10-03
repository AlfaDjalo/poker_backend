"""
tests/test_hole_card_count.py

Regression tests for the deals_hole/card_count derivation bug: the
previous implementation asserted against hand-constructed nodes with
fake `metadata={"deals_hole": True, "card_count": 2}` dicts — exactly
the wrong assumption that let the bug ship, since real compiled graphs
never populate metadata that way at all (the real signal is
node.kind == NodeKind.AUTO / node.auto_type == "deal_hole_cards" /
node.config["count"]).

These tests instead load REAL variant YAMLs through the graph-native
loader and assert against the actual compiled graph, so a future
regression in either the flow compiler's real node shape or in these
call sites' assumptions about it will be caught by construction rather
than by a fixture that can silently drift from reality.
"""

import pytest

from poker_engine.graph.graph_hole_cards import (
    first_deal_hole_cards_count,
    hole_cards_per_player,
)
from poker_engine.rules.graph_loader import load_game_graph

from app.services.equity_service import EquityService
from app.services.game_service import _summarize_flow


# ---------------------------------------------------------------------------
# Canonical helper, against real compiled graphs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant_name,expected_hole_cards",
    [
        ("holdem", 2),
        ("plo8", 4),
    ],
)
def test_hole_cards_per_player_real_graph(variant_name, expected_hole_cards):
    _, _, graph = load_game_graph(variant_name)
    assert hole_cards_per_player(graph) == expected_hole_cards


@pytest.mark.parametrize(
    "variant_name,expected_hole_cards",
    [
        ("holdem", 2),
        ("plo8", 4),
    ],
)
def test_first_deal_hole_cards_count_real_graph(variant_name, expected_hole_cards):
    _, _, graph = load_game_graph(variant_name)
    assert first_deal_hole_cards_count(graph) == expected_hole_cards


def test_hole_cards_per_player_raises_on_graph_with_no_deal_node():
    """
    hole_cards_per_player() must hard-fail (ValueError) rather than
    silently return 0/None when a graph genuinely has no
    deal_hole_cards node — equity_service.EquityService.calculate()
    depends on this propagating as a real error rather than being
    masked by a fallback (see equity_service.py's module docstring on
    why the old dry-run fallback was removed).
    """
    class _FakeNode:
        kind = type("Kind", (), {"name": "AUTO"})()
        auto_type = "deal_board_cards"  # NOT deal_hole_cards
        config = {"count": 1}
        metadata = {}

    class _FakeGraph:
        start_node = "n0"

        def node(self, node_id):
            return _FakeNode()

        def outgoing(self, node_id):
            return []

    with pytest.raises(ValueError):
        hole_cards_per_player(_FakeGraph())


# ---------------------------------------------------------------------------
# game_service._summarize_flow — real graph, both the aggregate field
# and the per-phase `deals_hole` flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant_name,expected_hole_cards",
    [
        ("holdem", 2),
        ("plo8", 4),
    ],
)
def test_summarize_flow_hole_cards_field(variant_name, expected_hole_cards):
    game_def, rules, graph = load_game_graph(variant_name)
    summary = _summarize_flow(game_def, rules, graph)
    assert summary["hole_cards"] == expected_hole_cards


@pytest.mark.parametrize("variant_name", ["holdem", "plo8"])
def test_summarize_flow_always_starts_with_setup_phase(variant_name):
    game_def, rules, graph = load_game_graph(variant_name)
    summary = _summarize_flow(game_def, rules, graph)
    phases = summary["creation_phases"]
    assert phases[0]["id"] == "SETUP"
    assert phases[0]["deals_hole"] is False
    assert phases[0]["deals_board"] is False
    assert phases[0]["allows_betting"] is False
    assert phases[0]["decision_domain"] is None


@pytest.mark.parametrize("variant_name", ["holdem", "plo8"])
def test_summarize_flow_marks_real_deal_phase(variant_name):
    """
    At least one non-SETUP phase in creation_phases must be flagged
    deals_hole=True for a real hold'em/PLO8 graph — this is the exact
    per-phase flag the Hand Creator wizard/tutorial_api.py rely on,
    and the one the old metadata-based walk silently always returned
    False for (every phase looked like deals_hole=False, since the
    metadata key it checked was never actually present).
    """
    game_def, rules, graph = load_game_graph(variant_name)
    summary = _summarize_flow(game_def, rules, graph)
    assert any(p["deals_hole"] for p in summary["creation_phases"])


# ---------------------------------------------------------------------------
# equity_service.EquityService — real graph, end-to-end through
# total_hole_cards derivation (not calling cap_equity itself, just
# confirming the ValueError paths and the derived count are correct
# before it would reach the C++ engine)
# ---------------------------------------------------------------------------


def test_equity_service_derives_hole_cards_from_real_graph():
    from app.services.equity_service import _hole_cards_per_player_from_metadata

    _, _, graph = load_game_graph("holdem")
    assert _hole_cards_per_player_from_metadata(graph) == 2


def test_equity_service_unknown_variant_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_game_graph("definitely_not_a_real_variant_xyz")