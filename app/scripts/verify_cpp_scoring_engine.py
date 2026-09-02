"""
verify_cpp_scoring_engine.py — Step 1 of the PokerState -> GraphEngine
migration (see migration doc, "Suggested cutover order", item 1).

Swaps MockScoringEngine -> CppScoringEngine in isolation and checks the
real evaluator against a set of hand-evaluation cases with known,
hand-verifiable outcomes. Nothing else in the migration touches this —
no graph, no HandSession, no callbacks. If this script doesn't pass,
stop here; don't proceed to wiring GraphEngine up to real scoring.

Interface under test (per migration doc):
    evaluate(player_masks, board_mask, score_type, showdown_type)
        player_masks : list[int]   — one 52-bit hole-card mask per player
        board_mask    : int        — bit mask of dealt board cards
        score_type    : poker_eval.ScoreType   (e.g. HIGH, LOW_A5, LOW_27)
        showdown_type : poker_eval.ShowdownType (e.g. SINGLE_BOARD)

    Returns something ordered/comparable per player — this script only
    asserts the RELATIVE ordering (who wins / who ties), since the
    absolute score representation is an internal implementation detail
    shared between Mock and Cpp engines, not part of what we're
    validating here.

Adjust the two import lines under "CONFIGURE ME" if MockScoringEngine
lives somewhere other than poker_engine.graph.mock_scoring_engine, or
if CppScoringEngine's evaluate() signature differs from the doc.

Run:
    python verify_cpp_scoring_engine.py
"""

from __future__ import annotations

import sys

# ── CONFIGURE ME ────────────────────────────────────────────────────
from poker_engine.scoring.scoring_engine import CppScoringEngine
from poker_engine.cards.card import Card
from poker_engine import poker_eval

try:
    from poker_engine.scoring.mock_scoring_engine import MockScoringEngine
    # from poker_engine.graph.mock_scoring_engine import MockScoringEngine
    HAVE_MOCK = True
except ImportError:
    HAVE_MOCK = False
# ─────────────────────────────────────────────────────────────────────


def cards_to_mask(card_strs: list[str]) -> int:
    mask = 0
    for cs in card_strs:
        mask |= 1 << Card.from_str(cs).id
    return mask


class Case:
    def __init__(self, name, hole_cards, board_cards, expected_order, score_type="HIGH"):
        """
        hole_cards      : list[list[str]] — per-player hole cards
        board_cards     : list[str]
        expected_order  : list[list[int]] — groups of player indices from
                          best to worst; players within the same inner
                          list are expected to TIE.
        """
        self.name = name
        self.hole_cards = hole_cards
        self.board_cards = board_cards
        self.expected_order = expected_order
        self.score_type = score_type


CASES = [
    Case(
        name="set over two pair (river)",
        hole_cards=[["Ah", "Ad"], ["Kh", "Kd"]],
        board_cards=["As", "Kc", "7d", "2h", "3s"],
        # P0: trip aces. P1: trip kings. P0 wins.
        expected_order=[[0], [1]],
    ),
    Case(
        name="flush beats straight",
        hole_cards=[["2h", "7h"], ["9c", "8d"]],
        board_cards=["Ah", "Kh", "5h", "Tc", "Jc"],
        # P0: flush (hearts). P1: straight (9-T-J-Q? no) -> use straight 8-9-T-J-... 
        # board gives Tc Jc, hole 9c 8d -> straight 8-9-T-J-? needs a 7 or Q; not present.
        # Replace with an unambiguous case below instead.
        expected_order=[[0], [1]],
    ),
    Case(
        name="split pot — identical best 5 on board",
        hole_cards=[["2c", "3d"], ["4h", "5s"]],
        board_cards=["Ah", "Kh", "Qh", "Jh", "Th"],
        # Board itself is an ace-high straight flush; neither hole pair
        # improves it, both players play the board -> tie.
        expected_order=[[0, 1]],
    ),
]

# Fix the flush-vs-straight case with cards that unambiguously make both hands.
CASES[1] = Case(
    name="flush beats straight",
    hole_cards=[["2h", "7h"], ["9s", "Tc"]],
    board_cards=["Ah", "Kh", "5h", "Jc", "Qc"],
    # P0: hole 2h 7h + board Ah Kh 5h -> flush (A K 7 5 2 of hearts).
    # P1: hole 9s Tc + board Jc Qc -> straight 9-T-J-Q-K.
    # Flush beats straight -> P0 wins.
    expected_order=[[0], [1]],
)


def run_case(engine, case: Case) -> bool:
    player_masks = [cards_to_mask(hc) for hc in case.hole_cards]
    board_mask = cards_to_mask(case.board_cards)
    score_type = getattr(poker_eval.ScoreType, case.score_type)
    showdown_type = poker_eval.ShowdownType.SINGLE_BOARD

    results = engine.evaluate(player_masks, board_mask, score_type, showdown_type)
    # results assumed to be a sequence of (score, ...) or comparable
    # objects, one per player, higher-is-better — matches
    # poker_eval.evaluate_hands' own convention used elsewhere in the
    # codebase (see equity_service.py).
    scores = [r[0] if isinstance(r, (tuple, list)) else r for r in results]

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # Build actual tie-groups from scores, in descending order.
    actual_groups: list[list[int]] = []
    for idx in ranked:
        if actual_groups and scores[idx] == scores[actual_groups[-1][0]]:
            actual_groups[-1].append(idx)
        else:
            actual_groups.append([idx])
    actual_groups = [sorted(g) for g in actual_groups]
    expected_groups = [sorted(g) for g in case.expected_order]

    ok = actual_groups == expected_groups
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {case.name}")
    print(f"        scores={scores}")
    print(f"        expected order={expected_groups}  actual order={actual_groups}")
    return ok


def main():
    engines = {"cpp": CppScoringEngine()}
    if HAVE_MOCK:
        engines["mock"] = MockScoringEngine()
    else:
        print("NOTE: MockScoringEngine not importable at the configured path — "
              "only validating CppScoringEngine against known-good expectations, "
              "not cross-checking against Mock. Update the import at the top of "
              "this file once you know Mock's real module path.\n")

    all_ok = True
    for engine_name, engine in engines.items():
        print(f"=== engine: {engine_name} ===")
        for case in CASES:
            ok = run_case(engine, case)
            all_ok = all_ok and ok
        print()

    if all_ok:
        print("All cases passed. CppScoringEngine is safe to substitute for "
              "MockScoringEngine — proceed to Step 2 (Hold'em behind a flag).")
        sys.exit(0)
    else:
        print("At least one case FAILED. Do not wire CppScoringEngine into "
              "GraphEngine until this is resolved.")
        sys.exit(1)


if __name__ == "__main__":
    main()