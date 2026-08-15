"""
equity_service.py
Bridges CAP game state (variant, players, board) to the C++ equity engine.

Responsibilities:
  - Load a game variant via the existing loader
  - Convert card strings ("Ah", "Kd") to integer ids via Card.from_str
  - Convert PointDefinitions and ScoreType enums to the plain ints cap_equity expects
  - Call cap_equity.calculate_equity()
  - Return a clean dict suitable for the FastAPI response

The service intentionally has NO FastAPI dependency — it can be called
from tests, notebooks, or the API layer.

── REDESIGN (rich equity reporting) ────────────────────────────────────
cap_equity.calculate_equity now returns a much richer structure per
player: overall pot equity (fraction + currency), scoop/split
probability, and a full per-point breakdown (win/tie probability +
currency contribution). To make that possible the C++ engine needs to
run the SAME resolution rules real showdowns use, so this service now
also passes through:

  - payout_type          (GameRules.payout_type: "points" | "split_pot")
  - no_qualify_action     (GameRules.no_qualify_action: "scoop" | "eliminate")
  - pot_size              (caller-supplied; 0 disables currency fields)
  - per point: is_low, low_qualifier, scoop_from
      (mirrors PointDefinition / GameRules so the C++ engine can apply
      the same no-qualify handling as ShowdownResolver._handle_no_qualify)

evaluator_wrapper is unchanged from the previous fix: it re-runs
rules.qualifies() on every raw score before it reaches the C++ engine,
so a non-qualifying low reading can never "win" a low point's equity
cell. low_qualifier / is_low are passed to C++ as well, purely as a
safety net in case a future evaluator implementation doesn't pre-filter.
─────────────────────────────────────────────────────────────────────────
"""

from typing import Any

from poker_engine import cap_equity, poker_eval
from poker_engine.cards.card import Card
from poker_engine.games.loader import load_game


def _card_srt_to_id(card_str: str | None) -> int | None:
    """Convert "Ah" -> 48, None -> None."""
    if card_str is None:
        return None
    return Card.from_str(card_str).id


def _card_id_to_none_or_int(card_id: int | None) -> int:
    """Cap equity uses -1 for unknown; convert None -> -1."""
    return -1 if card_id is None else card_id


def _score_type_int(score_type) -> int:
    """Convert poker_eval.ScoreType enum to int."""
    # pybind11 enums support int() cast
    return int(score_type)


def _showdown_type_int(showdown_type) -> int:
    return int(showdown_type)


class _RawScore:
    """
    Minimal stand-in for scoring.score_result.ScoreResult, just enough
    to satisfy GameRules.qualifies()'s `score.score[0]` access without
    pulling in the full evaluation-engine ScoreResult type here. Used
    only to re-run the qualifier check against a raw int returned by
    poker_eval.evaluate_hands.
    """

    __slots__ = ("score",)

    def __init__(self, value: int):
        self.score = (value,)


class EquityService:
    """
    Stateless equity calculator service.
    Thread-safe (each call is independent).
    """

    def calculate(
        self,
        variant_name: str,
        players_input: list[
            dict
        ],  # [{"seat": int, "hole_cards": ["Ah", "Kd" | None, ...]}]
        board_nodes_input: list[
            dict | None
        ],  # [{"node": int, "card": "7s" | None}, ...]
        street: int = 0,
        pot_size: float = 0.0,
        exact_threshold: int = 50000,
        mc_iterations: int = 20000,
    ) -> dict[str, Any]:
        """
        Calculate rich equity for all players across all scoring points.

        Parameters
        ----------
        variant_name        : registered game variant (e.g. "holdem")
        players_input       : list of {seat, hole_cards: ["Ah", None, ...]}
                              None entries = unknown hole card slots
        board_nodes_input   : list of {node: int, card: str|None}
                              all nodes the variant uses; None card = not dealt yet
        street              : current street index (informational only)
        pot_size            : current pot in chips. If 0/omitted, all currency
                              fields in the result are 0.0 — only fractions
                              and probabilities are meaningful.
        exact_threshold     : switch to MC above this many combinations
        mc_iterations       : Monte Carlo iterations when above threshold

        Returns
        -------
        {
          "players": {
            seat: {
              "overall_equity_fraction": float,
              "overall_equity_currency": float,
              "scoop_probability": float,
              "split_probability": float,
              "points": {
                point_name: {
                  "win_share": float,
                  "win_probability": float,
                  "tie_probability": float,
                  "equity_currency": float,
                  "equity_percent": float,
                }, ...
              }
            }, ...
          },
          "method": "exact" | "monte_carlo",
          "iterations": int,
          "elapsed_ms": float,
        }
        """

        # ── Load game definition ──────────────────────────────────
        game_def, rules = load_game(variant_name)

        # ── Build node_count-length board_nodes list ──────────────
        # Start with all unknown (-1), fill in known cards.
        node_count = game_def.node_count
        board_nodes: list[int] = [-1] * node_count

        for entry in board_nodes_input or []:
            if entry is None:
                continue
            n = entry.get("node")
            c = entry.get("card")
            if n is None or n >= node_count:
                continue
            board_nodes[n] = _card_id_to_none_or_int(_card_srt_to_id(c))

        # ── Build player list ─────────────────────────────────────
        # For each player, collect known card ids only.
        # None slots are NOT added to known_cards — they represent
        # unknown draws and are handled by the C++ engine.
        players_c: list[dict] = []
        for p in players_input:
            seat = p["seat"]
            raw_cards: list[str | None] = p.get("hole_cards") or []
            known_ids: list[int] = []
            for c in raw_cards:
                cid = _card_srt_to_id(c)
                if cid is not None:
                    known_ids.append(cid)
            players_c.append(
                {
                    "seat": seat,
                    "known_cards": known_ids,
                    "total_hole_cards": game_def.hole_cards,
                }
            )

        # ── Build points list ─────────────────────────────────────
        # Use the default showdown_type from rules; honour per-point overrides.
        # "is_low" / "low_qualifier" / "scoop_from" mirror GameRules /
        # PointDefinition so the C++ engine can replicate
        # ShowdownResolver._handle_no_qualify's scoop/eliminate behaviour
        # and correctly compute scoop/split probabilities at the pot level.
        default_showdown = rules.showdown_type
        points_c: list[dict] = []
        for pt in rules.points:
            showdown = (
                pt.showdown_type_override
                if pt.showdown_type_override is not None
                else default_showdown
            )
            is_low = rules.is_low_type(pt.score_type)
            points_c.append(
                {
                    "name": pt.name,
                    "score_type": pt.score_type,
                    "showdown_type": showdown,
                    "node_sets": [list(ns) for ns in pt.node_sets],
                    "is_low": is_low,
                    "low_qualifier": (
                        rules.low_qualifier
                        if is_low and rules.low_qualifier is not None
                        else -1
                    ),
                    "scoop_from": pt.scoop_from or "",
                }
            )

        # ── Wrap evaluator to re-cast int args back to enums, and to
        # apply the point's qualifier (e.g. 8-or-better for PLO8 low) ──
        # cap_equity passes score_type / showdown_type as plain ints;
        # poker_eval.evaluate_hands requires the actual enum types.
        #
        # rules.qualifies() is the SAME check showdown_resolver.py uses
        # at real showdown, so a hand that wouldn't win/split the low
        # half of the pot in an actual hand can no longer "win" the low
        # equity cell here either. For non-low score types this is a
        # cheap no-op (qualifies() just checks score != 0).
        def evaluator_wrapper(hands, board_mask, score_type, showdown_type):
            st_enum = poker_eval.ScoreType(score_type)
            sd_enum = poker_eval.ShowdownType(showdown_type)

            raw = poker_eval.evaluate_hands(hands, board_mask, st_enum, sd_enum)

            if not rules.is_low_type(st_enum):
                return raw

            filtered = []
            for item in raw:
                # item is (score_int, best_hand_mask, ...) per
                # poker_eval.evaluate_hands / CppScoringEngine.evaluate.
                score_val = item[0]
                if score_val != 0 and rules.qualifies(st_enum, _RawScore(score_val)):
                    filtered.append(item)
                else:
                    # Does not qualify (e.g. no 8-or-better low) — treat
                    # as "no hand" so it can't win the low equity cell.
                    filtered.append((0,) + tuple(item[1:]))
            return filtered

        # ── Call C++ engine ───────────────────────────────────────
        result = cap_equity.calculate_equity(
            variant_name=variant_name,
            total_hole_cards=game_def.hole_cards,
            players=players_c,
            board_nodes=board_nodes,
            points=points_c,
            evaluator=evaluator_wrapper,
            payout_type=rules.payout_type,
            no_qualify_action=rules.no_qualify_action,
            pot_size=pot_size,
            exact_threshold=exact_threshold,
            mc_iterations=mc_iterations,
        )

        return result


# Singleton - the service is stateless so sharing is fine
equity_service = EquityService()
