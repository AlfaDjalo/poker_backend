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
"""
 
import time
from typing import Any

import cap_equity
import poker_eval

from cards.card import Card
from games.loader import load_game


def _card_srt_to_id(card_str: str | None) -> int | None:
    """ Convert "Ah" -> 48, None -> None. """
    if card_str is None:
        return None
    return Card.from_str(card_str).id


def _card_id_to_none_or_int(card_id: int | None) -> int:
    """ Cap equity uses -1 for unknown; convert None -> -1. """
    return -1 if card_id is None else card_id


def _score_type_int(score_type) -> int:
    """ Convert poker_eval.ScoreType enum to int. """
    # pybind11 enums support int() cast
    return int(score_type)


def _showdown_type_int(showdown_type) -> int:
    return int(showdown_type)


class EquityService:
    """
    Stateless equity calculator service.
    Thread-safe (each call is independent).
    """
        
    def calculate(
        self,
        variant_name: str,
        players_input: list[dict],    # [{"seat": int, "hole_cards": ["Ah", "Kd" | None, ...]}]
        board_nodes_input: list[dict | None], # [{"node": int, "card": "7s" | None}, ...]
        street: int = 0,
        exact_threshold: int = 50000,
        mc_iterations: int = 20000,
    ) -> dict[str, Any]:
        """
        Calculate equity for all players across all scoring points.
 
        Parameters
        ----------
        variant_name        : registered game variant (e.g. "holdem")
        players_input       : list of {seat, hole_cards: ["Ah", None, ...]}
                              None entries = unknown hole card slots
        board_nodes_input   : list of {node: int, card: str|None}
                              all nodes the variant uses; None card = not dealt yet
        street              : current street index (informational only)
        exact_threshold     : switch to MC above this many combinations
        mc_iterations       : Monte Carlo iterations when above threshold
 
        Returns
        -------
        {
          "equity": {seat: {point_name: float}},
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

        for entry in (board_nodes_input or []):
             if entry is None:
                  continue
             n = entry.get("node")
             c = entry.get("card")
             if n is None or n >= node_count:
                  continue
             board_nodes[n] = _card_id_to_none_or_int(
                  _card_srt_to_id(c)
             )     

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
            players_c.append({
                "seat": seat,
                "known_cards": known_ids,
                "total_hole_cards": game_def.hole_cards,
            })             
             
        # ── Build points list ─────────────────────────────────────
        # Use the default showdown_type from rules; honour per-point overrides.
        default_showdown = rules.showdown_type
        points_c: list[dict] = []
        for pt in rules.points:
             showdown = (
                  pt.showdown_type_override
                  if pt.showdown_type_override is not None
                  else default_showdown
             )       
             points_c.append({
                  "name": pt.name,
                  "score_type": pt.score_type,
                  "showdown_type": showdown,
                  "node_sets": [list(ns) for ns in pt.node_sets],
             })

        # ── Wrap evaluator to re-cast int args back to enums ─────
        # cap_equity passes score_type / showdown_type as plain ints;
        # poker_eval.evaluate_hands requires the actual enum types.
        def evaluator_wrapper(hands, board_mask, score_type, showdown_type):
            return poker_eval.evaluate_hands(
                hands,
                board_mask,
                poker_eval.ScoreType(score_type),
                poker_eval.ShowdownType(showdown_type),
            )

        # ── Call C++ engine ───────────────────────────────────────
        result = cap_equity.calculate_equity(
            variant_name=variant_name,
            total_hole_cards=game_def.hole_cards,
            players=players_c,
            board_nodes=board_nodes,
            points=points_c,
            evaluator=evaluator_wrapper,
            exact_threshold=exact_threshold,
            mc_iterations=mc_iterations
        )

        return result


# Singleton - the service is stateles so sharing is fine
equity_service = EquityService()