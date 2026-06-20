"""
equity_api.py
FastAPI router for the equity calculator.
 
Mount in main.py:
    from app.api.equity_api import router as equity_router
    app.include_router(equity_router)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator

from app.services.equity_service import equity_service

router = APIRouter(prefix="/equity")


# ─────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────

class PlayerEquityInput(BaseModel):
    seat: int
    # Each entry is a known card string ("Ah") or null for an unknown slot.
    # The list length may be less than game's hole_cards count — missing
    # entries are treated as unknown (drawn during runout).
    hole_cards: List[Optional[str]] = []


class BoardNodeInput(BaseModel):
    node: int
    card: Optional[str] = None  # null = not yet dealt


class EquityRequest(BaseModel):
    variant_name: str
    players: List[PlayerEquityInput]
    board_nodes: List[BoardNodeInput] = []
    street: int = 0
    # Optional overrides for tuning
    exact_threshold: int = 50000
    mc_iterations: int = 20000

    @model_validator(mode="after")
    def validate_input(self) -> "EquityRequest":
        # At least one player must have at least one known card
        any_known = any(
            any(c is not None for c in p.hole_cards)
            for p in self.players
        )
        if not any_known:
            raise ValueError(
                "At least one player must have at least one known hole card"
            )
    
        # No duplicate cards
        seen: set[str] = set()
        for p in self.players:
            for c in p.hole_cards:
                if c is None:
                    continue
                if c in seen:
                    raise ValueError(f"Duplicate card: {c!r}")
                seen.add(c)
        for bn in self.board_nodes:
            if bn.card is None:
                continue
            if bn.card in seen:
                raise ValueError(f"Duplicate card: {bn.card!r}")
            seen.add(bn.card)

        return self
    

class EquityResponse(BaseModel):
    # {seat_str: {point_name: equity_fraction}}
    equity: Dict[str, Dict[str, float]]
    method: str
    iterations: int
    elapsed_ms: float


# ─────────────────────────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────────────────────────

@router.post("/calculate", response_model=EquityResponse)
def calculate_equity(req: EquityRequest) -> Any:
    """
    Calculate per-player, per-point equity for a CAP game variant.
 
    - Exact enumeration when combinations ≤ exact_threshold (default 50 000)
    - Monte Carlo (default 20 000 iterations) otherwise
    - Target latency: < 2 s for typical mid-hand positions with 2–4 players
    """
    try:
        raw = equity_service.calculate(
            variant_name=req.variant_name,
            players_input=[p.model_dump() for p in req.players],
            board_nodes_input=[bn.model_dump() for bn in req.board_nodes],
            street=req.street,
            exact_threshold=req.exact_threshold,
            mc_iterations=req.mc_iterations,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown game variant: {req.variant_name!r}"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Equity calculation failed: {exc}"
        )
    
    # Convert integer seat keys to string for JSON serialisation
    equity_str_keys: Dict[str, Dict[str, float]] = {
        str(seat): points
        for seat, points in raw["equity"].items()
    }

    return EquityResponse(
        equity=equity_str_keys,
        method=raw["method"],
        iterations=raw["iterations"],
        elapsed_ms=raw["elapsed_ms"],
    )