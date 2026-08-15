"""
equity_api.py
FastAPI router for the equity calculator.

Mount in main.py:
    from app.api.equity_api import router as equity_router
    app.include_router(equity_router)

── REDESIGN (rich equity reporting) ────────────────────────────────────
Response model replaced the old flat {seat: {point_name: fraction}}
shape with PlayerEquityDTO, matching the richer dict cap_equity now
returns (via equity_service.calculate): overall pot equity, scoop/split
probability, and a per-point breakdown (win/tie probability + currency
contribution). Added `pot_size` to the request so currency fields can
be populated — omit it (or pass 0) to get fraction/probability-only
results.
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator

from app.services.equity_service import equity_service

router = APIRouter(prefix="/equity")


# ─────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────


class PlayerEquityInput(BaseModel):
    seat: int
    # Each entry is a known card string ("Ah") or null for an unknown slot.
    # The list length may be less than game's hole_cards count — missing
    # entries are treated as unknown (drawn during runout).
    hole_cards: list[str | None] = []


class BoardNodeInput(BaseModel):
    node: int
    card: str | None = None  # null = not yet dealt


class EquityRequest(BaseModel):
    variant_name: str
    players: list[PlayerEquityInput]
    board_nodes: list[BoardNodeInput] = []
    street: int = 0
    # Current pot in chips. 0 (default) disables currency fields in the
    # response — overall_equity_currency / equity_currency stay 0.0 and
    # only fraction/probability fields are populated.
    pot_size: float = 0.0
    # Optional overrides for tuning
    exact_threshold: int = 200000  # 50000
    mc_iterations: int = 50000  # 20000

    @model_validator(mode="after")
    def validate_input(self) -> EquityRequest:
        # At least one player must have at least one known card
        any_known = any(any(c is not None for c in p.hole_cards) for p in self.players)
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


# ─────────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────────


class PointEquityDTO(BaseModel):
    win_share: float
    win_probability: float = 0.0
    tie_probability: float = 0.0
    equity_currency: float = 0.0
    equity_percent: float = 0.0


class PlayerEquityDTO(BaseModel):
    overall_equity_fraction: float
    overall_equity_currency: float = 0.0
    scoop_probability: float = 0.0
    split_probability: float = 0.0
    points: dict[str, PointEquityDTO] = {}


class EquityResponse(BaseModel):
    # seat_str -> player equity breakdown
    players: dict[str, PlayerEquityDTO]
    method: str
    iterations: int
    elapsed_ms: float


# ─────────────────────────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────────────────────────


@router.post("/calculate", response_model=EquityResponse)
def calculate_equity(req: EquityRequest) -> Any:
    """
    Calculate rich per-player, per-point equity for a CAP game variant.

    - Exact enumeration when combinations ≤ exact_threshold (default 50 000)
    - Monte Carlo (default 20 000 iterations) otherwise
    - Target latency: < 2 s for typical mid-hand positions with 2–4 players
    - Pass `pot_size` to populate currency fields; omit for
      fraction/probability-only results.
    """
    # variant = req.get("variant_name")
    # if variant not in VALID_VARIANTS:
    #     raise HTTPException(status_code=404, detail="Unknown variant") # Or 422

    try:
        raw = equity_service.calculate(
            variant_name=req.variant_name,
            players_input=[p.model_dump() for p in req.players],
            board_nodes_input=[bn.model_dump() for bn in req.board_nodes],
            street=req.street,
            pot_size=req.pot_size,
            exact_threshold=req.exact_threshold,
            mc_iterations=req.mc_iterations,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Unknown game variant: {req.variant_name!r}"
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Equity calculation failed: {exc}")

    # Convert integer seat keys to string for JSON serialisation
    players_str_keys: dict[str, dict] = {
        str(seat): payload for seat, payload in raw["players"].items()
    }

    return EquityResponse(
        players=players_str_keys,
        method=raw["method"],
        iterations=raw["iterations"],
        elapsed_ms=raw["elapsed_ms"],
    )
