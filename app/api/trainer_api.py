"""
app/api/trainer_api.py — REST endpoints for the frontend Trainer
component.

GET  /trainer/scenarios              — list available scenario types (dropdown)
POST /trainer/scenarios/{key}/new    — start a fresh scenario of that type
POST /trainer/action                 — hero submits a decision, graded + scored
GET  /trainer/state                  — current scenario state (reconnect/refresh)
POST /trainer/scoreboard/reset       — reset the running scoreboard
GET  /trainer/grid                   — 169-hand action-probability grid for the live scenario
GET  /trainer/history/{entry_id}     — read-only replay of a completed scenario
"""

from __future__ import annotations

import traceback

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.trainer_service import trainer_service

router = APIRouter(prefix="/trainer")


class TrainerActionRequest(BaseModel):
    action_type: str  # "fold" | "check" | "call" | "bet" | "all_in"


@router.get("/scenarios")
def list_scenarios():
    """Scenario types available in the Trainer dropdown, from training_config.yaml."""
    return trainer_service.list_scenarios()


@router.post("/scenarios/{scenario_key}/new")
def new_scenario(scenario_key: str):
    """Start a fresh, independent scenario of the given type (no persisted stacks)."""
    try:
        return trainer_service.new_scenario(scenario_key)
    except KeyError as e:
        print("Error: ", e)
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        print("Error: ", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Anything unexpected here (e.g. a dealer-position invariant
        # assertion) previously surfaced as an opaque network error the
        # frontend couldn't explain, and — because the failed request
        # never replaced `scenario` state — could leave a stale
        # "hand complete" screen stuck on screen with nothing the user
        # could do about it. Logging + a real message at least makes
        # the failure diagnosable and gives the frontend text to show.
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("/state")
def get_state():
    """Fetch the current trainer scenario state (e.g. on reconnect/refresh)."""
    state = trainer_service.get_state()
    if not state.get("active"):
        raise HTTPException(status_code=404, detail="No active trainer scenario.")
    return state


@router.post("/action")
def apply_action(req: TrainerActionRequest):
    """Submit the hero's decision; graded against the scenario's evaluator."""
    try:
        return trainer_service.apply_hero_action(req.action_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/scoreboard/reset")
def reset_scoreboard():
    """Reset correct/incorrect counts and history."""
    return trainer_service.reset_scoreboard()


@router.get("/grid")
def get_hand_grid():
    """
    Full hand action-probability grid for the CURRENT trainer scenario
    (same stack/position/villain state the live hand is using), with
    every hole-card combo swapped in one at a time. Used by the
    frontend's "Show Hand Grid" toggle — see TrainerHandGrid.jsx.
    """
    try:
        return trainer_service.get_hand_grid()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Previously any unexpected exception here (encoder/adapter
        # errors, a missing checkpoint edge case, etc.) surfaced to the
        # frontend only as "Failed to fetch trainer hand grid" with no
        # way to tell what actually broke. Log the real traceback
        # server-side and return its message so it's diagnosable.
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("/debug/live")
def get_live_debug_info():
    """
    Same as /trainer/debug/{scenario_key}, plus a real forward pass at
    the CURRENT scenario's actual decision point — reports the exact
    observation fields fed to the network (pot, bet_to_call, legal
    mask, hole/board cards) alongside the raw action-probability
    output. Use this when /trainer/debug/{scenario_key} says the
    checkpoint and hand encoder both loaded fine but the grid still
    looks wrong — that points at observation construction rather than
    the checkpoint itself.
    """
    try:
        return trainer_service.get_live_debug_info()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("/debug/{scenario_key}")
def get_debug_info(scenario_key: str):
    """
    Diagnostic snapshot of what's actually backing a scenario's agent
    (checkpoint path/existence, loaded architecture class, whether it's
    a real network or the uniform-random fallback, pretrained hand
    encoder status). Use this to check whether a hand grid that looks
    wrong is actually running a real trained checkpoint at all.
    """
    try:
        return trainer_service.get_debug_info(scenario_key)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("/history/{entry_id}")
def get_history_entry(entry_id: int):
    """
    Read-only replay of a completed scenario from the scoreboard
    history. Returns the same shape as /trainer/state, but with
    awaiting_hero forced False and is_history_replay=True so the
    frontend never re-offers a decision on an already-graded hand.
    """
    try:
        return trainer_service.get_history_snapshot(entry_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))