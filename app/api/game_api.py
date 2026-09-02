from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
import traceback

from app.api.deps import get_db
from app.services.game_service import game_service

router = APIRouter(prefix="/game")


# --------------------------------------------------
# Request models
# --------------------------------------------------


class ActionRequest(BaseModel):
    """
    Generic decision-response body. Which fields are used depends on
    the DOMAIN of the currently pending decision (GameStateDTO.decision
    .domain) — the frontend should only ever populate the field(s)
    that domain's renderer collected input for:

      BETTING      -> type ("fold"/"check"/"call"/"bet"/"raise"/"all_in"), amount
      CARD_SELECT  -> selected_cards (card strings, e.g. ["Ah","2c"]) —
                       the cards the player is choosing to discard/select
                       from their own hand. Count must satisfy whatever
                       min_count/max_count the pending option declares —
                       these are TOP-LEVEL fields on decision.options[0]
                       (decision.options[0].min_count /.max_count), NOT
                       nested under .metadata, which is reserved for
                       genuinely free-form extras. The engine's own
                       CardSelectResolver re-validates this regardless,
                       so an out-of-range count is rejected with a real
                       error rather than silently clamped.
      CARD_PASS    -> selected_cards, same shape as CARD_SELECT — target
                       seat is resolved automatically by the engine
                       (config-fixed "left"/"right"), never chosen here.
      BOOLEAN      -> bool_value (true/false)
      CHOICE       -> choice (the action_name of the selected option)

    type/amount are left BETTING-specific (not renamed to something
    generic) since every existing BETTING caller already sends exactly
    this shape — renaming would be a breaking change for no benefit.
    """

    type: str | None = None
    amount: int | None = None
    selected_cards: list[str] | None = None
    bool_value: bool | None = None
    choice: str | None = None


class RestartRequest(BaseModel):
    game_name: str | None = None


class NewHandRequest(BaseModel):
    game_name: str | None = None


class SelectGameRequest(BaseModel):
    game_name: str


# --------------------------------------------------
# Routes
# --------------------------------------------------


@router.get("/variants")
def get_variants():
    """Return all available game variants and the currently active one."""
    return game_service.get_variants()


@router.get("/variants/{game_name}/config")
def get_variant_config(game_name: str):
    """
    Return the per-variant config block (board layout + creation phases)
    that the Hand Creation wizard needs to drive its phase machine.
    """
    config = game_service.get_variant_config(game_name)
    if config is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown game variant: {game_name!r}"
        )
    return config


@router.post("/select-game")
def select_game(req: SelectGameRequest):
    """
    Queue a game variant to be used on the next hand or restart.
    Safe to call between hands; rejected mid-hand by the frontend.
    """
    try:
        game_service.select_game(req.game_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "pending_game": req.game_name}


@router.post("/new-hand")
def new_hand(req: NewHandRequest = NewHandRequest()):
    """
    Previously had no exception handling at all — RuntimeError("No
    active game session. Call /game/restart first.") or any other
    failure inside GameService.new_hand() (e.g. GraphEngine rebuild
    issues on a variant switch) surfaced as an opaque unhandled 500
    with no detail, and left the frontend holding whatever state it
    had before the call with no clean signal to recover from. Mirrors
    the ValueError/RuntimeError -> 400 pattern already used by
    /select-game and /restart below.
    """
    print("New hand starting with game ", req.game_name)
    try:
        return game_service.new_hand(game_name=req.game_name)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/restart")
def restart(req: RestartRequest = RestartRequest(), db: Session = Depends(get_db)):
    print("game_name: ", req.game_name)
    try:
        dto_state = game_service.restart(db, game_name=req.game_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return dto_state


@router.get("/state")
def get_state():
    state = game_service.get_state()
    print("State: ", state) 
    return state #game_service.get_state()


@router.post("/action")
def apply_action(req: ActionRequest):
    """
    Previously had NO exception handling — every ValueError out of
    GameService.apply_action() (illegal action, or "No decision is
    currently pending" when the frontend submits against a hand that
    GraphEngine has already advanced past, e.g. a stale action-panel
    click after a prior request already closed the betting round)
    surfaced as an unhandled 500 with a full traceback instead of a
    clean 400 the frontend could actually react to (re-fetch
    /game/state and resync) — a likely contributor to the simulator
    appearing to get stuck / new hands failing to load afterward,
    since a crashed request leaves the client with no signal to
    recover state from.
    """
    try:
        action = game_service.apply_action(req)
        print("Action: ", action)
        return action #game_service.apply_action(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")