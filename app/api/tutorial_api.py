"""
tutorial_api.py — Hypothetical hand storage and retrieval.

POST   /tutorial/hands                       — save a hypothetical hand (always creates new)
PUT    /tutorial/hands/{hand_id}              — overwrite an existing hypothetical hand
GET    /tutorial/hands                       — list hypothetical hands
GET    /tutorial/hands/{hand_id}              — fetch one hypothetical hand for replay
GET    /tutorial/hands/{hand_id}/edit-state   — fetch hand decomposed into per-phase edit state
DELETE /tutorial/hands/{hand_id}              — delete a hypothetical hand
"""

from __future__ import annotations

# import sys
# from pathlib import Path
# project_root = Path(__file__).resolve().parents[3]
# engine_root = project_root / "poker_engine"
# if str(engine_root) not in sys.path:
#     sys.path.insert(0, str(engine_root))
from fastapi import APIRouter, Depends, HTTPException, Query
from poker_engine.cards.card import Card
from pydantic import BaseModel, ConfigDict
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.db.models.actions import Action
from app.db.models.board_cards import BoardCard
from app.db.models.hand_points import HandPoint
from app.db.models.hands import Hand
from app.db.models.hole_cards import HoleCard
from app.db.models.payouts import Payout
from app.db.models.players import Player
from app.db.models.point_cards import PointCard
from app.db.models.point_results import PointResult
from app.services.game_service import game_service

router = APIRouter(prefix="/tutorial")


# ── Helpers ───────────────────────────────────────────────────────


def card_str(card_id: int) -> str:
    return str(Card(card_id))


def _evaluate_and_persist_showdown(
    db: Session,
    hand: Hand,
    req: SaveHypotheticalHandRequest,
    seat_to_pid: dict[int, int],
) -> None:
    """
    Reconstruct a PokerState from `req`, run the showdown phase, then persist
    HandPoint / PointResult / PointCard / Payout rows for `hand`.

    seat_to_pid maps 1-based seat numbers to the (negative) player_id values
    already written to the DB by _write_hand_payload.

    Skips silently if the engine raises any exception — the hand is still saved,
    just without showdown data.
    """
    try:
        from poker_engine.cards.card import Card as CardObj
        from poker_engine.games.loader import load_game
        from poker_engine.scoring.scoring_engine import CppScoringEngine
        from poker_engine.state.player_state import PlayerState
        from poker_engine.state.poker_state import Phase, PokerState

        game_name = req.game_name or req.variant_name
        game_def, rules = load_game(game_name)

        # ── Build PlayerState list ────────────────────────────────
        # Players ordered by seat number (1-based → 0-based index)
        players_by_seat: dict[int, HypotheticalPlayerInput] = {
            p.seat: p for p in req.players
        }
        # Use hole_cards sets if provided, else fall back to players[].hole_cards
        hole_cards_by_seat: dict[int, list[str | None]] = {}
        if req.hole_cards:
            for hc in req.hole_cards:
                hole_cards_by_seat[hc.player_seat] = hc.cards
        else:
            for p in req.players:
                hole_cards_by_seat[p.seat] = p.hole_cards

        seats_sorted = sorted(players_by_seat.keys())
        player_states: list[PlayerState] = []
        seat_order: list[int] = []  # engine index → seat number

        for seat in seats_sorted:
            p_input = players_by_seat[seat]
            ps = PlayerState(stack=p_input.stack)
            # Build hand_mask from hole cards
            mask = 0
            for cs in hole_cards_by_seat.get(seat, []):
                if cs is not None:
                    mask |= 1 << CardObj.from_str(cs).id
            ps.hand_mask = mask
            player_states.append(ps)
            seat_order.append(seat)

        # ── Build PokerState and fast-forward to showdown ─────────
        scoring_engine = CppScoringEngine()
        state = PokerState(
            player_states,
            game_def,
            rules,
            scoring_engine,
            callbacks=None,
        )

        # Inject board node cards directly (skip normal dealing)
        node_count = game_def.node_count
        node_cards_ids: list[int | None] = [None] * node_count

        # Prefer explicit board_cards (have per-node info); fall back to node_cards
        if req.board_cards:
            for bc in req.board_cards:
                if bc.node < node_count and bc.card:
                    node_cards_ids[bc.node] = CardObj.from_str(bc.card).id
        else:
            for node_idx, cs in enumerate(req.node_cards or []):
                if node_idx < node_count and cs:
                    node_cards_ids[node_idx] = CardObj.from_str(cs).id

        state.game.node_cards = node_cards_ids

        # Mark all non-folded players as active; no bets outstanding
        for ps in state.game.players:
            ps.has_folded = False
            ps.current_bet = 0
        state.game.pot = req.pot
        state.game.street_index = len(game_def.street_nodes)  # past last street

        # Force the phase to SHOWDOWN so state.step(None) triggers evaluation
        state._phase = Phase.SHOWDOWN
        state.step(None)  # runs scoring engine → populates state.last_showdown

        result = getattr(state, "last_showdown", None)
        if result is None:
            return

        # ── Persist HandPoint / PointResult / PointCard rows ──────
        engine_idx_to_pid: dict[int, int] = {
            i: seat_to_pid.get(seat, -seat) for i, seat in enumerate(seat_order)
        }

        for point in result.points:
            hp = HandPoint(
                hand_id=hand.hand_id,
                name=point.name,
                showdown_type=point.showdown_type,
                score_type=point.score_type,
                node_set=point.node_mask,
            )
            db.add(hp)
            db.flush()

            for pr in point.results:
                pid = engine_idx_to_pid.get(pr.player_index, -1)
                pr_row = PointResult(
                    point_id=hp.point_id,
                    player_id=pid,
                    best_hand_mask=pr.best_hand_mask,
                    rank=pr.rank,
                    hand_value=pr.value,
                    hand_category=pr.category,
                    point_share=pr.share,
                )
                db.add(pr_row)
                db.flush()

                for c in pr.hole_cards_used:
                    db.add(
                        PointCard(
                            point_result_id=pr_row.point_result_id,
                            card=c,
                            source="hole",
                        )
                    )
                for c in pr.board_cards_used:
                    db.add(
                        PointCard(
                            point_result_id=pr_row.point_result_id,
                            card=c,
                            source="board",
                        )
                    )

        # ── Persist Payout rows ───────────────────────────────────
        for engine_idx, amt in result.payouts.items():
            pid = engine_idx_to_pid.get(engine_idx, -1)
            db.add(
                Payout(
                    hand_id=hand.hand_id,
                    player_id=pid,
                    amount=amt,
                    point_id=None,
                )
            )

        # Update pot on Hand row with total paid out
        hand.pot = sum(result.payouts.values()) or req.pot

    except Exception as exc:  # noqa: BLE001
        # Never block a save because showdown evaluation fails
        print(f"[tutorial] showdown evaluation skipped: {exc}")


# ── Request / Response models ─────────────────────────────────────


class HypotheticalPlayerInput(BaseModel):
    seat: int
    name: str
    stack: int
    hole_cards: list[str | None]


class HoleCardSetInput(BaseModel):
    player_seat: int
    cards: list[str | None]


class BoardCardInput(BaseModel):
    node: int
    card: str
    street: int  # 1-based: 1=flop, 2=turn, 3=river


class HypotheticalActionInput(BaseModel):
    street: int
    action_index: int = 0  # was missing — now explicit
    player_seat: int
    player_name: str | None = None
    action_type: str
    amount: int | None = None
    stack_before: int | None = None
    pot_before: int | None = None


class PhaseSnapshotDTO(BaseModel):
    phase_index: int
    phase_id: str
    deals_hole: bool
    deals_board_street: int | None = None
    allows_betting: bool
    pot_at_start: int = 0
    hole_cards_dealt: dict[str, list[str | None]] = {}
    board_cards_dealt: dict[str, str | None] = {}
    actions: list[HypotheticalActionInput] = []


class TutorialEditStateDTO(BaseModel):
    editing_hand_id: int
    game_name: str
    variant_name: str
    layout_name: str
    dealer_seat: int
    pot: int
    players: list[HypotheticalPlayerInput]
    node_cards: list[str | None]
    discard_pile: list[str] = []
    actions: list[HypotheticalActionInput]
    phase_snapshots: list[PhaseSnapshotDTO]
    # New: fields useHandEditor.loadExistingHand expects
    initial_stacks: dict[str, int] = {}
    furthest_phase_idx: int = 0


class SaveHypotheticalHandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_name: str | None = None
    variant_name: str
    layout_name: str
    dealer_seat: int
    pot: int = 0

    # NEW: seat label map  { seat_int: name }
    seats: dict[int, str] = {}
    # NEW: pre-blind starting stacks  { seat_int: stack }
    initial_stacks: dict[int, int] = {}

    players: list[HypotheticalPlayerInput]
    # NEW: explicit hole-card sets (real-hand shape); overrides players[].hole_cards
    hole_cards: list[HoleCardSetInput] = []
    node_cards: list[str | None]
    # NEW: board cards with per-card street stamps
    board_cards: list[BoardCardInput] = []
    # NEW
    discard_pile: list[str] = []
    actions: list[HypotheticalActionInput] = []
    # NEW
    street_names: dict[int, str] | None = None


class HypotheticalHandSummaryDTO(BaseModel):
    hand_id: int
    variant_name: str
    layout_name: str
    pot: int
    dealer_seat: int
    started_at: str | None
    player_names: list[str]


# ── Persistence helpers ───────────────────────────────────────────


def _write_hand_payload(db: Session, hand: Hand, req: SaveHypotheticalHandRequest):
    """
    Write hole cards / board cards / actions for `req` onto an already-
    created-and-flushed `hand` row.
    Shared by POST (new) and PUT (overwrite). Does not commit — caller commits.
    """
    from poker_engine.cards.card import Card as CardObj

    def parse_card(s: str | None) -> int | None:
        if s is None:
            return None
        return CardObj.from_str(s).id

    # Seat → dummy player_id (negative to avoid FK conflicts with real players)
    seat_to_pid: dict[int, int] = {p.seat: -p.seat for p in req.players}

    all_needed_ids: dict[int, int] = dict(seat_to_pid)
    for act in req.actions:
        if act.player_seat not in all_needed_ids:
            all_needed_ids[act.player_seat] = -act.player_seat
    for hc in req.hole_cards:
        if hc.player_seat not in all_needed_ids:
            all_needed_ids[hc.player_seat] = -hc.player_seat

    existing_ids = {
        pid
        for (pid,) in db.query(Player.player_id)
        .filter(Player.player_id.in_(all_needed_ids.values()))
        .all()
    }

    # Name resolution priority: req.seats > players[].name > "Seat N"
    name_by_seat: dict[int, str] = {}
    for seat_key, name in req.seats.items():
        name_by_seat[int(seat_key)] = name
    for p in req.players:
        name_by_seat.setdefault(p.seat, p.name)

    for seat, pid in all_needed_ids.items():
        if pid not in existing_ids:
            label = name_by_seat.get(seat, f"Seat {seat}")
            db.add(
                Player(
                    player_id=pid,
                    username=f"[Hypothetical] {label}",
                    is_bot=True,
                )
            )
            existing_ids.add(pid)
    db.flush()

    # ── Hole cards ───────────────────────────────────────────────
    if req.hole_cards:
        hole_card_sets = [(hc.player_seat, hc.cards) for hc in req.hole_cards]
    else:
        hole_card_sets = [(p.seat, p.hole_cards) for p in req.players]

    for seat, cards in hole_card_sets:
        pid = seat_to_pid.get(seat, -seat)
        for card_str_val in cards:
            cid = parse_card(card_str_val)
            if cid is not None:
                db.add(
                    HoleCard(
                        hand_id=hand.hand_id,
                        player_id=pid,
                        street=0,
                        card=cid,
                        visible=True,
                    )
                )

    # ── Board cards ──────────────────────────────────────────────
    if req.board_cards:
        for bc in req.board_cards:
            cid = parse_card(bc.card)
            if cid is not None:
                db.add(
                    BoardCard(
                        hand_id=hand.hand_id,
                        street=bc.street,  # real 1-based street, not always 1
                        node=bc.node,
                        card=cid,
                    )
                )
    else:
        for node_idx, card_str_val in enumerate(req.node_cards):
            cid = parse_card(card_str_val)
            if cid is not None:
                db.add(
                    BoardCard(
                        hand_id=hand.hand_id,
                        street=1,
                        node=node_idx,
                        card=cid,
                    )
                )

    # ── Actions ──────────────────────────────────────────────────
    seats_with_actions = {act.player_seat for act in req.actions}

    for act in req.actions:
        pid = seat_to_pid.get(act.player_seat, -act.player_seat)
        db.add(
            Action(
                hand_id=hand.hand_id,
                street=act.street,
                action_index=act.action_index,  # persisted correctly now
                player_id=pid,
                action_type=act.action_type,
                amount=act.amount,
                pot_before=act.pot_before,
                stack_before=act.stack_before,
            )
        )

    # Backfill synthetic SEAT actions for players with no logged actions
    # so initial_stacks survive the replay round-trip via stack_before.
    for seat_key, stack in req.initial_stacks.items():
        seat_int = int(seat_key)
        if seat_int in seats_with_actions:
            continue
        pid = seat_to_pid.get(seat_int, -seat_int)
        db.add(
            Action(
                hand_id=hand.hand_id,
                street=0,
                action_index=-1,  # sorts before real actions; filtered on read
                player_id=pid,
                action_type="SEAT",
                amount=None,
                pot_before=0,
                stack_before=stack,
            )
        )

    return seat_to_pid


def _delete_hand_children(db: Session, hand_id: int):
    """Delete all child rows for a hand (FK-safe order). Does not delete Hand row."""
    point_ids = [
        r[0]
        for r in db.query(HandPoint.point_id).filter(HandPoint.hand_id == hand_id).all()
    ]
    if point_ids:
        pr_ids = [
            r[0]
            for r in db.query(PointResult.point_result_id)
            .filter(PointResult.point_id.in_(point_ids))
            .all()
        ]
        if pr_ids:
            db.query(PointCard).filter(PointCard.point_result_id.in_(pr_ids)).delete(
                synchronize_session=False
            )
        db.query(PointResult).filter(PointResult.point_id.in_(point_ids)).delete(
            synchronize_session=False
        )

    db.query(Payout).filter(Payout.hand_id == hand_id).delete(synchronize_session=False)
    db.query(HandPoint).filter(HandPoint.hand_id == hand_id).delete(
        synchronize_session=False
    )
    db.query(Action).filter(Action.hand_id == hand_id).delete(synchronize_session=False)
    db.query(HoleCard).filter(HoleCard.hand_id == hand_id).delete(
        synchronize_session=False
    )
    db.query(BoardCard).filter(BoardCard.hand_id == hand_id).delete(
        synchronize_session=False
    )


# ── Endpoints ─────────────────────────────────────────────────────


@router.post("/hands", response_model=HypotheticalHandSummaryDTO)
def save_hypothetical_hand(
    req: SaveHypotheticalHandRequest,
    db: Session = Depends(get_db),
):
    """Save a hand created in the Hand Creation wizard as a new hypothetical hand."""
    hand = Hand(
        session_id=None,
        variant_name=req.variant_name,
        layout_name=req.layout_name,
        split_pot=False,
        dealer_seat=req.dealer_seat,
        pot=req.pot,
        is_hypothetical=True,
    )
    db.add(hand)
    db.flush()

    seat_to_pid = _write_hand_payload(db, hand, req)
    db.flush()

    _evaluate_and_persist_showdown(db, hand, req, seat_to_pid)

    db.commit()
    db.refresh(hand)

    return HypotheticalHandSummaryDTO(
        hand_id=hand.hand_id,
        variant_name=hand.variant_name,
        layout_name=hand.layout_name,
        pot=hand.pot or 0,
        dealer_seat=hand.dealer_seat or 1,
        started_at=hand.started_at.isoformat() if hand.started_at else None,
        player_names=[p.name for p in req.players],
    )


@router.put("/hands/{hand_id}", response_model=HypotheticalHandSummaryDTO)
def update_hypothetical_hand(
    hand_id: int,
    req: SaveHypotheticalHandRequest,
    db: Session = Depends(get_db),
):
    """Overwrite an existing hypothetical hand. Refuses to overwrite real hands."""
    hand = (
        db.query(Hand)
        .filter(
            Hand.hand_id == hand_id,
            Hand.is_hypothetical == True,
        )
        .first()
    )
    if not hand:
        raise HTTPException(status_code=404, detail="Hypothetical hand not found")

    _delete_hand_children(db, hand_id)

    hand.variant_name = req.variant_name
    hand.layout_name = req.layout_name
    hand.dealer_seat = req.dealer_seat
    hand.pot = req.pot
    db.flush()

    seat_to_pid = _write_hand_payload(db, hand, req)
    db.flush()

    _evaluate_and_persist_showdown(db, hand, req, seat_to_pid)

    db.commit()
    db.refresh(hand)

    return HypotheticalHandSummaryDTO(
        hand_id=hand.hand_id,
        variant_name=hand.variant_name,
        layout_name=hand.layout_name,
        pot=hand.pot or 0,
        dealer_seat=hand.dealer_seat or 1,
        started_at=hand.started_at.isoformat() if hand.started_at else None,
        player_names=[p.name for p in req.players],
    )


@router.get("/hands", response_model=list[HypotheticalHandSummaryDTO])
def list_hypothetical_hands(
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    variant: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """List saved hypothetical hands for the Tutorial browser."""
    q = (
        db.query(Hand)
        .filter(Hand.is_hypothetical == True)
        .order_by(desc(Hand.started_at))
    )
    if variant:
        q = q.filter(Hand.variant_name == variant)

    hands = q.offset(offset).limit(limit).all()

    result = []
    for hand in hands:
        result.append(
            HypotheticalHandSummaryDTO(
                hand_id=hand.hand_id,
                variant_name=hand.variant_name,
                layout_name=hand.layout_name,
                pot=hand.pot or 0,
                dealer_seat=hand.dealer_seat or 1,
                started_at=hand.started_at.isoformat() if hand.started_at else None,
                player_names=[],
            )
        )

    return result


@router.get("/hands/{hand_id}")
def get_hypothetical_hand(hand_id: int, db: Session = Depends(get_db)):
    """
    Return full replay data for a hypothetical hand, in the same shape as
    GET /replay/hands/{hand_id}, with additional `street_names` injected.

    Filters out synthetic SEAT actions (action_index == -1) that are used
    only to persist initial_stacks; those values are surfaced via
    `initial_stacks` instead.
    """
    hand = (
        db.query(Hand)
        .filter(
            Hand.hand_id == hand_id,
            Hand.is_hypothetical == True,
        )
        .first()
    )
    if not hand:
        raise HTTPException(status_code=404, detail="Hypothetical hand not found")

    from app.api.replay_api import get_hand as replay_get_hand

    replay_data = replay_get_hand(hand_id, db)

    # Convert Pydantic model → dict so we can inject / patch extra fields
    if hasattr(replay_data, "model_dump"):
        data = replay_data.model_dump()
    else:
        data = dict(replay_data)

    # ── Filter out synthetic SEAT rows from actions list ──────────
    data["actions"] = [
        a for a in data.get("actions", []) if a.get("action_type") != "SEAT"
    ]

    # ── Inject street_names from variant YAML ─────────────────────
    config = game_service.get_variant_config(hand.variant_name)
    if config:
        raw_sn = (config.get("board_layout") or {}).get("street_names") or {}
        data["street_names"] = {str(k): v for k, v in raw_sn.items()}
    else:
        data["street_names"] = {}

    # ── Ensure seats dict is string-keyed ─────────────────────────
    # replay_get_hand already returns seats as {"1": "Alice", ...}
    # but guard in case it comes back int-keyed from model_dump
    if data.get("seats"):
        data["seats"] = {str(k): v for k, v in data["seats"].items()}

    return data


@router.get("/hands/{hand_id}/edit-state", response_model=TutorialEditStateDTO)
def get_hypothetical_hand_edit_state(hand_id: int, db: Session = Depends(get_db)):
    """
    Fetch a hypothetical hand decomposed into per-phase edit state, for
    re-opening in the Creator.

    Returns phase_snapshots, initial_stacks, and furthest_phase_idx
    in addition to the base fields.
    """
    hand = (
        db.query(Hand)
        .filter(
            Hand.hand_id == hand_id,
            Hand.is_hypothetical == True,
        )
        .first()
    )
    if not hand:
        raise HTTPException(status_code=404, detail="Hypothetical hand not found")

    config = game_service.get_variant_config(hand.variant_name)
    if config is None:
        raise HTTPException(
            status_code=500,
            detail=f"Variant config not found for {hand.variant_name!r}.",
        )
    creation_phases = config.get("creation_phases") or []
    streets_map = {
        int(k): v
        for k, v in ((config.get("board_layout") or {}).get("streets") or {}).items()
    }

    # ── Hole cards ─────────────────────────────────────────────────
    hc_raw = (
        db.query(HoleCard)
        .filter(HoleCard.hand_id == hand_id)
        .order_by(HoleCard.player_id, HoleCard.card)
        .all()
    )
    cards_by_seat: dict[int, list[int]] = {}
    for hc in hc_raw:
        seat = -hc.player_id if hc.player_id < 0 else hc.player_id
        cards_by_seat.setdefault(seat, []).append(hc.card)

    seats = sorted(cards_by_seat.keys())

    names_by_seat: dict[int, str] = {}
    pids = [-s for s in seats]
    if pids:
        players_db = db.query(Player).filter(Player.player_id.in_(pids)).all()
        for p in players_db:
            seat = -p.player_id
            names_by_seat[seat] = (p.username or "").replace("[Hypothetical] ", "")

    # ── Actions ────────────────────────────────────────────────────
    actions_raw = (
        db.query(Action)
        .filter(Action.hand_id == hand_id)
        .order_by(Action.street, Action.action_index)
        .all()
    )
    starting_stack_by_seat: dict[int, int] = {}
    for a in actions_raw:
        seat = -a.player_id if a.player_id < 0 else a.player_id
        if seat not in starting_stack_by_seat and a.stack_before is not None:
            starting_stack_by_seat[seat] = a.stack_before

    players_dto = [
        HypotheticalPlayerInput(
            seat=seat,
            name=names_by_seat.get(seat, f"Seat {seat}"),
            stack=starting_stack_by_seat.get(seat, 0),
            hole_cards=[card_str(c) for c in cards_by_seat.get(seat, [])],
        )
        for seat in seats
    ]
    hole_cards_by_seat_str: dict[str, list[str | None]] = {
        str(p.seat): p.hole_cards for p in players_dto
    }
    initial_stacks_str: dict[str, int] = {
        str(seat): stack for seat, stack in starting_stack_by_seat.items()
    }

    # ── Board cards ────────────────────────────────────────────────
    bc_raw = (
        db.query(BoardCard)
        .filter(BoardCard.hand_id == hand_id)
        .order_by(BoardCard.node)
        .all()
    )
    max_node = max([bc.node for bc in bc_raw], default=-1)
    node_cards: list[str | None] = [None] * (max_node + 1)
    card_str_by_node: dict[int, str] = {}
    for bc in bc_raw:
        s = card_str(bc.card)
        node_cards[bc.node] = s
        card_str_by_node[bc.node] = s

    # ── Actions DTO (filter synthetic SEAT rows) ───────────────────
    actions_dto = [
        HypotheticalActionInput(
            street=a.street,
            action_index=a.action_index,
            player_seat=(-a.player_id if a.player_id < 0 else a.player_id),
            action_type=a.action_type,
            amount=a.amount,
            stack_before=a.stack_before,
            pot_before=a.pot_before,
        )
        for a in actions_raw
        if a.action_index != -1
    ]
    actions_by_street: dict[int, list[HypotheticalActionInput]] = {}
    for a in actions_dto:
        actions_by_street.setdefault(a.street, []).append(a)

    # ── Walk creation_phases ───────────────────────────────────────
    phase_snapshots: list[PhaseSnapshotDTO] = []
    betting_phase_counter = 0
    hole_cards_emitted = False
    furthest_phase_idx = 0

    for idx, phase in enumerate(creation_phases):
        phase_id = phase.get("id", f"PHASE_{idx}")
        deals_hole = bool(phase.get("deals_hole", False))
        deals_board_street = phase.get("deals_board_street")
        allows_betting = bool(phase.get("allows_betting", False))

        snap = PhaseSnapshotDTO(
            phase_index=idx,
            phase_id=phase_id,
            deals_hole=deals_hole,
            deals_board_street=deals_board_street,
            allows_betting=allows_betting,
        )

        if deals_hole and not hole_cards_emitted:
            snap.hole_cards_dealt = hole_cards_by_seat_str
            hole_cards_emitted = True
            if any(cards for cards in hole_cards_by_seat_str.values()):
                furthest_phase_idx = max(furthest_phase_idx, idx)

        if deals_board_street is not None:
            nodes_for_street = streets_map.get(deals_board_street, [])
            snap.board_cards_dealt = {
                str(n): card_str_by_node.get(n) for n in nodes_for_street
            }
            if any(v is not None for v in snap.board_cards_dealt.values()):
                furthest_phase_idx = max(furthest_phase_idx, idx)

        if allows_betting:
            street_actions = actions_by_street.get(betting_phase_counter, [])
            snap.actions = street_actions
            snap.pot_at_start = (
                street_actions[0].pot_before
                if street_actions and street_actions[0].pot_before is not None
                else 0
            )
            if street_actions:
                furthest_phase_idx = max(furthest_phase_idx, idx)
            betting_phase_counter += 1

        phase_snapshots.append(snap)

    return TutorialEditStateDTO(
        editing_hand_id=hand.hand_id,
        game_name=config.get("game_name", hand.variant_name),
        variant_name=hand.variant_name,
        layout_name=hand.layout_name,
        dealer_seat=hand.dealer_seat or 1,
        pot=hand.pot or 0,
        players=players_dto,
        node_cards=node_cards,
        discard_pile=[],
        actions=actions_dto,
        phase_snapshots=phase_snapshots,
        initial_stacks=initial_stacks_str,
        furthest_phase_idx=furthest_phase_idx,
    )


@router.delete("/hands/{hand_id}", status_code=204)
def delete_hypothetical_hand(hand_id: int, db: Session = Depends(get_db)):
    """Delete a hypothetical hand. Refuses to delete real hands."""
    hand = (
        db.query(Hand)
        .filter(
            Hand.hand_id == hand_id,
            Hand.is_hypothetical == True,
        )
        .first()
    )
    if not hand:
        raise HTTPException(status_code=404, detail="Hypothetical hand not found")

    _delete_hand_children(db, hand_id)
    db.delete(hand)
    db.commit()
