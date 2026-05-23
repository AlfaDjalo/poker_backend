"""
replay_api.py  —  FastAPI router for the Hand Replayer
Mount this alongside game_api.py:
    app.include_router(replay_api.router)
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Optional, List
from pydantic import BaseModel

from app.api.deps import get_db

# DB models
from app.db.models.hands import Hand
from app.db.models.actions import Action
from app.db.models.hole_cards import HoleCard
from app.db.models.board_cards import BoardCard
from app.db.models.hand_points import HandPoint
from app.db.models.point_results import PointResult
from app.db.models.point_cards import PointCard
from app.db.models.payouts import Payout
from app.db.models.players import Player
from app.db.models.table_seating import TableSeat
from app.db.models.poker_sessions import PokerSession
from app.db.models.annotations import Annotation

from cards.card import Card

router = APIRouter(prefix="/replay")

# ─────────────────────────────────────────────
# Auth stub — replace with real auth later
# ─────────────────────────────────────────────
 
# TODO: Replace with real authentication (JWT / session)
STUB_USER_ID = 1
 
def get_current_user_id() -> int:
    """Stub: always returns user 1. Wire up real auth here."""
    return STUB_USER_ID
 
 
# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def card_str(card_id: int) -> str:
    return str(Card(card_id))


# ─────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────

class HandSummaryDTO(BaseModel):
    hand_id: int
    variant_name: str
    layout_name: str
    split_pot: bool
    pot: int
    dealer_seat: int
    started_at: Optional[str]
    player_names: List[str]


class ActionDTO(BaseModel):
    action_id: int
    street: int
    action_index: int
    player_seat: int
    player_name: str
    action_type: str
    amount: Optional[int]


class HoleCardSetDTO(BaseModel):
    player_seat: int
    player_name: str
    cards: List[str]


class BoardCardDTO(BaseModel):
    street: int
    node: int
    card: str


class PointResultDTO(BaseModel):
    point_name: str
    score_type: str
    player_seat: int
    player_name: str
    hand_category: str
    hand_value: int
    point_share: float
    hole_cards_used: List[str]
    board_cards_used: List[str]


class PayoutDTO(BaseModel):
    player_seat: int
    player_name: str
    amount: int


class HandReplayDTO(BaseModel):
    hand_id: int
    variant_name: str
    layout_name: str
    split_pot: bool
    pot: int
    dealer_seat: int
    started_at: Optional[str]
    seats: dict
    initial_stacks: dict
    actions: List[ActionDTO]
    hole_cards: List[HoleCardSetDTO]
    board_cards: List[BoardCardDTO]
    point_results: List[PointResultDTO]
    payouts: List[PayoutDTO]


class AnnotationDTO(BaseModel):
    annotation_id: int
    hand_id: int
    action_id: Optional[int]
    user_id: Optional[int]
    comment: str
    selected_cards: Optional[List[str]]
    created_at: Optional[str]


class CreateAnnotationRequest(BaseModel):
    action_id: Optional[int] = None
    comment: str
    selected_cards: Optional[List[str]] = None


class UpdateAnnotationRequest(BaseModel):
    comment: Optional[str] = None
    selected_cards: Optional[List[str]] = None


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@router.get("/hands", response_model=List[HandSummaryDTO])
def list_hands(
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    variant: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Return a paginated list of hands for the browser panel."""
    q = db.query(Hand).order_by(desc(Hand.started_at))
    if variant:
        q = q.filter(Hand.variant_name == variant)

    hands = q.offset(offset).limit(limit).all()

    result = []
    for hand in hands:
        seats = (
            db.query(TableSeat, Player)
            .join(Player, TableSeat.player_id == Player.player_id)
            .filter(TableSeat.session_id == hand.session_id)
            .order_by(TableSeat.seat_number)
            .all()
        )
        player_names = [p.username for _, p in seats]
        
        result.append(HandSummaryDTO(
            hand_id=hand.hand_id,
            variant_name=hand.variant_name,
            layout_name=hand.layout_name,
            split_pot=hand.split_pot,
            pot=hand.pot or 0,
            dealer_seat=hand.dealer_seat or 1,
            started_at=hand.started_at.isoformat() if hand.started_at else None,
            player_names=player_names,
        ))

    return result
    

@router.get("/variants", response_model=List[str])
def list_variants(db: Session = Depends(get_db)):
    """Return distinct variant names for the filter dropdown."""
    rows = db.query(Hand.variant_name).distinct().all()
    return [r[0] for r in rows]
        

@router.get("/hands/{hand_id}", response_model=HandReplayDTO)
def get_hand(hand_id: int, db: Session = Depends(get_db)):
    """Return full replay data for a single hand."""
    hand = db.query(Hand).filter(Hand.hand_id == hand_id).first()
    if not hand:
        raise HTTPException(status_code=404, detail="Hand not found")
    
    # ── Seat map ──────────────────────────────
    seat_rows = (
        db.query(TableSeat, Player)
        .join(Player, TableSeat.player_id == Player.player_id)
        .filter(TableSeat.session_id == hand.session_id)
        .order_by(TableSeat.seat_number)
        .all()
    )
    seat_map = {ts.seat_number: p.player_id for ts, p in seat_rows}
    seat_names = {ts.seat_number: p.username for ts, p in seat_rows}
    player_id_to_seat = {v: k for k, v in seat_map.items()}

    def seat_of(player_id):
        return player_id_to_seat.get(player_id, 0)
    
    def name_of(player_id):
        seat = seat_of(player_id)
        return seat_names.get(seat, f"Player {seat}")


    # ── Actions ───────────────────────────────
    actions_raw = (
        db.query(Action)
        .filter(Action.hand_id == hand_id)
        .order_by(Action.street, Action.action_index)
        .all()
    )

    initial_stacks = {}
    seen = set()
    for a in actions_raw:
        seat = seat_of(a.player_id)
        if seat not in seen and a.stack_before is not None:
            initial_stacks[str(seat)] = a.stack_before
            seen.add(seat)

    actions = [
        ActionDTO(
            action_id=a.action_id,
            street=a.street,
            action_index=a.action_index,
            player_seat=seat_of(a.player_id),
            player_name=name_of(a.player_id),
            action_type=a.action_type,
            amount=a.amount,
        )
        for a in actions_raw
    ]

    # ── Hole cards ────────────────────────────
    hc_raw = (
        db.query(HoleCard)
        .filter(HoleCard.hand_id == hand_id)
        .order_by(HoleCard.player_id)
        .all()
    )
    hc_by_player = {}
    for hc in hc_raw:
        hc_by_player.setdefault(hc.player_id, []).append(card_str(hc.card))

    hole_cards = [
        HoleCardSetDTO(
            player_seat=seat_of(pid),
            player_name=name_of(pid),
            cards=cards,
        )
        for pid, cards in hc_by_player.items()
    ]

    # ── Board cards ───────────────────────────
    bc_raw = (
        db.query(BoardCard)
        .filter(BoardCard.hand_id == hand_id)
        .order_by(BoardCard.street, BoardCard.node)
        .all()
    )
    board_cards = [
        BoardCardDTO(street=bc.street, node=bc.node, card=card_str(bc.card))
        for bc in bc_raw
    ]

    # ── Point results ─────────────────────────
    points_raw = db.query(HandPoint).filter(HandPoint.hand_id == hand_id).all()
    point_results = []
    for hp in points_raw:
        prs = db.query(PointResult).filter(PointResult.point_id == hp.point_id).all()
        for pr in prs:
            hole_used = [
                card_str(pc.card)
                for pc in db.query(PointCard).filter(
                    PointCard.point_result_id == pr.point_result_id,
                    PointCard.source == "hole"
                ).all()
            ]
            board_used = [
                card_str(pc.card)
                for pc in db.query(PointCard).filter(
                    PointCard.point_result_id == pr.point_result_id,
                    PointCard.source == "board"
                ).all()
            ]
            point_results.append(PointResultDTO(
                point_name=hp.name,
                score_type=hp.score_type,
                player_seat=seat_of(pr.player_id),
                player_name=name_of(pr.player_id),
                hand_category=pr.hand_category or "",
                hand_value=pr.hand_value or 0,
                point_share=pr.point_share or 0.0,
                hole_cards_used=hole_used,
                board_cards_used=board_used,
            ))

    # ── Payouts ───────────────────────────────
    payouts_raw = db.query(Payout).filter(Payout.hand_id == hand_id).all()
    payouts = [
        PayoutDTO(
            player_seat=seat_of(p.player_id),
            player_name=name_of(p.player_id),
            amount=p.amount or 0,
        )
        for p in payouts_raw
    ]

    return HandReplayDTO(
        hand_id=hand.hand_id,
        variant_name=hand.variant_name,
        layout_name=hand.layout_name,
        split_pot=hand.split_pot,
        pot=hand.pot or 0,
        dealer_seat=hand.dealer_seat or 1,
        started_at=hand.started_at.isoformat() if hand.started_at else None,
        seats={str(k): v for k, v in seat_names.items()},
        initial_stacks=initial_stacks,
        actions=actions,
        hole_cards=hole_cards,
        board_cards=board_cards,
        point_results=point_results,
        payouts=payouts,
    )
 
 
# ─────────────────────────────────────────────
# Annotation endpoints
# ─────────────────────────────────────────────

def _annotation_to_dto(ann: Annotation) -> AnnotationDTO:
    return AnnotationDTO(
        annotation_id=ann.annotation_id,
        hand_id=ann.hand_id,
        action_id=ann.action_id,
        user_id=ann.user_id,
        comment=ann.comment,
        selected_cards=ann.selected_cards or [],
        created_at=ann.created_at.isoformat() if ann.created_at else None,
    )


@router.get("/hands/{hand_id}/annotations", response_model=List[AnnotationDTO])
def get_annotations(
    hand_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Return all annotations for a hand belonging to the current user."""
    anns = (
        db.query(Annotation)
        .filter(Annotation.hand_id == hand_id, Annotation.user_id == user_id)
        .order_by(Annotation.action_id, Annotation.annotation_id)
        .all()
    )
    return [_annotation_to_dto(a) for a in anns]


@router.post("/hands/{hand_id}/annotations", response_model=AnnotationDTO)
def create_annotation(
    hand_id: int,
    body: CreateAnnotationRequest,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Create a new annotation on a hand (optionally tied to a specific action)."""
    hand = db.query(Hand).filter(Hand.hand_id == hand_id).first()
    if not hand:
        raise HTTPException(status_code=404, detail="Hand not found")
    
    ann = Annotation(
        hand_id=hand_id,
        action_id=body.action_id,
        user_id=user_id,
        comment=body.comment,
        selected_cards=body.selected_cards or [],
    )
    db.add(ann)
    db.commit()
    db.refresh(ann)
    return _annotation_to_dto(ann)


@router.patch("/annotations/{annotation_id}", response_model=AnnotationDTO)
def update_annotation(
    annotation_id: int,
    body: UpdateAnnotationRequest,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Edit an existing annotation (owner only)."""
    ann = db.query(Annotation).filter(
        Annotation.annotation_id == annotation_id,
        Annotation.user_id == user_id,
    ).first()
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")

    if body.comment is not None:
        ann.comment = body.comment
    if body.selected_cards is not None:
        ann.selected_cards = body.selected_cards

    db.commit()
    db.refresh(ann)
    return _annotation_to_dto(ann)


@router.delete("/annotations/{annotation_id}", status_code=204)
def delete_annotation(
    annotation_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Delete an annotation (owner only)."""
    ann = db.query(Annotation).filter(
        Annotation.annotation_id == annotation_id,
        Annotation.user_id == user_id,
    ).first()
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")

    db.delete(ann)
    db.commit()
