"""
tutorial_api.py — Hypothetical hand storage and retrieval.

POST /tutorial/hands           — save a hypothetical hand
GET  /tutorial/hands           — list hypothetical hands
GET  /tutorial/hands/{hand_id} — fetch one hypothetical hand for replay
"""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.api.deps import get_db
from app.db.models.hands import Hand
from app.db.models.actions import Action
from app.db.models.hole_cards import HoleCard
from app.db.models.board_cards import BoardCard
from app.db.models.players import Player
from app.db.models.hand_points import HandPoint
from app.db.models.point_results import PointResult
from app.db.models.point_cards import PointCard
from app.db.models.payouts import Payout
from app.services.game_service import game_service
from cards.card import Card

router = APIRouter(prefix="/tutorial")


# ── Helpers ───────────────────────────────────────────────────────

def card_str(card_id: int) -> str:
    return str(Card(card_id))


# ── Request / Response models ─────────────────────────────────────

class HypotheticalPlayerInput(BaseModel):
    seat: int
    name: str
    stack: int
    hole_cards: List[Optional[str]]


class HypotheticalActionInput(BaseModel):
    street: int
    action_index: int
    player_seat: int
    action_type: str
    amount: Optional[int] = None
    stack_before: Optional[int] = None
    pot_before: Optional[int] = None


class SaveHypotheticalHandRequest(BaseModel):
    game_name: str
    variant_name: str
    layout_name: str
    dealer_seat: int
    pot: int = 0
    players: List[HypotheticalPlayerInput]
    node_cards: List[Optional[str]]
    actions: List[HypotheticalActionInput] = []


class HypotheticalHandSummaryDTO(BaseModel):
    hand_id: int
    variant_name: str
    layout_name: str
    pot: int
    dealer_seat: int
    started_at: Optional[str]
    player_names: List[str]


# ── Endpoint

@router.post ("/hands", response_model=HypotheticalHandSummaryDTO)
def save_hypothetical_hand(
    req: SaveHypotheticalHandRequest,
    db: Session = Depends(get_db),
):
    """Save a hand created in the Hand Creation wizard as a hypothetical hand."""
    from cards.card import Card as CardObj

    def parse_card(s: Optional[str]) -> Optional[int]:
        if s is None:
            return None
        return CardObj.from_str(s).id
    
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

    # Seat → dummy player_id mapping (seats are 1-based)
    # Hypothetical hands use negative player ids (-seat) to avoid FK conflicts
    # when no real player rows exist.  Adjust if you FK-link to players.
    seat_to_pid = {p.seat: -p.seat for p in req.players}

    all_needed_ids = dict(seat_to_pid)
    for act in req.actions:
        if act.player_seat not in all_needed_ids:
            all_needed_ids[act.player_seat] = -act.player_seat

    existing_ids = {
        pid for (pid,) in db.query(Player.player_id)
            .filter(Player.player_id.in_(seat_to_pid.values()))
            .all()
    }

    name_by_seat = {p.seat: p.name for p in req.players}

    for seat, pid in all_needed_ids.items():
        if pid not in existing_ids:
            label = name_by_seat.get(seat, f"Seat {seat}")
            db.add(Player(
                player_id=pid,
                username=f"[Hypothetical] {label}",
                is_bot=True,
            ))
            existing_ids.add(pid)
    db.flush()

    # for p in req.players:
    #     pid = seat_to_pid[p.seat]
    #     if pid not in existing_ids:
    #         db.add(Player(
    #             player_id=pid,
    #             username=f"[Hypothetical] {p.name}",
    #             is_bot=True,
    #         ))
    #         existing_ids.add(pid)
    # db.flush()

    # Hole cards
    for p in req.players:
        pid = seat_to_pid[p.seat]
        for card_str_val in p.hole_cards:
            cid = parse_card(card_str_val)
            if cid is not None:
                db.add(HoleCard(
                    hand_id=hand.hand_id,
                    player_id=pid,
                    street=0,
                    card=cid,
                    visible=True,
                ))

    # Board cards
    for node_idx, card_str_val in enumerate(req.node_cards):
        cid = parse_card(card_str_val)
        if cid is not None:
            db.add(BoardCard(
                hand_id=hand.hand_id,
                street=1,
                node=node_idx,
                card=cid,
            ))

    # Actions
    for act in req.actions:
        pid = seat_to_pid.get(act.player_seat, -act.player_seat)
        db.add(Action(
            hand_id=hand.hand_id,
            street=act.street,
            action_index=act.action_index,
            player_id=pid,
            action_type=act.action_type,
            amount=act.amount,
            pot_before=act.pot_before,
            stack_before=act.stack_before,
        ))

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

@router.get("/hands", response_model=List[HypotheticalHandSummaryDTO])
def list_hypothetical_hands(
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    variant: Optional[str] = Query(None),
    db: Session = Depends(get_db)
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
        result.append(HypotheticalHandSummaryDTO(
            hand_id=hand.hand_id,
            variant_name=hand.variant_name,
            layout_name=hand.layout_name,
            pot=hand.pot or 0,
            dealer_seat=hand.dealer_seat or 1,
            started_at=hand.started_at.isoformat() if hand.started_at else None,
            player_names=[],
        ))

    return result


@router.get("/hands/{hand_id}")
def get_hypothetical_hand(hand_id: int, db: Session = Depends(get_db)):
    """
    Return full replay data for a hypothetical hand.
    Reuses the same HandReplayDTO shape as /replay/hands/{hand_id}
    so the frontend HandReplayer can consume it unchanged.
    """
    hand = db.query(Hand).filter(
        Hand.hand_id == hand_id,
        Hand.is_hypothetical == True,
    ).first()
    if not hand:
        raise HTTPException(status_code=404, detail="Hypothetical hand not found")
    
    # Delegate to replay_api helper - import inline to avoid circular deps
    from app.api.replay_api import get_hand as replay_get_hand
    return replay_get_hand(hand_id, db)


