"""
hands_api.py — Unified hand listing endpoint.

GET    /hands                      — combined real + hypothetical hands
DELETE /tutorial/hands/{hand_id}   — delete a hypothetical hand
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.db.models.hands import Hand
from app.db.models.players import Player
from app.db.models.table_seating import TableSeat

router = APIRouter()


class CombinedHandSummaryDTO(BaseModel):
    hand_id: int
    variant_name: str
    layout_name: str
    split_pot: bool
    pot: int
    dealer_seat: int
    started_at: str | None
    player_names: list[str]
    is_hypothetical: bool


def _player_names_for_hand(db: Session, hand: Hand) -> list[str]:
    """
    Real hands: look up seats via session_id.
    Hypothetical hands: session_id is None, so fall back to hole-card
    player rows (named "[Hypothetical] ..."), in player_id order.
    """
    if hand.session_id is not None:
        seats = (
            db.query(TableSeat, Player)
            .join(Player, TableSeat.player_id == Player.player_id)
            .filter(TableSeat.session_id == hand.session_id)
            .order_by(TableSeat.seat_number)
            .all()
        )
        return [p.username for _, p in seats]

    # Hypothetical: derive from negative player_ids used by hole cards
    from app.db.models.hole_cards import HoleCard

    pids = (
        db.query(HoleCard.player_id)
        .filter(HoleCard.hand_id == hand.hand_id)
        .distinct()
        .all()
    )
    pid_list = sorted({p[0] for p in pids}, reverse=True)  # -1, -2... -> seat order
    if not pid_list:
        return []
    players = db.query(Player).filter(Player.player_id.in_(pid_list)).all()
    by_id = {p.player_id: p.username for p in players}
    return [
        by_id[pid].replace("[Hypothetical] ", "") for pid in pid_list if pid in by_id
    ]


@router.get("/hands", response_model=list[CombinedHandSummaryDTO])
def list_all_hands(
    source: str = Query("all", pattern="^(all|real|hypothetical)$"),
    limit: int = Query(20, le=200),
    offset: int = Query(0, ge=0),
    variant: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """Combined paginated list of real + hypothetical hands."""
    q = db.query(Hand)

    if source == "real":
        q = q.filter(Hand.is_hypothetical == False)
    elif source == "hypothetical":
        q = q.filter(Hand.is_hypothetical == True)
    # "all" -> no filter

    if variant:
        q = q.filter(Hand.variant_name == variant)

    q = q.order_by(desc(Hand.started_at))
    hands = q.offset(offset).limit(limit).all()

    result = []
    for hand in hands:
        result.append(
            CombinedHandSummaryDTO(
                hand_id=hand.hand_id,
                variant_name=hand.variant_name,
                layout_name=hand.layout_name,
                split_pot=hand.split_pot,
                pot=hand.pot or 0,
                dealer_seat=hand.dealer_seat or 1,
                started_at=hand.started_at.isoformat() if hand.started_at else None,
                player_names=_player_names_for_hand(db, hand),
                is_hypothetical=bool(hand.is_hypothetical),
            )
        )

    return result


@router.delete("/tutorial/hands/{hand_id}", status_code=204)
def delete_hypothetical_hand(hand_id: int, db: Session = Depends(get_db)):
    """Delete a hypothetical hand. Refuses to delete real hands."""
    from app.db.models.actions import Action
    from app.db.models.board_cards import BoardCard
    from app.db.models.hand_points import HandPoint
    from app.db.models.hole_cards import HoleCard
    from app.db.models.payouts import Payout
    from app.db.models.point_cards import PointCard
    from app.db.models.point_results import PointResult

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
    db.query(Hand).filter(Hand.hand_id == hand_id).delete(synchronize_session=False)
    db.commit()
