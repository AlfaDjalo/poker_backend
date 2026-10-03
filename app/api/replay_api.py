"""
replay_api.py  —  FastAPI router for the Hand Replayer
Mount this alongside game_api.py:
    app.include_router(replay_api.router)
"""


from fastapi import APIRouter, Depends, HTTPException, Query
from poker_engine.cards.card import Card
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.db.models.actions import Action
from app.db.models.annotations import Annotation
from app.db.models.board_cards import BoardCard
from app.db.models.card_events import CardEvent
from app.db.models.hand_points import HandPoint

# DB models
from app.db.models.hands import Hand
from app.db.models.hole_cards import HoleCard
from app.db.models.payouts import Payout
from app.db.models.players import Player
from app.db.models.point_cards import PointCard
from app.db.models.point_results import PointResult
from app.db.models.table_seating import TableSeat

# Reused so the Hand Replayer's showdown payload is STRUCTURALLY
# IDENTICAL to what the live Game Simulator gets on GameStateDTO.showdown
# (see engine_adapter.build_showdown_dto / graph_engine_adapter.py) —
# aliased to avoid colliding with this file's own legacy flat
# PointResultDTO (kept below for backward compatibility).
from app.dto.state_dto import (
    PlayerBoardResultDTO,
    PointResultDTO as NestedPointResultDTO,
    ShowdownDTO as NestedShowdownDTO,
)

router = APIRouter(prefix="/replay")

# NOTE: hands_api.py's delete_hypothetical_hand() has its own copy of this
# same child-deletion logic, scoped to is_hypothetical == True hands only
# (the Tutorial browser only ever deletes hands it created). This route is
# the Hand Replayer's delete — it must work for REAL hands too, and it also
# cleans up Annotation rows, which hands_api.py's version does not (Hand
# Editor annotations are keyed on hand_id and were never deleted by
# anything before this endpoint existed — a real hand's annotations would
# silently orphan on delete). Not consolidated into one shared helper only
# because the two call sites have different real-vs-hypothetical filters on
# the Hand row itself; the child-row deletion statements are otherwise
# identical.

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


def _mask_to_card_strs(mask: int | None) -> list[str]:
    """
    LSB-iterate a 52-bit card bitmask into card strings — same idiom
    engine_adapter.py uses. PointResult.best_hand_mask is persisted
    verbatim from the live scoring engine's own best_hand_mask (see
    session_logger.finish_hand()), so this reconstructs
    PlayerBoardResultDTO.best_hand_cards for a replayed hand exactly
    the way the live path derives it.
    """
    cards = []
    m = mask or 0
    while m:
        lsb = m & -m
        cid = lsb.bit_length() - 1
        cards.append(card_str(cid))
        m ^= lsb
    return cards


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
    started_at: str | None
    player_names: list[str]


class ActionDTO(BaseModel):
    action_id: int
    street: int
    action_index: int
    player_seat: int
    player_name: str
    action_type: str
    amount: int | None
    stack_before: int | None = None
    pot_before: int | None = None


class HoleCardSetDTO(BaseModel):
    player_seat: int
    player_name: str
    cards: list[str]


class HoleCardEventDTO(BaseModel):
    """
    One row of the real CardEvent ledger (card_movement.py) — a deal,
    draw, discard, or pass, in TRUE chronological order (sequence) and
    stamped with the street it happened on. Unlike hole_cards below
    (a flat, timing-free snapshot of each player's FINAL cards), this
    is what the Replayer needs to correctly stage extra-card mechanics
    (ESG/Catchup ESG dealing a bonus card mid-street, a Drawmaha
    draw, a Pass-the-Trash transfer) on the actual street they
    occurred, rather than showing them as present from the very first
    (preflop) frame.
    """

    sequence: int
    street: int
    event_type: str  # "DEALT" | "DRAWN" | "DISCARDED" | "PASSED"
    card: str
    from_seat: int | None  # None for DEALT/DRAWN (card came from the deck)
    to_seat: int | None  # None for DISCARDED (card went to the muck)


class BoardCardDTO(BaseModel):
    street: int
    node: int
    card: str


class PointResultDTO(BaseModel):
    """
    LEGACY flat shape — one row per (point, board, player), no
    is_winner field. Kept for backward compatibility with any existing
    consumer of this exact shape. New frontend code should read
    `HandReplayDTO.showdown` instead (see that field's own docstring)
    — it's the shape that actually carries winner information and
    matches the live Game Simulator's GameStateDTO.showdown 1:1.
    """

    point_name: str
    score_type: str
    player_seat: int
    player_name: str
    hand_category: str
    hand_value: int
    point_share: float
    hole_cards_used: list[str]
    board_cards_used: list[str]


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
    started_at: str | None
    seats: dict
    initial_stacks: dict
    actions: list[ActionDTO]
    hole_cards: list[HoleCardSetDTO]
    hole_card_events: list[HoleCardEventDTO] = []
    board_cards: list[BoardCardDTO]
    point_results: list[PointResultDTO]
    payouts: list[PayoutDTO]

    # NEW — structurally identical to GameStateDTO.showdown (see
    # app/dto/state_dto.py's ShowdownDTO / PointResultDTO /
    # PlayerBoardResultDTO, and engine_adapter.build_showdown_dto,
    # which the live Game Simulator's response is built from). This is
    # the fix for the Hand Replayer's showdown panel not being able to
    # highlight the winning hand: the OLD flat `point_results` above
    # has no `is_winner` field at all, so a frontend component written
    # against the live shape (which DOES carry `is_winner` per player
    # per board) had nothing to read when rendering a replayed hand.
    # None only for a hand with no persisted showdown data at all
    # (e.g. every player but one folded before showdown).
    showdown: NestedShowdownDTO | None = None

    # NEW — mirrors GameStateDTO's own top-level `winners` field
    # (1-based seats that received a positive payout), computed the
    # same way engine_adapter.state_to_dto /
    # graph_engine_adapter.graph_state_to_dto already do it for a live
    # hand: `[p + 1 for p, amt in result.payouts.items() if amt > 0]`.
    winners: list[int] = []


class AnnotationDTO(BaseModel):
    annotation_id: int
    hand_id: int
    action_id: int | None
    user_id: int | None
    comment: str
    selected_cards: list[str] | None
    created_at: str | None


class CreateAnnotationRequest(BaseModel):
    action_id: int | None = None
    comment: str
    selected_cards: list[str] | None = None


class UpdateAnnotationRequest(BaseModel):
    comment: str | None = None
    selected_cards: list[str] | None = None


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────


@router.get("/hands", response_model=list[HandSummaryDTO])
def list_hands(
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    variant: str | None = Query(None),
    db: Session = Depends(get_db),
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

        result.append(
            HandSummaryDTO(
                hand_id=hand.hand_id,
                variant_name=hand.variant_name,
                layout_name=hand.layout_name,
                split_pot=hand.split_pot,
                pot=hand.pot or 0,
                dealer_seat=hand.dealer_seat or 1,
                started_at=hand.started_at.isoformat() if hand.started_at else None,
                player_names=player_names,
            )
        )

    return result


@router.get("/variants", response_model=list[str])
def list_variants(db: Session = Depends(get_db)):
    """Return distinct variant names for the filter dropdown."""
    rows = db.query(Hand.variant_name).distinct().all()
    return [r[0] for r in rows]


def _delete_replay_hand_children(db: Session, hand_id: int) -> None:
    """
    Delete every child row for `hand_id`, FK-safe order — used by the
    Hand Replayer's DELETE below. Covers everything
    tutorial_api._delete_hand_children does (PointCard -> PointResult ->
    HandPoint, Payout, Action, CardEvent, HoleCard, BoardCard), PLUS
    Annotation, which nothing deleted before this endpoint existed (see
    the module-level NOTE above this router's DELETE route).

    Works identically for a real hand or a hypothetical one — this route
    has no is_hypothetical filter of its own (see the route's own
    docstring for why deletion here isn't restricted the way
    hands_api.delete_hypothetical_hand() is).
    """
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
    db.query(Annotation).filter(Annotation.hand_id == hand_id).delete(
        synchronize_session=False
    )
    db.query(Action).filter(Action.hand_id == hand_id).delete(synchronize_session=False)
    # CardEvent is the ledger backing HoleCard (see card_movement.py /
    # hole_cards.py) — must go before/alongside HoleCard, same rationale
    # as tutorial_api._delete_hand_children's own note on this.
    db.query(CardEvent).filter(CardEvent.hand_id == hand_id).delete(
        synchronize_session=False
    )
    db.query(HoleCard).filter(HoleCard.hand_id == hand_id).delete(
        synchronize_session=False
    )
    db.query(BoardCard).filter(BoardCard.hand_id == hand_id).delete(
        synchronize_session=False
    )


@router.delete("/hands/{hand_id}", status_code=204)
def delete_hand(hand_id: int, db: Session = Depends(get_db)):
    """
    Delete a hand and all of its child rows from the Hand Replayer.

    Unlike hands_api.delete_hypothetical_hand() (Tutorial browser, which
    only ever deletes hands IT created and refuses real ones), this route
    has no is_hypothetical restriction — the Replayer lists and can delete
    BOTH real and hypothetical hands, so its delete action needs to work
    for both. If a caller specifically needs "only ever delete hypothetical
    hands" semantics, hands_api.py's existing endpoint is still there for
    that.

    Does NOT delete the parent PokerSession/TableSeat rows for a real
    hand's session — a session can span multiple hands, and deleting one
    hand from history shouldn't tear down the session it belonged to.
    """
    hand = db.query(Hand).filter(Hand.hand_id == hand_id).first()
    if not hand:
        raise HTTPException(status_code=404, detail="Hand not found")

    _delete_replay_hand_children(db, hand_id)
    db.delete(hand)
    db.commit()


@router.get("/hands/{hand_id}", response_model=HandReplayDTO)
def get_hand(hand_id: int, db: Session = Depends(get_db)):
    """Return full replay data for a single hand."""
    hand = db.query(Hand).filter(Hand.hand_id == hand_id).first()
    if not hand:
        raise HTTPException(status_code=404, detail="Hand not found")

    # ── Seat map ──────────────────────────────
    # Real hands: look up seats via session -> TableSeat -> Player.
    # Hypothetical hands: session_id is None; seats are encoded as negative
    # player_ids (-seat_number) by tutorial_api._write_hand_payload.
    player_id_to_seat = {}
    seat_names = {}

    if hand.session_id is not None:
        seat_rows = (
            db.query(TableSeat, Player)
            .join(Player, TableSeat.player_id == Player.player_id)
            .filter(TableSeat.session_id == hand.session_id)
            .order_by(TableSeat.seat_number)
            .all()
        )
        for ts, p in seat_rows:
            player_id_to_seat[p.player_id] = ts.seat_number
            seat_names[ts.seat_number] = p.username
    else:
        # Hypothetical: negative player_ids encode seat numbers directly.
        # Collect distinct player_ids referenced by this hand.
        hyp_pids = (
            db.query(HoleCard.player_id)
            .filter(HoleCard.hand_id == hand_id)
            .distinct()
            .all()
        )
        hyp_pid_list = [r[0] for r in hyp_pids]
        if hyp_pid_list:
            hyp_players = (
                db.query(Player).filter(Player.player_id.in_(hyp_pid_list)).all()
            )
            for p in hyp_players:
                seat = -p.player_id  # negative pid -> seat number
                player_id_to_seat[p.player_id] = seat
                seat_names[seat] = p.username.replace("[Hypothetical] ", "")

    def seat_of(player_id):
        return player_id_to_seat.get(player_id, abs(player_id) if player_id < 0 else 0)

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
            stack_before=a.stack_before,
            pot_before=a.pot_before,
        )
        for a in actions_raw
        if a.action_index != -1  # exclude synthetic SEAT rows used for initial_stacks
    ]

    # ── Hole cards ────────────────────────────
    # Only cards still IN_HAND count toward a player's displayed hand —
    # a card that was DISCARDED or PASSED_OUT during the hand (see
    # hole_cards.py's own docstring) is no longer part of what this
    # view means by "this player's hole cards". Full card-by-card
    # history (deal/discard/draw/pass) lives in CardEvent, not here.
    hc_raw = (
        db.query(HoleCard)
        .filter(HoleCard.hand_id == hand_id, HoleCard.status == "IN_HAND")
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

    # ── Hole card event timeline ──────────────
    # See HoleCardEventDTO's own docstring — this is the real,
    # sequenced, street-stamped ledger the Replayer needs to stage
    # extra-card / draw / pass mechanics correctly, instead of the
    # timing-free hole_cards snapshot above. `street` on each row is
    # now SessionLogger's own reveal-order counter (see
    # session_logger.py / graph_engine_callbacks.py) — the SAME
    # numbering board_cards[].street uses, guaranteed self-consistent
    # by construction.
    ce_raw = (
        db.query(CardEvent)
        .filter(CardEvent.hand_id == hand_id)
        .order_by(CardEvent.sequence)
        .all()
    )
    hole_card_events = [
        HoleCardEventDTO(
            sequence=ce.sequence,
            street=ce.street,
            event_type=ce.event_type,
            card=card_str(ce.card),
            from_seat=(
                seat_of(ce.from_player_id) if ce.from_player_id is not None else None
            ),
            to_seat=(
                seat_of(ce.to_player_id) if ce.to_player_id is not None else None
            ),
        )
        for ce in ce_raw
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
    # Two shapes are built from the SAME underlying rows: the legacy
    # flat `point_results` list (kept for any existing consumer), and
    # a nested `showdown` object using the EXACT SAME ShowdownDTO/
    # PointResultDTO/PlayerBoardResultDTO shapes GameStateDTO.showdown
    # already uses for the live Game Simulator (see
    # app/dto/state_dto.py and engine_adapter.build_showdown_dto).
    #
    # This is the fix for "winning-hand highlighting works in the Game
    # Simulator but not the Hand Replayer": the live shape carries an
    # explicit `is_winner: bool` per player per board — the OLD flat
    # replay shape had no such field at all, so a frontend showdown
    # component written against the live shape had nothing to read for
    # a replayed hand. Persisted PointResult.point_share > 0 is the
    # same "did this player take part of this board's pot" signal the
    # live path's own PlayerPointResult.is_winner is built from
    # (mirrors this file's own payouts>0 convention used for
    # `winners` below).
    #
    # session_logger.finish_hand() writes ONE HandPoint row per entry
    # in the live result.points list — i.e. a multi-board point name
    # (double board, etc.) produces MULTIPLE HandPoint rows sharing the
    # same `name`, in the order they were scored (point_id is
    # insertion order). Grouping HandPoint rows by `name` here
    # reconstructs exactly the same "boards" list
    # build_showdown_dto groups a live result.points list by .name
    # into — so both paths produce identically-shaped output,
    # board-for-board.
    points_raw = (
        db.query(HandPoint)
        .filter(HandPoint.hand_id == hand_id)
        .order_by(HandPoint.point_id)
        .all()
    )

    point_results: list[PointResultDTO] = []  # legacy flat shape
    boards_by_name: dict[str, list[HandPoint]] = {}
    score_type_by_name: dict[str, str] = {}

    for hp in points_raw:
        boards_by_name.setdefault(hp.name, []).append(hp)
        score_type_by_name.setdefault(hp.name, hp.score_type)

        prs = db.query(PointResult).filter(PointResult.point_id == hp.point_id).all()
        for pr in prs:
            hole_used = [
                card_str(pc.card)
                for pc in db.query(PointCard)
                .filter(
                    PointCard.point_result_id == pr.point_result_id,
                    PointCard.source == "hole",
                )
                .all()
            ]
            board_used = [
                card_str(pc.card)
                for pc in db.query(PointCard)
                .filter(
                    PointCard.point_result_id == pr.point_result_id,
                    PointCard.source == "board",
                )
                .all()
            ]
            point_results.append(
                PointResultDTO(
                    point_name=hp.name,
                    score_type=hp.score_type,
                    player_seat=seat_of(pr.player_id),
                    player_name=name_of(pr.player_id),
                    hand_category=pr.hand_category or "",
                    hand_value=pr.hand_value or 0,
                    point_share=pr.point_share or 0.0,
                    hole_cards_used=hole_used,
                    board_cards_used=board_used,
                )
            )

    nested_point_results: list[NestedPointResultDTO] = []
    for name, hp_rows in boards_by_name.items():
        board_results: list[list[PlayerBoardResultDTO]] = []
        board_winners: list[list[int]] = []
        no_qualify: list[bool] = []

        for hp in hp_rows:
            prs = (
                db.query(PointResult)
                .filter(PointResult.point_id == hp.point_id)
                .all()
            )
            players_dto: list[PlayerBoardResultDTO] = []
            winners_this_board: list[int] = []

            for pr in prs:
                p_index = seat_of(pr.player_id) - 1  # engine's own 0-based convention
                is_winner = (pr.point_share or 0.0) > 0
                if is_winner:
                    winners_this_board.append(p_index)

                hole_used = [
                    card_str(pc.card)
                    for pc in db.query(PointCard)
                    .filter(
                        PointCard.point_result_id == pr.point_result_id,
                        PointCard.source == "hole",
                    )
                    .all()
                ]
                board_used = [
                    card_str(pc.card)
                    for pc in db.query(PointCard)
                    .filter(
                        PointCard.point_result_id == pr.point_result_id,
                        PointCard.source == "board",
                    )
                    .all()
                ]

                players_dto.append(
                    PlayerBoardResultDTO(
                        player_index=p_index,
                        hand_category=pr.hand_category,
                        hand_value=pr.hand_value or 0,
                        best_hand_cards=_mask_to_card_strs(pr.best_hand_mask),
                        hole_cards_used=hole_used,
                        board_cards_used=board_used,
                        is_winner=is_winner,
                    )
                )

            board_results.append(players_dto)
            board_winners.append(winners_this_board)
            no_qualify.append(len(winners_this_board) == 0)

        nested_point_results.append(
            NestedPointResultDTO(
                name=name,
                score_type=score_type_by_name[name],
                board_winners=board_winners,
                board_results=board_results,
                no_qualify=no_qualify,
                # Scoop flags aren't persisted anywhere today (no
                # scoop_from / "scooped from paired high" column on
                # PointResult) — defaulted False rather than guessed,
                # the same gap the live path would have if
                # result.scoop_flags were ever absent (see
                # build_showdown_dto's own `if result.scoop_flags:`
                # guard).
                scoop=[False] * len(hp_rows),
            )
        )

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

    # winners (1-based seats) mirrors GameStateDTO's own top-level
    # `winners` field exactly — engine_adapter.state_to_dto /
    # graph_engine_adapter.graph_state_to_dto both compute it as
    # `[p + 1 for p, amt in result.payouts.items() if amt > 0]`.
    winners = [p.player_seat for p in payouts if p.amount and p.amount > 0]

    showdown = None
    if nested_point_results:
        showdown = NestedShowdownDTO(
            payout_type=("split_pot" if hand.split_pot else "points"),
            point_results=nested_point_results,
            # Per-player running point tallies aren't persisted
            # anywhere (no dedicated table) — left None, same as the
            # live path leaves this None whenever it has nothing to
            # report for a non-"points"-payout hand.
            point_tallies=None,
            payouts={p.player_seat - 1: p.amount for p in payouts},
            pot_winners=[w - 1 for w in winners],
        )

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
        hole_card_events=hole_card_events,
        board_cards=board_cards,
        point_results=point_results,
        payouts=payouts,
        showdown=showdown,
        winners=winners,
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


@router.get("/hands/{hand_id}/annotations", response_model=list[AnnotationDTO])
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
    ann = (
        db.query(Annotation)
        .filter(
            Annotation.annotation_id == annotation_id,
            Annotation.user_id == user_id,
        )
        .first()
    )
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
    ann = (
        db.query(Annotation)
        .filter(
            Annotation.annotation_id == annotation_id,
            Annotation.user_id == user_id,
        )
        .first()
    )
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")

    db.delete(ann)
    db.commit()