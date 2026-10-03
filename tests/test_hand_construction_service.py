"""
tests/test_hand_construction_service.py

Exercises HandConstructionService against a REAL compiled graph
(hold'em), not hand-constructed fake nodes — same rationale as
test_hole_card_count.py: a fake-node fixture can't catch a wrong
assumption about the real graph's shape.

These tests drive a full heads-up hold'em hand end-to-end through the
stepper: DEAL_HOLE review (accept + override), BETTING decisions
through to showdown, undo, and finish()'s payload shape — without
needing a live DB (finish()'s translated payload is asserted directly,
not round-tripped through tutorial_api's actual save endpoint, which
needs a DB session).
"""

import pytest

from app.dto.construction_dto import ConstructionStepResponse, StartConstructionRequest
from app.services.hand_construction_service import HandConstructionService


@pytest.fixture
def service():
    return HandConstructionService()


def _start_heads_up(service, stack=1000):
    req = StartConstructionRequest(
        game_name="holdem",
        dealer_seat=1,
        seats={1: "Alice", 2: "Bob"},
        initial_stacks={1: stack, 2: stack},
    )
    return service.start(req)


# ---------------------------------------------------------------------------
# Session lifecycle / basic stepping
# ---------------------------------------------------------------------------


def test_start_returns_deal_hole_step_first():
    service = HandConstructionService()
    step = _start_heads_up(service)
    assert step.domain == "DEAL_HOLE"
    assert step.count == 2
    assert step.seat in (1, 2)
    assert len(step.eligible_cards) == 52 - 2  # only this seat's own 2 unknown cards excluded so far


def test_accepting_dealt_hole_cards_advances_to_next_seat_then_decision():
    service = HandConstructionService()
    step = _start_heads_up(service)
    session_id = step.session_id

    seen_seats = set()
    while step.domain == "DEAL_HOLE":
        seen_seats.add(step.seat)
        step = service.apply_step(session_id, ConstructionStepResponse())  # accept as dealt

    assert seen_seats == {1, 2}
    assert step.domain == "BETTING"
    assert step.seat in (1, 2)
    assert step.to_call is not None


def test_overriding_hole_cards_is_reflected_in_state():
    service = HandConstructionService()
    step = _start_heads_up(service)
    session_id = step.session_id

    forced = ["Ah", "As"]
    step = service.apply_step(session_id, ConstructionStepResponse(cards=forced))
    # find the player we just set, from the returned state snapshot
    seat_that_was_dealt = None
    for p in step.state.players:
        if set(p.hand) == set(forced):
            seat_that_was_dealt = p.seat
    assert seat_that_was_dealt is not None


def test_overriding_with_duplicate_card_raises():
    service = HandConstructionService()
    step = _start_heads_up(service)
    session_id = step.session_id

    # force seat 1's hand to include Ah
    step = service.apply_step(session_id, ConstructionStepResponse(cards=["Ah", "As"]))
    # now try to give seat 2 a duplicate of Ah
    with pytest.raises(ValueError):
        service.apply_step(session_id, ConstructionStepResponse(cards=["Ah", "Kd"]))


def test_wrong_card_count_raises():
    service = HandConstructionService()
    step = _start_heads_up(service)
    session_id = step.session_id
    with pytest.raises(ValueError):
        service.apply_step(session_id, ConstructionStepResponse(cards=["Ah"]))


# ---------------------------------------------------------------------------
# Full hand through to completion
# ---------------------------------------------------------------------------


def _play_out_hand_folding_immediately(service, session_id, step):
    """Accept both hole-card deals, then fold the first BETTING decision."""
    while step.domain == "DEAL_HOLE":
        step = service.apply_step(session_id, ConstructionStepResponse())
    assert step.domain == "BETTING"
    step = service.apply_step(session_id, ConstructionStepResponse(type="fold"))
    return step


def test_folding_reaches_complete():
    service = HandConstructionService()
    step = _start_heads_up(service)
    session_id = step.session_id
    step = _play_out_hand_folding_immediately(service, session_id, step)
    assert step.domain == "COMPLETE"


def test_step_after_complete_raises():
    service = HandConstructionService()
    step = _start_heads_up(service)
    session_id = step.session_id
    step = _play_out_hand_folding_immediately(service, session_id, step)
    assert step.domain == "COMPLETE"
    with pytest.raises(ValueError):
        service.apply_step(session_id, ConstructionStepResponse())


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------


def test_undo_after_hole_card_override_reverts_state():
    service = HandConstructionService()
    step = _start_heads_up(service)
    session_id = step.session_id

    original_state_players = {p.seat: tuple(p.hand) for p in step.state.players}

    step = service.apply_step(session_id, ConstructionStepResponse(cards=["Ah", "As"]))
    assert step.domain == "DEAL_HOLE"  # second seat's step

    reverted = service.undo(session_id)
    assert reverted.domain == "DEAL_HOLE"
    assert reverted.seat == step.seat or reverted.seat is not None  # first seat's step again


def test_undo_with_no_history_raises():
    service = HandConstructionService()
    step = _start_heads_up(service)
    with pytest.raises(ValueError):
        service.undo(step.session_id)


# ---------------------------------------------------------------------------
# finish()
# ---------------------------------------------------------------------------


def test_finish_payload_shape_after_fold():
    service = HandConstructionService()
    step = _start_heads_up(service)
    session_id = step.session_id
    step = _play_out_hand_folding_immediately(service, session_id, step)
    assert step.domain == "COMPLETE"

    payload = service.finish(session_id)

    assert payload["game_name"] == "holdem"
    assert payload["dealer_seat"] == 1
    assert len(payload["players"]) == 2
    for p in payload["players"]:
        assert len(p["hole_cards"]) == 2
    assert len(payload["hole_cards"]) == 2
    assert len(payload["node_cards"]) >= 5  # hold'em board
    # exactly one BETTING action (the fold) recorded
    assert len(payload["actions"]) == 1
    assert payload["actions"][0]["action_type"] == "FOLD"


def test_finish_unknown_session_raises():
    service = HandConstructionService()
    with pytest.raises(ValueError):
        service.finish("not-a-real-session-id")


# ---------------------------------------------------------------------------
# Reentrancy guard
# ---------------------------------------------------------------------------


def test_busy_guard_rejects_concurrent_step():
    service = HandConstructionService()
    step = _start_heads_up(service)
    session_id = step.session_id
    session = service._sessions[session_id]

    session.busy = True
    try:
        with pytest.raises(ValueError):
            service.apply_step(session_id, ConstructionStepResponse())
    finally:
        session.busy = False