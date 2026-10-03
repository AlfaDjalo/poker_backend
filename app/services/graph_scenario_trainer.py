"""
app/services/graph_scenario_trainer.py — GraphScenarioEnv-backed Trainer
sessions, for scenarios auto-detected as migrated via
ScenarioConfig.is_graph_migrated() (see trainer_config.py's directory-
presence auto-detection).

Parallel path to the legacy PokerState machinery in trainer_service.py
— TrainerService dispatches here at exactly three points
(_new_scenario_once, apply_hero_action, get_state) when
cfg.is_graph_migrated() is True, and otherwise never touches this
module. Every unmigrated scenario's behavior stays byte-for-byte
unchanged.

Written against the REAL GraphScenarioEnv/graph_observation_adapter
source (poker_rl_lab), not the earlier speculative draft. Key facts
that shaped this rewrite:

  - GraphScenarioEnv is a SELF-PLAY env: no hero/villain split at the
    env level. reset()/step() hand back a GraphDecisionObs for
    WHICHEVER seat the graph asks to act next — "hero" is a pure
    Trainer-side bookkeeping concept layered on top (which physical
    seat the human occupies this hand), never passed into the env.
  - step(action_index) returns GraphScenarioStepResult(obs, rewards,
    done, info) — there is no way to re-query "what's pending" after
    the fact, so this module tracks the last-returned obs itself
    (GraphScenarioSession.current_obs) rather than re-reading engine
    state.
  - obs.legal_mask is padded out to config.max_action_options (fixed
    width, matching the policy network's action head); obs.options is
    the REAL, unpadded tuple. action_index into step() must be a valid
    index into obs.options specifically (env validates this itself).
  - GraphActionOption carries engine_action_type (BETTING),
    choice_label (CHOICE), is_stop (CARD_SELECT/CARD_PASS), or neither
    (BOOLEAN — label is "TRUE"/"FALSE" directly) — _option_display_label()
    below picks the right one for logging/grading/matching.
  - Dealer position for a 2-seat scenario is FIXED by
    StackConfig.fixed_dealer_position (not rotated hand-to-hand) — so
    unlike the legacy PokerState path, this module does NOT need to
    compute a dealer_position offset to control who's SB/BB; it only
    needs to pick which of the two fixed physical seats the human
    plays each hand (hero_rl_seat, drawn uniformly). SB/BB labeling for
    display is derived from env.state.dealer_position directly (same
    (dealer+1)%n formula push_fold_service.py already uses).

Open items, status per RL Lab's confirmation:
  - GraphScenarioEnv.engine: LANDING (RL Lab side, tracked separately)
    — a public read-only property, same shape as `.state`. Until it
    ships, _engine_for() below falls back to the private `_engine`
    attribute so this file works either way.
  - action_probabilities(seat, obs, legal_mask) -> torch.Tensor:
    LANDING on DualSeatActorCritic and all three concrete
    implementations (IndependentSeatActorCritic,
    SharedSingleHeadActorCritic, SharedTrunkTwoHeadsActorCritic) — full
    masked softmax, no sampling. trainer_agent_cache._GraphAgentWrapper
    calls this by the confirmed name/signature already; its uniform
    fallback stays in place only for the window before this lands.
  - load_graph_scenario(name, path=None) -> GraphScenarioConfig:
    CONFIRMED (scenarios/graph_scenario_config.py). `path` is passed
    explicitly here rather than relying on name-only resolution, since
    that only checks the package's own graph_scenarios/ subdir — see
    new_graph_scenario()'s own comment.
  - graph_engine_adapter.graph_state_to_dto(): CONFIRMED reusable as-is
    (Backend-side audit — no DB reads, no GameService-instance reads,
    no fixed-flow-shape assumptions). No changes needed there.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Any, Dict, List, Optional

from app.config.trainer_config import ScenarioConfig
from app.graph_engine_adapter import graph_state_to_dto


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class GraphScenarioSession:
    """Everything TrainerService needs to track for one in-progress
    GraphScenarioEnv hand — the graph-path analog of TrainerService's
    own self.state / self.hero_position / self._display_hole_cards."""

    cfg_key: str
    env: Any  # poker_rl_lab.envs.graph_scenario_env.GraphScenarioEnv
    hero_rl_seat: int
    villain_rl_seat: int
    hand_id: int

    current_obs: Optional[Any] = None  # GraphDecisionObs | None (None once done)
    done: bool = False
    rewards: Dict[int, float] = field(default_factory=dict)

    hero_position: str = ""
    hero_effective_bb: float = 0.0
    awaiting_hero: bool = False
    action_log: List[Dict[str, Any]] = field(default_factory=list)

    # Snapshot at deal time, before any fold zeroes a hand_mask — same
    # rationale as TrainerService._display_hole_cards.
    display_hole_cards: Dict[int, List[int]] = field(default_factory=dict)

    # Set by _grade_decision_graph(), consumed once TrainerService
    # records the graded result into the shared Scoreboard.
    pending_grade_extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Option display helpers
# ---------------------------------------------------------------------------


def _option_display_label(opt) -> str:
    """Raw label off a GraphActionOption — used ONLY for non-BETTING
    domains (BOOLEAN/CHOICE/CARD_SELECT/CARD_PASS), where there is no
    established Trainer-level vocabulary to translate onto yet. BETTING
    options go through _trainer_action_label() instead — see that
    function's own docstring for why raw engine_action_type names
    (RAISE, BET) can't be used directly here."""
    if opt is None:
        return "?"
    if opt.engine_action_type is not None:
        return opt.engine_action_type.name.lower()
    if opt.choice_label is not None:
        return opt.choice_label
    if opt.is_stop:
        return "stop"
    return opt.label.lower()


def _trainer_action_label(env, seat: int, opt) -> str:
    """
    Maps a GraphActionOption onto the Trainer-level action vocabulary
    the frontend actually sends/renders buttons for — fold/check/call/
    bet/all_in (TrainerActionRequest.action_type's own docstring in
    trainer_api.py; training_config.yaml's per-scenario `actions:`
    blocks are what the frontend's buttons are built from, and that
    frontend hasn't itself been migrated — it knows nothing about raw
    engine ActionType names). Using opt.engine_action_type.name.lower()
    directly was wrong: push-fold's SB shove is internally a RAISE
    (there's already a posted blind to raise over), never "all_in" as
    an engine_action_type — but the ONLY Trainer button push_fold_duo
    has ever offered for that decision is "all_in".

    BET/RAISE-shaped options are "all_in" when the option's amount
    equals the acting player's full remaining commitment (current_bet
    + stack) — true for every bet-like option push_fold_duo offers,
    since push-fold has no partial-bet concept at all — and "bet"
    otherwise (a real sized bet, for any future migrated scenario that
    actually has one).
    """
    if opt is None:
        return "?"
    if opt.engine_action_type is not None:
        name = opt.engine_action_type.name
        if name == "FOLD":
            return "fold"
        if name == "CHECK":
            return "check"
        if name == "CALL":
            return "call"
        if name in ("BET", "RAISE"):
            player = env.state.players[seat]
            full_commit = player.current_bet + player.stack
            if opt.amount is not None and opt.amount >= full_commit:
                return "all_in"
            return "bet"
        return name.lower()
    return _option_display_label(opt)


def _hole_card_ids(env, seat: int) -> List[int]:
    from poker_engine.cards.mask import mask_to_card_ids

    return list(mask_to_card_ids(env.state.players[seat].hand_mask))


def _position_for_seat(env, seat: int) -> str:
    """SB/BB label for display, derived from the env's own
    dealer_position — same (dealer+1)%n / (dealer+2)%n convention
    push_fold_service.py already uses. Scenario-specific (assumes a
    heads-up SB/BB game); generalize if a non-push-fold scenario
    migrates later."""
    n = len(env.state.players)
    sb_seat = (env.state.dealer_position + 1) % n
    return "SB" if seat == sb_seat else "BB"


# ---------------------------------------------------------------------------
# New scenario
# ---------------------------------------------------------------------------


def new_graph_scenario(cfg: ScenarioConfig, hand_id: int) -> GraphScenarioSession:
    from poker_rl_lab.envs.graph_scenario_env import GraphScenarioEnv
    from poker_rl_lab.scenarios.graph_scenario_config import load_graph_scenario

    # `path` passed explicitly — RL Lab confirmed load_graph_scenario's
    # own name-only resolution only checks the poker_rl_lab package's
    # OWN graph_scenarios/ subdir (the default tier of
    # _default_graph_scenarios_dir()'s three-tier lookup in
    # trainer_config.py). An operator-configured graph_scenarios_dir
    # (env var or rl_lab_paths.yaml) would silently miss by name alone,
    # so this always uses the already-resolved path from ScenarioConfig
    # rather than re-deriving it.
    graph_cfg = load_graph_scenario(
        cfg.resolved_graph_scenario_key(), path=cfg.graph_scenario_path
    )

    env = GraphScenarioEnv(graph_cfg)
    obs = env.reset()

    n = len(env.state.players)
    hero_rl_seat = random.randrange(n)
    villain_rl_seat = 1 - hero_rl_seat if n == 2 else (hero_rl_seat + 1) % n

    session = GraphScenarioSession(
        cfg_key=cfg.key,
        env=env,
        hero_rl_seat=hero_rl_seat,
        villain_rl_seat=villain_rl_seat,
        hand_id=hand_id,
        current_obs=obs,
        done=False,
    )

    session.hero_position = _position_for_seat(env, hero_rl_seat)
    big_blind = cfg.big_blind
    session.hero_effective_bb = round(
        min(p.stack for p in env.state.players) / big_blind, 2
    )
    session.display_hole_cards = {
        seat: _hole_card_ids(env, seat) for seat in range(n)
    }

    _advance_through_villain(session, cfg)
    session.awaiting_hero = _current_seat_is(session, hero_rl_seat)

    return session


def _current_seat_is(session: GraphScenarioSession, seat: int) -> bool:
    return (
        not session.done
        and session.current_obs is not None
        and session.current_obs.seat == seat
    )


def _advance_through_villain(session: GraphScenarioSession, cfg: ScenarioConfig) -> None:
    while _current_seat_is(session, session.villain_rl_seat):
        _villain_act_graph(session, cfg)


# ---------------------------------------------------------------------------
# Villain (AI) action
# ---------------------------------------------------------------------------


def _villain_act_graph(session: GraphScenarioSession, cfg: ScenarioConfig) -> None:
    from app.services.trainer_agent_cache import get_graph_agent
    from poker_rl_lab.representation.graph_observation_adapter import (
        graph_observation_to_representation_input,
    )

    env = session.env
    obs = session.current_obs
    rep_input = graph_observation_to_representation_input(env, obs)

    agent = get_graph_agent(cfg)
    result = agent.act(
        seat=session.villain_rl_seat, obs=rep_input, legal_mask=obs.legal_mask, deterministic=False
    )

    chosen = obs.options[result.action_type] if result.action_type < len(obs.options) else None
    pot_before = env.state.pot

    step_result = env.step(result.action_type)

    session.action_log.append(
        {
            "actor": "ai",
            "position": _position_for_seat(env, session.villain_rl_seat),
            "action": _trainer_action_label(env, session.villain_rl_seat, chosen),
            "amount": getattr(chosen, "amount", None),
            "street": 0,
            "pot_before": pot_before,
        }
    )

    session.current_obs = step_result.obs
    session.done = step_result.done
    if step_result.done:
        session.rewards = step_result.rewards


# ---------------------------------------------------------------------------
# Hero action
# ---------------------------------------------------------------------------


def hero_allowed_labels_graph(session: GraphScenarioSession) -> List[str]:
    if not _current_seat_is(session, session.hero_rl_seat):
        return []
    env = session.env
    return [
        _trainer_action_label(env, session.hero_rl_seat, o)
        for o in session.current_obs.options
    ]


def hero_action_options_graph(session: GraphScenarioSession) -> Optional[List[Dict[str, Any]]]:
    """Serialize the current discrete graph options, retaining sized
    BETTING option identity and exact engine amount for the Trainer UI."""
    if not _current_seat_is(session, session.hero_rl_seat):
        return None
    if session.current_obs.domain != "BETTING":
        return None

    options = []
    for option in session.current_obs.options:
        action_type = _trainer_action_label(session.env, session.hero_rl_seat, option)
        engine_type = getattr(option.engine_action_type, "name", None)
        is_sized_bet = engine_type in ("BET", "RAISE")
        options.append({
            "action_type": action_type,
            "amount": option.amount if is_sized_bet else None,
            "label": option.label,
        })
    return options


def _match_hero_option(session: GraphScenarioSession, action_label: str, amount: int | None = None):
    """Returns (index, option). Matches against the SAME Trainer-level
    vocabulary hero_allowed_labels_graph() reports (fold/check/call/
    bet/all_in — see _trainer_action_label's own docstring), not raw
    engine_action_type names. No amount tiebreak: push_fold_duo's
    BETTING options never collide under this vocabulary (FOLD vs
    RAISE-as-all_in, or FOLD vs CALL), so this is unambiguous for it.
    A scenario with a real sized-bet option ALONGSIDE an all-in option
    (both mapping to different labels already, "bet" vs "all_in", per
    _trainer_action_label's full_commit check) stays unambiguous too;
    only two same-labeled bet sizes would need a tiebreak, and no
    migrated scenario has that yet."""
    if not _current_seat_is(session, session.hero_rl_seat):
        raise ValueError("Not the hero's turn to act.")

    env = session.env
    options = session.current_obs.options
    label_lower = (action_label or "").lower()
    matches = []
    for i, opt in enumerate(options):
        if _trainer_action_label(env, session.hero_rl_seat, opt) != label_lower:
            continue
        if amount is not None and opt.amount != amount:
            continue
        matches.append((i, opt))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Action {action_label!r} matches multiple legal bet sizes; "
            "submit the selected option's exact amount."
        )

    legal = [_trainer_action_label(env, session.hero_rl_seat, o) for o in options]
    raise ValueError(
        f"Action {action_label!r} with amount {amount!r} is not among the legal options right now "
        f"(legal: {legal})"
    )


def apply_hero_action_graph(
    session: GraphScenarioSession, cfg: ScenarioConfig, action_label: str,
    amount: int | None = None,
) -> Dict[str, Any]:
    """Apply the hero's decision, grade it, run the villain's
    response(s). Does NOT touch the Scoreboard — caller (TrainerService)
    records the returned dict there, same as the legacy path."""
    obs = session.current_obs
    index, option = _match_hero_option(session, action_label, amount)

    result = _grade_decision_graph(session, cfg, obs, option)

    pot_before = session.env.state.pot
    step_result = session.env.step(index)

    session.action_log.append(
        {
            "actor": "hero",
            "position": session.hero_position,
            "action": _trainer_action_label(session.env, session.hero_rl_seat, option),
            "amount": getattr(option, "amount", None),
            "street": 0,
            "pot_before": pot_before,
        }
    )

    session.current_obs = step_result.obs
    session.done = step_result.done
    if step_result.done:
        session.rewards = step_result.rewards

    _advance_through_villain(session, cfg)
    session.awaiting_hero = _current_seat_is(session, session.hero_rl_seat)

    return result


def _grade_decision_graph(
    session: GraphScenarioSession, cfg: ScenarioConfig, obs, chosen_option
) -> Dict[str, Any]:
    """Real forward pass at the hero's decision point, graded
    "highest-probability legal action wins" — same logic
    decision_evaluator.RLPolicyFrequencyEvaluator applies, reimplemented
    inline since inputs here are keyed by obs.options[i] labels, not a
    config-mapped trainer string (see this module's own docstring on
    the unconfirmed action_probabilities() method this depends on)."""
    from app.services.trainer_agent_cache import get_graph_agent
    from poker_rl_lab.representation.graph_observation_adapter import (
        graph_observation_to_representation_input,
    )

    env = session.env
    rep_input = graph_observation_to_representation_input(env, obs)
    agent = get_graph_agent(cfg)
    probs = agent.action_probabilities(seat=obs.seat, obs=rep_input, legal_mask=obs.legal_mask)

    labels = [
        (
            f"bet_{o.amount}"
            if o.engine_action_type is not None
            and o.engine_action_type.name in ("BET", "RAISE")
            and _trainer_action_label(env, obs.seat, o) == "bet"
            else _trainer_action_label(env, obs.seat, o)
        )
        for o in obs.options
    ]
    probs_by_label = {label: float(probs[i]) for i, label in enumerate(labels)}

    chosen_label = (
        f"bet_{chosen_option.amount}"
        if chosen_option.engine_action_type is not None
        and chosen_option.engine_action_type.name in ("BET", "RAISE")
        and _trainer_action_label(env, obs.seat, chosen_option) == "bet"
        else _trainer_action_label(env, obs.seat, chosen_option)
    )
    best_label = max(labels, key=lambda l: probs_by_label.get(l, 0.0)) if labels else chosen_label
    correct = best_label == chosen_label

    hole_ids = _hole_card_ids(env, session.hero_rl_seat)
    from poker_engine.cards.card import Card as CardObj

    hole_strs = [str(CardObj(cid)) for cid in hole_ids]
    hand_display = _canonicalize_2(hole_strs) if len(hole_strs) == 2 else " ".join(hole_strs)

    probs_str = ", ".join(f"{l}={probs_by_label.get(l, 0.0):.3f}" for l in labels)

    session.pending_grade_extra = {
        "hand": hand_display,
        "position": session.hero_position,
        "effective_stack_bb": session.hero_effective_bb,
        "scenario": cfg.key,
        "policy_probs": probs_by_label,
    }

    return {
        "correct": correct,
        "best_action": best_label,
        "hero_action": chosen_label,
        "explanation": f"{session.hero_position} policy frequencies: [{probs_str}]",
    }


def _canonicalize_2(hole_strs: List[str]) -> str:
    from app.services.decision_evaluator import canonicalize_hole_cards

    return canonicalize_hole_cards(hole_strs)


# ---------------------------------------------------------------------------
# State serialization
# ---------------------------------------------------------------------------


def _engine_for(env) -> Any:
    """
    RL Lab has confirmed a public, read-only `engine` property is
    landing on GraphScenarioEnv (same shape as the existing `.state`
    property — raises RuntimeError before reset()), tracked separately
    from this migration. Prefer it the moment it exists; fall back to
    the private `_engine` attribute until then so this file keeps
    working across that landing without needing a synchronized deploy.
    Simplify to `env.engine` directly once the property has shipped and
    this fallback is confirmed dead.
    """
    engine = getattr(env, "engine", None)
    if engine is not None:
        return engine
    return env._engine  # noqa: SLF001 — fallback only; see docstring above


def get_state_dto_graph(session: GraphScenarioSession, cfg: ScenarioConfig) -> Dict[str, Any]:
    """Reuses app.graph_engine_adapter.graph_state_to_dto() directly —
    confirmed reusable as-is against an ephemeral engine/state pair (no
    DB reads, no GameService-instance reads, no fixed-flow-shape
    assumptions — see the coordination thread's audit)."""
    env = session.env
    engine = _engine_for(env)
    game_def, rules, graph = env.game_def, env.rules, env.graph

    dto = graph_state_to_dto(engine, game_def, rules, graph=graph)
    payload = dto.dict()

    for p in payload["players"]:
        engine_seat = p["seat"] - 1
        if engine_seat in session.display_hole_cards:
            live_player = env.state.players[engine_seat]
            if getattr(live_player, "has_folded", False):
                from poker_engine.cards.card import Card as CardObj

                p["hand"] = [str(CardObj(cid)) for cid in session.display_hole_cards[engine_seat]]

    # ------------------------------------------------------------
    # Seat remap — MUST mirror TrainerService.get_state()'s own remap
    # block (trainer_service.py: engine_seat_to_display /
    # _HERO_DISPLAY_SEAT=1 / _AI_DISPLAY_SEAT=4) exactly, or the
    # frontend — which hardcodes hero at display seat 1 and AI at
    # display seat 4, and reads payload["hero_seat"]/["ai_seat"] to
    # know which is which — has no way to find either player, which is
    # why no cards were rendering. The legacy path can hardcode
    # cfg.hero_seat/cfg.villain_seat (fixed per that path's own
    # convention); this path can't, since session.hero_rl_seat is
    # drawn randomly per hand (see new_graph_scenario) rather than
    # fixed by config — use the session's own seats instead.
    #
    # _HERO_DISPLAY_SEAT/_AI_DISPLAY_SEAT are duplicated here rather
    # than imported from trainer_service.py to avoid a circular import
    # (trainer_service imports this module). Keep these two literals in
    # sync with trainer_service.py's own constants if either changes.
    # ------------------------------------------------------------
    _HERO_DISPLAY_SEAT = 1
    _AI_DISPLAY_SEAT = 4

    engine_seat_to_display = {
        session.hero_rl_seat: _HERO_DISPLAY_SEAT,
        session.villain_rl_seat: _AI_DISPLAY_SEAT,
    }
    for p in payload["players"]:
        engine_seat = p["seat"] - 1
        p["seat"] = engine_seat_to_display.get(engine_seat, p["seat"])

    if payload.get("winners"):
        payload["winners"] = [
            engine_seat_to_display.get(w - 1, w) for w in payload["winners"]
        ]

    payload.update(
        {
            "active": True,
            "scenario": cfg.key,
            "hero_seat": _HERO_DISPLAY_SEAT,
            "ai_seat": _AI_DISPLAY_SEAT,
            "villain_seat": _AI_DISPLAY_SEAT,
            "hero_position": session.hero_position,
            "hero_effective_bb": session.hero_effective_bb,
            "awaiting_hero": session.awaiting_hero,
            "hero_allowed_actions": hero_allowed_labels_graph(session),
            "hero_action_options": hero_action_options_graph(session),
            "hand_over": session.done,
            "action_log": list(session.action_log),
            "hand_id": session.hand_id,
        }
    )
    return payload


# ---------------------------------------------------------------------------
# Hand grid
# ---------------------------------------------------------------------------

_RANK_ORDER_HIGH_TO_LOW = "AKQJT98765432"
_RANK_CHAR_BY_ENGINE_RANK = {
    0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8",
    7: "9", 8: "T", 9: "J", 10: "Q", 11: "K", 12: "A",
}
# Duplicated from trainer_service._canonical_hand_and_cell rather than
# imported (trainer_service imports THIS module — importing back would
# be circular). Same poker_engine.cards.card.Card id scheme throughout
# the codebase (rank = id % 13, suit = id // 13), so this stays correct
# as long as that scheme doesn't change. Worth extracting into a small
# shared util module if a third caller ever needs it.


def _canonical_hand_and_cell(c1: int, c2: int) -> tuple[str, int, int]:
    r1, s1 = c1 % 13, c1 // 13
    r2, s2 = c2 % 13, c2 // 13

    hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
    hi_char = _RANK_CHAR_BY_ENGINE_RANK[hi]
    lo_char = _RANK_CHAR_BY_ENGINE_RANK[lo]

    row = _RANK_ORDER_HIGH_TO_LOW.index(hi_char)
    col = _RANK_ORDER_HIGH_TO_LOW.index(lo_char)

    if hi == lo:
        return f"{hi_char}{lo_char}", row, col

    suited = s1 == s2
    if suited:
        return f"{hi_char}{lo_char}s", row, col
    return f"{hi_char}{lo_char}o", col, row


def compute_hand_grid_graph(session: GraphScenarioSession, cfg: ScenarioConfig) -> Dict[str, Any]:
    """
    Graph-native counterpart of TrainerService._compute_hand_grid() —
    swaps the hero's hole cards through all 1326 combos at the CURRENT
    decision point and asks the loaded agent for action probabilities
    at each combo, via the same agent.action_probabilities(seat, obs,
    legal_mask) RL Lab confirmed and _grade_decision_graph() already
    uses for live grading.

    PERFORMANCE NOTE: unlike the legacy path's
    _get_action_probabilities_batch() (one batched forward pass via
    representation_model.forward_batch()), this calls
    action_probabilities() ONCE PER COMBO — 1326 individual forward
    passes — since no batched equivalent has been confirmed for
    IndependentSeatActorCritic/friends yet. Functionally correct, just
    slower. Isolated to this one function: swap the inner loop for a
    real batch call the moment RL Lab confirms one, nothing else here
    needs to change.

    ASSUMPTION (flagged, not confirmed against source): RepresentationInput
    is a real @dataclass supporting dataclasses.replace() the same way
    StructuredObservation does for the legacy path's own hand-grid combo
    substitution. This mirrors the exact pattern already proven for the
    legacy path (_compute_hand_grid's
    `_dataclasses_replace(template_obs, hole_cards=...)`), so it's a
    reasonable bet — but if RepresentationInput turns out to be frozen
    differently or not a dataclass at all, this raises a clear
    AttributeError/TypeError at the dataclasses.replace() call below
    rather than silently producing wrong output.
    """
    if not _current_seat_is(session, session.hero_rl_seat):
        raise ValueError("Not the hero's turn to act — cannot build a hand grid.")

    from app.services.trainer_agent_cache import get_graph_agent
    from poker_rl_lab.representation.graph_observation_adapter import (
        graph_observation_to_representation_input,
    )

    env = session.env
    obs = session.current_obs
    seat = session.hero_rl_seat

    template_rep = graph_observation_to_representation_input(env, obs)
    agent = get_graph_agent(cfg)
    action_order = [_trainer_action_label(env, seat, o) for o in obs.options]

    combos = list(itertools.combinations(range(52), 2))

    action_sum: List[List[Dict[str, float]]] = [[dict() for _ in range(13)] for _ in range(13)]
    cell_count = [[0] * 13 for _ in range(13)]
    combos_by_hand: Dict[str, List[Dict[str, Any]]] = {}

    from poker_engine.cards.card import Card as CardObj

    for c1, c2 in combos:
        rep_input = _dc_replace(template_rep, hole_card_ids=(c1, c2))
        probs = agent.action_probabilities(seat=seat, obs=rep_input, legal_mask=obs.legal_mask)
        probs_by_label = {label: float(probs[i]) for i, label in enumerate(action_order)}

        hand_str, row, col = _canonical_hand_and_cell(c1, c2)
        cell_count[row][col] += 1
        cell_sums = action_sum[row][col]
        for a, p in probs_by_label.items():
            cell_sums[a] = cell_sums.get(a, 0.0) + p

        combos_by_hand.setdefault(hand_str, []).append(
            {"cards": [str(CardObj(c1)), str(CardObj(c2))], "probs": probs_by_label}
        )

    action_grid = [
        [
            (
                {a: (action_sum[r][c].get(a, 0.0) / cell_count[r][c]) for a in action_order}
                if cell_count[r][c]
                else {a: 0.0 for a in action_order}
            )
            for c in range(13)
        ]
        for r in range(13)
    ]

    return {
        "position": session.hero_position,
        "action_order": action_order,
        "hero_effective_bb": session.hero_effective_bb,
        "action_grid": action_grid,
        "combos": {hand: {"combos": entries} for hand, entries in combos_by_hand.items()},
    }