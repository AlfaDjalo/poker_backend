"""
app/services/trainer_service.py — Backend for the frontend Trainer
component (scenario-agnostic drilling: push-fold today, more later).

Differs from push_fold_service.py in the ways requested:
  - No persistent stacks — every scenario is a fresh, independent spot.
  - Position (SB/BB, or OOP/IP) and effective stack are drawn randomly,
    not alternated/fixed.
  - Hero decisions are graded against a pluggable BaseDecisionEvaluator
    (app/services/decision_evaluator.py), not just played out.
  - A running scoreboard (correct/incorrect counts) is kept per
    session, resettable.
  - Everything scenario-specific (stack range, blinds, checkpoint path,
    evaluator) comes from training_config.yaml via TrainingConfig — see
    that module. Adding a new scenario type requires no changes here as
    long as it fits the same "engine_variant + betting_type + policy +
    evaluator" shape.

Reuses the same engine machinery as push_fold_service.py (PokerState,
loader.load_game, CppScoringEngine) and the same RL inference chain
(ObservationAdapter, DualSeatActorCritic) — see that file's own
docstring for the game_name-vs-engine_variant rationale, which applies
identically here.

Observation construction (push-fold vs. generic)
--------------------------------------------------
_build_obs() branches on cfg.betting_type:

  - "push_fold" -> _build_obs_push_fold(): a hand-tuned observation
    matching exactly how the trained push-fold-duo policy was built
    (push_fold_service.py's own _build_sb_obs/_build_bb_obs) — fixed
    SB={FOLD,ALL_IN} / BB={FOLD,CALL} legal masks, and a synthetic
    "push" HistoryEvent fed to BB. Left untouched so an already-trained
    checkpoint's input distribution doesn't shift.

  - anything else (river, and any future generic-tree-shaped scenario)
    -> _build_obs_generic(): asks the ENGINE what's actually legal
    right now (GameState.legal_actions()) and maps those names
    directly onto the abstract action vocabulary the RL Lab trained
    against, rather than assuming a fixed position-keyed action set.
    Board cards come straight from GameState.node_cards, so whatever
    the variant's yaml actually deals is what the model sees.

This split is what fixes river's previously-empty hand grid (the old
code reused push-fold's hardcoded FOLD/CALL/ALL_IN-only mask for every
scenario, which never matched river's real CHECK/BET_100/CALL/FOLD
legal set).

Hand Grid feature
------------------
get_hand_grid() (near the bottom of TrainerService) computes the
model's action probabilities for all 169 canonical starting hands at
the CURRENT scenario's decision point (same stack/position/villain
state), by swapping the hero's hole cards through every one of the
C(52,2) = 1326 possible combos and running one batched forward pass.
Results are bucketed into the 13x13 grid and averaged per cell, plus
kept per-combo for the frontend's cell-detail view. See
_canonical_hand_and_cell() and _get_action_probabilities_batch() below.

Scoreboard history + read-only replay
----------------------------------------
Each graded decision is recorded with a monotonically increasing id and
a full state snapshot taken AFTER the hand has fully resolved (showdown
included). get_history_snapshot(entry_id) returns that snapshot with
awaiting_hero forced False and is_history_replay=True, so the frontend
can render a past hand without offering new decisions on it — see
trainer_api.py's GET /trainer/history/{entry_id}.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field, replace as _dataclasses_replace
from typing import Any, Dict, List, Optional, Tuple

from poker_engine.state.poker_state import PokerState, Phase
from poker_engine.state.player_state import PlayerState
from poker_engine.scoring.scoring_engine import CppScoringEngine
from poker_engine.games.loader import load_game
from poker_engine.actions.action import Action as EngineAction
from poker_engine.actions.action_type import ActionType as EngineActionType
from poker_engine.cards.mask import mask_to_card_ids
from poker_engine.cards.card import Card as CardObj

from poker_rl_lab.envs.observations import HoleCardsObs

from app.config.trainer_config import ScenarioConfig, get_training_config
from app.services.decision_evaluator import (
    DecisionContext,
    DecisionResult,
    canonicalize_hole_cards,
    get_evaluator,
)

HERO_ROLE_TO_ENGINE_ACTION = {
    ("SB", "fold"): "fold",
    ("SB", "all_in"): "all_in",
    ("BB", "fold"): "fold",
    ("BB", "call"): "call",
}

# Fixed visual seats for the Trainer table, independent of which
# underlying engine seat (0/1) hero/AI occupy this hand (that flips
# with SB/BB — see new_scenario()'s dealer_position handling). The
# Trainer is always heads-up hero-vs-AI, so these two constants are
# the only seats that will ever be populated.
_HERO_DISPLAY_SEAT = 1
_AI_DISPLAY_SEAT = 4


# ---------------------------------------------------------------------------
# Fallback agent (mirrors push_fold_service.RandomFallbackAgent)
# ---------------------------------------------------------------------------

class _FallbackActionResult:
    __slots__ = ("action_type", "log_prob", "value", "entropy")

    def __init__(self, action_type: int):
        self.action_type = action_type
        self.log_prob = 0.0
        self.value = 0.0
        self.entropy = 0.0


class RandomFallbackAgent:
    def act(self, seat, obs, legal_mask, deterministic: bool = False):
        legal_indices = [i for i, ok in enumerate(legal_mask) if ok]
        return _FallbackActionResult(action_type=random.choice(legal_indices))


def _get_action_probabilities(agent, rl_seat: int, rep_input, legal_mask) -> Dict[str, float]:
    """
    Real forward pass through the loaded policy: representation_model
    -> the correct seat's policy head -> masked softmax. Returns a dict
    of engine action-type strings -> probability, restricted to the
    action types recognised by _ACTION_TYPE_TO_ENGINE below.

    Falls back to a uniform distribution over legal actions when no
    real network is loaded (RandomFallbackAgent) — the Trainer still
    grades consistently, just against a meaningless baseline until a
    real checkpoint is in place.
    """
    from poker_rl_lab.actions.abstract_action import ActionType, NUM_ACTIONS

    _ACTION_TYPE_TO_ENGINE = {
        int(ActionType.FOLD): "fold",
        int(ActionType.CHECK): "check",
        int(ActionType.CALL): "call",
        int(ActionType.BET_25): "bet_25",
        int(ActionType.BET_50): "bet_50",
        int(ActionType.BET_75): "bet_75",
        int(ActionType.BET_100): "bet_100",
        int(ActionType.BET_150): "bet_150",
        int(ActionType.ALL_IN): "all_in",
    }

    representation_model = getattr(agent, "representation_model", None)
    if representation_model is None and hasattr(agent, "actor_critic"):
        representation_model = agent.actor_critic.representation_model

    if representation_model is None:
        # RandomFallbackAgent (no real network) — uniform over legal actions.
        legal_indices = [i for i, ok in enumerate(legal_mask) if ok]
        n = len(legal_indices) or 1
        return {
            _ACTION_TYPE_TO_ENGINE[i]: 1.0 / n
            for i in legal_indices
            if i in _ACTION_TYPE_TO_ENGINE
        }

    import torch
    from torch.distributions import Categorical
    from poker_rl_lab.models.policy_value_network import apply_legal_mask

    if hasattr(agent, "head_sb") and hasattr(agent, "head_bb"):
        head = agent.head_sb if rl_seat == 0 else agent.head_bb
    elif hasattr(agent, "actor_critic"):
        head = agent.actor_critic.policy_value_network
    else:
        head = agent.policy_value_network  # plain PokerActorCritic

    with torch.no_grad():
        latent = representation_model(rep_input)
        logits, _ = head(latent)
        mask_t = torch.as_tensor(legal_mask, dtype=torch.bool, device=logits.device)
        masked_logits = apply_legal_mask(logits, mask_t)
        probs_t = Categorical(logits=masked_logits).probs

    return {
        _ACTION_TYPE_TO_ENGINE[i]: float(probs_t[i].item())
        for i in range(NUM_ACTIONS)
        if legal_mask[i] and i in _ACTION_TYPE_TO_ENGINE
    }


# ---------------------------------------------------------------------------
# Hand Grid helpers
# ---------------------------------------------------------------------------

_RANK_ORDER_HIGH_TO_LOW = "AKQJT98765432"
_RANK_CHAR_BY_ENGINE_RANK = {
    0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8",
    7: "9", 8: "T", 9: "J", 10: "Q", 11: "K", 12: "A",
}


def _canonical_hand_and_cell(c1: int, c2: int) -> Tuple[str, int, int]:
    """
    Map two engine card ids (0-51, rank = id % 13, suit = id // 13) to
    (canonical_hand_str, row, col) using the frontend's grid convention
    (ranks high-to-low A..2 on both axes; row==col pair, row<col
    suited, row>col offsuit).
    """
    r1, s1 = c1 % 13, c1 // 13
    r2, s2 = c2 % 13, c2 // 13

    hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
    hi_char = _RANK_CHAR_BY_ENGINE_RANK[hi]
    lo_char = _RANK_CHAR_BY_ENGINE_RANK[lo]

    row = _RANK_ORDER_HIGH_TO_LOW.index(hi_char)
    col = _RANK_ORDER_HIGH_TO_LOW.index(lo_char)

    if hi == lo:
        return f"{hi_char}{lo_char}", row, col  # pair, row == col

    suited = s1 == s2
    if suited:
        return f"{hi_char}{lo_char}s", row, col  # upper triangle
    else:
        return f"{hi_char}{lo_char}o", col, row  # lower triangle


def _get_action_probabilities_batch(
    agent, rl_seat: int, rep_inputs: list, legal_masks: list
) -> List[Dict[str, float]]:
    """
    Batched sibling of _get_action_probabilities(): one forward pass for
    every (rep_input, legal_mask) pair instead of one call each.

    Returns
    -------
    List[Dict[str, float]]
        One {"fold": p, "call": p, "all_in": p, ...} dict per example
        (only the legal actions for THAT row are included), same order
        as rep_inputs.
    """
    from poker_rl_lab.actions.abstract_action import ActionType, NUM_ACTIONS

    _ACTION_TYPE_TO_ENGINE = {
        int(ActionType.FOLD): "fold",
        int(ActionType.CHECK): "check",
        int(ActionType.CALL): "call",
        int(ActionType.BET_25): "bet_25",
        int(ActionType.BET_50): "bet_50",
        int(ActionType.BET_75): "bet_75",
        int(ActionType.BET_100): "bet_100",
        int(ActionType.BET_150): "bet_150",
        int(ActionType.ALL_IN): "all_in",
    }

    representation_model = getattr(agent, "representation_model", None)
    if representation_model is None and hasattr(agent, "actor_critic"):
        representation_model = agent.actor_critic.representation_model

    n = len(rep_inputs)

    if representation_model is None:
        # RandomFallbackAgent — uniform over each row's own legal actions.
        out = []
        for mask in legal_masks:
            legal_indices = [i for i, ok in enumerate(mask) if ok]
            k = len(legal_indices) or 1
            out.append({
                _ACTION_TYPE_TO_ENGINE[i]: 1.0 / k
                for i in legal_indices
                if i in _ACTION_TYPE_TO_ENGINE
            })
        return out

    import torch
    from torch.distributions import Categorical
    from poker_rl_lab.models.policy_value_network import apply_legal_mask

    if hasattr(agent, "head_sb") and hasattr(agent, "head_bb"):
        head = agent.head_sb if rl_seat == 0 else agent.head_bb
    elif hasattr(agent, "actor_critic"):
        head = agent.actor_critic.policy_value_network
    else:
        head = agent.policy_value_network

    with torch.no_grad():
        latents = representation_model.forward_batch(rep_inputs)          # (n, latent_dim)
        logits, _ = head(latents)                                          # (n, action_dim)
        mask_t = torch.as_tensor(legal_masks, dtype=torch.bool, device=logits.device)
        masked_logits = apply_legal_mask(logits, mask_t)
        probs_t = Categorical(logits=masked_logits).probs                  # (n, action_dim)

    out = []
    for row in range(n):
        row_probs = {}
        for i in range(NUM_ACTIONS):
            if legal_masks[row][i] and i in _ACTION_TYPE_TO_ENGINE:
                row_probs[_ACTION_TYPE_TO_ENGINE[i]] = float(probs_t[row, i].item())
        out.append(row_probs)
    return out


# ---------------------------------------------------------------------------
# Scoreboard
# ---------------------------------------------------------------------------

@dataclass
class Scoreboard:
    correct: int = 0
    incorrect: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Internal — not surfaced via as_dict(). Assigns each recorded
    # decision a stable id and keeps a full post-hand state snapshot
    # per id so a past hand can be replayed read-only later (see
    # get_snapshot() / TrainerService.get_history_snapshot()).
    _next_id: int = field(default=0, repr=False)
    _snapshots: Dict[int, Dict[str, Any]] = field(default_factory=dict, repr=False)

    def record(
        self,
        result: DecisionResult,
        extra: Dict[str, Any],
        state_snapshot: Dict[str, Any],
    ) -> int:
        """Records one graded decision. Returns the entry's id."""
        entry_id = self._next_id
        self._next_id += 1

        if result.correct:
            self.correct += 1
        else:
            self.incorrect += 1

        self.history.append({
            "id": entry_id,
            "correct": result.correct,
            "best_action": result.best_action,
            "hero_action": result.hero_action,
            "explanation": result.explanation,
            **extra,
        })
        self._snapshots[entry_id] = state_snapshot

        # cap history so it doesn't grow unbounded in a long session
        if len(self.history) > 200:
            dropped = self.history[:-200]
            self.history = self.history[-200:]
            for d in dropped:
                self._snapshots.pop(d["id"], None)

        return entry_id

    def get_snapshot(self, entry_id: int) -> Optional[Dict[str, Any]]:
        return self._snapshots.get(entry_id)

    def as_dict(self) -> Dict[str, Any]:
        total = self.correct + self.incorrect
        accuracy = round(self.correct / total, 4) if total else None
        return {
            "correct": self.correct,
            "incorrect": self.incorrect,
            "total": total,
            "accuracy": accuracy,
            "history": self.history[-20:],  # most recent 20 for display
        }

    def reset(self):
        self.correct = 0
        self.incorrect = 0
        self.history = []
        self._snapshots = {}
        # _next_id intentionally NOT reset — ids stay unique for the
        # life of the process even across scoreboard resets, so a stale
        # frontend reference to an id from before a reset fails cleanly
        # (404 via get_history_snapshot) instead of silently resolving
        # to an unrelated new entry that happens to reuse the same id.


# ---------------------------------------------------------------------------
# Per-scenario-type agent/adapter cache
# ---------------------------------------------------------------------------

class _AgentCache:
    def __init__(self):
        self._agents: Dict[str, Any] = {}
        self._adapter = None
        # One trace entry per pipeline step, per scenario key — see
        # _try_build_real_agent()'s docstring. Kept alongside the agent
        # so get_debug_info() can report exactly what happened at
        # build time rather than re-inferring it from the final object
        # (which can't distinguish "real checkpoint loaded" from "an
        # untrained network of the same class was silently built
        # instead", since both produce the same agent_class).
        self._traces: Dict[str, List[Dict[str, Any]]] = {}

    def get_agent(self, scenario: ScenarioConfig):
        if scenario.key not in self._agents:
            agent, trace = _try_build_real_agent(scenario)
            if agent is None:
                trace.append({
                    "step": "final_fallback",
                    "status": "fallback",
                    "detail": "Using RandomFallbackAgent (uniform random over legal actions).",
                })
                agent = RandomFallbackAgent()
            self._agents[scenario.key] = agent
            self._traces[scenario.key] = trace
            for entry in trace:
                print(f"[trainer_service:{scenario.key}] [{entry['status'].upper()}] "
                      f"{entry['step']}: {entry['detail']}")
        return self._agents[scenario.key]

    def get_trace(self, scenario_key: str) -> List[Dict[str, Any]]:
        return self._traces.get(scenario_key, [])

    def get_adapter(self):
        if self._adapter is None:
            from poker_rl_lab.representation.observation_adapter import ObservationAdapter
            self._adapter = ObservationAdapter()
        return self._adapter


def _try_build_real_agent(scenario: ScenarioConfig) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Attempt to build/load the real DualSeatActorCritic for this
    scenario's policy config, recording a step-by-step trace of every
    point along the way that could silently substitute a fallback.

    Why a trace and not just a final status
    -------------------------------------------
    There are THREE independent things that can each silently degrade
    without raising, and a caller only checking "is agent_class
    RandomFallbackAgent?" misses two of them entirely:

      1. Checkpoint weights: real file loaded vs. missing/failed vs.
         RandomFallbackAgent (no network at all).
      2. Hand encoder: the PRETRAINED base card-set encoder
         (representation/pretrained_hand_encoder.py) that the loaded
         checkpoint's own adapter layers (attn_scorer/project) were
         TRAINED ON TOP OF. If scenario.policy.encoder_path fails to
         load, the checkpoint's policy/value weights can still load
         successfully (they don't include the frozen TF base encoder's
         weights at all — see composite_hand_encoder.py's own
         docstring: "NOT registered as an nn.Module (TF-backed,
         frozen)") while the base encoder underneath silently becomes
         an UNTRAINED one. The resulting agent_class and
         checkpoint_exists_on_disk both look completely normal in that
         case — only the hand-encoder step of this trace catches it.
      3. Architecture variant mismatch: a checkpoint saved with one
         PolicyVariant loaded against the other raises inside
         load_state_dict (a real error, caught and traced below) — but
         a MATCHING variant with a genuinely wrong/stale checkpoint
         path would load "successfully" while pointing at the wrong
         model entirely; that can only be caught by cross-checking
         checkpoint_path against what you expect it to be, which is
         why get_debug_info() reports the full resolved absolute path.

    This function ALWAYS resolves hand_encoder explicitly (never
    passes hand_encoder=None into build_dual_seat_actor_critic /
    load_dual_seat_actor_critic) specifically so it never silently
    falls through to those functions' own hidden default — which
    reads from a hardcoded path
    (generic_tree_experiment._DEFAULT_ENCODER_PATH) that has nothing
    to do with scenario.policy.encoder_path — without this trace ever
    knowing that happened.

    Returns
    -------
    (agent_or_None, trace)
        agent is None if every step failed and the caller should fall
        back to RandomFallbackAgent (see _AgentCache.get_agent).
    """
    trace: List[Dict[str, Any]] = []

    def step(name, status, detail):
        trace.append({"step": name, "status": status, "detail": detail})

    try:
        from poker_rl_lab.experiments.generic_tree_experiment import (
            build_dual_seat_actor_critic,
            load_dual_seat_actor_critic,
            PolicyVariant,
        )
        step("import_rl_lab", "ok", "poker_rl_lab.experiments.generic_tree_experiment imported.")
    except ImportError as e:
        step("import_rl_lab", "error", f"RL Lab not importable: {e}")
        return None, trace

    import os

    checkpoint_path = scenario.policy.checkpoint_path
    abs_checkpoint_path = os.path.abspath(checkpoint_path)

    try:
        variant = PolicyVariant(scenario.policy.variant)
        step("parse_variant", "ok", f"variant={variant.value!r}")
    except ValueError as e:
        step("parse_variant", "error", f"unknown variant {scenario.policy.variant!r}: {e}")
        return None, trace

    # -- Hand encoder: resolved explicitly, never left as None --------
    hand_encoder = None
    encoder_source = None
    if not scenario.policy.encoder_path:
        step("hand_encoder", "warning",
             "no encoder_path configured for this scenario.")
    elif not os.path.exists(scenario.policy.encoder_path):
        step("hand_encoder", "warning",
             f"encoder_path {os.path.abspath(scenario.policy.encoder_path)!r} "
             f"does not exist on disk.")
    else:
        try:
            from poker_rl_lab.representation.pretrained_hand_encoder import PretrainedHandEncoder
            hand_encoder = PretrainedHandEncoder.from_file(
                scenario.policy.encoder_path,
                expected_embedding_dim=scenario.policy.encoder_embedding_dim,
            )
            encoder_source = "configured_pretrained_file"
            step("hand_encoder", "ok",
                 f"loaded pretrained hand encoder from "
                 f"{os.path.abspath(scenario.policy.encoder_path)} "
                 f"(embedding_dim={hand_encoder.embedding_dim}).")
        except Exception as e:
            step("hand_encoder", "error",
                 f"failed to load hand encoder from "
                 f"{os.path.abspath(scenario.policy.encoder_path)!r}: "
                 f"{type(e).__name__}: {e}")

    if hand_encoder is None:
        from poker_rl_lab.representation.pretrained_hand_encoder import PretrainedHandEncoder
        hand_encoder = PretrainedHandEncoder.untrained()
        encoder_source = "untrained_fallback"
        step("hand_encoder_fallback", "fallback",
             "Using an UNTRAINED base hand encoder — even if the "
             "checkpoint below loads without error, its policy/value "
             "weights were trained on top of REAL card embeddings and "
             "are now seeing random ones instead. This will NOT raise "
             "an exception and will NOT show up as agent_class == "
             "RandomFallbackAgent — check this step specifically.")

    # -- Checkpoint weights ---------------------------------------------
    if not os.path.exists(checkpoint_path):
        step("checkpoint", "warning", f"no file at {abs_checkpoint_path!r}.")
    else:
        try:
            agent = load_dual_seat_actor_critic(
                checkpoint_path, variant=variant, hand_encoder=hand_encoder
            )
            step("checkpoint", "ok",
                 f"loaded checkpoint from {abs_checkpoint_path} "
                 f"(variant={variant.value}, hand_encoder={encoder_source}, "
                 f"architecture=current).")
            agent._trainer_debug_encoder_source = encoder_source  # noqa: SLF001 — debug-only tag
            agent._trainer_debug_architecture = "current"  # noqa: SLF001
            return agent, trace
        except RuntimeError as e:
            msg = str(e)
            if "Missing key" in msg or "Unexpected key" in msg:
                # Not a corrupt file or wrong variant — the checkpoint's
                # actual saved shapes don't match what
                # actor_critic_factory.build_actor_critic() currently
                # constructs (CompositeHandEncoder/CompositeBoardEncoder).
                # This is exactly the signature of a checkpoint saved
                # BEFORE that refactor — see
                # generic_tree_experiment.py's "Legacy-architecture
                # builders/loaders" section. Try reconstructing that
                # older shape instead of giving up straight to
                # untrained.
                step("checkpoint", "warning",
                     f"file EXISTS at {abs_checkpoint_path!r} but its "
                     f"state_dict doesn't match the CURRENT architecture "
                     f"(missing/unexpected keys — see raw error below). "
                     f"This is the signature of a checkpoint saved before "
                     f"the CompositeHandEncoder/CompositeBoardEncoder "
                     f"refactor. Attempting a legacy-architecture load. "
                     f"Raw error: {msg}")
                try:
                    from poker_rl_lab.experiments.generic_tree_experiment import (
                        load_dual_seat_actor_critic_legacy,
                    )
                    agent = load_dual_seat_actor_critic_legacy(
                        checkpoint_path, variant=variant, hand_encoder=hand_encoder
                    )
                    step("checkpoint_legacy", "ok",
                         f"loaded checkpoint from {abs_checkpoint_path} "
                         f"using the LEGACY (pre-composite-encoder) "
                         f"architecture. This is a best-effort "
                         f"reconstruction using default hyperparameters "
                         f"for every component the mismatch doesn't "
                         f"directly implicate — if this checkpoint used "
                         f"non-default encoder sizes anywhere, this may "
                         f"still be silently wrong. Consider re-exporting "
                         f"a checkpoint against the current architecture "
                         f"when possible.")
                    agent._trainer_debug_encoder_source = encoder_source  # noqa: SLF001
                    agent._trainer_debug_architecture = "legacy"  # noqa: SLF001
                    return agent, trace
                except ImportError as e2:
                    step("checkpoint_legacy", "error",
                         f"legacy loader not available — add "
                         f"build_dual_seat_actor_critic_legacy/"
                         f"load_dual_seat_actor_critic_legacy to "
                         f"generic_tree_experiment.py. ({e2})")
                except Exception as e2:
                    step("checkpoint_legacy", "error",
                         f"legacy-architecture load also failed: "
                         f"{type(e2).__name__}: {e2}")
            else:
                step("checkpoint", "error",
                     f"file EXISTS at {abs_checkpoint_path!r} but failed "
                     f"to load: {type(e).__name__}: {e}")
        except Exception as e:
            step("checkpoint", "error",
                 f"file EXISTS at {abs_checkpoint_path!r} but failed to "
                 f"load: {type(e).__name__}: {e}")

    if not scenario.policy.allow_untrained_fallback:
        step("untrained_fallback", "error",
             "allow_untrained_fallback is False — no agent could be built.")
        return None, trace

    try:
        agent = build_dual_seat_actor_critic(variant, hand_encoder=hand_encoder)
        agent.eval()
        step("untrained_fallback", "fallback",
             f"Using an UNTRAINED policy network (architecture matches "
             f"variant={variant.value}, but random init — no real poker "
             f"knowledge) in place of {abs_checkpoint_path!r}.")
        agent._trainer_debug_encoder_source = encoder_source  # noqa: SLF001
        agent._trainer_debug_architecture = "current (untrained)"  # noqa: SLF001
        return agent, trace
    except Exception as e:
        step("untrained_fallback", "error",
             f"failed to build even an untrained agent: {type(e).__name__}: {e}")
        return None, trace


# ---------------------------------------------------------------------------
# TrainerService
# ---------------------------------------------------------------------------

class TrainerService:
    """
    Stateful (one active scenario at a time) per backend process —
    same singleton pattern as GameService / PushFoldService. If
    multi-user sessions are needed later, key this dict by session id;
    not required yet.
    """

    def __init__(self):

        self._cache = _AgentCache()
        self.scoreboard = Scoreboard()

        self.scenario_key: Optional[str] = None
        self.state: Optional[PokerState] = None
        self.hero_position: Optional[str] = None
        self.hero_effective_bb: Optional[float] = None
        self.awaiting_hero: bool = False

        # Cards as originally dealt this scenario, keyed by engine seat.
        # GameState.apply_action(FOLD) zeroes a folded player's hand_mask
        # (moving cards to the discard pile) — that's correct for the
        # Engine's own bookkeeping, but it means state_to_dto() would
        # render an empty hand for anyone who folded. The Trainer wants
        # folded hole cards to stay visible until "Next Scenario" is
        # clicked, so we snapshot them here at deal time and splice them
        # back into every get_state() response for the life of this
        # scenario.
        self._display_hole_cards: Dict[int, List[int]] = {}

        # Set by _grade_decision(), consumed by apply_hero_action() once
        # the hand has fully resolved — see both methods' docstrings for
        # why recording is deferred until after _progress_engine().
        self._pending_grade_result: Optional[DecisionResult] = None
        self._pending_grade_extra: Dict[str, Any] = {}

        # Reentrancy guard — apply_hero_action() mutates self.state in
        # place with no locking. A double-submitted action (double click,
        # a retried request after a slow/lost response, etc.) hitting this
        # singleton concurrently could step the engine twice for one hero
        # decision, desyncing whatever the frontend still thinks the
        # current state is (symptoms: action buttons stuck doing nothing,
        # or the hand silently completing with no "Next Scenario" button
        # ever rendered because the response the UI is holding is stale).
        self._busy: bool = False

    # ------------------------------------------------------------
    # Scenario listing
    # ------------------------------------------------------------

    def list_scenarios(self) -> List[Dict[str, Any]]:
        cfg = get_training_config()
        return [
            {"key": s.key, "label": s.label, "description": s.description}
            for s in cfg.list_scenarios()
        ]

    # ------------------------------------------------------------
    # New scenario
    # ------------------------------------------------------------

    def new_scenario(self, scenario_key: str) -> Dict[str, Any]:
        cfg = get_training_config().get_scenario(scenario_key)
        self.scenario_key = scenario_key

        game_def, rules = load_game(cfg.engine_variant)

        hero_bb = round(random.uniform(cfg.stack_bb.min, cfg.stack_bb.max), 1)
        if cfg.villain_matches_hero_range:
            villain_bb = round(random.uniform(cfg.stack_bb.min, cfg.stack_bb.max), 1)
        else:
            villain_bb = hero_bb

        big_blind = game_def.big_blind
        hero_chips = max(big_blind, round(hero_bb * big_blind))
        villain_chips = max(big_blind, round(villain_bb * big_blind))

        players = [None, None]
        players[cfg.hero_seat] = PlayerState(stack=hero_chips)
        players[cfg.villain_seat] = PlayerState(stack=villain_chips)

        state = PokerState(players, game_def, rules, CppScoringEngine(), callbacks=None)

        # Random position: dealer_position determines who's SB (SB =
        # (dealer_position + 1) % n, heads-up convention already used
        # by push_fold_service.py / poker_state.py). Choose dealer_position
        # so hero lands on the requested position.
        #
        # IMPORTANT: PokerState.start_hand() unconditionally does
        # `g.dealer_position = (g.dealer_position + 1) % n` before using
        # it (see poker_state.py) — it's designed to rotate the button
        # hand-to-hand on a persistent PokerState. Since we build a
        # brand-new PokerState per scenario here, that implicit +1 would
        # silently shift dealer_position by one seat, which in heads-up
        # (n=2) FLIPS SB and BB relative to what we just computed below.
        # push_fold_service.py accounts for this via its own
        # `_dealer_parity` pre-increment; we do the equivalent one-seat
        # pre-compensation here so the hero actually lands on
        # `hero_position` once start_hand() has run.
        hero_position = random.choice(cfg.positions)
        n = len(players)
        if hero_position == cfg.positions[0]:
            # "SB"/"OOP"-style first-position seat should be hero_seat
            # -> dealer_position = hero_seat - 1
            target_dealer_position = (cfg.hero_seat - 1) % n
        else:
            # "BB"/"IP"-style second-position seat should be hero_seat
            # -> dealer_position = hero_seat - 2
            target_dealer_position = (cfg.hero_seat - 2) % n

        state.game.dealer_position = (target_dealer_position - 1) % n

        state.start_hand()

        # river (and any future scenario declaring pot_bb) doesn't use
        # posted blinds at all — the "pot" represents dead money from
        # streets this drill doesn't model (see training_config.yaml's
        # own note on river_duo). start_hand() unconditionally posts
        # blinds regardless of betting_type, so undo that here and
        # replace it with a randomized dead pot before anyone acts.
        # This was documented in training_config.yaml but never
        # actually implemented — river hands were silently being dealt
        # as if 1/2 blinds had been posted instead of a randomized
        # stack-to-pot ratio.
        self._apply_dead_pot(cfg, game_def)

        # Defensive check: fail loudly in dev if this ever drifts again,
        # rather than silently serving the wrong action set to the
        # frontend (which is exactly how this bug manifested before).
        actual_sb_seat = (state.game.dealer_position + 1) % n
        actual_hero_position = cfg.positions[0] if actual_sb_seat == cfg.hero_seat else cfg.positions[1]
        assert actual_hero_position == hero_position, (
            f"dealer_position compensation drifted: wanted hero_position="
            f"{hero_position!r} but engine landed on {actual_hero_position!r} "
            f"(dealer_position={state.game.dealer_position})"
        )

        self.state = state
        self.hero_position = hero_position
        self.hero_effective_bb = round(
            min(hero_chips, villain_chips) / big_blind, 2
        )
        self.awaiting_hero = False
        self._pending_grade_result = None
        self._pending_grade_extra = {}

        # Snapshot hole cards now, before any fold can zero them out —
        # see _display_hole_cards' docstring above.
        self._display_hole_cards = {
            seat: mask_to_card_ids(self.state.game.players[seat].hand_mask)
            for seat in range(len(players))
        }

        # If hero is second-to-act, villain acts first — auto-run the agent.
        self._maybe_run_villain_turn(cfg)
        self.awaiting_hero = self._is_hero_turn(cfg)

        return self.get_state()

    def _apply_dead_pot(self, cfg: ScenarioConfig, game_def) -> None:
        """
        Overrides start_hand()'s blind-posting with a randomized dead
        pot, per a scenario's PotConfig(mode="randomized_bb") semantics
        (river_duo.yaml). No-op for scenarios that don't declare
        cfg.pot_bb (push-fold, which genuinely does use blinds).

        Must be called AFTER state.start_hand() (hole cards + board
        already dealt) but BEFORE any hero/villain action — refunds
        whatever was posted as blinds back to each player's stack,
        zeroes bet_to_call/current_bet, and sets game.pot to a fresh
        randomized dead-money amount with no per-seat attribution.
        """
        if cfg.pot_bb is None:
            return

        g = self.state.game
        big_blind = game_def.big_blind

        for p in g.players:
            g.pot -= p.current_bet
            p.stack += p.current_bet
            p.total_contribution -= p.current_bet
            p.current_bet = 0
            p.is_all_in = False

        dead_pot = round(random.uniform(cfg.pot_bb.min, cfg.pot_bb.max) * big_blind)
        g.pot += dead_pot
        g.bet_to_call = 0
        g.min_raise = big_blind
        g.raises_this_street = 0

    # ------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------

    def get_debug_info(self, scenario_key: str) -> Dict[str, Any]:
        """
        Full step-by-step provenance of what's actually backing a
        scenario's agent — see _try_build_real_agent()'s docstring for
        why a single "agent_class" check isn't enough: a checkpoint can
        load "successfully" while its hand encoder silently fell back
        to untrained weights, which raises nothing and doesn't change
        agent_class at all.

        `pipeline_trace` is the ordered list of every step that ran
        while building this scenario's agent (import, variant parse,
        hand encoder resolution, checkpoint load, any fallback taken),
        each tagged "ok" / "warning" / "fallback" / "error". Read it
        top to bottom — the first non-"ok" entry is where things
        actually diverged from "the real trained checkpoint, exactly
        as configured".

        If the agent was already built earlier in this process's
        lifetime (cached in _AgentCache), this returns the trace
        recorded at THAT build time — it does not rebuild or reload
        anything, so it's safe to call repeatedly without triggering
        redundant checkpoint loads.
        """
        import os

        cfg = get_training_config().get_scenario(scenario_key)
        agent = self._cache.get_agent(cfg)  # builds + caches trace on first call
        trace = self._cache.get_trace(scenario_key)

        agent_class = type(agent).__name__
        is_fallback = isinstance(agent, RandomFallbackAgent)

        representation_model = getattr(agent, "representation_model", None)
        if representation_model is None and hasattr(agent, "actor_critic"):
            representation_model = agent.actor_critic.representation_model

        hand_encoder_info = None
        if representation_model is not None:
            hand_encoder = getattr(representation_model, "hand_encoder", None)
            if hand_encoder is not None:
                base = getattr(hand_encoder, "base_encoder", hand_encoder)
                hand_encoder_info = {
                    "class": type(hand_encoder).__name__,
                    "base_class": type(base).__name__,
                    "embedding_dim": getattr(base, "embedding_dim", None),
                    # Set explicitly by _try_build_real_agent — the only
                    # reliable way to know whether this is the
                    # configured pretrained file or the untrained
                    # fallback, since both produce the same class.
                    "source": getattr(agent, "_trainer_debug_encoder_source", "unknown"),
                }

        # Overall verdict, derived from the trace rather than just
        # agent_class — this is what actually answers "is the grid
        # I'm looking at backed by the real trained model end-to-end".
        used_legacy = any(e["step"] == "checkpoint_legacy" and e["status"] == "ok" for e in trace)
        has_fallback_step = any(e["status"] in ("fallback", "error") for e in trace)
        if is_fallback:
            verdict = "RANDOM — no network at all, every action equally likely."
        elif used_legacy:
            verdict = "LEGACY — trained checkpoint weights loaded, but via a reconstructed pre-refactor architecture (see pipeline_trace) rather than the current one. Predictions come from real trained weights, but may not exactly match what a checkpoint re-exported against current code would produce."
        elif has_fallback_step:
            verdict = "PARTIAL — a real network is running, but at least one component (see pipeline_trace) fell back to untrained/default instead of your configured checkpoint/encoder."
        else:
            verdict = "REAL — checkpoint and hand encoder both loaded exactly as configured."

        checkpoint_path = cfg.policy.checkpoint_path
        return {
            "scenario_key": scenario_key,
            "verdict": verdict,
            "architecture": getattr(agent, "_trainer_debug_architecture", "unknown"),
            "betting_type": cfg.betting_type,
            "positions": cfg.positions,
            "checkpoint_path": checkpoint_path,
            "checkpoint_abs_path": os.path.abspath(checkpoint_path),
            "checkpoint_exists_on_disk": os.path.exists(checkpoint_path),
            "configured_variant": cfg.policy.variant,
            "allow_untrained_fallback": cfg.policy.allow_untrained_fallback,
            "agent_class": agent_class,
            "is_random_fallback": is_fallback,
            "hand_encoder": hand_encoder_info,
            "encoder_path": cfg.policy.encoder_path,
            "encoder_path_exists_on_disk": (
                os.path.exists(cfg.policy.encoder_path) if cfg.policy.encoder_path else False
            ),
            "pipeline_trace": trace,
        }

    def get_live_debug_info(self) -> Dict[str, Any]:
        """
        Same as get_debug_info(), plus a live forward pass at the
        CURRENT scenario's actual decision point, reporting the raw
        action-probability distribution the network just produced.

        This catches a failure mode none of the build-time trace can:
        everything loads without error (real checkpoint, real hand
        encoder, matching variant) but the OBSERVATION being fed in is
        subtly wrong (wrong pot/stack units, wrong legal_mask, wrong
        history event) — the trace above would show a clean "REAL"
        verdict while the grid is still nonsense, because that's a
        runtime-input problem, not a load-time problem. A close-to-
        uniform distribution here despite a REAL verdict from
        get_debug_info() points at the observation construction
        (_build_obs*) rather than the checkpoint/encoder.
        """
        if self.state is None or self.scenario_key is None:
            raise ValueError("No active trainer scenario. Call new_scenario() first.")

        cfg = get_training_config().get_scenario(self.scenario_key)
        info = self.get_debug_info(self.scenario_key)

        hero_rl_seat = self._rl_seat_for_position(cfg, self.hero_position)
        obs = self._build_obs(
            cfg, seat=cfg.hero_seat, rl_seat=hero_rl_seat, position=self.hero_position
        )
        agent = self._cache.get_agent(cfg)
        rep_input = self._cache.get_adapter().adapt(obs)
        action_probs = _get_action_probabilities(agent, hero_rl_seat, rep_input, obs.legal_mask)

        info["live_observation"] = {
            "hero_position": self.hero_position,
            "hero_rl_seat": hero_rl_seat,
            "legal_mask_true_indices": [i for i, ok in enumerate(obs.legal_mask) if ok],
            "pot": obs.betting.pot,
            "bet_to_call": obs.betting.bet_to_call,
            "hole_cards": list(obs.hole_cards.cards),
            "board_cards": list(obs.board.node_cards),
        }
        info["live_action_probs"] = action_probs
        return info

    # ------------------------------------------------------------
    # Hero decision
    # ------------------------------------------------------------

    def apply_hero_action(self, action_type: str) -> Dict[str, Any]:
        if self.state is None or self.scenario_key is None:
            raise ValueError("No active trainer scenario. Call new_scenario() first.")

        if self._busy:
            raise ValueError(
                "A previous action is still being processed for this scenario."
            )
        self._busy = True
        try:
            cfg = get_training_config().get_scenario(self.scenario_key)

            if not self._is_hero_turn(cfg):
                raise ValueError("Not the hero's turn to act.")

            allowed = self._live_allowed_actions(cfg)
            if action_type not in allowed:
                raise ValueError(
                    f"Action {action_type!r} not legal right now for "
                    f"{self.hero_position} in scenario {self.scenario_key!r} "
                    f"(currently legal: {allowed})"
                )

            # Grades against the model's forward pass at the DECISION
            # point (before chips move) — see _grade_decision(). The
            # scoreboard entry itself is recorded further below, once
            # the hand has fully resolved, so its snapshot reflects the
            # finished hand rather than the split-second before the
            # hero's action was applied.
            result = self._grade_decision(cfg, action_type)

            engine_action = self._build_engine_action(cfg.hero_seat, action_type)
            self.state.step(engine_action)

            self._maybe_run_villain_turn(cfg)
            self._progress_engine()
            self.awaiting_hero = self._is_hero_turn(cfg)

            payload = self.get_state()
            payload["last_decision"] = {
                "correct": result.correct,
                "best_action": result.best_action,
                "hero_action": result.hero_action,
                "explanation": result.explanation,
            }

            entry_id = self.scoreboard.record(
                result,
                extra=self._pending_grade_extra,
                state_snapshot=payload,
            )
            payload["scoreboard_entry_id"] = entry_id
            payload["scoreboard"] = self.scoreboard.as_dict()
            return payload
        finally:
            self._busy = False

    def reset_scoreboard(self) -> Dict[str, Any]:
        self.scoreboard.reset()
        return self.scoreboard.as_dict()

    def get_history_snapshot(self, entry_id: int) -> Dict[str, Any]:
        """
        Read-only replay of a completed scenario from the scoreboard
        history. Forces awaiting_hero False (and flags
        is_history_replay) so the frontend never re-offers a decision
        on an already-graded hand.
        """
        snap = self.scoreboard.get_snapshot(entry_id)
        if snap is None:
            raise ValueError(f"No scoreboard history entry with id {entry_id}")
        replay = dict(snap)
        replay["awaiting_hero"] = False
        replay["is_history_replay"] = True
        return replay

    # ------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------

    def _grade_decision(self, cfg: ScenarioConfig, hero_action: str) -> DecisionResult:
        hole_ids = mask_to_card_ids(self.state.game.players[cfg.hero_seat].hand_mask)
        hole_strs = [str(CardObj(cid)) for cid in hole_ids]

        hero_rl_seat = self._rl_seat_for_position(cfg, self.hero_position)
        obs = self._build_obs(
            cfg, seat=cfg.hero_seat, rl_seat=hero_rl_seat, position=self.hero_position
        )

        agent = self._cache.get_agent(cfg)
        rep_input = self._cache.get_adapter().adapt(obs)
        action_probs = _get_action_probabilities(agent, hero_rl_seat, rep_input, obs.legal_mask)

        ctx = DecisionContext(
            hole_cards=hole_strs,
            position=self.hero_position,
            effective_stack_bb=self.hero_effective_bb,
            hero_action=hero_action,
            allowed_actions=cfg.actions.get(self.hero_position, []),
            action_probs=action_probs,
        )

        evaluator = get_evaluator(cfg.evaluator)
        result = evaluator.evaluate(ctx)

        # Stashed for apply_hero_action() to record on the scoreboard
        # once the hand is fully resolved — see that method.
        self._pending_grade_extra = {
            "hand": canonicalize_hole_cards(hole_strs),
            "position": self.hero_position,
            "effective_stack_bb": self.hero_effective_bb,
            "scenario": cfg.key,
            "policy_probs": action_probs,
        }
        self._pending_grade_result = result
        return result

    # ------------------------------------------------------------
    # Hand Grid
    # ------------------------------------------------------------

    def get_hand_grid(self) -> Dict[str, Any]:
        """
        Compute the model's per-action probabilities for all 169
        canonical starting hands, evaluated at the CURRENT scenario's
        decision point (same position, same stack/pot, same villain
        state) — with the hero's actual hole cards swapped out for
        every possible combo.

        Returns
        -------
        dict
            {
              "position": "SB" | "BB" | "OOP" | "IP",
              "action_order": [...],                   # stable render order
              "hero_effective_bb": float,
              "action_grid": [                          # 13x13, row-major
                  [ {"fold": 0.9, "all_in": 0.1}, ... x13 ],
                  ...
              ],
              "combos": {
                  "AKs": {
                      "combos": [
                          {"cards": ["Ah","Kh"], "probs": {"fold":.., "all_in":..}},
                          ...
                      ]
                  },
                  ...
              }
            }
        """
        if self.state is None or self.scenario_key is None:
            raise ValueError("No active trainer scenario. Call new_scenario() first.")

        cfg = get_training_config().get_scenario(self.scenario_key)
        hero_rl_seat = self._rl_seat_for_position(cfg, self.hero_position)
        action_order = self._live_allowed_actions(cfg)
        if not action_order:
            raise ValueError(
                f"No legal hero actions at the current decision point for "
                f"scenario {self.scenario_key!r} — cannot build a hand grid "
                f"(is it actually the hero's turn?)."
            )

        agent = self._cache.get_agent(cfg)
        adapter = self._cache.get_adapter()

        # Template observation at the CURRENT decision point (real
        # stack/pot/villain state) — only hole_cards differs per combo.
        template_obs = self._build_obs(
            cfg, seat=cfg.hero_seat, rl_seat=hero_rl_seat, position=self.hero_position
        )

        combos = list(itertools.combinations(range(52), 2))

        obs_list = [
            _dataclasses_replace(template_obs, hole_cards=HoleCardsObs(cards=(c1, c2)))
            for (c1, c2) in combos
        ]
        rep_inputs = [adapter.adapt(o) for o in obs_list]
        legal_masks = [o.legal_mask for o in obs_list]

        probs_per_combo = _get_action_probabilities_batch(
            agent, hero_rl_seat, rep_inputs, legal_masks
        )

        # ---- bucket into the 169-cell grid, summing per action --------
        action_sum: List[List[Dict[str, float]]] = [
            [dict() for _ in range(13)] for _ in range(13)
        ]
        cell_count = [[0] * 13 for _ in range(13)]
        combos_by_hand: Dict[str, List[Dict[str, Any]]] = {}

        for (c1, c2), probs in zip(combos, probs_per_combo):
            hand_str, row, col = _canonical_hand_and_cell(c1, c2)

            cell_count[row][col] += 1
            cell_sums = action_sum[row][col]
            for action, p in probs.items():
                cell_sums[action] = cell_sums.get(action, 0.0) + p

            combos_by_hand.setdefault(hand_str, []).append({
                "cards": [str(CardObj(c1)), str(CardObj(c2))],
                "probs": probs,
            })

        action_grid = [
            [
                (
                    {
                        action: (action_sum[r][c].get(action, 0.0) / cell_count[r][c])
                        for action in action_order
                    }
                    if cell_count[r][c] else {action: 0.0 for action in action_order}
                )
                for c in range(13)
            ]
            for r in range(13)
        ]

        return {
            "position": self.hero_position,
            "action_order": action_order,
            "hero_effective_bb": self.hero_effective_bb,
            "action_grid": action_grid,
            "combos": {
                hand: {"combos": entries} for hand, entries in combos_by_hand.items()
            },
        }

    # ------------------------------------------------------------
    # Engine helpers
    # ------------------------------------------------------------

    def _live_allowed_actions(self, cfg: ScenarioConfig) -> List[str]:
        """
        The Trainer-level action strings actually legal for the hero
        RIGHT NOW, derived from live betting state rather than
        training_config.yaml's static per-position action list.

        training_config.yaml's `actions:` block is documentation of the
        UNION of actions a position could ever face across the whole
        scenario (e.g. river's OOP sees {check, bet} on the opening
        node but {call, fold} if it checked and IP bet) — it was
        previously being sent to the frontend and validated against
        directly, which is why every river decision offered all four
        buttons regardless of node, and why picking an actually-illegal
        one (e.g. "bet" while already facing a bet) 400'd out of
        _build_engine_action.

        push-fold is unaffected: there is exactly one hero decision
        point per hand for that scenario type, so the static list and
        the live list always coincide — kept as a direct passthrough
        rather than re-deriving it from to_call, since push-fold's
        static list already matches push_fold_service.py's own
        SB={fold,all_in}/BB={fold,call} convention exactly.
        """
        if cfg.betting_type == "push_fold":
            return cfg.actions.get(self.hero_position, [])

        if self.state is None or self.state.phase != Phase.BETTING:
            return []

        g = self.state.game
        player = g.players[cfg.hero_seat]
        to_call = max(g.bet_to_call - player.current_bet, 0)

        candidates = ("call", "fold") if to_call > 0 else ("check", "bet")
        declared = cfg.actions.get(self.hero_position, [])
        return [a for a in candidates if a in declared]

    def _rl_seat_for_position(self, cfg: ScenarioConfig, position: str) -> int:
        """
        Map a position label ("SB"/"BB", "OOP"/"IP", ...) to the RL
        seat index (0/1) the policy was trained against, using the
        scenario's own cfg.positions ordering rather than assuming
        SB/BB specifically — this is what lets river's OOP/IP labels
        (and any future scenario's own labels) work without special
        casing here.
        """
        try:
            return cfg.positions.index(position)
        except ValueError:
            raise ValueError(
                f"position {position!r} is not one of scenario "
                f"{cfg.key!r}'s declared positions {cfg.positions!r}"
            ) from None

    def _is_hero_turn(self, cfg: ScenarioConfig) -> bool:
        return (
            self.state.phase == Phase.BETTING
            and self.state.game.current_player == cfg.hero_seat
        )

    def _is_villain_turn(self, cfg: ScenarioConfig) -> bool:
        return (
            self.state.phase == Phase.BETTING
            and self.state.game.current_player == cfg.villain_seat
        )

    def _progress_engine(self):
        """
        Auto-advance past states nobody needs to act in.

        Two situations drive this loop:

          1. DEAL_BOARD -> deal the next street automatically.
          2. BETTING with every remaining player already all-in ->
             nobody CAN act (PokerState.step() requires a real action
             for BETTING, so step(None) isn't valid here); force the
             phase to DEAL_BOARD ourselves so case 1 deals out the rest
             of the board and the engine reaches SHOWDOWN on its own.

        The Trainer grades the hero's decision the instant it's made
        and does not display a board runout — so once SHOWDOWN is
        reached we immediately close the hand out to HAND_COMPLETE
        ourselves (PokerState.step() deliberately leaves that last
        transition to the caller — see its own comment — since other
        callers, e.g. the main GameSimulator, want to pause on
        SHOWDOWN to reveal the board turn by turn; the Trainer doesn't).
        """
        while True:
            if self.state.phase == Phase.DEAL_BOARD:
                self.state.step(None)
                continue

            if self.state.phase == Phase.BETTING:
                g = self.state.game
                remaining = [p for p in g.players if not p.has_folded]
                actionable = [p for p in remaining if not p.is_all_in]
                if len(remaining) > 1 and not actionable:
                    self.state.phase = Phase.DEAL_BOARD
                    continue

            if self.state.phase == Phase.SHOWDOWN:
                self.state.phase = Phase.HAND_COMPLETE

            break

    def _build_engine_action(self, seat: int, action_type: str) -> EngineAction:
        """
        Translate a Trainer-level action string into a concrete Engine
        Action, sized correctly for whatever the acting player actually
        faces.

        Handles every action the various scenario configs can offer:
        fold / check / call / bet (single pot-sized bet, river-style,
        only legal as the opening action of a betting round) / all_in
        (full-stack shove, used by push-fold and as river's "raise").

        Previously this method only had explicit branches for "fold"
        and "call" — "check" and "bet" both silently fell through to
        the trailing all-in branch, meaning a river Check click was
        actually shoving the hero's entire stack. That's fixed here.
        """
        g = self.state.game
        player = g.players[seat]
        to_call = max(g.bet_to_call - player.current_bet, 0)

        if action_type == "fold":
            return EngineAction(type=EngineActionType.FOLD, amount=None)

        if action_type == "check":
            if to_call > 0:
                raise ValueError("Cannot check while facing a bet.")
            return EngineAction(type=EngineActionType.CHECK, amount=None)

        if action_type == "call":
            if to_call <= 0:
                return EngineAction(type=EngineActionType.CHECK, amount=None)
            return EngineAction(type=EngineActionType.CALL, amount=min(to_call, player.stack))

        if action_type == "bet":
            # river_only: a single pot-sized bet — the only bet size the
            # river_duo-trained policy ever saw (BET_100 in
            # GenericTreeEnv._SUPPORTED_BET_TYPES) — only ever legal as
            # the OPENING action of the street.
            if to_call > 0:
                raise ValueError("Cannot 'bet' while facing a bet — use 'call' or 'fold'.")
            total = player.current_bet + min(g.pot, player.stack)
            return EngineAction(type=EngineActionType.BET, amount=total)

        if action_type == "all_in":
            shove_total = player.current_bet + player.stack
            act_type = EngineActionType.RAISE if to_call > 0 else EngineActionType.BET
            return EngineAction(type=act_type, amount=shove_total)

        raise ValueError(f"Unknown action_type: {action_type!r}")

    # ------------------------------------------------------------
    # Villain (RL agent) turn
    # ------------------------------------------------------------

    def _maybe_run_villain_turn(self, cfg: ScenarioConfig):
        while self._is_villain_turn(cfg):
            self._villain_act(cfg)

    def _villain_act(self, cfg: ScenarioConfig):
        villain_position = cfg.positions[1] if self.hero_position == cfg.positions[0] else cfg.positions[0]
        rl_seat = self._rl_seat_for_position(cfg, villain_position)

        obs = self._build_obs(cfg, seat=cfg.villain_seat, rl_seat=rl_seat, position=villain_position)

        agent = self._cache.get_agent(cfg)
        rep_input = self._cache.get_adapter().adapt(obs)
        result = agent.act(seat=rl_seat, obs=rep_input, legal_mask=obs.legal_mask, deterministic=True)

        engine_action_type = self._abstract_action_to_engine_string(result.action_type, obs.legal_mask)
        self.state.step(self._build_engine_action(cfg.villain_seat, engine_action_type))

    @staticmethod
    def _abstract_action_to_engine_string(action_type_value: int, legal_mask) -> str:
        """
        Map the RL Lab's sampled ActionType (an int) back onto one of
        the Trainer-level action strings _build_engine_action()
        understands. Falls back to "fold" if the sampled index somehow
        isn't one of the strings this Trainer speaks (defensive — every
        legal_mask this Trainer builds only ever marks a subset of
        {FOLD, CHECK, CALL, BET_100, ALL_IN} as legal in the first
        place, so this should be unreachable in practice).
        """
        from poker_rl_lab.actions.abstract_action import ActionType

        mapping = {
            int(ActionType.FOLD): "fold",
            int(ActionType.CHECK): "check",
            int(ActionType.CALL): "call",
            int(ActionType.BET_100): "bet",
            int(ActionType.ALL_IN): "all_in",
        }
        return mapping.get(action_type_value, "fold")

    # ------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------

    def _build_obs(self, cfg: ScenarioConfig, seat: int, rl_seat: int, position: str):
        """Build a StructuredObservation for `seat`, playing `position`."""
        if cfg.betting_type == "push_fold":
            return self._build_obs_push_fold(cfg, seat, rl_seat, position)
        return self._build_obs_generic(cfg, seat, rl_seat, position)

    # -- push-fold: hand-tuned obs matching the trained policy's own --
    # -- observation construction (push_fold_service.py's own          --
    # -- _build_sb_obs/_build_bb_obs) — unchanged from before.          --
    def _build_obs_push_fold(self, cfg: ScenarioConfig, seat: int, rl_seat: int, position: str):
        from poker_rl_lab.actions.abstract_action import ActionType, NUM_ACTIONS
        from poker_rl_lab.envs.observations import (
            BettingObs, BoardObs, HistoryEventObs, HistoryObs, HoleCardsObs,
            PlayerObs, StructuredObservation,
        )

        g = self.state.game
        sb_seat = cfg.hero_seat if self.hero_position == "SB" else cfg.villain_seat
        bb_seat = cfg.villain_seat if sb_seat == cfg.hero_seat else cfg.hero_seat
        sb_player, bb_player = g.players[sb_seat], g.players[bb_seat]
        hole_ids = mask_to_card_ids(g.players[seat].hand_mask)

        if position == "SB":
            mask = [False] * NUM_ACTIONS
            mask[ActionType.FOLD] = True
            mask[ActionType.ALL_IN] = True
            legal_mask = tuple(mask)
            history = HistoryObs(events=())
            bet_to_call = g.bet_to_call - sb_player.current_bet
        else:
            mask = [False] * NUM_ACTIONS
            mask[ActionType.FOLD] = True
            mask[ActionType.CALL] = True
            legal_mask = tuple(mask)

            pot_before_push = cfg.small_blind + cfg.big_blind
            push_amount = sb_player.current_bet - cfg.small_blind
            pot_fraction = max(0.0, push_amount / pot_before_push) if pot_before_push > 0 else 0.0
            push_event = HistoryEventObs(
                seat=0, action_type=int(ActionType.ALL_IN), street_index=0,
                pot_fraction=pot_fraction, is_aggressor=True,
            )
            history = HistoryObs(events=(push_event,))
            bet_to_call = g.bet_to_call - bb_player.current_bet

        players = (
            PlayerObs(seat=0, stack=sb_player.stack, current_bet=sb_player.current_bet,
                      has_folded=False, is_all_in=sb_player.is_all_in, is_hero=(sb_seat == seat)),
            PlayerObs(seat=1, stack=bb_player.stack, current_bet=bb_player.current_bet,
                      has_folded=False, is_all_in=bb_player.is_all_in, is_hero=(bb_seat == seat)),
        )

        return StructuredObservation(
            hero_seat=rl_seat,
            hole_cards=HoleCardsObs(cards=tuple(hole_ids)),
            board=BoardObs(node_cards=tuple()),
            players=players,
            betting=BettingObs(
                pot=g.pot, bet_to_call=bet_to_call, min_raise=g.min_raise,
                street_index=0, raises_this_street=g.raises_this_street,
            ),
            legal_mask=legal_mask,
            game_name=cfg.rl_game_name,
            history=history,
            dealer_position=0,
            is_terminal=False,
        )

    # -- generic (river and anything else): ask the ENGINE what's --
    # -- legal rather than hardcoding a position-keyed mask.       --
    def _build_obs_generic(self, cfg: ScenarioConfig, seat: int, rl_seat: int, position: str):
        from poker_rl_lab.actions.abstract_action import ActionType, NUM_ACTIONS
        from poker_rl_lab.envs.observations import (
            BettingObs, BoardObs, HistoryObs, HoleCardsObs, PlayerObs, StructuredObservation,
        )

        g = self.state.game
        player = g.players[seat]
        to_call = max(g.bet_to_call - player.current_bet, 0)

        engine_actions = (
            [a.name.lower() for a in g.legal_actions()]
            if self.state.phase.name == "BETTING"
            else []
        )

        # Map engine legal-action names directly onto the abstract
        # action vocabulary the RL Lab actually trained against for
        # generic-tree-shaped games (see GenericTreeEnv._SUPPORTED_BET_TYPES
        # — only BET_100 and ALL_IN are ever real bet-type actions in
        # these configs). Deliberately NOT using
        # actions.abstract_action.legal_action_mask() here: that helper
        # marks every BET_25/50/75/150 pot-fraction as "legal" too,
        # which is out-of-distribution input for a policy that has
        # never seen those sizes as options and would skew the grid.
        _ENGINE_TO_ABSTRACT = {
            "fold": ActionType.FOLD,
            "check": ActionType.CHECK,
            "call": ActionType.CALL,
            "bet": ActionType.BET_100,
            "raise": ActionType.ALL_IN,
        }
        mask = [False] * NUM_ACTIONS
        for name in engine_actions:
            atype = _ENGINE_TO_ABSTRACT.get(name)
            if atype is not None:
                mask[atype] = True
        legal_mask = tuple(mask)

        players = tuple(
            PlayerObs(
                seat=i,
                stack=p.stack,
                current_bet=p.current_bet,
                has_folded=p.has_folded,
                is_all_in=p.is_all_in,
                is_hero=(i == seat),
            )
            for i, p in enumerate(g.players)
        )

        return StructuredObservation(
            hero_seat=rl_seat,
            hole_cards=HoleCardsObs(cards=tuple(mask_to_card_ids(player.hand_mask))),
            board=BoardObs(node_cards=tuple(g.node_cards)),
            players=players,
            betting=BettingObs(
                pot=g.pot, bet_to_call=to_call, min_raise=g.min_raise,
                street_index=g.street_index, raises_this_street=g.raises_this_street,
            ),
            legal_mask=legal_mask,
            game_name=cfg.rl_game_name,
            history=HistoryObs(events=()),
            dealer_position=0,
            is_terminal=False,
        )

    # ------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        if self.state is None:
            return {"active": False}

        from app.engine_adapter import state_to_dto

        cfg = get_training_config().get_scenario(self.scenario_key)
        dto = state_to_dto(self.state)

        if self.state.phase in (Phase.SHOWDOWN, Phase.HAND_COMPLETE):
            dto.winners = [w + 1 for w in getattr(self.state, "last_winners", [])]

        # Restore hole cards for anyone who folded — the Engine zeroes
        # a folded player's hand_mask (see _display_hole_cards' note
        # in __init__), but the Trainer wants cards to stay on screen
        # until the user explicitly starts the next scenario.
        for i, p in enumerate(dto.players):
            engine_player = self.state.game.players[i]
            if engine_player.has_folded and i in self._display_hole_cards:
                p.hand = [str(CardObj(cid)) for cid in self._display_hole_cards[i]]

        payload = dto.dict()

        # push-fold deals a full board internally so ShowdownResolver
        # can settle all-in equity, but that runout is never meant to
        # be shown (the hand is decided preflop) — blank it. Any OTHER
        # scenario (river, ...) deals a real board the hero is supposed
        # to see, so only push-fold gets blanked here.
        if cfg.betting_type == "push_fold":
            payload["nodes"] = []
            payload["discard_pile"] = []

        # ------------------------------------------------------------
        # Seat remap: engine_adapter.state_to_dto (shared with the main
        # GameSimulator) numbers seats sequentially as engine_seat + 1.
        # The Trainer instead wants a FIXED visual convention — hero
        # always seat 1, the AI opponent always seat 4 — regardless of
        # which underlying engine seat (0 or 1) each currently sits in
        # (that flips hand-to-hand along with SB/BB). Remapped here
        # rather than in engine_adapter.py, since that module's generic
        # convention is still used elsewhere and shouldn't change.
        # ------------------------------------------------------------
        engine_seat_to_display = {
            cfg.hero_seat: _HERO_DISPLAY_SEAT,
            cfg.villain_seat: _AI_DISPLAY_SEAT,
        }
        for p in payload["players"]:
            engine_seat = p["seat"] - 1  # dto seat is engine_seat + 1
            p["seat"] = engine_seat_to_display.get(engine_seat, p["seat"])

        if payload.get("winners"):
            payload["winners"] = [
                engine_seat_to_display.get(w - 1, w) for w in payload["winners"]
            ]

        payload.update({
            "active": True,
            "scenario": self.scenario_key,
            "hero_seat": _HERO_DISPLAY_SEAT,
            "ai_seat": _AI_DISPLAY_SEAT,
            "villain_seat": _AI_DISPLAY_SEAT,  # kept as an alias for anything still reading this name
            "hero_position": self.hero_position,
            "hero_effective_bb": self.hero_effective_bb,
            "awaiting_hero": self.awaiting_hero,
            "hero_allowed_actions": self._live_allowed_actions(cfg),
            "hand_over": self.state.phase == Phase.HAND_COMPLETE,
            "scoreboard": self.scoreboard.as_dict(),
        })
        return payload


trainer_service = TrainerService()