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

  - anything else (river, single_bet, and any future generic-tree-
    shaped scenario) -> _build_obs_generic(): asks the ENGINE what's
    actually legal right now (GameState.legal_actions()) and maps
    those names directly onto the abstract action vocabulary the RL
    Lab trained against, rather than assuming a fixed position-keyed
    action set. Board cards come straight from GameState.node_cards,
    so whatever the variant's yaml actually deals is what the model
    sees.

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
    format_hero_hand_display,
    get_evaluator,
)

HERO_ROLE_TO_ENGINE_ACTION = {
    ("SB", "fold"): "fold",
    ("SB", "all_in"): "all_in",
    ("BB", "fold"): "fold",
    ("BB", "call"): "call",
}


# ---------------------------------------------------------------------------
# Abstract action vocabulary helpers
# ---------------------------------------------------------------------------
#
# The RL Lab's abstract ActionType vocabulary (poker_rl_lab.actions.
# abstract_action.ActionType) is fixed — one name per network output head
# (fold, check, call, bet_25 .. bet_150, all_in) — and is NOT scenario
# specific. What IS scenario specific is which of those abstract names a
# given Trainer-level action string (the button the frontend shows, the
# name in training_config.yaml's `actions:`/`action_map:` blocks) actually
# corresponds to for THIS scenario — see ScenarioConfig.action_map /
# .abstract_to_trainer_action() / .trainer_to_abstract() /
# .pot_fraction_for() in trainer_config.py.
#
# These two helpers only deal with the fixed, scenario-independent half
# (int index <-> abstract name); every place that needs the scenario-
# specific half goes through the relevant ScenarioConfig method instead of
# hardcoding a name here.

_ABSTRACT_ACTION_NAMES_CACHE: Dict[int, str] = {}


def _abstract_action_names() -> Dict[int, str]:
    """{ActionType int -> abstract name}, e.g. {..., 5: "bet_100", ...}."""
    global _ABSTRACT_ACTION_NAMES_CACHE
    if not _ABSTRACT_ACTION_NAMES_CACHE:
        from poker_rl_lab.actions.abstract_action import ActionType
        _ABSTRACT_ACTION_NAMES_CACHE = {
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
    return _ABSTRACT_ACTION_NAMES_CACHE


_ABSTRACT_ACTION_NAME_TO_INDEX_CACHE: Dict[str, int] = {}


def _abstract_action_name_to_index() -> Dict[str, int]:
    """Inverse of _abstract_action_names(): {abstract name -> ActionType int}."""
    global _ABSTRACT_ACTION_NAME_TO_INDEX_CACHE
    if not _ABSTRACT_ACTION_NAME_TO_INDEX_CACHE:
        _ABSTRACT_ACTION_NAME_TO_INDEX_CACHE = {
            name: idx for idx, name in _abstract_action_names().items()
        }
    return _ABSTRACT_ACTION_NAME_TO_INDEX_CACHE


def _probs_to_trainer_vocabulary(
    cfg: ScenarioConfig, probs_by_abstract_name: Dict[str, float]
) -> Dict[str, float]:
    """
    Translate {abstract_name: prob} (e.g. "bet_100") into this scenario's
    own Trainer-level vocabulary (e.g. "bet"), via cfg.action_map — see
    ScenarioConfig.abstract_to_trainer_action(). An abstract name this
    scenario's action_map never maps onto (e.g. bet_50 for a scenario
    that only declares "bet": bet_100) is dropped rather than raising —
    it was never a real option for this scenario's UI in the first place.
    """
    out: Dict[str, float] = {}
    for abstract_name, p in probs_by_abstract_name.items():
        trainer_name = cfg.abstract_to_trainer_action(abstract_name)
        if trainer_name is not None:
            out[trainer_name] = out.get(trainer_name, 0.0) + p
    return out

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


def _get_action_probabilities(
    cfg: ScenarioConfig, agent, rl_seat: int, rep_input, legal_mask
) -> Dict[str, float]:
    """
    Real forward pass through the loaded policy: representation_model
    -> the correct seat's policy head -> masked softmax. Returns a dict
    of THIS SCENARIO's Trainer-level action strings -> probability —
    translated from the network's fixed abstract vocabulary via
    cfg.action_map (see _probs_to_trainer_vocabulary / ScenarioConfig.
    abstract_to_trainer_action in trainer_config.py), so a caller never
    needs to know "bet_100" (or any other abstract name) exists.

    Falls back to a uniform distribution over legal actions when no
    real network is loaded (RandomFallbackAgent) — the Trainer still
    grades consistently, just against a meaningless baseline until a
    real checkpoint is in place.
    """
    from poker_rl_lab.actions.abstract_action import NUM_ACTIONS

    action_names = _abstract_action_names()

    representation_model = getattr(agent, "representation_model", None)
    if representation_model is None and hasattr(agent, "actor_critic"):
        representation_model = agent.actor_critic.representation_model

    if representation_model is None:
        # RandomFallbackAgent (no real network) — uniform over legal actions.
        legal_indices = [i for i, ok in enumerate(legal_mask) if ok]
        n = len(legal_indices) or 1
        raw = {
            action_names[i]: 1.0 / n
            for i in legal_indices
            if i in action_names
        }
        return _probs_to_trainer_vocabulary(cfg, raw)

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

    raw = {
        action_names[i]: float(probs_t[i].item())
        for i in range(NUM_ACTIONS)
        if legal_mask[i] and i in action_names
    }
    return _probs_to_trainer_vocabulary(cfg, raw)


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
    cfg: ScenarioConfig, agent, rl_seat: int, rep_inputs: list, legal_masks: list
) -> List[Dict[str, float]]:
    """
    Batched sibling of _get_action_probabilities(): one forward pass for
    every (rep_input, legal_mask) pair instead of one call each. Returns
    dicts already translated into THIS SCENARIO's Trainer-level action
    vocabulary — see _get_action_probabilities()'s own docstring.

    Returns
    -------
    List[Dict[str, float]]
        One {"fold": p, "call": p, "bet": p, ...} dict per example (only
        the legal actions for THAT row are included), same order as
        rep_inputs.
    """
    from poker_rl_lab.actions.abstract_action import NUM_ACTIONS

    action_names = _abstract_action_names()

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
            raw = {
                action_names[i]: 1.0 / k
                for i in legal_indices
                if i in action_names
            }
            out.append(_probs_to_trainer_vocabulary(cfg, raw))
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
        raw = {}
        for i in range(NUM_ACTIONS):
            if legal_masks[row][i] and i in action_names:
                raw[action_names[i]] = float(probs_t[row, i].item())
        out.append(_probs_to_trainer_vocabulary(cfg, raw))
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

    # Also internal, also not surfaced via as_dict() — the
    # decision-point observation context (rl_seat + serialized
    # StructuredObservation minus hole cards) needed to recreate a hand
    # grid for a PAST decision. Kept separate from `extra` in record()
    # rather than merged into the public history entry: decision_obs is
    # a fairly heavy nested structure, and every /trainer/action or
    # /trainer/scoreboard/reset response already carries the last 20
    # history entries — bloating each of those with full observation
    # data for every entry would be wasteful. Fetched on demand instead
    # via TrainerService.get_history_hand_grid().
    _grid_contexts: Dict[int, Dict[str, Any]] = field(default_factory=dict, repr=False)

    def record(
        self,
        result: DecisionResult,
        extra: Dict[str, Any],
        state_snapshot: Dict[str, Any],
        grid_context: Optional[Dict[str, Any]] = None,
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
        if grid_context is not None:
            self._grid_contexts[entry_id] = grid_context

        # cap history so it doesn't grow unbounded in a long session
        if len(self.history) > 200:
            dropped = self.history[:-200]
            self.history = self.history[-200:]
            for d in dropped:
                self._snapshots.pop(d["id"], None)
                self._grid_contexts.pop(d["id"], None)

        return entry_id

    def get_snapshot(self, entry_id: int) -> Optional[Dict[str, Any]]:
        return self._snapshots.get(entry_id)

    def get_entry(self, entry_id: int) -> Optional[Dict[str, Any]]:
        """Looks up one history entry (public fields only) by id."""
        for h in self.history:
            if h["id"] == entry_id:
                return h
        return None

    def get_grid_context(self, entry_id: int) -> Optional[Dict[str, Any]]:
        return self._grid_contexts.get(entry_id)

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
        self._grid_contexts = {}
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
        # Keyed on (scenario_key, resolved_checkpoint_path) — not just
        # scenario_key — so different user-selected checkpoints for the
        # same scenario type are each built/cached independently (see
        # get_agent's checkpoint_path param / TrainerService.select_checkpoint).
        self._agents: Dict[Tuple[str, str], Any] = {}
        self._adapter = None
        # One trace entry per pipeline step, per (scenario_key,
        # checkpoint_path) — see _try_build_real_agent()'s docstring.
        # Kept alongside the agent so get_debug_info() can report
        # exactly what happened at build time rather than re-inferring
        # it from the final object (which can't distinguish "real
        # checkpoint loaded" from "an untrained network of the same
        # class was silently built instead", since both produce the
        # same agent_class).
        self._traces: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    def get_agent(self, scenario: ScenarioConfig, checkpoint_path: str | None = None):
        """
        checkpoint_path: absolute path to a specific .pt file to load
        instead of scenario.policy.checkpoint_path (see
        TrainerService.select_checkpoint). Cache key includes it so
        switching checkpoints for a scenario doesn't reuse a
        previously-loaded agent for a *different* file, and switching
        back to an already-tried checkpoint doesn't reload it either.
        """
        resolved = checkpoint_path or scenario.policy.checkpoint_path
        cache_key = (scenario.key, resolved)
        if cache_key not in self._agents:
            agent, trace = _try_build_real_agent(scenario, checkpoint_path=resolved)
            if agent is None:
                trace.append({
                    "step": "final_fallback",
                    "status": "fallback",
                    "detail": "Using RandomFallbackAgent (uniform random over legal actions).",
                })
                agent = RandomFallbackAgent()
            self._agents[cache_key] = agent
            self._traces[cache_key] = trace
            for entry in trace:
                print(f"[trainer_service:{scenario.key}:{resolved}] [{entry['status'].upper()}] "
                      f"{entry['step']}: {entry['detail']}")
        return self._agents[cache_key]

    def get_trace(self, scenario_key: str, checkpoint_path: str) -> List[Dict[str, Any]]:
        return self._traces.get((scenario_key, checkpoint_path), [])

    def get_adapter(self):
        if self._adapter is None:
            from poker_rl_lab.representation.observation_adapter import ObservationAdapter
            self._adapter = ObservationAdapter()
        return self._adapter


def _try_build_real_agent(
    scenario: ScenarioConfig, checkpoint_path: str | None = None
) -> Tuple[Any, List[Dict[str, Any]]]:
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

    # Caller (get_agent) always resolves a concrete path — defaulting
    # here too only as a defensive fallback for any direct caller that
    # skips that resolution step.
    checkpoint_path = checkpoint_path or scenario.policy.checkpoint_path
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

        # scenario_key -> absolute checkpoint path the user has picked
        # via select_checkpoint(). Absent entries fall back to that
        # scenario's cfg.policy.checkpoint_path default. Persists across
        # new_scenario() calls (deliberately not reset there) so picking
        # a checkpoint sticks for "New Scenario"/auto-advance within the
        # same scenario type — only changed by an explicit
        # select_checkpoint() call.
        self._selected_checkpoints: Dict[str, str] = {}

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
        self._pending_grid_context: Dict[str, Any] = {}

        # Reentrancy guard — apply_hero_action() mutates self.state in
        # place with no locking. A double-submitted action (double click,
        # a retried request after a slow/lost response, etc.) hitting this
        # singleton concurrently could step the engine twice for one hero
        # decision, desyncing whatever the frontend still thinks the
        # current state is (symptoms: action buttons stuck doing nothing,
        # or the hand silently completing with no "Next Scenario" button
        # ever rendered because the response the UI is holding is stale).
        self._busy: bool = False

        # Chronological log of every action (hero's and the AI's) taken
        # so far in the CURRENT scenario, for display purposes only —
        # lets the frontend show "AI checks" / "AI bets $12" etc. when
        # a scenario has more than one hero decision point (e.g. river
        # OOP checks -> AI bets -> hero now faces call/fold), which was
        # previously invisible: the hero's second decision would just
        # silently appear with no indication of what the AI did to
        # cause it. Reset at the start of every new_scenario(); appended
        # to by both _villain_act() and apply_hero_action(). Included
        # verbatim in get_state()'s payload (and therefore also in
        # scoreboard history snapshots, since those are just a stored
        # get_state() payload).
        self._action_log: List[Dict[str, Any]] = []

        # Monotonically increasing id, bumped once per _new_scenario_once()
        # call — i.e. once per actual PokerState built, including retries
        # inside new_scenario()'s MAX_ATTEMPTS loop that never reach the
        # hero. Included in get_state()'s payload so the frontend (or a
        # human watching the UI) can tell "this really is the same hand
        # asking twice" apart from "this is a brand-new hand that just
        # happens to look identical" — river_duo's pot/stack are now
        # fixed (100/100bb) rather than randomized, so two consecutive
        # hands can otherwise be visually indistinguishable at a glance
        # (same pot, same stack, same position 50% of the time, only the
        # hole cards/board actually differ).
        self._hand_id: int = 0

    # ------------------------------------------------------------
    # Scenario listing
    # ------------------------------------------------------------

    def list_scenarios(self) -> List[Dict[str, Any]]:
        cfg = get_training_config()
        result = []
        for s in cfg.list_scenarios():
            # hole_cards drives the frontend's "Show Hand Grid" gating
            # (the grid enumerates all 1326 2-card combos and only
            # makes sense for a 2-hole-card game — Omaha's 4 hole cards
            # make the whole "canonical 169-hand grid" concept
            # undefined). Read from the scenario's own engine_variant
            # yaml rather than duplicating a hole-card count in
            # training_config.yaml, since the Engine yaml is already
            # the source of truth for that.
            print("Scenario: ", s)
            game_def, _ = load_game(s.engine_variant)
            result.append({
                "key": s.key,
                "label": s.label,
                "description": s.description,
                "hole_cards": game_def.hole_cards,
            })
        return result

    # ------------------------------------------------------------
    # Checkpoint selection
    # ------------------------------------------------------------

    def _resolved_checkpoint(self, cfg: ScenarioConfig) -> str:
        """The checkpoint path to actually load for this scenario right
        now — the user's selection if one was made, else the config
        default."""
        return self._selected_checkpoints.get(cfg.key, cfg.policy.checkpoint_path)

    def list_checkpoints(self, scenario_key: str) -> Dict[str, Any]:
        """
        List every .pt file in this scenario's configured
        checkpoint_dir, for the Trainer UI's model-picker dropdown.

        Returns
        -------
        dict
            {
              "checkpoint_dir": str | None,     # absolute path, or None if unconfigured
              "default_checkpoint": str,        # filename of cfg.policy.checkpoint_path's file
              "selected_checkpoint": str | None,# filename currently selected, if any
              "checkpoints": [str, ...]         # filenames, sorted
            }
        """
        import os

        cfg = get_training_config().get_scenario(scenario_key)
        checkpoint_dir = cfg.policy.checkpoint_dir

        checkpoints: List[str] = []
        if checkpoint_dir and os.path.isdir(checkpoint_dir):
            checkpoints = sorted(
                f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")
            )

        selected_path = self._selected_checkpoints.get(scenario_key)

        return {
            "checkpoint_dir": checkpoint_dir,
            "default_checkpoint": os.path.basename(cfg.policy.checkpoint_path),
            "selected_checkpoint": os.path.basename(selected_path) if selected_path else None,
            "checkpoints": checkpoints,
        }

    def select_checkpoint(self, scenario_key: str, filename: str | None) -> Dict[str, Any]:
        """
        Select which .pt file (by filename, not full path — the
        frontend only ever sees filenames from list_checkpoints) this
        scenario should load on its NEXT agent build. filename=None (or
        empty string) clears the selection, reverting to
        cfg.policy.checkpoint_path.

        Does not rebuild the agent eagerly — _AgentCache.get_agent()
        lazily builds (and caches) it the next time it's needed
        (new_scenario / grading / villain act / hand grid), keyed on
        the resolved path, so switching back to a previously-loaded
        checkpoint within the same process is instant.
        """
        import os

        cfg = get_training_config().get_scenario(scenario_key)  # raises KeyError if unknown

        if not filename:
            self._selected_checkpoints.pop(scenario_key, None)
            return self.list_checkpoints(scenario_key)

        checkpoint_dir = cfg.policy.checkpoint_dir
        if not checkpoint_dir:
            raise ValueError(
                f"Scenario {scenario_key!r} has no checkpoint_dir configured — "
                f"cannot select a checkpoint by filename."
            )

        # Reject path separators outright — filename must name a file
        # directly inside checkpoint_dir, never an escape via ../ or an
        # absolute path smuggled in from the request body.
        if os.path.basename(filename) != filename:
            raise ValueError(f"Invalid checkpoint filename: {filename!r}")

        full_path = os.path.join(checkpoint_dir, filename)
        if not os.path.isfile(full_path):
            raise ValueError(
                f"No checkpoint file {filename!r} in {checkpoint_dir!r} "
                f"for scenario {scenario_key!r}."
            )
        if not filename.endswith(".pt"):
            raise ValueError(f"Checkpoint filename must end in .pt: {filename!r}")

        self._selected_checkpoints[scenario_key] = full_path
        return self.list_checkpoints(scenario_key)

    # ------------------------------------------------------------
    # New scenario
    # ------------------------------------------------------------

    def new_scenario(self, scenario_key: str) -> Dict[str, Any]:
        """
        Start a fresh scenario, retrying if the very first hero decision
        point never actually arrives — e.g. push-fold's villain SB
        folding immediately (a legal, if unusual, action for the model
        to take) means the hand ends before hero (BB) ever gets to act
        at all. Per-request: "I only want scenarios where the player
        has a decision" — rather than surface a hand-complete state
        with nothing for the hero to do (which previously also had no
        frontend auto-advance path at all, since that only existed on
        the /trainer/action response, not /new — see
        _new_scenario_once callers), just generate another one.

        Bounded retry count as a safety net against a pathological
        config where the hero could never possibly get a turn (e.g. a
        scenario type with zero real decision points) — that should
        raise loudly rather than hang.
        """
        MAX_ATTEMPTS = 50
        last_payload = None
        for _ in range(MAX_ATTEMPTS):
            last_payload = self._new_scenario_once(scenario_key)
            if self.awaiting_hero:
                return last_payload
        raise ValueError(
            f"Could not generate a scenario for {scenario_key!r} where the "
            f"hero gets a decision after {MAX_ATTEMPTS} attempts — check "
            f"this scenario's action tree actually gives the hero a turn."
        )

    def _new_scenario_once(self, scenario_key: str) -> Dict[str, Any]:
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
        self._action_log = []
        self._hand_id += 1

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

    # Fallback dead-pot range (in big blinds) used when a non-push-fold
    # scenario has no cfg.pot_bb configured at all — guarantees a real
    # randomized pot instead of silently leaving start_hand()'s posted
    # blinds in place (which is what produced pot=3/to_call=1 — the
    # raw small_blind+big_blind sum — for a river scenario whose
    # training_config.yaml pot_bb apparently wasn't actually reaching
    # this code, whatever the cause).
    _DEFAULT_DEAD_POT_BB_RANGE = (10.0, 40.0)

    def _apply_dead_pot(self, cfg: ScenarioConfig, game_def) -> None:
        """
        Overrides start_hand()'s blind-posting with a randomized dead
        pot, per a scenario's PotConfig(mode="randomized_bb") semantics
        (river_duo.yaml). No-op for push_fold AND single_bet — both
        genuinely use posted blinds as the pot (push_fold: the shove
        itself; single_bet: real multi-street hold'em/Omaha, e.g.
        holdem_full_duo/omaha_full_duo, which have no dead-money
        concept at all). Every OTHER betting_type is guaranteed a real
        dead pot, using cfg.pot_bb when configured and
        _DEFAULT_DEAD_POT_BB_RANGE otherwise (with a loud log line so a
        missing config is diagnosable rather than silently producing a
        degenerate 1-2 chip pot).

        Must be called AFTER state.start_hand() (hole cards + board
        already dealt) but BEFORE any hero/villain action — refunds
        whatever was posted as blinds back to each player's stack,
        zeroes bet_to_call/current_bet, and sets game.pot to a fresh
        randomized dead-money amount with no per-seat attribution.
        """
        if cfg.betting_type in ("push_fold", "single_bet"):
            return

        g = self.state.game
        big_blind = game_def.big_blind

        if cfg.pot_bb is not None:
            pot_min, pot_max = cfg.pot_bb.min, cfg.pot_bb.max
        else:
            pot_min, pot_max = self._DEFAULT_DEAD_POT_BB_RANGE
            print(
                f"[trainer_service] scenario {cfg.key!r} "
                f"(betting_type={cfg.betting_type!r}) has no pot_bb "
                f"configured in training_config.yaml — using a default "
                f"{pot_min}-{pot_max}bb dead pot instead of leaving "
                f"posted blinds as the pot."
            )

        for p in g.players:
            g.pot -= p.current_bet
            p.stack += p.current_bet
            p.total_contribution -= p.current_bet
            p.current_bet = 0
            p.is_all_in = False

        dead_pot = round(random.uniform(pot_min, pot_max) * big_blind)
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
        resolved_checkpoint = self._resolved_checkpoint(cfg)
        agent = self._cache.get_agent(cfg, checkpoint_path=resolved_checkpoint)  # builds + caches trace on first call
        trace = self._cache.get_trace(scenario_key, resolved_checkpoint)

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

        checkpoint_path = resolved_checkpoint
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
        agent = self._cache.get_agent(cfg, checkpoint_path=self._resolved_checkpoint(cfg))
        rep_input = self._cache.get_adapter().adapt(obs)
        action_probs = _get_action_probabilities(cfg, agent, hero_rl_seat, rep_input, obs.legal_mask)

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

            engine_action = self._build_engine_action(cfg, cfg.hero_seat, action_type)

            hero_player = self.state.game.players[cfg.hero_seat]
            hero_bet_before = hero_player.current_bet
            hero_street_before = self.state.game.street_index
            hero_pot_before = self.state.game.pot

            self.state.step(engine_action)

            hero_bet_after = self.state.game.players[cfg.hero_seat].current_bet
            # is_bet_like() catches sizing "pot" OR "stack" — the
            # previous pot_fraction_for(...) is not None check missed
            # river_duo's "bet" (sizing: stack), so a hero shove there
            # was logged with amount=None instead of the real chip delta.
            is_chip_moving_action = (
                action_type in ("call", "all_in")
                or cfg.is_bet_like(action_type)
            )
            hero_amount = (
                (hero_bet_after - hero_bet_before)
                if is_chip_moving_action else None
            )
            self._action_log.append({
                "actor": "hero",
                "position": self.hero_position,
                "action": action_type,
                "amount": hero_amount,
                "street": hero_street_before,
                "pot_before": hero_pot_before,
            })

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

    def _serialize_obs(self, obs) -> Dict[str, Any]:
        """
        Plain-JSON-serializable snapshot of a StructuredObservation,
        EXCLUDING hole_cards (the hand grid always replaces those per
        combo, so they're never needed for reconstruction) — used to
        store the exact decision-point context in a scoreboard history
        entry so get_history_hand_grid() can rebuild it later without
        touching self.state (which has since moved on to a different
        hand entirely).
        """
        return {
            "hero_seat": obs.hero_seat,
            "board_node_cards": list(obs.board.node_cards),
            "players": [
                {
                    "seat": p.seat, "stack": p.stack, "current_bet": p.current_bet,
                    "has_folded": p.has_folded, "is_all_in": p.is_all_in, "is_hero": p.is_hero,
                }
                for p in obs.players
            ],
            "pot": obs.betting.pot,
            "bet_to_call": obs.betting.bet_to_call,
            "min_raise": obs.betting.min_raise,
            "street_index": obs.betting.street_index,
            "raises_this_street": obs.betting.raises_this_street,
            "legal_mask": list(obs.legal_mask),
            "game_name": obs.game_name,
            "history_events": [
                {
                    "seat": e.seat, "action_type": e.action_type, "street_index": e.street_index,
                    "pot_fraction": e.pot_fraction, "is_aggressor": e.is_aggressor,
                }
                for e in obs.history.events
            ],
            "dealer_position": obs.dealer_position,
        }

    def _deserialize_obs(self, data: Dict[str, Any], hole_cards: Tuple[int, ...]):
        """Inverse of _serialize_obs(), with hole_cards supplied by the caller."""
        from poker_rl_lab.envs.observations import (
            BettingObs, BoardObs, HistoryEventObs, HistoryObs, HoleCardsObs,
            PlayerObs, StructuredObservation,
        )

        players = tuple(
            PlayerObs(
                seat=p["seat"], stack=p["stack"], current_bet=p["current_bet"],
                has_folded=p["has_folded"], is_all_in=p["is_all_in"], is_hero=p["is_hero"],
            )
            for p in data["players"]
        )
        history = HistoryObs(events=tuple(
            HistoryEventObs(
                seat=e["seat"], action_type=e["action_type"], street_index=e["street_index"],
                pot_fraction=e["pot_fraction"], is_aggressor=e["is_aggressor"],
            )
            for e in data["history_events"]
        ))
        return StructuredObservation(
            hero_seat=data["hero_seat"],
            hole_cards=HoleCardsObs(cards=tuple(hole_cards)),
            board=BoardObs(node_cards=tuple(data["board_node_cards"])),
            players=players,
            betting=BettingObs(
                pot=data["pot"], bet_to_call=data["bet_to_call"], min_raise=data["min_raise"],
                street_index=data["street_index"], raises_this_street=data["raises_this_street"],
            ),
            legal_mask=tuple(data["legal_mask"]),
            game_name=data["game_name"],
            history=history,
            dealer_position=data["dealer_position"],
            is_terminal=False,
        )

    def _grade_decision(self, cfg: ScenarioConfig, hero_action: str) -> DecisionResult:
        hole_ids = mask_to_card_ids(self.state.game.players[cfg.hero_seat].hand_mask)
        hole_strs = [str(CardObj(cid)) for cid in hole_ids]

        hero_rl_seat = self._rl_seat_for_position(cfg, self.hero_position)
        obs = self._build_obs(
            cfg, seat=cfg.hero_seat, rl_seat=hero_rl_seat, position=self.hero_position
        )

        agent = self._cache.get_agent(cfg, checkpoint_path=self._resolved_checkpoint(cfg))
        rep_input = self._cache.get_adapter().adapt(obs)
        action_probs = _get_action_probabilities(cfg, agent, hero_rl_seat, rep_input, obs.legal_mask)

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

        # Public scoreboard-entry fields (shown in the sidebar/history
        # list) — kept separate from the grid-replay context (rl_seat/
        # decision_obs), which is stored privately (see Scoreboard's
        # _grid_contexts) and only fetched on demand.
        self._pending_grade_extra = {
            "hand": format_hero_hand_display(hole_strs),
            "position": self.hero_position,
            "effective_stack_bb": self.hero_effective_bb,
            "scenario": cfg.key,
            "policy_probs": action_probs,
        }
        self._pending_grid_context = {
            "rl_seat": hero_rl_seat,
            "decision_obs": self._serialize_obs(obs),
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

        if self.state.game_def.hole_cards != 2:
            raise ValueError(
                f"Hand grid is only available for 2-hole-card games — "
                f"scenario {self.scenario_key!r} deals "
                f"{self.state.game_def.hole_cards} hole cards per player "
                f"(e.g. Omaha), for which the 169-combo canonical grid "
                f"isn't a meaningful concept."
            )

        cfg = get_training_config().get_scenario(self.scenario_key)
        hero_rl_seat = self._rl_seat_for_position(cfg, self.hero_position)
        action_order = self._live_allowed_actions(cfg)
        if not action_order:
            raise ValueError(
                f"No legal hero actions at the current decision point for "
                f"scenario {self.scenario_key!r} — cannot build a hand grid "
                f"(is it actually the hero's turn?)."
            )

        agent = self._cache.get_agent(cfg, checkpoint_path=self._resolved_checkpoint(cfg))
        adapter = self._cache.get_adapter()

        # Template observation at the CURRENT decision point (real
        # stack/pot/villain state) — only hole_cards differs per combo.
        template_obs = self._build_obs(
            cfg, seat=cfg.hero_seat, rl_seat=hero_rl_seat, position=self.hero_position
        )

        action_grid, combos_by_hand = self._compute_hand_grid(
            cfg, agent, adapter, hero_rl_seat, template_obs, action_order
        )

        return {
            "position": self.hero_position,
            "action_order": action_order,
            "hero_effective_bb": self.hero_effective_bb,
            "action_grid": action_grid,
            "combos": {
                hand: {"combos": entries} for hand, entries in combos_by_hand.items()
            },
        }

    def get_history_hand_grid(self, entry_id: int) -> Dict[str, Any]:
        """
        Recreate the 169-hand grid for a PAST, already-graded
        scoreboard decision, using the decision-point context stashed
        by _grade_decision() at the time (Scoreboard._grid_contexts —
        see its own docstring for why this is stored separately from
        the public history entry). Same response shape as
        get_hand_grid(), so the frontend can reuse the exact same
        rendering path for both live and review modes.

        Uses whatever checkpoint is CURRENTLY selected for that
        scenario type (self._selected_checkpoints), not necessarily
        the one that was active when the decision was originally
        graded — the grid_context doesn't pin a checkpoint identity,
        so a checkpoint switched after the fact will show this past
        hand through the new model instead. Acceptable for a
        review/debugging tool; if this ever needs to be historically
        exact, the resolved checkpoint path would need to be stored
        in the grid_context at grading time too.
        """
        entry = self.scoreboard.get_entry(entry_id)
        if entry is None:
            raise ValueError(f"No scoreboard history entry with id {entry_id}")

        grid_context = self.scoreboard.get_grid_context(entry_id)
        if grid_context is None:
            raise ValueError(
                f"No hand-grid context stored for history entry {entry_id} "
                f"(it may predate this feature, or its scoreboard entry "
                f"has since been evicted)."
            )

        scenario_key = entry.get("scenario")
        if not scenario_key:
            raise ValueError(f"History entry {entry_id} has no recorded scenario key.")

        cfg = get_training_config().get_scenario(scenario_key)

        game_def, _ = load_game(cfg.engine_variant)
        if game_def.hole_cards != 2:
            raise ValueError(
                f"Hand grid is only available for 2-hole-card games — "
                f"scenario {scenario_key!r} deals {game_def.hole_cards} "
                f"hole cards per player (e.g. Omaha), for which the "
                f"169-combo canonical grid isn't a meaningful concept."
            )

        rl_seat = grid_context["rl_seat"]
        decision_obs = grid_context["decision_obs"]

        action_order = self._trainer_action_names_from_legal_mask(cfg, decision_obs["legal_mask"])
        if not action_order:
            raise ValueError(
                f"Stored decision context for history entry {entry_id} has no "
                f"legal actions — cannot rebuild a hand grid."
            )

        # hole_cards is irrelevant here — _compute_hand_grid replaces it
        # per combo — pass an empty tuple as a harmless placeholder.
        template_obs = self._deserialize_obs(decision_obs, hole_cards=())

        agent = self._cache.get_agent(cfg, checkpoint_path=self._resolved_checkpoint(cfg))
        adapter = self._cache.get_adapter()

        action_grid, combos_by_hand = self._compute_hand_grid(
            cfg, agent, adapter, rl_seat, template_obs, action_order
        )

        return {
            "position": entry.get("position"),
            "action_order": action_order,
            "hero_effective_bb": entry.get("effective_stack_bb"),
            "action_grid": action_grid,
            "combos": {
                hand: {"combos": entries} for hand, entries in combos_by_hand.items()
            },
        }

    @staticmethod
    def _trainer_action_names_from_legal_mask(cfg: ScenarioConfig, legal_mask) -> List[str]:
        """
        Map a stored abstract-action legal_mask back onto THIS SCENARIO's
        Trainer-level action name strings, in stable NUM_ACTIONS order —
        used to derive get_history_hand_grid()'s action_order from a
        serialized decision context, via cfg.action_map (see
        ScenarioConfig.abstract_to_trainer_action in trainer_config.py)
        rather than a hardcoded {ActionType -> string} table.
        """
        from poker_rl_lab.actions.abstract_action import NUM_ACTIONS

        action_names = _abstract_action_names()
        result = []
        for i in range(NUM_ACTIONS):
            if i >= len(legal_mask) or not legal_mask[i]:
                continue
            abstract_name = action_names.get(i)
            if abstract_name is None:
                continue
            trainer_name = cfg.abstract_to_trainer_action(abstract_name)
            if trainer_name is not None:
                result.append(trainer_name)
        return result

    def _compute_hand_grid(self, cfg: ScenarioConfig, agent, adapter, rl_seat, template_obs, action_order):
        """
        Shared core of get_hand_grid()/get_history_hand_grid(): swap
        every one of the 1326 hole-card combos into template_obs, run
        one batched forward pass, and bucket the results into the
        13x13 canonical-hand grid (averaged per cell) plus a per-combo
        breakdown. template_obs's own hole_cards are ignored/replaced
        entirely — only its board/betting/player state is used.

        Returns
        -------
        (action_grid, combos_by_hand)
            action_grid: 13x13 nested list of {action: avg_prob} dicts.
            combos_by_hand: {canonical_hand_str: [{"cards": [..], "probs": {..}}, ...]}
        """
        combos = list(itertools.combinations(range(52), 2))

        obs_list = [
            _dataclasses_replace(template_obs, hole_cards=HoleCardsObs(cards=(c1, c2)))
            for (c1, c2) in combos
        ]
        rep_inputs = [adapter.adapt(o) for o in obs_list]
        legal_masks = [o.legal_mask for o in obs_list]

        probs_per_combo = _get_action_probabilities_batch(
            cfg, agent, rl_seat, rep_inputs, legal_masks
        )

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

        return action_grid, combos_by_hand

    # ------------------------------------------------------------
    # Engine helpers
    # ------------------------------------------------------------

    def _live_legal_trainer_actions(self, cfg: ScenarioConfig, seat: int) -> List[str]:
        """
        Trainer-level action strings actually legal for `seat` right
        now, for non-push-fold (generic-tree-shaped) scenarios —
        derived from betting math, never from GameState.legal_actions()
        directly.

        Why not just use the engine's legal_actions()
        --------------------------------------------------
        That generic branch (game_state.py, the non-push_fold/
        non-river_only/non-single_bet path) unconditionally includes
        FOLD regardless of to_call — even when nothing is owed and the
        only real choice is check-or-bet. _build_obs_generic() previously
        built its network-facing legal_mask straight from that engine
        output, which meant a check/bet-only decision was actually fed
        to the policy as a THREE-way {fold, check, bet} distribution.
        The Trainer's UI (correctly) only ever showed check/bet as
        buttons — so whatever probability mass the network assigned to
        the never-shown FOLD silently vanished from what the hand grid
        displayed, which is exactly why reported percentages didn't sum
        to 100% (e.g. check=36%, bet=0%, ~64% missing — that 64% was
        FOLD's share, computed but never surfaced).

        Using this same restricted candidate-set function for BOTH the
        buttons (_live_allowed_actions) AND the network's legal_mask
        (_build_obs_generic) guarantees the two can never drift apart.

        single_bet (holdem_full_duo / omaha_full_duo) adds one more
        wrinkle beyond river_only: to_call > 0 does not by itself mean
        "facing a bet — call or fold only", because a street can start
        with a live to_call purely from POSTED BLINDS (preflop) before
        anyone has voluntarily bet. raises_this_street (mirrors
        game_state.py's own single_bet branch in legal_actions() — the
        two MUST stay in lockstep, or the buttons offered here and what
        the engine will actually accept can drift apart) is what
        actually tells the two apart:
          - raises_this_street == 0 and to_call > 0 (blind gap, nobody
            has raised): fold / call(-the-blind) / open a bet are all
            live — e.g. preflop SB facing the BB.
          - raises_this_street == 0 and to_call == 0: check or bet —
            e.g. any street's first-to-act with nothing owed.
          - raises_this_street > 0: someone already bet this street —
            call or fold only, same as river_only's facing-a-bet case.
        """
        if self.state is None or self.state.phase != Phase.BETTING:
            return []
        g = self.state.game
        player = g.players[seat]
        to_call = max(g.bet_to_call - player.current_bet, 0)

        if g.betting_type == "single_bet" and g.raises_this_street == 0:
            if to_call > 0:
                return ["fold", "call"] + cfg.bet_like_actions()
            return ["check"] + cfg.bet_like_actions()

        if to_call > 0:
            return ["call", "fold"]
        return ["check"] + cfg.bet_like_actions()

    def _live_allowed_actions(self, cfg: ScenarioConfig) -> List[str]:
        """
        The Trainer-level action strings actually legal for the HERO
        right now, further intersected with training_config.yaml's
        declared per-position action list (documentation of the UNION
        of actions that position could ever face — see
        _live_legal_trainer_actions' own docstring for why the live
        candidate set alone, not the engine's raw legal_actions(), is
        the right source of truth here).

        push-fold is unaffected: there is exactly one hero decision
        point per hand for that scenario type, so the static list and
        the live list always coincide — kept as a direct passthrough
        rather than re-deriving it from to_call, since push-fold's
        static list already matches push_fold_service.py's own
        SB={fold,all_in}/BB={fold,call} convention exactly.
        """
        if cfg.betting_type == "push_fold":
            return cfg.actions.get(self.hero_position, [])

        candidates = self._live_legal_trainer_actions(cfg, cfg.hero_seat)
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

    def _build_engine_action(self, cfg: ScenarioConfig, seat: int, action_type: str) -> EngineAction:
        """
        Translate a Trainer-level action string into a concrete Engine
        Action, sized correctly for whatever the acting player actually
        faces.

        Handles fold / check / call / all_in (full-stack shove) directly,
        plus ANY other action name this scenario declares in its own
        action_map (training_config.yaml) as long as it has a
        pot_fraction configured — see ScenarioConfig.pot_fraction_for()
        in trainer_config.py. That's what lets a scenario offer more than
        one bet size (e.g. "bet" at pot_fraction 1.0 AND "bet_half" at
        0.5) with zero changes needed here: this method never hardcodes
        the string "bet" as special, only the fact that it's a bet-like
        action per this scenario's own config.

        Note: for river_only betting_type, GameState.apply_action()
        (game_state.py) itself clamps any BET to min(pot, stack)
        regardless of the amount passed in — so today, with every
        declared bet action at pot_fraction 1.0, this is consistent.
        A future river scenario declaring a bet action at a fraction
        OTHER than 1.0 would need that engine-side clamp updated too;
        this method computes the config-requested size correctly, but
        can't override what the engine does with it afterward.

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

        if action_type == "all_in":
            shove_total = player.current_bet + player.stack
            act_type = EngineActionType.RAISE if to_call > 0 else EngineActionType.BET
            return EngineAction(type=act_type, amount=shove_total)

        # Any other declared action must be bet-like per this scenario's
        # own action_map (sizing "pot" or "stack" — see
        # ScenarioConfig.is_bet_like/sizing_for/pot_fraction_for in
        # trainer_config.py). Two sizing modes:
        #   "stack" — a full-stack shove, same math as the "all_in"
        #             branch above, just under a different Trainer-level
        #             name (river's "bet" action_map entry uses this).
        #   "pot"   — sized to pot_fraction * current pot, capped at stack.
        if not cfg.is_bet_like(action_type):
            raise ValueError(
                f"Unknown action_type {action_type!r} for scenario "
                f"{cfg.key!r} — not fold/check/call/all_in, and this "
                f"scenario's action_map doesn't declare it as bet-like."
            )

        # single_bet scenarios: a bet/raise is still legal while facing
        # a to_call > 0 IF that gap is purely from posted blinds and
        # nobody has voluntarily bet yet this street (raises_this_street
        # == 0) — e.g. preflop SB opening/raising over the BB's blind.
        # Must match _live_legal_trainer_actions' own single_bet branch
        # exactly, or this raises on an action the UI just offered.
        facing_a_real_bet = to_call > 0 and not (
            g.betting_type == "single_bet" and g.raises_this_street == 0
        )
        if facing_a_real_bet:
            raise ValueError(
                f"Cannot '{action_type}' while facing a bet — use 'call' or 'fold'."
            )

        sizing = cfg.sizing_for(action_type)

        if sizing == "stack":
            shove_total = player.current_bet + player.stack
            return EngineAction(type=EngineActionType.BET, amount=shove_total)

        # sizing == "pot"
        pot_fraction = cfg.pot_fraction_for(action_type)
        desired = round(g.pot * pot_fraction)
        total = player.current_bet + min(desired, player.stack)
        return EngineAction(type=EngineActionType.BET, amount=total)

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

        agent = self._cache.get_agent(cfg, checkpoint_path=self._resolved_checkpoint(cfg))
        rep_input = self._cache.get_adapter().adapt(obs)
        # Sample from the policy's own action distribution rather than
        # always taking the argmax (deterministic=True) — the villain
        # should play the same mixed strategy the network actually
        # learned, not a deterministic best-response. This also matters
        # for scenarios with more than one hero decision point (river):
        # a deterministic villain effectively never bets if check has
        # even slightly higher probability, so the hero would almost
        # never see the second (call/fold) decision at all.
        result = agent.act(seat=rl_seat, obs=rep_input, legal_mask=obs.legal_mask, deterministic=False)

        engine_action_type = self._abstract_action_to_engine_string(cfg, result.action_type)

        player = self.state.game.players[cfg.villain_seat]
        bet_before = player.current_bet
        street_before = self.state.game.street_index
        pot_before = self.state.game.pot

        self.state.step(self._build_engine_action(cfg, cfg.villain_seat, engine_action_type))

        bet_after = self.state.game.players[cfg.villain_seat].current_bet
        # Same is_bet_like() fix as apply_hero_action() above — the
        # villain's own river shove was previously mislogged the same way.
        is_chip_moving_action = (
            engine_action_type in ("call", "all_in")
            or cfg.is_bet_like(engine_action_type)
        )
        amount = (bet_after - bet_before) if is_chip_moving_action else None
        self._action_log.append({
            "actor": "ai",
            "position": villain_position,
            "action": engine_action_type,
            "amount": amount,
            "street": street_before,
            "pot_before": pot_before,
        })

    @staticmethod
    def _abstract_action_to_engine_string(cfg: ScenarioConfig, action_type_value: int) -> str:
        """
        Map the RL Lab's sampled ActionType (an int) onto one of THIS
        SCENARIO's Trainer-level action strings, via cfg.action_map (see
        ScenarioConfig.abstract_to_trainer_action in trainer_config.py)
        rather than a hardcoded {ActionType -> string} table — so a
        scenario declaring a different bet size (e.g. "bet_half" ->
        bet_50) is understood here with no code change. Falls back to
        "fold" if the sampled index isn't part of this scenario's
        vocabulary at all (defensive — every legal_mask this Trainer
        builds only ever marks a subset of this scenario's own declared
        actions as legal in the first place, so this should be
        unreachable in practice).
        """
        abstract_name = _abstract_action_names().get(action_type_value)
        if abstract_name is None:
            return "fold"
        return cfg.abstract_to_trainer_action(abstract_name) or "fold"

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

    # -- generic (river, single_bet, and anything else): ask the ---
    # -- ENGINE what's legal rather than hardcoding a position-keyed --
    # -- mask.                                                        --
    def _build_obs_generic(self, cfg: ScenarioConfig, seat: int, rl_seat: int, position: str):
        from poker_rl_lab.actions.abstract_action import NUM_ACTIONS
        from poker_rl_lab.envs.observations import (
            BettingObs, BoardObs, HistoryObs, HoleCardsObs, PlayerObs, StructuredObservation,
        )

        g = self.state.game
        player = g.players[seat]
        to_call = max(g.bet_to_call - player.current_bet, 0)

        # Trainer-level candidates for THIS seat right now — the same
        # restricted {call,fold} / {check,bet} / single_bet-aware set
        # _live_allowed_actions uses for the hero's buttons (see
        # _live_legal_trainer_actions' docstring for why the raw engine
        # legal_actions() output is NOT used here: its generic branch
        # always marks FOLD legal even when nothing is owed, which would
        # otherwise steal probability mass from a decision that's only
        # ever supposed to be check-or-bet).
        trainer_actions = self._live_legal_trainer_actions(cfg, seat)

        name_to_index = _abstract_action_name_to_index()
        mask = [False] * NUM_ACTIONS
        for name in trainer_actions:
            abstract_name = cfg.trainer_to_abstract(name)
            idx = name_to_index.get(abstract_name)
            if idx is not None:
                mask[idx] = True
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
            # Real action history, not always-empty — see
            # _history_events_for_obs' docstring. Without this, a
            # decision facing a prior check (e.g. IP acting after OOP
            # checks) was observationally IDENTICAL to OOP's own
            # original check-or-bet decision (same board/pot/legal_mask,
            # nothing distinguishing "I'm first to act" from "opponent
            # just checked to me") — the model had no way to condition
            # on what actually happened, and effectively re-faced the
            # same decision instead of responding to it.
            history=HistoryObs(events=self._history_events_for_obs(cfg)),
            dealer_position=0,
            is_terminal=False,
        )

    def _history_events_for_obs(self, cfg: ScenarioConfig):
        """
        Convert self._action_log (this hand's actions so far, in order
        — see its docstring in __init__) into HistoryEventObs entries
        for _build_obs_generic(). Each logged action records enough
        (position, trainer-level action name, street, pot_before,
        chip amount) to reconstruct the same seat/action_type/
        pot_fraction/is_aggressor shape push-fold's own hand-tuned
        synthetic history event uses (_build_obs_push_fold) — just
        derived from whatever actually happened instead of a single
        hardcoded push event.
        """
        from poker_rl_lab.envs.observations import HistoryEventObs

        name_to_index = _abstract_action_name_to_index()

        events = []
        for entry in self._action_log:
            trainer_action = entry.get("action")
            abstract_name = cfg.trainer_to_abstract(trainer_action)
            idx = name_to_index.get(abstract_name) if abstract_name else None
            if idx is None:
                continue
            try:
                seat = cfg.positions.index(entry["position"])
            except (KeyError, ValueError):
                continue

            pot_before = entry.get("pot_before") or 0
            amount = entry.get("amount") or 0
            pot_fraction = (amount / pot_before) if pot_before > 0 else 0.0

            # Aggressive = a real bet-like action (this scenario's
            # action_map declares sizing "pot" OR "stack" for it — see
            # ScenarioConfig.is_bet_like) or an outright shove — never a
            # hardcoded "bet" string check, so a scenario with a
            # differently-named bet action (e.g. "bet_half", or
            # river_duo's own "bet" at sizing: stack) is still correctly
            # flagged as the street's aggressor. Previously this checked
            # pot_fraction_for(...) is not None, which missed sizing:
            # stack entirely.
            is_aggressor = (
                cfg.is_bet_like(trainer_action)
                or trainer_action == "all_in"
            )

            events.append(HistoryEventObs(
                seat=seat,
                action_type=idx,
                street_index=entry.get("street", 0),
                pot_fraction=pot_fraction,
                is_aggressor=is_aggressor,
            ))
        return tuple(events)

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
        # scenario (river, single_bet, ...) deals a real board the hero
        # is supposed to see, so only push-fold gets blanked here.
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
            # Chronological log of every action so far this hand — see
            # self._action_log's docstring in __init__. Frozen into
            # whatever the payload was at the time for scoreboard
            # history snapshots (get_history_snapshot just returns a
            # stored get_state() payload), so a past hand's replay
            # still shows its own action-by-action log correctly.
            "action_log": list(self._action_log),
            # See self._hand_id's docstring in __init__ — bumped once per
            # PokerState actually built, so two payloads with the same
            # hand_id are provably the same hand asking again, and two
            # payloads with different hand_ids are provably different
            # hands regardless of how similar pot/stack/position look.
            "hand_id": self._hand_id,
        })
        return payload


trainer_service = TrainerService()
