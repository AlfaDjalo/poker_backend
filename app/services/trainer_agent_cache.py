"""
app/services/trainer_agent_cache.py — checkpoint cache for
GraphScenarioEnv-native agents (graph_scenario_trainer.py), kept
separate from trainer_service._AgentCache on purpose: that cache's
get_agent() wraps everything in the legacy 9-slot abstract_action
vocabulary translation (_get_action_probabilities /
_probs_to_trainer_vocabulary), which a graph-native checkpoint has no
use for at all — its action space IS obs.options, resolved live, no
translation layer.

FLAG: the exact loader for a graph-native-trained checkpoint
(poker_rl_lab.trainers.graph_training_loop's own save/load functions,
per graph_scenario_env's action space rather than DualSeatActorCritic's
fixed abstract vocabulary) hasn't been confirmed against real RL Lab
source yet. `_load_graph_agent()` below is written against the same
shape trainer_service._try_build_real_agent() already uses successfully
for the legacy path (load_dual_seat_actor_critic_for_graph), on the
assumption RL Lab's graph-scenario checkpoints load the same way — this
needs confirming/fixing once poker_rl_lab's real API for this is
available to read directly.
"""

from __future__ import annotations

from typing import Any, Dict

from app.config.trainer_config import ScenarioConfig


class _GraphAgentWrapper:
    """
    Thin wrapper giving graph_scenario_trainer.py a stable
    `.act(seat, obs, legal_mask, deterministic)` /
    `.action_probabilities(seat, obs, legal_mask)` surface, regardless
    of the real loaded object's own shape — mirrors how
    trainer_service._get_action_probabilities() already isolates the
    legacy path's own head_sb/head_bb/actor_critic duck-typing into one
    place rather than spreading it across call sites.
    """

    def __init__(self, raw_agent: Any):
        self._raw = raw_agent

    def act(self, seat: int, obs, legal_mask, deterministic: bool = False):
        return self._raw.act(seat=seat, obs=obs, legal_mask=legal_mask, deterministic=deterministic)

    def action_probabilities(self, seat: int, obs, legal_mask):
        """
        Returns a plain list/array of per-option probabilities, indexed
        identically to legal_mask/obs.options. Falls back to uniform
        over legal options if the raw agent exposes no direct
        probability method (e.g. a random fallback with no real
        network) — mirrors _get_action_probabilities()'s own
        RandomFallbackAgent branch in trainer_service.py.
        """
        if hasattr(self._raw, "action_probabilities"):
            return self._raw.action_probabilities(seat=seat, obs=obs, legal_mask=legal_mask)

        legal_indices = [i for i, ok in enumerate(legal_mask) if ok]
        n = len(legal_indices) or 1
        probs = [0.0] * len(legal_mask)
        for i in legal_indices:
            probs[i] = 1.0 / n
        return probs


class _RandomGraphFallbackAgent:
    """Uniform-random fallback — same rationale as
    trainer_service.RandomFallbackAgent, for a graph scenario with no
    loadable checkpoint yet."""

    def act(self, seat: int, obs, legal_mask, deterministic: bool = False):
        import random

        class _Result:
            def __init__(self, action_type: int):
                self.action_type = action_type

        legal_indices = [i for i, ok in enumerate(legal_mask) if ok]
        return _Result(action_type=random.choice(legal_indices))


_cache: Dict[str, _GraphAgentWrapper] = {}


def get_graph_agent(cfg: ScenarioConfig) -> _GraphAgentWrapper:
    """
    Cached per ScenarioConfig.key (not per-checkpoint-path like
    TrainerService._AgentCache — checkpoint selection for graph
    scenarios isn't wired up yet; add a cache-key extension here if/when
    it is, mirroring _AgentCache's own (scenario_key, checkpoint_path)
    tuple key).
    """
    if cfg.key not in _cache:
        raw = _load_graph_agent(cfg) or _RandomGraphFallbackAgent()
        _cache[cfg.key] = _GraphAgentWrapper(raw)
    return _cache[cfg.key]


def _load_graph_agent(cfg: ScenarioConfig):
    import os

    try:
        from poker_rl_lab.trainers.graph_training_loop import (
            load_dual_seat_actor_critic_for_graph,
        )
    except ImportError as e:
        print(f"[trainer_agent_cache] RL Lab not importable, using random fallback: {e}")
        return None

    checkpoint_path = cfg.policy.checkpoint_path
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(
            f"[trainer_agent_cache] no checkpoint at {checkpoint_path!r} for "
            f"graph scenario {cfg.key!r} — using random fallback."
        )
        return None

    try:
        return load_dual_seat_actor_critic_for_graph(checkpoint_path, device="cpu")
    except Exception as e:
        print(
            f"[trainer_agent_cache] failed to load graph checkpoint "
            f"{checkpoint_path!r} for {cfg.key!r}: {type(e).__name__}: {e}"
        )
        return None