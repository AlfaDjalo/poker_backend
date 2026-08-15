"""
app/services/decision_evaluator.py — Pluggable "what was the correct
decision" evaluators for the Trainer.

Default evaluator: RLPolicyFrequencyEvaluator. "Best" = whichever
scenario-allowed action the trained push_fold_duo policy assigns
higher probability to, from a real forward pass at the hero's actual
decision point (real hole cards, real stack, real position). No chart,
no separate equity solve — the model's own softmax output over legal
actions IS the ground truth. TrainerService computes those
probabilities (it already has the loaded agent + RepresentationInput)
and hands them in as plain data via DecisionContext, so this module
stays framework-agnostic (no torch import here).

get_evaluator() is a registry keyed on training_config.yaml's
evaluator.type, so a different "best answer" method can be swapped in
later (e.g. grading against a solved chart, or an ICM-adjusted
threshold) without touching TrainerService.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.config.trainer_config import EvaluatorConfig

_RANK_ORDER = "23456789TJQKA"


def canonicalize_hole_cards(card_strs: list[str]) -> str:
    """
    Convert two hole-card strings (e.g. ["Ah", "Kd"]) into standard
    notation: "AKs" (suited), "AKo" (offsuit), or "TT" (pocket pair).
    Used only for display (scoreboard history), not for grading.
    """
    if len(card_strs) != 2:
        raise ValueError(f"canonicalize_hole_cards expects 2 cards, got {card_strs!r}")

    (r1, s1), (r2, s2) = card_strs[0], card_strs[1]
    ranks = sorted(
        [r1.upper(), r2.upper()], key=lambda r: _RANK_ORDER.index(r), reverse=True
    )

    if ranks[0] == ranks[1]:
        return f"{ranks[0]}{ranks[1]}"

    suited = s1.lower() == s2.lower()
    return f"{ranks[0]}{ranks[1]}{'s' if suited else 'o'}"


@dataclass(frozen=True)
class DecisionContext:
    """Everything an evaluator needs to judge one hero decision."""

    hole_cards: list[str]  # e.g. ["Ah", "Kd"]
    position: str  # "SB" | "BB"
    effective_stack_bb: float
    hero_action: str  # what the hero actually chose
    allowed_actions: list[str] = field(default_factory=list)
    # Policy's own probability for each allowed action, computed by
    # TrainerService from a real forward pass. e.g. {"fold": 0.82, "all_in": 0.18}
    action_probs: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionResult:
    correct: bool
    best_action: str
    hero_action: str
    explanation: str = ""


class BaseDecisionEvaluator(ABC):
    @abstractmethod
    def evaluate(self, ctx: DecisionContext) -> DecisionResult: ...


# ---------------------------------------------------------------------------
# rl_policy_frequency — default: highest-probability legal action wins
# ---------------------------------------------------------------------------


class RLPolicyFrequencyEvaluator(BaseDecisionEvaluator):
    """
    Best action = argmax over ctx.allowed_actions by ctx.action_probs.

    For push_fold_duo this is exactly:
        SB: higher of P(all_in) vs P(fold)
        BB: higher of P(call)   vs P(fold)

    Ties (exact float equality) are broken by taking the first action
    in allowed_actions' configured order — trivially unlikely with a
    continuous softmax output, and the simplest correct tie-break.
    """

    def evaluate(self, ctx: DecisionContext) -> DecisionResult:
        if not ctx.allowed_actions:
            raise ValueError("RLPolicyFrequencyEvaluator requires ctx.allowed_actions")

        best = max(ctx.allowed_actions, key=lambda a: ctx.action_probs.get(a, 0.0))

        probs_str = ", ".join(
            f"{a}={ctx.action_probs.get(a, 0.0):.3f}" for a in ctx.allowed_actions
        )
        explanation = f"{ctx.position} policy frequencies: [{probs_str}]"

        return DecisionResult(
            correct=(best == ctx.hero_action),
            best_action=best,
            hero_action=ctx.hero_action,
            explanation=explanation,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_evaluator(config: EvaluatorConfig) -> BaseDecisionEvaluator:
    if config.type == "rl_policy_frequency":
        return RLPolicyFrequencyEvaluator()

    raise ValueError(f"Unknown evaluator type: {config.type!r}")
