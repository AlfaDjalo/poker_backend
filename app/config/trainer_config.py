"""
app/config/trainer_config.py — Generic loader for training_config.yaml.

Deliberately scenario-agnostic: this module knows nothing about
"push_fold_duo" specifically. It parses the YAML into plain dataclasses
mirroring the file's structure, and exposes lookup helpers. Adding a
new scenario to training_config.yaml requires zero changes here, as
long as it fits the same shape (policy/evaluator/actions blocks).

Any scenario-specific interpretation (e.g. "what does stack_bb.min
mean for push_fold_duo") lives in the service that consumes it
(trainer_service.py / push_fold evaluators), not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "training_config.yaml"


# training_config.yaml's checkpoint_path/encoder_path are documented as
# "relative to the poker_rl_lab base" — i.e. relative to the INSTALLED
# package directory, not whatever the backend process's cwd happens to
# be. This mirrors how poker_engine.games.loader resolves variant YAMLs
# relative to poker_engine's own package __file__ rather than cwd, so
# `pip install`-ing either package from a different working directory
# doesn't break path resolution. Resolved once here at parse time so
# every downstream consumer (trainer_service.py's os.path.exists /
# open() calls) just sees a plain absolute path and never needs to know
# about this convention itself.
def _rl_lab_root() -> Path | None:
    try:
        import poker_rl_lab
    except ImportError:
        return None
    # poker_rl_lab/__init__.py -> parent is the poker_rl_lab package dir,
    # which is exactly what "relative to the poker_rl_lab base" means.
    return Path(poker_rl_lab.__file__).resolve().parent


def _resolve_rl_lab_path(raw: str | None) -> str | None:
    """
    Resolve a config path relative to the installed poker_rl_lab package
    root. Leaves already-absolute paths untouched (so an operator can
    still override with a fully-qualified path, e.g. pointing at a
    checkpoint stored outside the package entirely). Falls back to the
    raw string unresolved if poker_rl_lab isn't importable yet — callers
    (trainer_service.py) already handle a missing/bad path gracefully
    (allow_untrained_fallback / RandomFallbackAgent), so this must never
    raise.
    """
    if raw is None:
        return None
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    root = _rl_lab_root()
    if root is None:
        return raw
    return str((root / p).resolve())


@dataclass(frozen=True)
class StackRange:
    min: float
    max: float


@dataclass(frozen=True)
class PotRange:
    """Dead-money pot range in big blinds for scenarios whose hero is
    dropped in mid-hand (e.g. river_duo) rather than starting from
    posted blinds. None for scenarios (push_fold_duo) with no such
    concept — the pot there is built entirely from posted blinds /
    the push itself."""

    min: float
    max: float


@dataclass(frozen=True)
class PolicyConfig:
    checkpoint_path: str
    encoder_path: str | None = None
    encoder_embedding_dim: int | None = None
    variant: str = "shared_trunk_two_heads"
    allow_untrained_fallback: bool = True


@dataclass(frozen=True)
class EvaluatorConfig:
    type: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioConfig:
    key: str
    label: str
    description: str
    engine_variant: str
    rl_game_name: str
    betting_type: str
    hero_seat: int
    villain_seat: int
    stack_bb: StackRange
    villain_matches_hero_range: bool
    small_blind: int
    big_blind: int
    positions: list[str]
    policy: PolicyConfig
    evaluator: EvaluatorConfig
    actions: dict[str, list[str]]
    pot_bb: PotRange | None = None


def _parse_scenario(key: str, raw: dict) -> ScenarioConfig:
    stack_raw = raw["stack_bb"]
    policy_raw = raw["policy"]
    evaluator_raw = raw["evaluator"]
    pot_raw = raw.get("pot_bb")

    return ScenarioConfig(
        key=key,
        label=raw.get("label", key),
        description=raw.get("description", ""),
        engine_variant=raw["engine_variant"],
        rl_game_name=raw.get("rl_game_name", raw["engine_variant"]),
        betting_type=raw["betting_type"],
        hero_seat=raw.get("hero_seat", 0),
        villain_seat=raw.get("villain_seat", 1),
        stack_bb=StackRange(min=float(stack_raw["min"]), max=float(stack_raw["max"])),
        villain_matches_hero_range=raw.get("villain_matches_hero_range", True),
        small_blind=raw.get("small_blind", 1),
        big_blind=raw.get("big_blind", 2),
        positions=raw.get("positions", ["SB", "BB"]),
        policy=PolicyConfig(
            checkpoint_path=_resolve_rl_lab_path(policy_raw["checkpoint_path"]),
            encoder_path=_resolve_rl_lab_path(policy_raw.get("encoder_path")),
            encoder_embedding_dim=policy_raw.get("encoder_embedding_dim"),
            variant=policy_raw.get("variant", "shared_trunk_two_heads"),
            allow_untrained_fallback=policy_raw.get("allow_untrained_fallback", True),
        ),
        evaluator=EvaluatorConfig(
            type=evaluator_raw["type"],
            params=evaluator_raw.get("params", {}),
        ),
        actions=raw.get("actions", {}),
        pot_bb=(
            PotRange(min=float(pot_raw["min"]), max=float(pot_raw["max"]))
            if pot_raw
            else None
        ),
    )


class TrainingConfig:
    """
    Loads and caches training_config.yaml. Call TrainingConfig.load()
    (module-level singleton via get_training_config()) rather than
    constructing directly in most call sites.
    """

    def __init__(self, path: Path = _CONFIG_PATH):
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        self._scenarios: dict[str, ScenarioConfig] = {
            key: _parse_scenario(key, val)
            for key, val in (raw.get("scenarios") or {}).items()
        }

    def list_scenarios(self) -> list[ScenarioConfig]:
        return list(self._scenarios.values())

    def get_scenario(self, key: str) -> ScenarioConfig:
        if key not in self._scenarios:
            raise KeyError(f"Unknown trainer scenario: {key!r}")
        return self._scenarios[key]

    def has_scenario(self, key: str) -> bool:
        return key in self._scenarios


_singleton: TrainingConfig | None = None


def get_training_config() -> TrainingConfig:
    global _singleton
    if _singleton is None:
        _singleton = TrainingConfig()
    return _singleton


def reload_training_config() -> TrainingConfig:
    """Force a re-read of training_config.yaml (e.g. after editing charts/checkpoints)."""
    global _singleton
    _singleton = TrainingConfig()
    return _singleton
