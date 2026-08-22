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
    # Directory the user can pick a checkpoint from at runtime (see
    # TrainerService.list_checkpoints/select_checkpoint). Optional —
    # scenarios that don't set this only ever use checkpoint_path, same
    # as before. Resolved the same way as checkpoint_path itself
    # (relative to the poker_rl_lab package root unless absolute).
    checkpoint_dir: str | None = None


@dataclass(frozen=True)
class EvaluatorConfig:
    type: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ActionMapEntry:
    """
    One Trainer-level action string's mapping onto the RL Lab's abstract
    ActionType vocabulary (poker_rl_lab.actions.abstract_action.ActionType),
    plus — for bet-like actions — how TrainerService should size the
    concrete engine Action it builds (see trainer_service.py's
    _build_engine_action).

    abstract : str
        The abstract action's name, e.g. "bet_100", "fold", "all_in".
        Must be a name NUM_ACTIONS/ActionType actually defines.
    sizing : str | None
        How this action's bet amount is computed. One of:
          - None        : not a bet-like action (fold/check/call/all_in) —
                           sizing is either meaningless or already fully
                           determined by the engine itself.
          - "pot"        : sized to `pot_fraction` * the current pot,
                           capped at the player's stack. 1.0 = pot-sized.
          - "stack"      : sized to the player's ENTIRE remaining stack —
                           a true shove — regardless of pot size.
        Per-scenario, not hardcoded: a river-style scenario training
        pot-bet-or-check decisions declares "pot"; a scenario meant to
        train shove-or-check decisions (where "bet" should always commit
        the whole stack, independent of how big the dead pot happens to
        be) declares "stack" instead — see training_config.yaml's
        river_duo / double_board_plo_river_duo entries for both.
    pot_fraction : float | None
        Required (and only meaningful) when sizing == "pot". Ignored
        when sizing == "stack".

    A scenario offering a second bet size (e.g. half-pot alongside a
    full shove) declares a SECOND Trainer-level action name (e.g.
    "bet_half") with its own ActionMapEntry(abstract="bet_50",
    sizing="pot", pot_fraction=0.5) — no code change needed anywhere in
    trainer_service.py for that to work.
    """
    abstract: str
    sizing: str | None = None
    pot_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.sizing not in (None, "pot", "stack"):
            raise ValueError(
                f"ActionMapEntry.sizing must be None, 'pot', or 'stack' — "
                f"got {self.sizing!r} (abstract={self.abstract!r})"
            )
        if self.sizing == "pot" and self.pot_fraction is None:
            raise ValueError(
                f"ActionMapEntry(abstract={self.abstract!r}) declares "
                f"sizing='pot' but no pot_fraction — a pot-sized bet "
                f"needs a fraction (1.0 = pot-sized)."
            )


# Every scenario that predates action_map (push_fold_duo) only ever uses
# action names that are ALREADY valid abstract ActionType names 1:1
# (fold/call/check/all_in — no scenario before "bet" existed used a name
# that needed translating), so this default just maps each declared
# action's Trainer name onto ITSELF as the abstract name with no bet
# sizing. Any scenario introducing "bet" (or any other name that isn't
# already an abstract ActionType name) MUST declare its own action_map
# in training_config.yaml — see river_duo's entry for the pattern.
def _default_action_map(actions: dict[str, list[str]]) -> dict[str, ActionMapEntry]:
    names: set[str] = set()
    for action_list in actions.values():
        names.update(action_list)
    return {name: ActionMapEntry(abstract=name) for name in names}


def _parse_action_map(raw: dict | None, actions: dict[str, list[str]]) -> dict[str, ActionMapEntry]:
    if not raw:
        return _default_action_map(actions)
    parsed = {}
    for trainer_name, entry_raw in raw.items():
        if isinstance(entry_raw, str):
            # Shorthand: `fold: fold` — a non-bet action with no sizing
            # to spell out.
            parsed[trainer_name] = ActionMapEntry(abstract=entry_raw)
            continue

        sizing = entry_raw.get("sizing")
        pot_fraction = entry_raw.get("pot_fraction")
        if sizing is None and pot_fraction is not None:
            # Backward-compat shorthand: a bare `pot_fraction` with no
            # explicit `sizing` key implies sizing: pot (this was the
            # only mode that existed before `sizing` was introduced).
            sizing = "pot"

        parsed[trainer_name] = ActionMapEntry(
            abstract=entry_raw["abstract"],
            sizing=sizing,
            pot_fraction=pot_fraction,
        )
    return parsed


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
    action_map: dict[str, ActionMapEntry]
    pot_bb: PotRange | None = None

    def abstract_to_trainer_action(self, abstract_name: str) -> str | None:
        """
        Reverse lookup through this scenario's own action_map: abstract
        ActionType name -> Trainer-level action string, or None if this
        scenario never declares an action mapping onto that abstract
        name. Computed on demand (action_map is small, called
        infrequently — no need to cache) rather than duplicating this
        loop at every trainer_service.py call site.
        """
        for trainer_name, entry in self.action_map.items():
            if entry.abstract == abstract_name:
                return trainer_name
        return None

    def trainer_to_abstract(self, trainer_name: str) -> str | None:
        entry = self.action_map.get(trainer_name)
        return entry.abstract if entry else None

    def is_bet_like(self, trainer_name: str) -> bool:
        """True if this scenario's action_map declares `trainer_name`
        as a bet-like action (sizing is "pot" or "stack") — the
        general replacement for any hardcoded `== "bet"` check."""
        entry = self.action_map.get(trainer_name)
        return entry is not None and entry.sizing is not None

    def sizing_for(self, trainer_name: str) -> str | None:
        entry = self.action_map.get(trainer_name)
        return entry.sizing if entry else None

    def pot_fraction_for(self, trainer_name: str) -> float | None:
        entry = self.action_map.get(trainer_name)
        return entry.pot_fraction if entry else None

    def bet_like_actions(self) -> list[str]:
        """Every Trainer-level action name this scenario declares as
        bet-like (sizing is "pot" or "stack"), e.g. ["bet"], or
        ["bet_half", "bet_pot"] for a scenario offering more than one
        size. Order follows action_map's own declaration order."""
        return [
            name for name, entry in self.action_map.items()
            if entry.sizing is not None
        ]


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
            checkpoint_dir=_resolve_rl_lab_path(policy_raw.get("checkpoint_dir")),
        ),
        evaluator=EvaluatorConfig(
            type=evaluator_raw["type"],
            params=evaluator_raw.get("params", {}),
        ),
        actions=raw.get("actions", {}),
        action_map=_parse_action_map(raw.get("action_map"), raw.get("actions", {})),
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