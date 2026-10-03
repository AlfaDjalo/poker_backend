"""
app/config/trainer_config.py — Loader for Trainer scenario configs.

REWRITTEN: scenario config ownership has moved to RL Lab (see
RL_LAB_COORDINATION_REQUEST.md). This module no longer reads a single
backend-owned training_config.yaml; instead it discovers one YAML file
per scenario from RL Lab's own `scenarios/` directory, the same way
game_service.get_variants() discovers Engine variants from
poker_engine's own `games/` directory (resources.files(...) against
the installed package) — Engine and RL Lab are both directly
importable, so no new IPC/network boundary is introduced, just the
same "ask the package where its own data lives" pattern already used
for Engine.

Directory/path resolution (see rl_lab_paths.yaml + the coordination
doc's §3/§4) is three-tier, checked in this order for both
scenarios_dir and models_dir independently:
  1. POKER_RL_LAB_SCENARIOS_DIR / POKER_RL_LAB_MODELS_DIR env var
  2. scenarios_dir / models_dir in app/config/rl_lab_paths.yaml
  3. default: <poker_rl_lab package dir>/scenarios or /models

Engine remains the sole source of truth for which GAMES exist — each
scenario's engine_variant is validated against
poker_engine/games/<name>.yaml at load time; a scenario naming an
engine_variant Engine doesn't have is skipped (logged) rather than
crashing scenario discovery for every other scenario.

Deliberately still scenario-agnostic in the same sense as before: this
module knows nothing about "push_fold_duo" specifically. Any
scenario-specific interpretation lives in the service that consumes it
(trainer_service.py / push_fold evaluators), not here.

PUBLIC API IS UNCHANGED from the previous revision — StackRange,
PotRange, PolicyConfig, EvaluatorConfig, ActionMapEntry, ScenarioConfig
(including all its methods), TrainingConfig (list_scenarios/
get_scenario/has_scenario), get_training_config(), and
reload_training_config() all keep the same names and shapes, so
trainer_service.py / trainer_api.py need no changes at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).parent
_RL_LAB_PATHS_CONFIG_PATH = _CONFIG_DIR / "rl_lab_paths.yaml"


# ---------------------------------------------------------------------------
# Directory resolution — scenarios_dir / models_dir, three-tier per the
# coordination doc's §3/§4 (env var > rl_lab_paths.yaml > package default).
# ---------------------------------------------------------------------------


def _load_rl_lab_paths_config() -> dict:
    """
    Parses app/config/rl_lab_paths.yaml. Missing file (e.g. a dev
    checkout that hasn't added it yet) is NOT an error — it just means
    tier 2 has nothing to offer and resolution falls through to tier 3
    (or tier 1's env vars, checked independently beforehand).
    """
    if not _RL_LAB_PATHS_CONFIG_PATH.exists():
        return {}
    with open(_RL_LAB_PATHS_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


def _poker_rl_lab_pkg_dir() -> Path | None:
    """
    <poker_rl_lab package dir> — mirrors game_service.get_variants()'s
    own use of importlib.resources against poker_engine, applied here
    to poker_rl_lab instead. Returns None if poker_rl_lab isn't
    importable in this environment (e.g. a minimal deployment that
    only runs the live game, no Trainer) — callers treat that as "no
    scenarios available" rather than raising, same as before.
    """
    try:
        import poker_rl_lab
    except ImportError:
        return None
    return Path(poker_rl_lab.__file__).resolve().parent


def _resolve_configured_dir(
    env_var: str, configured_override: str | None, default_subdir: str
) -> Path | None:
    """
    One directory's three-tier resolution — shared logic for both
    scenarios_dir and models_dir, parameterized by which env var and
    which default subdirectory name each uses.
    """
    from_env = os.environ.get(env_var)
    if from_env:
        return Path(from_env).resolve()

    if configured_override:
        p = Path(configured_override)
        if not p.is_absolute():
            p = (_CONFIG_DIR / p).resolve()
        return p

    root = _poker_rl_lab_pkg_dir()
    if root is None:
        return None
    return root / default_subdir


def _default_scenarios_dir() -> Path | None:
    # "trainer_scenarios" — deliberately NOT "scenarios" (that's the RL
    # Lab's own Python package, containing graph_scenario_config.py etc,
    # not a data directory) and NOT "graph_scenarios" (RL Lab's own
    # TRAINING-time scenario configs — GraphScenarioConfig schema: scope/
    # action_abstraction/num_seats — a different schema entirely from
    # this file's ScenarioConfig, which is Backend/Trainer-UI-owned:
    # hero_seat/policy.checkpoint_path/evaluator/actions/action_map. The
    # two are intentionally separate files, see the Backend/RL Lab
    # coordination discussion this directory name resolves.
    cfg = _load_rl_lab_paths_config()
    return _resolve_configured_dir(
        "POKER_RL_LAB_SCENARIOS_DIR", cfg.get("scenarios_dir"), "trainer_scenarios"
    )


def _default_models_dir() -> Path | None:
    # "" (not "models") — resolves to the poker_rl_lab package ROOT,
    # since checkpoints/ and models/ are both real sibling directories
    # there (see rl_lab_paths.yaml's own comments) and scenario YAMLs
    # reference paths like "checkpoints/push_fold_duo/ckpt.pt" relative
    # to that root, not nested under a "models/" subdir that doesn't
    # actually contain them. Path("x") / "" == Path("x"), so this is a
    # safe no-op default_subdir, not a bug.
    cfg = _load_rl_lab_paths_config()
    return _resolve_configured_dir(
        "POKER_RL_LAB_MODELS_DIR", cfg.get("models_dir"), ""
    )


def _default_graph_scenarios_dir() -> Path | None:
    """
    RL Lab's own GraphScenarioConfig directory — one YAML per training
    scenario (poker_rl_lab.scenarios.graph_scenario_config), e.g.
    graph_scenarios/push_fold_duo.yaml. Deliberately a SEPARATE
    directory from _default_scenarios_dir()'s "trainer_scenarios" (this
    module's own ScenarioConfig schema: hero_seat/policy.checkpoint_path
    /evaluator/actions/action_map — Backend/Trainer-UI-owned) — same
    naming-collision rationale trainer_scenarios' own default-subdir
    comment already documents for why these two schemas live in
    separate directories with different names, not the same one.

    Used ONLY to auto-detect which scenarios have a GraphScenarioEnv
    counterpart available (see ScenarioConfig.graph_scenario_key) — the
    file's actual CONTENTS are never read here, only its existence.
    poker_rl_lab.scenarios.graph_scenario_config.load_graph_scenario()
    is what actually parses it, called lazily by
    graph_scenario_trainer.py only once a migrated scenario is started.
    """
    cfg = _load_rl_lab_paths_config()
    return _resolve_configured_dir(
        "POKER_RL_LAB_GRAPH_SCENARIOS_DIR",
        cfg.get("graph_scenarios_dir"),
        "graph_scenarios",
    )


def _resolve_model_path(raw: str | None, models_dir: Path | None) -> str | None:
    """
    Resolve one policy path field (checkpoint_path / checkpoint_dir /
    encoder_path) relative to models_dir — the direct replacement for
    the previous revision's _resolve_rl_lab_path(), which resolved
    against the poker_rl_lab PACKAGE root instead. An already-absolute
    path is still honored as-is (an operator can still point directly
    at a checkpoint stored anywhere). Falls back to the raw string
    unresolved if models_dir couldn't be determined at all — callers
    (trainer_service.py) already handle a missing/bad path gracefully
    (allow_untrained_fallback / RandomFallbackAgent), so this must
    never raise.
    """
    if raw is None:
        return None
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    if models_dir is None:
        return raw
    return str((models_dir / p).resolve())


def _engine_variant_exists(engine_variant: str) -> bool:
    """
    Cross-checks a scenario's engine_variant against Engine's own
    games directory — Engine is the source of truth for which games
    exist (see module docstring). Fails OPEN (returns True) if the
    check itself can't run (e.g. poker_engine not importable here) —
    that's a different, more fundamental problem than a single
    scenario's config, and shouldn't silently hide every scenario
    behind an unrelated import failure.
    """
    try:
        from importlib import resources

        games_dir = Path(str(resources.files("poker_engine") / "games"))
    except Exception:
        return True
    return (games_dir / f"{engine_variant}.yaml").exists()


# ---------------------------------------------------------------------------
# Dataclasses — UNCHANGED from the previous revision.
# ---------------------------------------------------------------------------


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
    # (relative to RL Lab's models_dir unless absolute — see
    # _resolve_model_path).
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
        be) declares "stack" instead.
    pot_fraction : float | None
        Required (and only meaningful) when sizing == "pot". Ignored
        when sizing == "stack".
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
            parsed[trainer_name] = ActionMapEntry(abstract=entry_raw)
            continue

        sizing = entry_raw.get("sizing")
        pot_fraction = entry_raw.get("pot_fraction")
        if sizing is None and pot_fraction is not None:
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

    # ── GraphEngine migration (see RL Lab coordination thread:
    # "Migrate trainer_service.py / push_fold_service.py off PokerState
    # onto GraphEngine") ──────────────────────────────────────────
    #
    # AUTO-DETECTED, never hand-set in a Backend YAML file: True exactly
    # when RL Lab's graph_scenarios/ directory (see
    # _default_graph_scenarios_dir()) contains a file named
    # "{key}.yaml" at TrainingConfig load time. No manual per-scenario
    # flag to maintain — dropping a new graph_scenarios/{key}.yaml file
    # into that directory (RL Lab's own, already-directory-discovered
    # per _default_scenarios_dir's existing pattern) is the entire
    # "migrate this scenario" action from Backend's point of view.
    # Populated by TrainingConfig.__init__ via _parse_scenario's
    # graph_scenario_available param — never set any other way.
    graph_scenario_key: str | None = None

    # Resolved absolute path to the matched graph_scenarios/{key}.yaml
    # file, populated alongside graph_scenario_key at TrainingConfig
    # load time. Passed explicitly as load_graph_scenario()'s `path`
    # argument — RL Lab confirmed name-only resolution only checks
    # <poker_rl_lab package>/graph_scenarios/{name}.yaml, which is just
    # the DEFAULT tier of _default_graph_scenarios_dir()'s own three-tier
    # resolution (env var / rl_lab_paths.yaml / package default) — an
    # operator-configured graph_scenarios_dir (tiers 1-2) would silently
    # not be found by name alone, so this must always be passed rather
    # than relying on load_graph_scenario("key")'s own default lookup.
    graph_scenario_path: Path | None = None

    def is_graph_migrated(self) -> bool:
        """True once this scenario should be driven via GraphScenarioEnv
        rather than PokerState — see graph_scenario_key's own docstring."""
        return self.graph_scenario_key is not None

    def resolved_graph_scenario_key(self) -> str:
        """The actual graph_scenarios/*.yaml stem to load — always
        `self.key` today (auto-detection only ever matches on identical
        stem), kept as its own method rather than inlining `self.key`
        at call sites in case a future override mechanism is added."""
        if self.graph_scenario_key is None:
            raise ValueError(
                f"Scenario {self.key!r} has no matching graph_scenarios/"
                f"{self.key}.yaml — it hasn't been migrated to "
                f"GraphScenarioEnv yet."
            )
        return self.graph_scenario_key

    def abstract_to_trainer_action(self, abstract_name: str) -> str | None:
        for trainer_name, entry in self.action_map.items():
            if entry.abstract == abstract_name:
                return trainer_name
        return None

    def trainer_to_abstract(self, trainer_name: str) -> str | None:
        entry = self.action_map.get(trainer_name)
        return entry.abstract if entry else None

    def is_bet_like(self, trainer_name: str) -> bool:
        entry = self.action_map.get(trainer_name)
        return entry is not None and entry.sizing is not None

    def sizing_for(self, trainer_name: str) -> str | None:
        entry = self.action_map.get(trainer_name)
        return entry.sizing if entry else None

    def pot_fraction_for(self, trainer_name: str) -> float | None:
        entry = self.action_map.get(trainer_name)
        return entry.pot_fraction if entry else None

    def bet_like_actions(self) -> list[str]:
        return [
            name for name, entry in self.action_map.items()
            if entry.sizing is not None
        ]


def _parse_scenario(
    key: str,
    raw: dict,
    models_dir: Path | None,
    graph_scenario_available: bool = False,
    graph_scenario_path: Path | None = None,
) -> ScenarioConfig:
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
            checkpoint_path=_resolve_model_path(policy_raw["checkpoint_path"], models_dir),
            encoder_path=_resolve_model_path(policy_raw.get("encoder_path"), models_dir),
            encoder_embedding_dim=policy_raw.get("encoder_embedding_dim"),
            variant=policy_raw.get("variant", "shared_trunk_two_heads"),
            allow_untrained_fallback=policy_raw.get("allow_untrained_fallback", True),
            checkpoint_dir=_resolve_model_path(policy_raw.get("checkpoint_dir"), models_dir),
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
        graph_scenario_key=(key if graph_scenario_available else None),
        graph_scenario_path=(graph_scenario_path if graph_scenario_available else None),
    )


# ---------------------------------------------------------------------------
# Registry — discovers one file per scenario from scenarios_dir, instead
# of parsing a single backend-owned YAML.
# ---------------------------------------------------------------------------


class TrainingConfig:
    """
    Loads and caches every scenario file found in RL Lab's scenarios
    directory. Call TrainingConfig.load() — actually, call the
    module-level singleton via get_training_config() — rather than
    constructing directly in most call sites (kept identical to the
    previous revision's own guidance).

    A single malformed/invalid scenario file is skipped (logged to
    `load_errors`) rather than taking down scenario discovery for
    every other scenario — this is new behavior versus the previous
    single-YAML revision, where one bad entry in training_config.yaml
    would have broken yaml parsing (or the dict comprehension) for the
    whole file.
    """

    def __init__(
        self,
        scenarios_dir: Path | None = None,
        models_dir: Path | None = None,
        graph_scenarios_dir: Path | None = None,
    ):
        self.scenarios_dir = (
            scenarios_dir if scenarios_dir is not None else _default_scenarios_dir()
        )
        self.models_dir = models_dir if models_dir is not None else _default_models_dir()
        self.graph_scenarios_dir = (
            graph_scenarios_dir
            if graph_scenarios_dir is not None
            else _default_graph_scenarios_dir()
        )

        # Stems present in graph_scenarios_dir, mapped to their resolved
        # file path — computed ONCE here. This is the entire "migrate a
        # scenario" surface from Backend's point of view (see
        # ScenarioConfig.graph_scenario_key's own docstring): dropping a
        # same-stemmed YAML file into RL Lab's directory is sufficient,
        # no Backend code or config edit needed. An unreadable/missing
        # directory yields an empty dict rather than raising — every
        # scenario just stays on the legacy PokerState path, same as
        # before this feature existed.
        self._graph_scenario_paths: dict[str, Path] = {}
        if self.graph_scenarios_dir is not None and self.graph_scenarios_dir.exists():
            self._graph_scenario_paths = {
                p.stem: p for p in self.graph_scenarios_dir.glob("*.yaml")
            }

        self._scenarios: dict[str, ScenarioConfig] = {}
        self._load_errors: list[str] = []

        if self.scenarios_dir is None or not self.scenarios_dir.exists():
            msg = (
                f"[trainer_config] No RL Lab scenarios directory found "
                f"(resolved to {self.scenarios_dir!r}) — the Trainer will "
                f"have zero scenarios available. Confirm poker_rl_lab is "
                f"installed and exposes a 'scenarios/' directory, or set "
                f"scenarios_dir in app/config/rl_lab_paths.yaml / the "
                f"POKER_RL_LAB_SCENARIOS_DIR env var."
            )
            print(msg)
            self._load_errors.append(msg)
            return

        for path in sorted(self.scenarios_dir.glob("*.yaml")):
            key = path.stem
            try:
                with open(path, "r") as f:
                    raw = yaml.safe_load(f) or {}

                declared_key = raw.get("key")
                if declared_key and declared_key != key:
                    raise ValueError(
                        f"declares key {declared_key!r}, which doesn't "
                        f"match its filename stem {key!r} — rename one "
                        f"to match the other."
                    )

                engine_variant = raw.get("engine_variant")
                if not engine_variant:
                    raise ValueError("missing required field 'engine_variant'")
                if not _engine_variant_exists(engine_variant):
                    raise ValueError(
                        f"engine_variant {engine_variant!r} has no matching "
                        f"poker_engine/games/{engine_variant}.yaml — Engine "
                        f"is the source of truth for available games; add "
                        f"that variant there before shipping this scenario."
                    )

                self._scenarios[key] = _parse_scenario(
                    key,
                    raw,
                    self.models_dir,
                    graph_scenario_available=(key in self._graph_scenario_paths),
                    graph_scenario_path=self._graph_scenario_paths.get(key),
                )
            except Exception as e:
                msg = f"[trainer_config] skipping scenario file {path.name!r}: {e}"
                print(msg)
                self._load_errors.append(msg)

    def list_scenarios(self) -> list[ScenarioConfig]:
        return list(self._scenarios.values())

    def get_scenario(self, key: str) -> ScenarioConfig:
        if key not in self._scenarios:
            raise KeyError(f"Unknown trainer scenario: {key!r}")
        return self._scenarios[key]

    def has_scenario(self, key: str) -> bool:
        return key in self._scenarios

    @property
    def load_errors(self) -> list[str]:
        """Every scenario file skipped at load time, with why — surfaced
        by trainer_api.py's diagnostics endpoint for ops visibility."""
        return list(self._load_errors)


_singleton: TrainingConfig | None = None


def get_training_config() -> TrainingConfig:
    global _singleton
    if _singleton is None:
        _singleton = TrainingConfig()
    return _singleton


def reload_training_config() -> TrainingConfig:
    """Force a re-read of RL Lab's scenario directory (e.g. after RL Lab
    ships a new scenario file or checkpoint without a backend restart)."""
    global _singleton
    _singleton = TrainingConfig()
    return _singleton