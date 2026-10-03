"""
app/services/push_fold_service.py — Backend service for playing
push-fold heads-up against the trained RL policy (see
RL_lab_integration_pushfoldduo.md).

Reuses the existing engine end-to-end
------------------------------------------
This uses exactly the same machinery as app/services/game_service.py:

    poker_engine.games.loader.load_game("push_fold")   — new variant,
        see push_fold.yaml. Same single-board HIGH/HOLDEM board+showdown
        definition as holdem.yaml, but betting.type: push_fold, which
        activates GameState's push-fold branch of legal_actions()/
        apply_action() (fold, or full-stack shove/call only — see
        game_state.py and betting_rules.py's AllInOrFoldRules).
    poker_engine.state.poker_state.PokerState / Phase — same hand
        lifecycle (start_hand(), step(), DEAL_BOARD auto-progression)
        GameService already drives.
    poker_engine.scoring.scoring_engine.CppScoringEngine +
        poker_engine.showdown.showdown_resolver.ShowdownResolver
        (used internally by PokerState) — same showdown resolution.
    app.engine_adapter.state_to_dto — same rich GameStateDTO the main
        GameSimulator already knows how to render (players, pot, phase,
        available_actions, showdown, winners, ...). The push-fold
        frontend mode reuses this shape rather than inventing a new one.

SB/BB alternates automatically
--------------------------------
PokerState.start_hand() increments dealer_position and computes
first-to-act preflop as (dealer_position + 1) % n — for heads-up that
is always whichever seat posts the SMALL blind (see the note added to
poker_state.py). So simply calling start_hand() again each hand
alternates which physical seat is SB/BB with no extra bookkeeping here
beyond seeding dealer_position's pre-increment parity once per hand so
a freshly-constructed PokerState (dealer_position always starts at 0)
doesn't always hand seat 0 the button.

Stacks are episodic, not a persistent bankroll
------------------------------------------------
Per spec: the human always starts a hand at 100bb, and the agent's
stack is redrawn uniformly in [5bb, 25bb] every hand — matching
scenarios.push_fold_duo_generator's own training distribution
(min_stack_bb=4.0, max_stack_bb=20.0, close enough to spec's 5-25bb
that reusing that shape here is intentional, not coincidental). This
means a fresh PokerState (fresh PlayerState stacks) is built for every
hand rather than carrying stacks forward, unlike the main game's
/game/new-hand.

RL observation game_name stays "holdem", not "push_fold"
------------------------------------------------------------
This is the one place the new push_fold variant is deliberately NOT
used: RL_lab_integration_pushfoldduo.md's own build_sb_obs()/
build_bb_obs() reference implementations hardcode game_name="holdem",
meaning the trained policy's GameEncoder token was computed from
holdem.yaml's GameDefinition/GameRules (betting_type="no_limit")
during training, not from a push-fold-flavored one. Feeding the new
push_fold.yaml's betting_type into GameEncoder at inference time would
map to its "other" bucket (see game_encoder.py's _betting_type_id()
fallback) — technically graceful, but out-of-distribution versus what
the policy actually trained against. So the ENGINE plays the hand using
push_fold.yaml (for correct legal-action restriction), while the
OBSERVATION fed to the policy still declares itself "holdem" (RL_GAME_NAME
below), matching training exactly. The action-space restriction itself
comes entirely from the legal_mask built in _sb_legal_mask()/
_bb_legal_mask(), not from game_name.
"""

from __future__ import annotations

import os
import random
from typing import Any

from poker_engine.actions.action import Action as EngineAction
from poker_engine.actions.action_type import ActionType as EngineActionType
from poker_engine.cards.mask import mask_to_card_ids
from poker_engine.games.loader import load_game
from poker_engine.scoring.scoring_engine import CppScoringEngine
from poker_engine.state.player_state import PlayerState
from poker_engine.state.poker_state import Phase, PokerState

HUMAN_SEAT = 0
AGENT_SEAT = 1

HUMAN_STACK_BB = 100.0
AGENT_MIN_STACK_BB = 5.0
AGENT_MAX_STACK_BB = 25.0

# Explicit now — GameDefinition (as returned by the now-unified
# load_game()) no longer carries small_blind/big_blind at all (see
# game_definition.py's own docstring: "betting type/blinds/ante... now
# flow-step config"). These MUST agree with whatever push_fold.yaml's
# own flow: post_blinds step actually posts — nothing cross-checks that
# for this legacy PokerState-based path anymore.
SMALL_BLIND = 1
BIG_BLIND = 2

ENGINE_VARIANT_NAME = "push_fold"  # drives real legal actions / showdown
RL_GAME_NAME = "holdem"  # matches the policy's training-time observations


# ---------------------------------------------------------------------------
# Fallback agent (used until a real checkpoint is available)
# ---------------------------------------------------------------------------


class _FallbackActionResult:
    __slots__ = ("action_type", "entropy", "log_prob", "value")

    def __init__(self, action_type: int):
        self.action_type = action_type
        self.log_prob = 0.0
        self.value = 0.0
        self.entropy = 0.0


class RandomFallbackAgent:
    """
    Stand-in for models.dual_seat_policy.DualSeatActorCritic used when
    no trained checkpoint / ML dependencies are available yet. Picks
    uniformly among legal actions so the API contract can be built and
    tested end-to-end before real weights exist. Swapped out
    automatically by PushFoldService._ensure_agent() the moment a real
    checkpoint / build succeeds.
    """

    def act(self, seat, obs, legal_mask, deterministic: bool = False):
        legal_indices = [i for i, ok in enumerate(legal_mask) if ok]
        return _FallbackActionResult(action_type=random.choice(legal_indices))


def _try_build_real_agent():
    """
    Attempt to build the real DualSeatActorCritic. Returns None if the
    RL Lab package or its ML dependencies aren't importable in this
    environment — caller falls back to RandomFallbackAgent.
    """
    try:
        from poker_rl_lab.experiments.push_fold_duo_experiment import (
            PolicyVariant,
            build_dual_seat_actor_critic,
            load_dual_seat_actor_critic,
        )
    except ImportError as e:
        print(
            f"[push_fold_service] RL Lab not importable, using random fallback agent: {e}"
        )
        return None

    checkpoint_path = os.environ.get("PUSH_FOLD_CHECKPOINT_PATH", "ckpt_duo.pt")

    if os.path.exists(checkpoint_path):
        try:
            agent = load_dual_seat_actor_critic(checkpoint_path)
            print(
                f"[push_fold_service] Loaded trained checkpoint from {checkpoint_path}"
            )
            return agent
        except Exception as e:
            print(
                f"[push_fold_service] Failed to load checkpoint '{checkpoint_path}': {e}"
            )

    try:
        agent = build_dual_seat_actor_critic(PolicyVariant.SHARED_TRUNK_TWO_HEADS)
        agent.eval()
        print(
            "[push_fold_service] No checkpoint found at "
            f"'{checkpoint_path}' — using an UNTRAINED policy network "
            "(no real poker knowledge yet). Set PUSH_FOLD_CHECKPOINT_PATH "
            "once a real checkpoint has been trained."
        )
        return agent
    except Exception as e:
        print(
            f"[push_fold_service] Failed to build untrained agent, using random fallback: {e}"
        )
        return None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class PushFoldService:
    """
    Stateful (one active hand at a time), mirroring GameService's own
    singleton pattern in app/services/game_service.py.
    """

    def __init__(self):
        self.agent = None
        self.adapter = None
        self.state: PokerState | None = None
        self._dealer_parity = -1  # incremented then %2'd each new_hand()
        self.hand_number = 0
        self.last_agent_action: str | None = None

    def _ensure_agent(self):
        if self.agent is None:
            self.agent = _try_build_real_agent() or RandomFallbackAgent()
        return self.agent

    def _ensure_adapter(self):
        if self.adapter is None:
            from poker_rl_lab.representation.observation_adapter import (
                ObservationAdapter,
            )

            self.adapter = ObservationAdapter()
        return self.adapter

    # ------------------------------------------------------------
    # Hand lifecycle
    # ------------------------------------------------------------

    def new_hand(self) -> dict[str, Any]:
        self._ensure_agent()

        game_def, rules, graph = load_game(ENGINE_VARIANT_NAME)
        big_blind = BIG_BLIND

        agent_bb = round(random.uniform(AGENT_MIN_STACK_BB, AGENT_MAX_STACK_BB), 1)
        human_chips = max(big_blind, round(HUMAN_STACK_BB * big_blind))
        agent_chips = max(big_blind, round(agent_bb * big_blind))

        players = [PlayerState(stack=human_chips), PlayerState(stack=agent_chips)]

        self.state = PokerState(
            players, game_def, rules, CppScoringEngine(), callbacks=None
        )

        # Seed the pre-increment dealer parity so SB/BB alternates hand
        # to hand instead of always handing seat 0 the button on a
        # freshly-constructed PokerState — see module docstring.
        self._dealer_parity += 1
        self.state.game.dealer_position = self._dealer_parity % 2

        self.state.start_hand()
        self.hand_number += 1
        self.last_agent_action = None

        self._maybe_run_agent_turn()
        self._progress_engine()

        return self.get_state()

    def apply_hero_action(self, action_type: str) -> dict[str, Any]:
        if self.state is None:
            raise ValueError("No active push-fold hand. Call new_hand() first.")
        if (
            self.state.phase != Phase.BETTING
            or self.state.game.current_player != HUMAN_SEAT
        ):
            raise ValueError("Not the hero's turn to act.")
        if action_type not in ("fold", "call", "all_in"):
            raise ValueError(f"Unknown action_type: {action_type!r}")

        action = self._build_engine_action(action_type)
        self.state.step(action)

        self._maybe_run_agent_turn()
        self._progress_engine()

        return self.get_state()

    def _progress_engine(self):
        """Auto-advance DEAL_BOARD phases, mirroring GameService._progress_engine()."""
        while self.state.phase == Phase.DEAL_BOARD:
            self.state.step(None)

    def _current_seat_is_agent(self) -> bool:
        return (
            self.state.phase == Phase.BETTING
            and self.state.game.current_player == AGENT_SEAT
        )

    def _maybe_run_agent_turn(self):
        while self._current_seat_is_agent():
            self._agent_act()

    # ------------------------------------------------------------
    # Engine action helpers
    # ------------------------------------------------------------

    def _build_engine_action(self, action_type: str) -> EngineAction:
        g = self.state.game
        player = g.players[g.current_player]
        to_call = g.bet_to_call - player.current_bet

        if action_type == "fold":
            return EngineAction(type=EngineActionType.FOLD, amount=None)

        if action_type == "call":
            if to_call <= 0:
                return EngineAction(type=EngineActionType.CHECK, amount=None)
            return EngineAction(
                type=EngineActionType.CALL, amount=min(to_call, player.stack)
            )

        # "all_in" — shove the entire remaining stack. GameState.apply_action
        # additionally clamps this to the full stack itself for push_fold
        # betting_type, so an off-by-one here can't leave chips behind.
        shove_total = player.current_bet + player.stack
        act_type = EngineActionType.RAISE if to_call > 0 else EngineActionType.BET
        return EngineAction(type=act_type, amount=shove_total)

    def _sb_seat(self) -> int:
        g = self.state.game
        return (g.dealer_position + 1) % len(g.players)

    def _bb_seat(self) -> int:
        g = self.state.game
        return (g.dealer_position + 2) % len(g.players)

    def _seat_role(self, seat: int) -> str:
        return "SB" if seat == self._sb_seat() else "BB"

    # ------------------------------------------------------------
    # Agent decision
    # ------------------------------------------------------------

    def _agent_act(self):
        from poker_rl_lab.actions.abstract_action import ActionType as RLActionType

        rl_seat = 0 if self._seat_role(AGENT_SEAT) == "SB" else 1  # SB=0, BB=1
        obs = self._build_sb_obs() if rl_seat == 0 else self._build_bb_obs()

        rep_input = self._ensure_adapter().adapt(obs)
        result = self._ensure_agent().act(
            seat=rl_seat, obs=rep_input, legal_mask=obs.legal_mask, deterministic=True
        )
        rl_action = RLActionType(result.action_type)

        if rl_action == RLActionType.FOLD:
            engine_action_type = "fold"
        elif rl_action == RLActionType.ALL_IN:
            engine_action_type = "all_in"
        elif rl_action == RLActionType.CALL:
            engine_action_type = "call"
        else:
            # legal_mask only ever allows FOLD/ALL_IN (SB) or FOLD/CALL
            # (BB) — see _sb_legal_mask/_bb_legal_mask — so this branch
            # should be unreachable; fold defensively rather than shove
            # an unintended amount if it ever is.
            engine_action_type = "fold"

        self.last_agent_action = engine_action_type
        self.state.step(self._build_engine_action(engine_action_type))

    # ------------------------------------------------------------
    # Observation building (mirrors RL_lab_integration_pushfoldduo.md §6)
    # ------------------------------------------------------------

    def _hole_cards(self, seat: int) -> list[int]:
        return mask_to_card_ids(self.state.game.players[seat].hand_mask)

    def _sb_legal_mask(self):
        from poker_rl_lab.actions.abstract_action import NUM_ACTIONS, ActionType

        mask = [False] * NUM_ACTIONS
        mask[ActionType.FOLD] = True
        mask[ActionType.ALL_IN] = True
        return tuple(mask)

    def _bb_legal_mask(self):
        from poker_rl_lab.actions.abstract_action import NUM_ACTIONS, ActionType

        mask = [False] * NUM_ACTIONS
        mask[ActionType.FOLD] = True
        mask[ActionType.CALL] = True
        return tuple(mask)

    def _build_sb_obs(self):
        from poker_rl_lab.envs.observations import (
            BettingObs,
            BoardObs,
            HistoryObs,
            HoleCardsObs,
            PlayerObs,
            StructuredObservation,
        )

        g = self.state.game
        sb_seat, bb_seat = self._sb_seat(), self._bb_seat()
        sb_player, bb_player = g.players[sb_seat], g.players[bb_seat]

        players = (
            PlayerObs(
                seat=0,
                stack=sb_player.stack,
                current_bet=sb_player.current_bet,
                has_folded=False,
                is_all_in=sb_player.is_all_in,
                is_hero=(sb_seat == AGENT_SEAT),
            ),
            PlayerObs(
                seat=1,
                stack=bb_player.stack,
                current_bet=bb_player.current_bet,
                has_folded=False,
                is_all_in=bb_player.is_all_in,
                is_hero=(bb_seat == AGENT_SEAT),
            ),
        )

        return StructuredObservation(
            hero_seat=0,
            hole_cards=HoleCardsObs(cards=tuple(self._hole_cards(sb_seat))),
            board=BoardObs(node_cards=tuple()),
            players=players,
            betting=BettingObs(
                pot=g.pot,
                bet_to_call=g.bet_to_call - sb_player.current_bet,
                min_raise=g.min_raise,
                street_index=0,
                raises_this_street=g.raises_this_street,
            ),
            legal_mask=self._sb_legal_mask(),
            game_name=RL_GAME_NAME,
            history=HistoryObs(events=()),
            dealer_position=0,
            is_terminal=False,
        )

    def _build_bb_obs(self):
        from poker_rl_lab.actions.abstract_action import ActionType
        from poker_rl_lab.envs.observations import (
            BettingObs,
            BoardObs,
            HistoryEventObs,
            HistoryObs,
            HoleCardsObs,
            PlayerObs,
            StructuredObservation,
        )

        g = self.state.game
        sb_seat, bb_seat = self._sb_seat(), self._bb_seat()
        sb_player, bb_player = g.players[sb_seat], g.players[bb_seat]

        small_blind = SMALL_BLIND
        big_blind = BIG_BLIND
        pot_before_push = small_blind + big_blind
        push_amount = sb_player.current_bet - small_blind
        pot_fraction = (push_amount / pot_before_push) if pot_before_push > 0 else 0.0

        push_event = HistoryEventObs(
            seat=0,
            action_type=int(ActionType.ALL_IN),
            street_index=0,
            pot_fraction=max(0.0, pot_fraction),
            is_aggressor=True,
        )

        players = (
            PlayerObs(
                seat=0,
                stack=sb_player.stack,
                current_bet=sb_player.current_bet,
                has_folded=False,
                is_all_in=sb_player.is_all_in,
                is_hero=(sb_seat == AGENT_SEAT),
            ),
            PlayerObs(
                seat=1,
                stack=bb_player.stack,
                current_bet=bb_player.current_bet,
                has_folded=False,
                is_all_in=bb_player.is_all_in,
                is_hero=(bb_seat == AGENT_SEAT),
            ),
        )

        return StructuredObservation(
            hero_seat=1,
            hole_cards=HoleCardsObs(cards=tuple(self._hole_cards(bb_seat))),
            board=BoardObs(node_cards=tuple()),
            players=players,
            betting=BettingObs(
                pot=g.pot,
                bet_to_call=g.bet_to_call - bb_player.current_bet,
                min_raise=g.min_raise,
                street_index=0,
                raises_this_street=g.raises_this_street,
            ),
            legal_mask=self._bb_legal_mask(),
            game_name=RL_GAME_NAME,
            history=HistoryObs(events=(push_event,)),
            dealer_position=0,
            is_terminal=False,
        )

    # ------------------------------------------------------------
    # State serialization — reuses app.engine_adapter.state_to_dto,
    # same shape the main GameSimulator already renders.
    # ------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        if self.state is None:
            return {"active": False}

        from app.engine_adapter import state_to_dto

        dto = state_to_dto(self.state)

        # state_to_dto only populates dto.winners from a real showdown
        # (poker_state.last_showdown); when the hand instead ends by a
        # fold, last_winners is set directly on PokerState without a
        # ShowdownResult. Patch it the same way game_service.py's own
        # _progress_engine() does for the main game.
        if self.state.phase in (Phase.SHOWDOWN, Phase.HAND_COMPLETE):
            dto.winners = [w + 1 for w in getattr(self.state, "last_winners", [])]

        payload = dto.dict()

        g = self.state.game
        big_blind = BIG_BLIND
        payload.update(
            {
                "active": True,
                "hand_number": self.hand_number,
                "human_seat": HUMAN_SEAT,
                "agent_seat": AGENT_SEAT,
                "human_role": self._seat_role(HUMAN_SEAT),
                "agent_role": self._seat_role(AGENT_SEAT),
                "human_stack_bb": round(g.players[HUMAN_SEAT].stack / big_blind, 2),
                "agent_stack_bb": round(g.players[AGENT_SEAT].stack / big_blind, 2),
                "last_agent_action": self.last_agent_action,
                "hand_over": self.state.phase == Phase.HAND_COMPLETE,
            }
        )
        return payload


push_fold_service = PushFoldService()