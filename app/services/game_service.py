import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
engine_root = project_root / "poker_engine"

if str(engine_root) not in sys.path:
    sys.path.insert(0, str(engine_root))

from state.poker_state import PokerState, Phase
from betting.betting_rules import BettingRules
from rules.game_definition import GameDefinition
from state.player_state import PlayerState
from games.loader import load_game
from scoring.scoring_engine import CppScoringEngine
from actions.action import Action
from actions.action_type import ActionType

from app.engine_adapter import state_to_dto

# GAME = "omaha"
GAME = "holdem"

class GameService:

    def __init__(self):
        self.state = None

    def restart(self):

        players = [
            PlayerState(stack=100),
            PlayerState(stack=100),
            PlayerState(stack=100),
            PlayerState(stack=100),
        ]

        game_def, rules = load_game(GAME)

        scoring_engine = CppScoringEngine()

        self.state = PokerState(players, game_def, rules, scoring_engine)

        self.state.start_hand()

        return state_to_dto(self.state)
    
    def get_state(self):

        if self.state is None:
            return None
        
        return state_to_dto(self.state)
    
    def apply_action(self, req):

        if self.state is None:
            return None
        
        action = to_engine_action(req)

        self.state.step(action)

        self._progress_engine()

        # print(state_to_dto(self.state))

        return state_to_dto(self.state)
    
    def advance_street(self):
        if self.state is None:
            return None
        
        self.state.step(None)
        # self.state.deal_next_street()

        return state_to_dto(self.state)

    def _progress_engine(self):
        """
        Automatically advance non-player phases:
        - DEAL_BOARD
        - SHOWDOWN
        """

        while self.state.phase in (Phase.DEAL_BOARD, Phase.SHOWDOWN):
            self.state.step(None)

        dto = state_to_dto(self.state)

        if self.state.phase == Phase.HAND_COMPLETE:
            dto.winners = self._get_winners()

        return dto


    def new_hand(self):

        self.state.reset_hand()

        return state_to_dto(self.state)


    def _get_winners(self):
        stacks = [p.stack for p in self.state.game.players]
        max_stack = max(stacks)
        return [i + 1 for i, s in enumerate(stacks) if s == max_stack]

def to_engine_action(req):

    return Action(
        type=ActionType[req.type.upper()],
        amount=req.amount
    )

    
game_service = GameService()