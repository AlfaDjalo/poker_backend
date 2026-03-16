from poker_engine.state.poker_state import PokerState
from poker_engine.rules.betting_rules import BettingRules
from poker_engine.game_def import GameDef
from poker_engine.player_state import PlayerState

from app.engine_adapter import state_to_dto


class GameService:

    def __init__(self):
        self.state = None

    def new_hand(self):

        players = [
            PlayerState(stack=100),
            PlayerState(stack=100),
            PlayerState(stack=100),
            PlayerState(stack=100),
        ]

        game_def = GameDef()
        rules = BettingRules()

        self.state = PokerState(players, game_def, rules, None)

        self.state.start_hand()

        return state_to_dto(self.state)
    
    def get_state(self):

        if self.state is None:
            return None
        
        return state_to_dto(self.state)
    
    def apply_action(self, action):

        self.state.step(action)

        return state_to_dto(self.state)
    

game_service = GameService()