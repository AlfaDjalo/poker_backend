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
from cards.mask import mask_to_card_ids

from app.engine_adapter import state_to_dto
from app.services.session_logger import SessionLogger
from app.services.engine_callbacks import BackendEngineCallbacks
from app.db.models.poker_tables import PokerTable
from app.db.models.players import Player

# GAME = "holdem"
# GAME = "omaha"
# GAME = "plo8"
# GAME = "double_board_plo_bomb_pot"
GAME = "five_points_high_low_high_low_hand"

class GameService:

    def __init__(self):
        self.state = None
        self.logger = None
        self.callback = None

    def restart(self, db):

        active_table = db.query(PokerTable).first()
        if not active_table:
            raise Exception("No poker tables found in database.")

        db_players = db.query(Player).filter(
            Player.username.in_([f"Player {i}" for i in range(1, 7)])
        ).order_by(Player.username).all()

        if len(db_players) < 6:
            raise Exception(f"Found only {len(db_players)} players.")
        
        players = [
            PlayerState(stack=100) for _ in db_players
        ]
        # players = [
        #     PlayerState(stack=100),
        #     PlayerState(stack=100),
        #     PlayerState(stack=100),
        #     PlayerState(stack=100),
        #     PlayerState(stack=100),
        #     PlayerState(stack=100),
        # ]

        game_def, rules = load_game(GAME)

        print("GameDef: ", game_def)

        scoring_engine = CppScoringEngine()

        self.logger = SessionLogger(db)
        self.callbacks = BackendEngineCallbacks(self.logger)

        self.state = PokerState(
            players,
            game_def,
            rules,
            scoring_engine,
            callbacks=self.callbacks
        )

        self.logger.start_game({
            "table_id": active_table.table_id,
            "player_ids": [p.player_id for p in db_players],
            # "players": len(players)
        })

        self.state.start_hand()

        self.logger.start_hand({
            "variant_name": game_def.game_name,
            "layout_name": game_def.layout_name,
            "split_pot": (rules.payout_type == "split_pot"),
            "betting_config_id": 1,
            "dealer_seat": 0,
            "pot": 0,
            "ended_at": None,   
            "players": self.state.game.players         
        })

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

        return self._progress_engine()

        # print(state_to_dto(self.state))

        # return state_to_dto(self.state)
    
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
        """

        while self.state.phase == Phase.DEAL_BOARD:
            self.state.step(None)

        dto = state_to_dto(self.state)

        if self.state.phase in [Phase.SHOWDOWN, Phase.HAND_COMPLETE]:
            dto.winners = self._get_winners()

        # print("DTO: ", dto)

        return dto


    def new_hand(self):

        self.state.start_hand()

        return state_to_dto(self.state)


    def _get_winners(self):
        # stacks = [p.stack for p in self.state.game.players]
        # max_stack = max(stacks)
        # return [i + 1 for i, s in enumerate(stacks) if s == max_stack]
        return [w + 1 for w in self.state.last_winners]

def to_engine_action(req):

    return Action(
        type=ActionType[req.type.upper()],
        amount=req.amount
    )

def decode_hand_mask(mask, node_mask, player_mask):
    cards = mask_to_card_ids(mask)
    board_cards = mask_to_card_ids(mask & node_mask)
    hole_cards = mask_to_card_ids(mask & player_mask)

    return cards, hole_cards, board_cards

    
game_service = GameService()