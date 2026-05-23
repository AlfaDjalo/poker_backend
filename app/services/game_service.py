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

DEFAULT_GAME = "holdem"

class GameService:

    def __init__(self):
        self.state = None
        self.logger = None
        self.callback = None
        self.current_game = DEFAULT_GAME
        self.pending_game = None

    # --------------------------------------------------
    # Variant discovery
    # --------------------------------------------------

    def get_variants(self):
        """Return all .yaml game definitions found in the engine's games directory."""
        games_dir = engine_root / "games"
        if not games_dir.exists():
            return {"variants": [], "current": self.current_game}
        names = sorted(p.stem for p in games_dir.glob("*.yaml"))
        print("Variants: ", names)
        return {"variants": names, "current": self.current_game}

    # --------------------------------------------------
    # Game selection (queued; applied between hands only)
    # --------------------------------------------------

    def select_game(self, game_name: str):
        """
        Queue a game change. Applied at the start of the next hand or on
        a full restart. Raises ValueError if the variant doesn't exist.
        """
        games_dir = engine_root / "games"
        if not (games_dir / f"{game_name}.yaml").exists():
            raise ValueError(f"Unknown game variant: '{game_name}'")
        self.pending_game = game_name

    def _apply_pending_game(self, game_name: str | None = None):
        """ Consume any explicit or queued game selection and return it."""
        if game_name:
            self.select_game(game_name)
        if self.pending_game:
            self.current_game = self.pending_game
            self.pending_game = None
        return self.current_game

    # --------------------------------------------------
    # Session restart (re-creates PokerState)
    # --------------------------------------------------

    def restart(self, db, game_name: str | None = None):

        self._apply_pending_game(game_name)

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

        game_def, rules = load_game(self.current_game)

        print("GameDef: ", game_def)
        print("Game variant: ", self.current_game)

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
        })

        self.state.start_hand()

        # self.logger.start_hand({
        #     "variant_name": game_def.game_name,
        #     "layout_name": game_def.layout_name,
        #     "split_pot": (rules.payout_type == "split_pot"),
        #     "betting_config_id": 1,
        #     "dealer_seat": 0,
        #     "pot": 0,
        #     "ended_at": None,
        #     "game_def": game_def,
        #     "players": self.state.game.players         
        # })

        return state_to_dto(self.state)
    
    # --------------------------------------------------
    # State
    # --------------------------------------------------

    def get_state(self):

        if self.state is None:
            return None
        
        return state_to_dto(self.state)
    
    # --------------------------------------------------
    # Player actions
    # --------------------------------------------------
        
    def apply_action(self, req):

        if self.state is None:
            return None
        
        action = to_engine_action(req)
        self.state.step(action)

        return self._progress_engine()

    
    def advance_street(self):
        if self.state is None:
            return None
        
        self.state.step(None)

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

        return dto

    # --------------------------------------------------
    # New hand (within the same session)
    # --------------------------------------------------

    def _build_continuation_players(self):
        return [PlayerState(stack=p.stack) for p in self.state.game.players]

    def _recreate_state_with_variant(self, game_def, rules):
        if self.state is None:
            raise RuntimeError("No active game session. Call /game/restart first.")

        self.state = PokerState(
            self._build_continuation_players(),
            game_def,
            rules,
            CppScoringEngine(),
            callbacks=self.callbacks
        )

    def new_hand(self, game_name: str | None = None):
        """
        Start a new hand.  If game_name is supplied (or a pending_game is
        queued) the variant is switched before dealing.
 
        Note: the API/frontend is responsible for only calling this once the
        previous hand is HAND_COMPLETE; we don't gate it here.
        """
        self._apply_pending_game(game_name)

        game_def, rules = load_game(self.current_game)

        if self.state is None:
            raise RuntimeError("No active game state. Call /game/restart first.")

        if self.state.game_def.game_name != game_def.game_name:
            self._recreate_state_with_variant(game_def, rules)
        else:
            self.state.game_def = game_def
            self.state.rules = rules

        self.state.start_hand()

        return state_to_dto(self.state)


    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _get_winners(self):
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