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
from cards.card import Card as CardObj

from app.db.models.actions import Action as ActionModel
from app.db.models.hole_cards import HoleCard as HoleCardModel
from app.db.models.board_cards import BoardCard as BoardCardModel
from app.db.models.hand_points import HandPoint as HandPointModel
from app.db.models.point_results import PointResult as PointResultModel
from app.db.models.point_cards import PointCard as PointCardModel
from app.db.models.payouts import Payout as PayoutModel
from app.db.models.hands import Hand as HandModel

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
        self.editing_mode = False
        self.pre_edit_snapshot = None

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
        self.callbacks = BackendEngineCallbacks(self.logger, game_service_ref=self)

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

    def begin_edit(self, db):
        """
        Enter editing mode.
        - Saves an in-memory snapshot of current PokerState
        - Deletes DB records for the current hand
        - Sets editing_mode = True so subsequent action callbacks skip logging
        """
        if self.state is None:
            raise RuntimeError("No active game state.")

        # Snapshot current state for cancel support
        self.pre_edit_snapshot = self._snapshot_state()
        self.editing_mode = True
        
        # Delete DB records for the current hand
        if self.logger and self.logger.hand_id:
            self._delete_hand_records(db, self.logger.hand_id)


    def apply_edit(self, req):
        """Apply an edited snapshot to the live game and resume."""               
        if not self.editing_mode:
            raise RuntimeError("Not in editing mode. Call /game/edit/begin first.")

        self._validate_edit_request(req)
        self._load_snapshot_from_request(req)
        self.editing_mode = False
        self.pre_edit_snapshot = None
        
        return self._progress_engine()
    

    def load_edit(self, req):
        """
        Load an arbitrary snapshot (from Replayer).
        Marks as editing_mode so the hand is never saved.
        """        
        self._validate_edit_request(req)

        game_def, rules = load_game(req.game_name)

        players = [
            PlayerState(stack=p.stack) for p in req.players
        ]

        self.state = PokerState(
            players,
            game_def,
            rules,
            CppScoringEngine(),
            callbacks=self.callbacks,
        )

        self._apply_snapshot_to_state(req)
        self.editing_mode = True

        return state_to_dto(self.state)


    def cancel_edit(self):
        """Restore the snapshot taken at begin_edit."""
        if self.pre_edit_snapshot is None:
            raise RuntimeError("No pre-edit snapshot available.")        
        
        self._restore_snapshot(self.pre_edit_snapshot)
        self.editing_mode = False
        self.pre_edit_snapshot = None

        return state_to_dto(self.state)
    


    # ── Snapshot helpers ──────────────────────────────────────────────

    def _snapshot_state(self):
        """Capture a minimal dict snapshot of the current PokerState."""
        g = self.state.game
        return {
            "game_name": self.current_game,
            "street_index": g.street_index,
            "pot": g.pot,
            "dealer_position": g.dealer_position,
            "current_player": g.current_player,
            "bet_to_call": g.bet_to_call,
            "min_raise": g.min_raise,
            "node_cards": list(g.node_cards),
            "discard_pile": list(getattr(g, "discard_pile", [])),
            "players": [
                {
                    "stack": p.stack,
                    "hand_mask": p.hand_mask,
                    "current_bet": p.current_bet,
                    "total_contribution": getattr(p, "total_contribution", 0),
                    "has_folded": p.has_folded,
                    "is_all_in": getattr(p, "is_all_in", False),
                }
                for p in g.players
            ],
        }

    def _restore_snapshot(self, snap):
        g = self.state.game
        g.street_index = snap["street_index"]
        g.pot = snap["pot"]
        g.dealer_position = snap["dealer_position"]
        g.current_player = snap["current_player"]
        g.bet_to_call = snap["bet_to_call"]
        g.min_raise = snap["min_raise"]
        g.node_cards = snap["node_cards"]
        if hasattr(g, "discard_pile"):
            g.discard_pile = snap["discard_pile"]
        for i, ps in enumerate(snap["players"]):
            p = g.players[i]
            p.stack = ps["stack"]
            p.hand_mask = ps["hand_mask"]
            p.current_bet = ps["current_bet"]
            if hasattr(p, "total_contribution"):
                p.total_contribution = ps["total_contribution"]
            p.has_folded = ps["has_folded"]
            if hasattr(p, "is_all_in"):
                p.is_all_in = ps["is_all_in"]

    def _apply_snapshot_to_state(self, req):
        """Write an EditStateRequest into the live PokerState."""
        g = self.state.game
        g.street_index = req.street_index
        g.pot = req.pot
        g.dealer_position = req.dealer_position
        g.current_player = req.current_player
        g.bet_to_call = req.bet_to_call
        g.min_raise = req.min_raise

        # node cards
        g.node_cards = [
            CardObj.from_str(c).id if c else None
            for c in req.node_cards
        ]

        # discard pile
        if hasattr(g, "discard_pile"):
            g.discard_pile = [CardObj.from_str(c).id for c in req.discard_pile]

        # players
        for p_input in req.players:
            idx = p_input.seat - 1
            if idx < 0 or idx >= len(g.players):
                continue
            p = g.players[idx]
            p.stack = p_input.stack
            p.current_bet = p_input.current_bet
            if hasattr(p, "total_contribution"):
                p.total_contribution = p_input.total_contribution
            p.has_folded = p_input.has_folded
            if hasattr(p, "is_all_in"):
                p.is_all_in = p_input.is_all_in
            # Rebuild hand_mask from hole_cards list
            mask = 0
            for cs in p_input.hole_cards:
                if cs is not None:
                    mask |= (1 << CardObj.from_str(cs).id)
            p.hand_mask = mask

    def _load_snapshot_from_request(self, req):
        """For apply_edit: load req into existing state (same game variant)."""
        if self.state.game_def.game_name != req.game_name:
            game_def, rules = load_game(req.game_name)
            self._recreate_state_with_variant(game_def, rules)
        self._apply_snapshot_to_state(req)

    def _validate_edit_request(self, req):
        seen: set[str] = set()
        errors = []
        for p in req.players:
            for c in p.hole_cards:
                if c is None:
                    continue
                if c in seen:
                    errors.append(f"Duplicate card: {c}")
                seen.add(c)
        for c in req.node_cards:
            if c is None:
                continue
            if c in seen:
                errors.append(f"Duplicate card: {c}")
            seen.add(c)
        for c in req.discard_pile:
            if c in seen:
                errors.append(f"Duplicate card: {c}")
            seen.add(c)
        if len(seen) > 52:
            errors.append("More than 52 cards accounted for.")
        if errors:
            raise ValueError("; ".join(errors))

    def _delete_hand_records(self, db, hand_id: int):
        """Remove all DB rows for a hand that is being edited."""

        # Delete in FK-safe order
        point_ids = [
            r[0] for r in db.query(HandPointModel.point_id)
                            .filter(HandPointModel.hand_id == hand_id).all()
        ]
        if point_ids:
            pr_ids = [
                r[0] for r in db.query(PointResultModel.point_result_id)
                                .filter(PointResultModel.point_id.in_(point_ids)).all()
            ]
            if pr_ids:
                db.query(PointCardModel).filter(
                    PointCardModel.point_result_id.in_(pr_ids)
                ).delete(synchronize_session=False)

            db.query(PointResultModel).filter(
                PointResultModel.point_id.in_(point_ids)
            ).delete(synchronize_session=False)
        db.query(PayoutModel).filter(PayoutModel.hand_id == hand_id).delete(synchronize_session=False)
        db.query(HandPointModel).filter(HandPointModel.hand_id == hand_id).delete(synchronize_session=False)
        db.query(ActionModel).filter(ActionModel.hand_id == hand_id).delete(synchronize_session=False)
        db.query(HoleCardModel).filter(HoleCardModel.hand_id == hand_id).delete(synchronize_session=False)
        db.query(BoardCardModel).filter(BoardCardModel.hand_id == hand_id).delete(synchronize_session=False)
        db.query(HandModel).filter(HandModel.hand_id == hand_id).delete(synchronize_session=False)
        db.commit()



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