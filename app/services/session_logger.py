import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
engine_root = project_root / "poker_engine"

if str(engine_root) not in sys.path:
    sys.path.insert(0, str(engine_root))

from sqlalchemy.orm import session

from app.db.models.poker_sessions import PokerSession
from app.db.models.table_seating import TableSeat
from app.db.models.players import Player
from app.db.models.hands import Hand
from app.db.models.actions import Action
from app.db.models.hole_cards import HoleCard
from app.db.models.board_cards import BoardCard
from app.db.models.hand_points import HandPoint
from app.db.models.point_results import PointResult
from app.db.models.point_cards import PointCard
from app.db.models.payouts import Payout

from cards.mask import mask_to_card_ids

def mask_to_cards(mask):
    return list(mask_to_card_ids(mask))

class SessionLogger:

    def __init__(self, db: session):
        self.db = db
        self.session_id = None
        self.hand_id = None
        self._action_index = 0
        self._finish_hand_called = False

    def start_game(self, config):
        """Start a new poker session for the given table."""
        t_id = config.get("table_id")

        # poker_session = PokerSession(config)
        poker_session = PokerSession(table_id=t_id)

        self.db.add(poker_session)
        self._safe_commit()
        self.db.refresh(poker_session)

        self.session_id = poker_session.session_id
        self.active_player_ids = config.get("player_ids", [])

        for seat_index, p_id in enumerate(self.active_player_ids):
            seat = TableSeat(
                session_id=poker_session.session_id,
                player_id=p_id,
                seat_number=seat_index + 1
            )
            self.db.add(seat)
        self._safe_commit()


    def start_hand(self, config):

        print("Config: ", config)

        self._action_index = 0
        self._finish_hand_called = False

        layout_name = config.get("layout_name") or "single_board"

        hand = Hand(
            session_id=self.session_id,
            variant_name=config.get("variant_name", "nlhe"),
            layout_name=layout_name,
            split_pot=config.get("split_pot", False),
            betting_config_id=config.get("betting_config_id", 1),
            dealer_seat=config.get("dealer_seat", 0),
            pot=config.get("pot", 0),
            ended_at=config.get("ended_at", None),
            # players=config.get("players", []),
        )

        self.db.add(hand)
        self.db.flush()
        # self.db.refresh(hand)

        self.hand_id = hand.hand_id

        players_list = config.get("players", [])
        for player_index, p in enumerate(players_list):
            try:
                actual_player_id = self.active_player_ids[player_index]
            except IndexError:
                print(f"Warning: no player_id found for engine index {player_index}")
                continue

            cards = mask_to_card_ids(p.hand_mask)
            # cards = mask_to_cards(p.hand_mask)

            for card in cards:

                self.db.add(HoleCard(
                    hand_id=self.hand_id,
                    player_id=actual_player_id,
                    street=0,
                    card=card,
                    visible=True
                ))

        self._safe_commit()


    def log_action(
            self,
            street,
            player_index,
            action,
            amount,
            # state
    ):
        self._recover_if_needed()

        actual_player_id = self.active_player_ids[player_index]
        a = Action(
            hand_id=self.hand_id,
            street=street,
            action_index=self._next_action_index(),
            player_id=actual_player_id,
            action_type=action,
            amount=amount,
        )

        self.db.add(a)
        self._safe_commit()

    def log_board(self, state):

        g = state.game

        for node, card in enumerate(g.node_cards):

            if card is not None:
                self.db.add(BoardCard(
                    hand_id=self.hand_id,
                    street=g.street_index,
                    node=node,
                    card=card
                ))


    def finish_hand(self, state):
        if self._finish_hand_called:
            print(f"SessionLogger.finish_hand: already called for hand_id={self.hand_id}, skipping.")
            return
        
        self._finish_hand_called = True

        self._recover_if_needed()

        self.log_board(state)

        result = state.last_showdown
        if not result:
            return

        for point in result.points:

            hp = HandPoint(
                hand_id=self.hand_id,
                name=point.name,
                showdown_type=point.showdown_type,
                score_type=point.score_type,
                node_set=point.node_mask
            )
        
            self.db.add(hp)
            self.db.flush()

            for pr in point.results:
                
                player_id = self.active_player_ids[pr.player_index]
                
                pr_row = PointResult(
                    point_id=hp.point_id,
                    player_id=player_id,
                    best_hand_mask=pr.best_hand_mask,
                    rank=pr.rank,
                    hand_value=pr.value,
                    hand_category=pr.category,
                    point_share=pr.share
                )

                self.db.add(pr_row)
                self.db.flush()

                for c in pr.hole_cards_used:
                    self.db.add(PointCard(
                        point_result_id=pr_row.point_result_id,
                        card=c,
                        source="hole"
                    ))

                for c in pr.board_cards_used:
                    self.db.add(PointCard(
                        point_result_id=pr_row.point_result_id,
                        card=c,
                        source="board"
                    ))

        for p, amt in result.payouts.items():
            player_id = self.active_player_ids[p]

            self.db.add(Payout(
                hand_id=self.hand_id,
                player_id=player_id,
                amount=amt,
                point_id=None
            ))

        self._safe_commit()

    def _next_action_index(self):
        # if not hasattr(self, "_action_index"):
        #     self._action_index = 0
        val = self._action_index
        self._action_index += 1
        return val
    
    def _safe_commit(self):
        """Commit, rolling back first if the session is in a failed state."""
        try:
            self.db.commit()
        except Exception as exc:
            print(f"SessionLogger._safe_commit: commit failed ({exc}), rolling back.")
            try:
                self.db.rollback()
            except Exception as rb_exc:
                print(f"SessionLogger._safe_commit: rollback also failed: {rb_exc}")
            raise

    def _recover_if_needed(self):
        """
        If a previous flush/commit left the session in PendingRollbackError
        state, issue a rollback so subsequent operations can proceed.
        """
        try:
            # A lightweight probe — if the session is broken this will raise
            self.db.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
        except Exception:
            print("SessionLogger._recover_if_needed: session broken, rolling back.")
            try:
                self.db.rollback()
            except Exception:
                pass