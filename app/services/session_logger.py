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
        self._logged_nodes = set()

    def start_game(self, config):
        """Start a new poker session for the given table."""
        t_id = config.get("table_id")

        poker_session = PokerSession(table_id=t_id)

        self.db.add(poker_session)
        self.db.commit()
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
        self.db.commit()


    def start_hand(self, config):

        print("Config: ", config)

        layout_name = config.get("layout_name") or "single_board"
        game_def = config.get("game_def")

        hand = Hand(
            session_id=self.session_id,
            variant_name=config.get("variant_name", "nlhe"),
            layout_name=layout_name,
            split_pot=config.get("split_pot", False),
            betting_config_id=config.get("betting_config_id", 1),
            dealer_seat=config.get("dealer_seat", 0),
            pot=config.get("pot", 0),
            ended_at=config.get("ended_at", None),
        )

        self.db.add(hand)
        self.db.flush()

        self.hand_id = hand.hand_id
        self._logged_nodes = set()
        self._node_to_street_map = self._build_node_street_map(game_def)

        players_list = config.get("players", [])
        for player_index, p in enumerate(players_list):
            try:
                actual_player_id = self.active_player_ids[player_index]
            except IndexError:
                print(f"Warning: no player_id found for engine index {player_index}")
                continue

            cards = mask_to_card_ids(p.hand_mask)

            for card in cards:

                self.db.add(HoleCard(
                    hand_id=self.hand_id,
                    player_id=actual_player_id,
                    street=0,
                    card=card,
                    visible=True
                ))

        self.db.commit()


    def log_action(
            self,
            street,
            player_index,
            action,
            amount,
            pot_before,
            stack_before,
    ):
        actual_player_id = self.active_player_ids[player_index]
        a = Action(
            hand_id=self.hand_id,
            street=street,
            action_index=self._next_action_index(),
            player_id=actual_player_id,
            action_type=action,
            amount=amount,
            pot_before=pot_before,
            stack_before=stack_before,
        )

        self.db.add(a)
        self.db.commit()

    def log_board(self, state):
        """
        Log any board cards that have been dealt but not yet recorded.
        Safe to call multiple times — tracks which nodes have already been logged
        and stamps each card with the street it was dealt on (from board layout config).
        """
        g = state.game

        for node, card in enumerate(g.node_cards):
            if card is not None and (self.hand_id, node) not in self._logged_nodes:
                card_street = self._node_to_street_map.get(node, 1)
                
                self.db.add(BoardCard(
                    hand_id=self.hand_id,
                    street=card_street,
                    node=node,
                    card=card
                ))
                self._logged_nodes.add((self.hand_id, node))

        self.db.commit()


    def finish_hand(self, state):

        # Log any board cards not yet captured.
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

        self.db.query(Hand).filter(Hand.hand_id == self.hand_id).update({
            "pot": sum(result.payouts.values())
        })

        self.db.commit()

    def _build_node_street_map(self, game_def):
        """
        Build a mapping of node index -> street (1-based) from the game definition.
 
        game_def.street_nodes is List[List[int]] where street_nodes[i] contains
        the node indices dealt on street i (0-based street index, i.e. index 0
        = flop).  We store as 1-based street numbers so they align with the
        board_cards.street column convention.
 
        Example — double board bomb pot:
            street_nodes = [[0,1,2,5,6,7], [3,8], [4,9]]
            → nodes 0,1,2,5,6,7 get street=1  (flop)
            → nodes 3,8         get street=2  (turn)
            → nodes 4,9         get street=3  (river)
        """
        node_map = {}
        if not game_def:
            return node_map
 
        try:
            for street_idx, nodes in enumerate(game_def.street_nodes, start=1):
                for node in nodes:
                    node_map[node] = street_idx
        except Exception as e:
            print(f"Warning: Failed to build node_street_map: {e}")
 
        return node_map
 
    def _next_action_index(self):
        if not hasattr(self, "_action_index"):
            self._action_index = 0
        val = self._action_index
        self._action_index += 1
        return val
 