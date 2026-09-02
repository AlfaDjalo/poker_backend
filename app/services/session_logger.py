from poker_engine.cards.mask import mask_to_card_ids
from sqlalchemy.orm import session

from app.db.models.actions import Action
from app.db.models.board_cards import BoardCard
from app.db.models.hand_points import HandPoint
from app.db.models.hands import Hand
from app.db.models.hole_cards import HoleCard
from app.db.models.payouts import Payout
from app.db.models.point_cards import PointCard
from app.db.models.point_results import PointResult
from app.db.models.poker_sessions import PokerSession
from app.db.models.table_seating import TableSeat
from app.services.card_movement import deal_card, discard_card, pass_card


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
                seat_number=seat_index + 1,
            )
            self.db.add(seat)
        self.db.commit()

    def start_hand(self, config):
        print("Config: ", config)

        layout_name = config.get("layout_name") or "single_board"
        game_def = config.get("game_def")

        variant_name = config.get("variant_name") or "nlhe"   # was: config.get("variant_name", "nlhe")

        hand = Hand(
            session_id=self.session_id,
            variant_name=variant_name,
            # variant_name=config.get("variant_name", "nlhe"),
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
        # BoardCard.street used to be a real 1-based street index,
        # derived either from game_def.street_nodes (legacy PokerState
        # path) or from a graph-walk reconstruction
        # (graph_engine_callbacks._node_street_map_from_graph, which
        # made unconfirmed assumptions about graph.node(...).metadata
        # key names and silently fell back to {} on any mismatch).
        # Under GraphEngine there IS no structural "street" — the
        # engine only knows node positions and decision points, not
        # street indices — so reconstructing one from graph metadata
        # was fighting the architecture, not modeling it. When that
        # reconstruction silently failed, log_board()'s old
        # `self._node_to_street_map.get(node, 1)` fallback stamped
        # EVERY board card (flop, turn, river alike) with street=1,
        # which is exactly why the Hand Replayer revealed the whole
        # board on the very first frame.
        #
        # Fixed by dropping street reconstruction entirely: `street`
        # on BoardCard now just records REVEAL ORDER — a small
        # monotonically increasing group number, incremented each time
        # log_board() observes a NEW batch of cards. Cards dealt
        # together (a simultaneous double-flop bomb pot, a single-node
        # river drop, etc.) naturally share a group; cards dealt in a
        # later engine pass naturally get a higher one. This is
        # derived purely from the actual order cards appeared in
        # node_cards, so it's correct for ANY flow graph — standard,
        # bomb pot, hopscotch, funnel, whatever — with zero graph
        # introspection and zero per-variant assumptions.
        #
        # `game_def`/`node_street_map` config keys are still accepted
        # (and still passed by graph_engine_callbacks.py /
        # engine_callbacks.py) but are no longer consumed here — kept
        # only so neither caller needs a matching change to stop
        # passing them.
        self._board_reveal_index = 0

        players_list = config.get("players", [])
        for player_index, p in enumerate(players_list):
            try:
                actual_player_id = self.active_player_ids[player_index]
            except IndexError:
                print(f"Warning: no player_id found for engine index {player_index}")
                continue

            cards = mask_to_card_ids(p.hand_mask)

            # Routed through card_movement.deal_card() rather than a
            # bare HoleCard insert — this is what also writes the
            # corresponding CardEvent(event_type="DEALT") row, so the
            # ledger and the current-state table start in sync from
            # the very first card of the hand. See hole_cards.py /
            # card_events.py for why both tables need to agree.
            for card in cards:
                deal_card(
                    self.db,
                    hand_id=self.hand_id,
                    player_id=actual_player_id,
                    card=card,
                    street=0,
                    visible=True,
                )

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

    def log_card_select(self, street, player_index, card_ids):
        """
        Log a CARD_SELECT decision (drawmaha-style discard) — every
        card id the player chose is discarded via
        card_movement.discard_card(), which flips its HoleCard row to
        DISCARDED and writes the matching CardEvent row. Called from
        graph_engine_callbacks.py's on_decision(); see that module's
        _log_card_decision() for the CARD_SELECT-vs-CARD_PASS
        dispatch and the pass_direction fallback that routes an
        unresolvable CARD_PASS target here too.
        """
        actual_player_id = self.active_player_ids[player_index]
        for cid in card_ids:
            discard_card(
                self.db,
                hand_id=self.hand_id,
                player_id=actual_player_id,
                card=cid,
                street=street,
            )
        self.db.commit()

    def log_card_pass(self, street, player_index, target_player_index, card_ids):
        """
        Log a CARD_PASS decision (pass-the-trash-style) — every card
        id the player chose moves directly from their hand to
        target_player_index's hand via card_movement.pass_card() (one
        CardEvent row per card, from_player_id/to_player_id both set).
        """
        from_player_id = self.active_player_ids[player_index]
        to_player_id = self.active_player_ids[target_player_index]
        for cid in card_ids:
            pass_card(
                self.db,
                hand_id=self.hand_id,
                from_player_id=from_player_id,
                to_player_id=to_player_id,
                card=cid,
                street=street,
                visible=True,
            )
        self.db.commit()

    def log_board(self, state):
        """
        Log any board cards that have been dealt but not yet recorded.
        Safe to call multiple times — tracks which nodes have already
        been logged and stamps every NEW batch of cards observed in a
        single call with the same reveal-order group number (see
        self._board_reveal_index's docstring in start_hand()). Cards
        dealt across separate calls (i.e. separate deal points in the
        flow graph) get strictly increasing group numbers, regardless
        of how many nodes each deal point fills.
        """
        g = state.game

        new_cards = [
            (node, card)
            for node, card in enumerate(g.node_cards)
            if card is not None and (self.hand_id, node) not in self._logged_nodes
        ]
        if not new_cards:
            return

        self._board_reveal_index += 1

        for node, card in new_cards:
            self.db.add(
                BoardCard(
                    hand_id=self.hand_id,
                    street=self._board_reveal_index,
                    node=node,
                    card=card,
                )
            )
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
                node_set=point.node_mask,
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
                    point_share=pr.share,
                )

                self.db.add(pr_row)
                self.db.flush()

                for c in pr.hole_cards_used:
                    self.db.add(
                        PointCard(
                            point_result_id=pr_row.point_result_id,
                            card=c,
                            source="hole",
                        )
                    )

                for c in pr.board_cards_used:
                    self.db.add(
                        PointCard(
                            point_result_id=pr_row.point_result_id,
                            card=c,
                            source="board",
                        )
                    )

        for p, amt in result.payouts.items():
            player_id = self.active_player_ids[p]

            self.db.add(
                Payout(
                    hand_id=self.hand_id, player_id=player_id, amount=amt, point_id=None
                )
            )

        self.db.query(Hand).filter(Hand.hand_id == self.hand_id).update(
            {"pot": sum(result.payouts.values())}
        )

        self.db.commit()

    def _next_action_index(self):
        if not hasattr(self, "_action_index"):
            self._action_index = 0
        val = self._action_index
        self._action_index += 1
        return val