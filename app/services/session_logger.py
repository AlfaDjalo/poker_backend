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
        self._logged_hole_cards: dict[int, set[int]] = {}

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

        # --------------------------------------------------------------
        # THE ONE STREET COUNTER — single source of truth for "what
        # street is this hand on right now", shared by every table that
        # stamps a `street` column (BoardCard, Action, HoleCard/
        # CardEvent via log_hole_cards). Starts at 0 (preflop — no board
        # cards revealed yet) and is incremented ONLY by log_board(),
        # exactly when a genuinely new batch of board cards is observed.
        #
        # Previously Action.street/hole_card_events[].street were
        # derived SEPARATELY, by walking the compiled GameGraph and
        # reading a `street_index` value off each node's metadata
        # (graph_engine_callbacks.py's now-removed
        # _betting_node_street_map/_card_node_street_map/
        # _all_nodes_street_map). That's fragile in two ways: (1) it
        # assumes the compiled graph node's metadata dict preserves the
        # flow YAML's `street_index` key verbatim, which was never
        # actually confirmed against the real graph loader and turned
        # out to still be wrong; and (2) even when it "worked", it was
        # computing street from a DIFFERENT signal than BoardCard.street
        # (reveal order) — the two numbering schemes are only
        # guaranteed to agree for the simplest single-board, standard-
        # street variants, and can drift apart for bomb pots, multi-
        # board/hopscotch layouts, or any custom flow.
        #
        # Deriving every street-stamped table from THIS SAME counter
        # instead removes the graph entirely from the question: no
        # matter what the flow graph/layout looks like, "which street
        # is this action on" is defined identically to "how many board
        # reveals have happened so far", which is exactly what the
        # frontend's replay reconstruction needs to key frames on.
        # --------------------------------------------------------------
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

        self._logged_hole_cards = {
            player_index: set(mask_to_card_ids(p.hand_mask))
            for player_index, p in enumerate(players_list)
        }

        self.db.commit()

    def current_street(self) -> int:
        """
        The street value every OTHER logging call should stamp right
        now — see self._board_reveal_index's docstring in start_hand()
        for why this single counter (not graph metadata) is the source
        of truth. Callers (graph_engine_callbacks.py) call this
        immediately after log_board() has had a chance to run for the
        current engine state, so a board reveal that just happened as
        part of this same AUTO-walk is already reflected here.
        """
        return self._board_reveal_index

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
        single call with the same reveal-order group number
        (self._board_reveal_index — see its docstring in start_hand()).
        Cards dealt across separate calls (i.e. separate deal points in
        the flow graph) get strictly increasing group numbers,
        regardless of how many nodes each deal point fills.

        THIS is the one place _board_reveal_index is ever incremented
        — every other street-stamped write (log_action,
        log_card_select, log_card_pass, log_hole_cards) reads it via
        current_street() but never advances it, so as long as callers
        call log_board() before those other methods for the same
        engine-state snapshot (graph_engine_callbacks.py already does
        this), everything stays in lockstep with zero possibility of
        drift between BoardCard.street and every other table's street.
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
        # No explicit street resolver is available here (finish_hand
        # is called from on_showdown, past the point any caller has a
        # meaningful "current node" to walk from) — fall back to
        # current_street() the same way every other caller does now,
        # rather than the engine's own g.street_index (a sequential
        # betting-round counter with completely different numbering —
        # see log_hole_cards()'s own note on why that fallback was
        # wrong).
        self.log_hole_cards(state)

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

    def log_hole_cards(self, state, street=None):
        """
        Log any hole cards that entered a player's hand SINCE the last
        call (or since start_hand()) — the mid-hand analog of
        log_board() for BOARD cards. Two mechanics rely on this:

            - Extra Card AUTO handlers (ESG, Catchup ESG, Christmas,
            Grinch's deal_extra_hole_card) hand a player one or more
            NEW cards directly into hand_mask with no DECISION node
            involved at all — nothing else in the callback chain ever
            observes this, so without this method those cards are
            live in-game (state_to_dto reads hand_mask fresh every
            call) but permanently missing from HoleCard/CardEvent, and
            therefore invisible on replay.
            - A CARD_SELECT redraw (replace_from_deck: true, e.g.
            Drawmaha) — the discarded cards are logged explicitly via
            log_card_select()/discard_card(), but the REPLACEMENT
            cards drawn back in were never logged by anything until
            now; they show up here as a same-shaped diff.

        Safe to call repeatedly (idempotent), same contract as
        log_board().

        `street` — the street ordinal (same convention BoardCard.street
        / Action.street already use — see current_street()'s
        docstring) to stamp on any NEW cards found this call. Callers
        that have already called log_board() for this same engine-
        state snapshot (graph_engine_callbacks.py's on_decision() /
        on_cards_distributed()) should pass self.current_street()
        explicitly, which is exactly what "the street this card was
        actually dealt on" means now.

        Falls back to current_street() when omitted (street=None) —
        this used to fall back to state.game.street_index (the
        engine's own sequential betting-round counter, a DIFFERENT
        numbering from BoardCard.street's reveal-order — see
        current_street()'s own docstring for why that mismatch was a
        bug in its own right), which is what made an ESG/Christmas/
        pass-the-trash extra card show up under the wrong street in
        the Hand Replayer. The only caller that still relies on this
        fallback is finish_hand()'s own trailing call, which has no
        specific engine-state snapshot to derive a street from other
        than "whatever the count is right now".
        """
        g = state.game

        if street is None:
            street = self.current_street()

        for player_index, actual_player_id in enumerate(self.active_player_ids):
            if player_index >= len(g.players):
                continue
            p = g.players[player_index]
            current = set(mask_to_card_ids(p.hand_mask))
            logged = self._logged_hole_cards.setdefault(player_index, set())
            new_cards = current - logged
            if not new_cards:
                continue

            for card in sorted(new_cards):
                deal_card(
                    self.db,
                    hand_id=self.hand_id,
                    player_id=actual_player_id,
                    card=card,
                    street=street,
                    visible=True,
                )
                logged.add(card)

        self.db.commit()