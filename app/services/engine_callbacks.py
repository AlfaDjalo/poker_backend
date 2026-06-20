from cards.mask import mask_to_card_ids

def mask_to_cards(mask):
    return list(mask_to_card_ids(mask))

class BackendEngineCallbacks:

    def __init__(self, logger, game_service_ref=None):
        self.logger = logger
        self._game_service = game_service_ref

    def _is_editing(self):
        return self._game_service and self._game_service.editing_mode

    def on_hand_start(self, state):

        if self._is_editing():
            return
        
        self.logger.start_hand({
            "variant_name": state.game_def.game_name,
            "layout_name": state.game_def.layout_name,
            "split_pot": (state.rules.payout_type == "split_pot"),
            "betting_config_id": 1,
            "dealer_seat": state.game.dealer_position,
            "pot": state.game.pot,
            "ended_at": None,    
            "players": state.game.players,
            "game_def": state.game_def,      
        })

        # self.logger.start_hand({
        #     # "hand_number": self.hand_number,
        #     "state": state
        # })

    def on_action(self, state, action, player_index, pot_before, stack_before):

        if self._is_editing():
            return
        
        g = state.game

        board = [
            c for c in g.node_cards
            if c is not None
        ]

        stacks = [p.stack for p in g.players]

        self.logger.log_action(
            street=g.street_index,
            player_index=player_index,
            action=action.type.name,
            amount=action.amount,
            pot_before=pot_before,
            stack_before=stack_before,
            # state={
            #     "board": board,
            #     "pot": g.pot,
            #     "stacks": stacks
            # }
        )

    def on_showdown(self, state, result):

        if self._is_editing():
            return

        self.logger.finish_hand(state)

