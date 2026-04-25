from cards.mask import mask_to_card_ids

def mask_to_cards(mask):
    return list(mask_to_card_ids(mask))

class BackendEngineCallbacks:

    def __init__(self, logger):
        self.logger = logger
        # self.hand_number = 0

    def on_hand_start(self, state):

        # self.hand_number += 1

        # hole_cards = [
        #     mask_to_cards(p.hand_mask)
        #     for p in state.game.players
        # ]

        self.logger.start_hand({
            "variant_name": state.game_def.game_name,
            "layout_name": state.game_def.layout_name,
            "split_pot": (state.rules.payout_type == "split_pot"),
            "betting_config_id": 1,
            "dealer_seat": state.game.dealer_position,
            "pot": state.game.pot,
            "ended_at": None,    
            "players": state.game.players,        
        })

        # self.logger.start_hand({
        #     # "hand_number": self.hand_number,
        #     "state": state
        # })

    def on_action(self, state, action, player_index):

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
            # state={
            #     "board": board,
            #     "pot": g.pot,
            #     "stacks": stacks
            # }
        )

    def on_showdown(self, state, result):

        self.logger.finish_hand(state)

        # g = state.game

        # board = [
        #     c for c in g.node_cards
        #     if c is not None
        # ]

        # payouts = result.payouts

        # self.logger.finish_hand(
        #     board=board,
        #     result={
        #         "payouts": payouts
        #     }
        # )