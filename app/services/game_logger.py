from sqlalchemy.orm import session

from app.db.models.game import Game
from app.db.models.hand import Hand
from app.db.models.action import Action

class GameLogger:

    def __init__(self, db: session):
        self.db = db
        self.game_id = None
        self.hand_id = None

    def start_game(self, config):

        game = Game(config=config)

        self.db.add(game)
        self.db.commit()
        self.db.refresh(game)

        self.game_id = game.id

    def start_hand(self, hand_number, hole_cards):

        hand = Hand(
            game_id=self.game_id,
            hand_number=hand_number,
            hole_cards=hole_cards
        )

        self.db.add(hand)
        self.db.commit()
        self.db.refresh(hand)

        self.hand_id = hand.id

    def log_action(
            self,
            street,
            player,
            action,
            amount,
            state
    ):
        a = Action(
            hand_id=self.hand_id,
            street=street,
            player_index=player,
            action=action,
            amount=amount,
            state=state
        )

        self.db.add(a)
        self.db.commit()

    def finish_hand(self, board, result):

        hand = self.db.query(Hand).get(self.hand_id)

        hand.board = board
        hand.result = result

        self.db.commit()

