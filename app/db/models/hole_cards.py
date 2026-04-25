from sqlalchemy import Column, Integer, Boolean, ForeignKey

from app.db.base import Base

class HoleCard(Base):
    __tablename__ = "hole_cards"

    hole_card_id = Column(Integer, primary_key = True)
    hand_id = Column(Integer, ForeignKey("hands.hand_id"))
    player_id = Column(Integer, ForeignKey("players.player_id"))
    street = Column(Integer)
    card = Column(Integer)
    visible = Column(Boolean, default = False)
