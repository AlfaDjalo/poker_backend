from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func

from app.db.base import Base

class BoardCard(Base):
    __tablename__ = "board_cards"

    board_card_id = Column(Integer, primary_key = True)
    hand_id = Column(Integer, ForeignKey("hands.hand_id"))
    street = Column(Integer)
    node = Column(Integer)
    card = Column(Integer)