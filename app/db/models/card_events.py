from sqlalchemy import Column, Integer, String, ForeignKey

from app.db.base import Base

class CardEvent(Base):
    __tablename__ = "card_events"

    card_event_id = Column(Integer, primary_key = True)
    hand_id = Column(Integer, ForeignKey("hands.hand_id"))
    player_id = Column(Integer, ForeignKey("players.player_id"))
    street = Column(Integer)
    card = Column(Integer)
    event_type = Column(String, nullable = False)
