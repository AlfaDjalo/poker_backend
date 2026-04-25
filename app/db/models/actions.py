from sqlalchemy import Column, Integer, ForeignKey, String

from app.db.base import Base

class Action(Base):
    __tablename__ = "actions"

    action_id = Column(Integer, primary_key=True)
    hand_id = Column(Integer, ForeignKey("hands.hand_id"), nullable=False)
    street  = Column(Integer)
    action_index = Column(Integer)
    player_id = Column(Integer, ForeignKey("players.player_id"))
    action_type = Column(String)
    amount = Column(Integer)
    pot_before = Column(Integer)
    stack_before = Column(Integer)
