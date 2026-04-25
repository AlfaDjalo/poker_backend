from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.sql import func

from app.db.base import Base

class Hand(Base):
    __tablename__ = "hands"

    hand_id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("poker_sessions.session_id"))
    variant_name = Column(String, nullable=False)
    layout_name = Column(String, nullable=False)
    split_pot = Column(Boolean, default=False)
    betting_config_id = Column(Integer, ForeignKey("betting_configs.betting_config_id"))
    dealer_seat = Column(Integer)
    pot = Column(Integer)
    started_at = Column(DateTime, server_default=func.now())
    ended_at = Column(DateTime, nullable=True)
