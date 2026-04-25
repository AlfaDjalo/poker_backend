from sqlalchemy import Column, Integer, DateTime, ForeignKey
from sqlalchemy.sql import func

from app.db.base import Base

class TableSeat(Base):
    __tablename__ = "table_seating"

    session_id = Column(Integer, ForeignKey("poker_sessions.session_id"), primary_key = True)
    seat_number = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.player_id"))
    stack = Column(Integer)
    joined_at = Column(DateTime, server_default=func.now())
    left_at = Column(DateTime, server_default=func.now()) # **** should be empty to start
