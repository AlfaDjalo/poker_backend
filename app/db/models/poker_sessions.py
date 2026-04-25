from sqlalchemy import Column, Integer, DateTime, ForeignKey
from sqlalchemy.sql import func

from app.db.base import Base

class PokerSession(Base):
    __tablename__ = "poker_sessions"

    session_id = Column(Integer, primary_key=True)
    table_id = Column(Integer, ForeignKey("poker_tables.table_id"))
    started_at = Column(DateTime, server_default=func.now())
    ended_at = Column(DateTime, nullable=True)