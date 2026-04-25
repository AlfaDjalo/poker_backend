from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.sql import func

from app.db.base import Base

class Player(Base):
    __tablename__ = "players"

    player_id = Column(Integer, primary_key = True)
    username = Column(String, unique=True)
    password = Column(String, nullable=True)
    is_bot = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, server_default=func.now())