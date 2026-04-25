from sqlalchemy import Column, Integer, DateTime, JSON
from sqlalchemy.sql import func

from app.db.base import Base

class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True)

    config = Column(JSON)

    started_at = Column(DateTime, server_default=func.now())

    finished_at = Column(DateTime, nullable=True)