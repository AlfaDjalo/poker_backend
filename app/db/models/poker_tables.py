from sqlalchemy import Column, Integer, String

from app.db.base import Base

class PokerTable(Base):
    __tablename__ = "poker_tables"

    table_id = Column(Integer, primary_key = True)
    table_name = Column(String)
    max_players = Column(Integer)
