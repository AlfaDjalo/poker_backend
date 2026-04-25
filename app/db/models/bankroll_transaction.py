from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func

from app.db.base import Base

class BankrollTransaction(Base):
    __tablename__ = "bankroll_transactions"

    transaction_id = Column(Integer, primary_key = True)
    player_id = Column(Integer, ForeignKey("players.player_id"))
    transaction_type = Column(String)
    hand_id = Column(Integer, ForeignKey("hands.hand_id"))
    amount = Column(Integer)
    balance_after = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())