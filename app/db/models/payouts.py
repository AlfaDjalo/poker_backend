from sqlalchemy import Column, Integer, ForeignKey

from app.db.base import Base


class Payout(Base):
    __tablename__ = "payouts"

    payout_id = Column(Integer, primary_key=True)
    hand_id = Column(Integer, ForeignKey("hands.hand_id"), nullable=False)
    point_id = Column(Integer, ForeignKey("hand_points.point_id"))
    player_id = Column(Integer, ForeignKey("players.player_id"))
    amount = Column(Integer)
