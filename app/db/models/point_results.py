from sqlalchemy import Column, Integer, BigInteger, Float, String, ForeignKey

from app.db.base import Base


class PointResult(Base):
    __tablename__ = "point_results"

    point_result_id = Column(Integer, primary_key=True)
    point_id = Column(Integer, ForeignKey("hand_points.point_id"))
    player_id = Column(Integer, ForeignKey("players.player_id"))
    best_hand_mask = Column(BigInteger)
    rank = Column(Integer)
    hand_value = Column(Integer)
    hand_category = Column(String, nullable=False)
    point_share = Column(Float, nullable=False)
