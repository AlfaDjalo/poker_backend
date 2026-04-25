from sqlalchemy import Column, Integer, BigInteger, String, ForeignKey
from sqlalchemy.orm import relationship

from app.db.base import Base


class HandPoint(Base):
    __tablename__ = "hand_points"

    point_id = Column(Integer, primary_key=True)
    hand_id = Column(Integer, ForeignKey("hands.hand_id"), nullable=False)
    name = Column(String, nullable=False)
    showdown_type = Column(String, nullable=False)
    score_type = Column(String, nullable=False)
    scoop_from_point_id = Column(Integer, ForeignKey("hand_points.point_id"), nullable=True)
    node_set = Column(BigInteger, nullable=False)

    scoop_from = relationship(
        "HandPoint",
        remote_side=[point_id],
        backref="scooped_points"
    )