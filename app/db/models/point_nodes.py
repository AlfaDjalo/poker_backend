from sqlalchemy import Column, Integer, ForeignKey

from app.db.base import Base

class PointNode(Base):
    __tablename__ = "point_nodes"

    point_node_id = Column(Integer, primary_key = True)
    point_id = Column(Integer, ForeignKey("hand_points.point_id"))
    node = Column(Integer)
