from sqlalchemy import Column, Integer, BigInteger, Float, String, ForeignKey

from app.db.base import Base


class PointCard(Base):
    __tablename__ = "point_cards"

    point_card_id = Column(Integer, primary_key=True)
    point_result_id = Column(Integer, ForeignKey("point_results.point_result_id"))
    card = Column(Integer)
    source = Column(String, nullable=False)
