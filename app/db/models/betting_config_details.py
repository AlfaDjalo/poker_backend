from sqlalchemy import Column, Integer, String, ForeignKey 

from app.db.base import Base

class BettingConfigDetails(Base):
    __tablename__ = "betting_config_details"

    betting_detail_id = Column(Integer, primary_key = True)
    betting_config_id = Column(Integer, ForeignKey("betting_configs.betting_config_id"))
    bet_name = Column(String, nullable=False)
    bet_amount = Column(Integer)