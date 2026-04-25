from sqlalchemy import Column, Integer, String

from app.db.base import Base

class BettingConfig(Base):
    __tablename__ = "betting_configs"

    betting_config_id = Column(Integer, primary_key=True)
    betting_config_name = Column(String, nullable=False)
