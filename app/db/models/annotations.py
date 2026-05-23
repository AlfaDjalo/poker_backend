from sqlalchemy import Column, Integer, ForeignKey, String, DateTime, JSON
from sqlalchemy.sql import func

from app.db.base import Base

class Annotation(Base):
    """
    Stores user annotations on individual actions in a hand replay.
 
    user_id is nullable for now — TODO: wire up real auth.
    selected_cards is a JSON list of card strings e.g. ["Ah", "Kd", "2c"]
    that the user highlighted when writing the annotation.
    """
    __tablename__ = "annotations"

    annotation_id       = Column(Integer,   primary_key=True, autoincrement=True)
    hand_id             = Column(Integer,   ForeignKey("hands.hand_id"),        nullable=False,     index=True)
    action_id           = Column(Integer,   ForeignKey("actions.action_id"),    nullable=True,      index=True)
    user_id             = Column(Integer,   nullable=False)     # TODO: FK to users table
    comment             = Column(String,    nullable=False)
    selected_cards      = Column(JSON,      nullable=True)
    created_at          = Column(DateTime,  server_default=func.now())
    updated_at          = Column(DateTime,  server_default=func.now(),          onupdate=func.now())