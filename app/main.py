import sys
from pathlib import Path

engine_path = str(Path(__file__).parent.parent / "poker_engine")
if engine_path not in sys.path:
    sys.path.append(engine_path)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.game_api import router as game_router
from app.api.replay_api import router as replay_router

from app.db.session import engine
from app.db.base import Base

# for app.api.game_api import router as game_router

from poker_eval import ScoreType, ShowdownType

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Poker Backend")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(game_router)
app.include_router(replay_router)