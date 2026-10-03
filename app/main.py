
# engine_path = str(Path(__file__).parent.parent / "poker_engine")
# if engine_path not in sys.path:
#     sys.path.append(engine_path)

# lab_path = str(Path(__file__).parent.parent.parent / "poker_rl_lab")
# if lab_path not in sys.path:
#     sys.path.append(lab_path)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from app.api.edit_api import router as edit_router
from app.api.equity_api import router as equity_router
from app.api.game_api import router as game_router
from app.api.construction_api import router as construction_router
from app.api.hands_api import router as hands_router
# from app.api.push_fold_api import router as push_fold_router
from app.api.replay_api import router as replay_router
from app.api.trainer_api import router as trainer_router
from app.api.tutorial_api import router as tutorial_router
from app.db.base import Base
from app.db.session import engine

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
app.include_router(equity_router)
# app.include_router(edit_router)
app.include_router(construction_router)
app.include_router(tutorial_router)
app.include_router(hands_router)
# app.include_router(push_fold_router)
app.include_router(trainer_router)
