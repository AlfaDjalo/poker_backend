from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

for app.api.game_api import router as game_router
    
app = FastAPI(title="Poker Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers["*"],
)

app.include_router(game_router)