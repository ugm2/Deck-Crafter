from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import game

app = FastAPI(
    title="Deck Crafter API",
    description="API for generating card games",
    version="1.0.0"
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(game.router, prefix="/api/v1/games", tags=["games"])

@app.get("/")
async def root():
    return {"message": "Welcome to Deck Crafter API"} 