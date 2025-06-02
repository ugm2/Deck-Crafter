from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import game

app = FastAPI(
    title="Deck Crafter API",
    description="API for generating card games",
    version="1.0.0"
)

# Configurar CORS para desarrollo local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir las rutas
app.include_router(game.router, prefix="/api/v1/games", tags=["games"])

@app.get("/")
async def root():
    return {"message": "Welcome to Deck Crafter API"} 