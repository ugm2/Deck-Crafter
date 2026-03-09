from enum import Enum
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel

from deck_crafter.models.card import Card
from deck_crafter.models.game_concept import GameConcept
from deck_crafter.models.rules import Rules
from deck_crafter.models.user_preferences import UserPreferences
from deck_crafter.models.evaluation import GameEvaluation

# --- LÓGICA DE ACTUALIZACIÓN PERSONALIZADA ---
def last_write_wins(a, b):
    return b

class GameStatus(str, Enum):
    CREATED = "created"
    CONCEPT_GENERATED = "concept_generated"
    RULES_GENERATED = "rules_generated"
    CARDS_GENERATED = "cards_generated"
    IMAGES_GENERATED = "images_generated"
    EVALUATED = "evaluated"

class CardGameState(BaseModel):
    game_id: str
    status: GameStatus
    preferences: UserPreferences
    concept: Optional[GameConcept] = None
    rules: Optional[Rules] = None
    cards: Optional[List[Card]] = None
    image_paths: Optional[dict[str, str]] = None
    evaluation: Optional[GameEvaluation] = None
    created_at: datetime
    updated_at: datetime
    
    critique: Optional[str] = None
    refinement_count: int = 0
