from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from deck_crafter.models.card import Card
from deck_crafter.models.game_concept import GameConcept
from deck_crafter.models.rules import Rules
from deck_crafter.models.user_preferences import UserPreferences

class GameStatus(str, Enum):
    CREATED = "created"
    CONCEPT_GENERATED = "concept_generated"
    RULES_GENERATED = "rules_generated"
    CARDS_GENERATED = "cards_generated"

class CardGameState(BaseModel):
    game_id: str
    status: GameStatus
    preferences: UserPreferences
    concept: Optional[GameConcept] = None
    rules: Optional[Rules] = None
    cards: Optional[list[Card]] = None
    created_at: datetime
    updated_at: datetime
