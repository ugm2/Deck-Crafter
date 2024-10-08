from typing import TypedDict, List

from deck_crafter.models.card import Card
from deck_crafter.models.game_concept import GameConcept
from deck_crafter.models.rules import Rules
from deck_crafter.models.user_preferences import UserPreferences


class CardGameState(TypedDict):
    game_concept: GameConcept
    cards: List[Card]
    rules: Rules
    user_preferences: UserPreferences
