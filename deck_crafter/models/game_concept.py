from typing import List, Optional
from pydantic import BaseModel, Field

from deck_crafter.models.card_type import CardType


class GameConcept(BaseModel):
    """
    Represents the core concept of a card-only game.
    Defines the essential elements required to generate and design the game's foundation,
    including theme, rules, and player dynamics.
    This model is intended for the initial game concept generation step.
    """

    title: str = Field(
        ...,
        description=(
            "A compelling and thematic title for the game (**required**). "
            "Should capture the essence of the game and attract interest. "
            "Examples: 'Realm of Legends', 'Battle Quest', 'Mystic Journeys', "
            "'Shadow Hunters', 'Elemental Wars', 'Galactic Conquest', 'Mystery Manor'."
        ),
    )
    theme: str = Field(
        ...,
        description=(
            "The central theme or setting of the game (**required**). "
            "Defines the game's atmosphere and aesthetic. "
            "Examples: 'Medieval Fantasy', 'Space Exploration', 'Mythological Creatures', "
            "'Steampunk Adventure', 'Post-Apocalyptic Survival', 'Superheroes vs. Villains', "
            "'Mystery and Detective Work'."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "A brief overview of the game's premise and objectives (**required**). "
            "Should provide players with an understanding of the game's core experience. "
            "Examples: 'Players build armies to conquer territories and defeat opponents', "
            "'A cooperative game where players team up to solve mysteries', "
            "'A fast-paced card game focused on resource management and strategy', "
            "'Players collect sets of artifacts while sabotaging opponents', "
            "'A bluffing game where deception leads to victory', "
            "'Navigate a haunted house to uncover secrets before time runs out'."
        ),
    )
    language: str = Field(
        ...,
        description=(
            "The primary language used in the game's text and instructions (**required**). "
            "Examples: 'English', 'Spanish', 'French', 'German', 'Chinese', 'Japanese', 'Korean'."
        ),
    )
    game_style: str = Field(
        ...,
        description=(
            "The core gameplay style or genre (**required**). "
            "Defines how players interact with the game and each other. "
            "Examples: 'Competitive', 'Cooperative', 'Deck-Building', 'Bluffing', "
            "'Set Collection', 'Strategy', 'Party Game', 'Role-Playing', 'Deduction'."
        ),
    )
    number_of_players: str = Field(
        ...,
        description=(
            "The recommended number of players (**required**). "
            "Should specify the minimum and maximum players suitable for the game. "
            "Examples: '2-4 players', '3-6 players', '1-5 players (includes solo mode)', "
            "'2-8 players', '4-10 players (ideal for parties)'."
        ),
    )
    game_duration: str = Field(
        ...,
        description=(
            "The estimated duration of a single game session (**required**). "
            "Provides players with an idea of how long the game takes to play. "
            "Examples: '15-30 minutes', '30-60 minutes', '1-2 hours', "
            "'Variable length depending on player count', 'Approximately 45 minutes'."
        ),
    )
    target_audience: Optional[str] = Field(
        None,
        description=(
            "The intended age group or demographic for the game (**optional**). "
            "Helps tailor the game's complexity and content appropriately. "
            "Examples: 'Family-friendly (Ages 8+)', 'Teens and Adults (Ages 13+)', "
            "'Adults Only (Ages 18+)', 'All Ages', 'Kids (Ages 6+)'."
        ),
    )
    rule_complexity: Optional[str] = Field(
        None,
        description=(
            "The complexity level of the game's rules (**optional**). "
            "Assists in setting expectations for learning and playing the game. "
            "Examples: 'Simple rules for quick learning', 'Moderate complexity with strategic depth', "
            "'Advanced rules for experienced players', 'Easy to learn, hard to master', "
            "'Designed for casual play'."
        ),
    )

    card_types: List[CardType] = Field(
        ...,
        description=(
            "A list of card types or categories included in the game (**required**). "
            "Each card type defines a group of cards with similar functions or roles. "
            "This field provides detailed guidance to the CardGenerationAgent."
        ),
    )

    @property
    def number_of_unique_cards(self) -> int:
        return sum(card_type.unique_cards for card_type in self.card_types)
