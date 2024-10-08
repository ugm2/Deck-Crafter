from typing import Optional
from pydantic import BaseModel, Field


class Card(BaseModel):
    """
    Represents a unique card in a card-only game.
    Each card has attributes that define its role, effects, and interactions within the game.
    """

    name: str = Field(
        ...,
        description=(
            "The name of the card (**required**). "
            "Should be unique within the game context. "
            "Examples: 'Fireball', 'Stealth Assassin', 'Healing Potion', 'Skip Turn', 'Wild Card'."
        ),
    )
    quantity: int = Field(
        ...,
        description=(
            "The number of copies of this card included in the game deck (**required**). "
            "Examples: 1, 2, 4, 10, 20."
        ),
    )
    type: str = Field(
        ...,
        description=(
            "The category or classification of the card (**required**). "
            "Must match one of the `CardType.name` values from the `GameConcept`. "
            "Examples: 'Organ', 'Virus', 'Medicine', 'Attack', 'Defense', 'Spell'."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "A concise explanation of the card's effect or role in the game (**required**). "
            "Clearly states what the card does when played. "
            "Examples: 'Deal 3 damage to an opponent', 'Skip your next turn', "
            "'Steal a random card from another player'."
        ),
    )
    cost: Optional[str] = Field(
        None,
        description=(
            "The resource or condition required to play the card (**optional**). "
            "Examples: '2 Mana', 'Discard a card', 'No cost'."
        ),
    )
    flavor_text: Optional[str] = Field(
        None,
        description=(
            "Narrative or thematic text that adds depth to the card (**optional**). "
            "Examples: 'An ancient spell whispered among the shadows', "
            "'A warrior with a mysterious past'."
        ),
    )
    rarity: Optional[str] = Field(
        None,
        description=(
            "The rarity level of the card within the game (**optional**). "
            "Examples: 'Common', 'Uncommon', 'Rare'."
        ),
    )
    interactions: Optional[str] = Field(
        None,
        description=(
            "Special interactions this card has with other cards or game mechanics (**optional**). "
            "Examples: 'Doubles the effect of healing cards', 'Cannot be blocked by defense cards'."
        ),
    )
    color: Optional[str] = Field(
        None,
        description=(
            "The color or suit associated with the card, if applicable (**optional**). "
            "Examples: 'Red', 'Blue', 'Hearts', 'Spades'."
        ),
    )
    image_description: Optional[str] = Field(
        None,
        description=(
            "A description of the card's artwork or illustration in English only (**optional**). "
            "Examples: 'A fierce dragon breathing fire', 'A knight in armor wielding a sword'."
        ),
    )
