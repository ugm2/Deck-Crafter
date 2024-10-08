from typing import Optional
from pydantic import BaseModel, Field


class Card(BaseModel):
    """
    A model representing a card used in a card-only game.
    This model can be called multiple times to generate different types of cards such as action, character, resource, etc.
    """

    name: str = Field(
        ...,
        description=(
            "The name of the card (required). "
            "Examples: 'Fireball', 'Warrior of the North', 'Elven Archer', 'Healing Potion', 'Ancient Spell'."
        ),
    )
    type: str = Field(
        ...,
        description=(
            "The type of card (required). It describes the role of the card, such as Action, Character, Resource, etc. "
            "Examples: 'Action', 'Character', 'Resource', 'Spell', 'Item', 'Event'."
        ),
    )
    effect: str = Field(
        ...,
        description=(
            "The effect or ability of the card (required). This describes what happens when the card is played. "
            "Examples: 'Deal 3 damage to any enemy', 'Heal 5 health points to an ally', "
            "'Draw 2 cards', 'Increase your defense by 2 for the next round', "
            "'Summon a creature with 4 attack and 3 defense'."
        ),
    )
    cost: Optional[str] = Field(
        None,
        description=(
            "The cost to play the card (optional). Some games might not use costs. "
            "Examples: '2 mana', '3 energy', '1 resource card', 'No cost'."
        ),
    )
    flavor_text: Optional[str] = Field(
        None,
        description=(
            "Flavor text that adds thematic or narrative depth to the card (optional). "
            "This text doesn’t affect gameplay but adds to the story. "
            "Examples: 'A blazing inferno summoned from the depths of the underworld', "
            "'The warrior stands tall, ready to defend his homeland', "
            "'A potion brewed by the ancient druids of the forest'."
        ),
    )
    rarity: Optional[str] = Field(
        None,
        description=(
            "The rarity of the card (optional). Not all games use rarity. "
            "Examples: 'Common', 'Uncommon', 'Rare', 'Legendary'."
        ),
    )
    interactions: Optional[str] = Field(
        None,
        description=(
            "Any special interactions the card might have with other cards or game mechanics (optional). "
            "Examples: 'Can only be played if a creature is already on the battlefield', "
            "'This card cannot be countered by trap cards', 'Gains +1 attack for each resource card you control'."
        ),
    )
