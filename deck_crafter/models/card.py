from typing import Optional, List, Literal
from pydantic import BaseModel, Field


# Effect types that can be simulated
EffectType = Literal["none", "draw", "damage", "heal", "gain_points", "gain_resource", "win_game"]
EffectTarget = Literal["self", "opponent", "any"]


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
    image_description: str = Field(
        ...,
        description=(
            "A description of the card's artwork or illustration in English only. "
            "Examples: 'A fierce dragon breathing fire', 'A knight in armor wielding a sword'."
        ),
    )
    image_data: Optional[str] = Field(
        None,
        description=(
            "Base64 encoded image data for the card (**optional**). "
            "Used to store the card's visual representation."
        ),
    )

    # Structured effect fields for simulation (optional but recommended)
    effect_type: Optional[EffectType] = Field(
        None,
        description=(
            "The primary mechanical effect of the card for simulation (**optional but recommended**). "
            "Must be one of: 'none', 'draw', 'damage', 'heal', 'gain_points', 'gain_resource', 'win_game'. "
            "Examples: A card that says 'Deal 3 damage' should have effect_type='damage'. "
            "A card that says 'Gain 2 points' should have effect_type='gain_points'."
        ),
    )
    effect_value: Optional[int] = Field(
        None,
        description=(
            "The numeric value for the effect (**optional but recommended when effect_type is set**). "
            "Examples: 'Draw 2 cards' = 2, 'Deal 5 damage' = 5, 'Gain 3 points' = 3."
        ),
    )
    effect_target: Optional[EffectTarget] = Field(
        None,
        description=(
            "Who the effect targets (**optional, defaults to 'self'**). "
            "Must be one of: 'self' (affects the player), 'opponent' (affects enemy), 'any' (player chooses). "
            "Examples: Damage usually targets 'opponent', healing usually targets 'self'."
        ),
    )


class CardBatch(BaseModel):
    """A batch of cards generated together in a single LLM call."""
    cards: List["Card"] = Field(
        ...,
        description="List of cards generated in this batch."
    )
