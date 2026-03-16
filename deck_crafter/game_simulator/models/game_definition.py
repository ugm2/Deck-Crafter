from pydantic import BaseModel, Field
from typing import Any, Callable, Literal
from enum import Enum


class CardEffect(str, Enum):
    """Standard card effects that the engine understands."""
    NONE = "none"  # Card does nothing
    DRAW = "draw"  # Draw N cards
    DAMAGE = "damage"  # Deal N damage to target
    HEAL = "heal"  # Heal N health
    GAIN_RESOURCE = "gain_resource"  # Gain N of resource type
    WIN_GAME = "win_game"  # Immediately win the game
    GAIN_POINTS = "gain_points"  # Gain N points


class CardDefinition(BaseModel):
    """Definition of a card type in the game."""
    name: str
    quantity: int = 1  # How many copies in deck
    cost: dict[str, int] = Field(default_factory=dict)  # Resource costs to play
    effect: CardEffect = CardEffect.NONE
    effect_value: int = 0  # Parameter for the effect (e.g., draw 3)
    effect_target: Literal["self", "opponent", "any"] = "self"
    properties: dict[str, Any] = Field(default_factory=dict)  # Custom properties


class WinCondition(BaseModel):
    """Defines how a player wins the game."""
    type: Literal["points", "elimination", "empty_deck", "property_threshold", "last_standing"]
    target_value: int = 0  # For points/threshold: value needed to win
    property_name: str = ""  # For property_threshold: which property to check


class RuleSet(BaseModel):
    """Rules that govern gameplay."""
    initial_hand_size: int = 5
    draw_per_turn: int = 1
    max_cards_per_turn: int = 1  # How many cards can be played per turn
    initial_resources: dict[str, int] = Field(default_factory=dict)
    resource_per_turn: dict[str, int] = Field(default_factory=dict)  # Resources gained each turn
    initial_properties: dict[str, Any] = Field(default_factory=dict)  # e.g., {"health": 20}


class GameDefinition(BaseModel):
    """Complete definition of a game that can be simulated."""
    name: str
    description: str = ""
    cards: list[CardDefinition]
    rules: RuleSet = Field(default_factory=RuleSet)
    win_condition: WinCondition
    num_players: int = 2

    def build_deck(self) -> list[dict]:
        """Build a full deck from card definitions."""
        deck = []
        card_id = 0
        for card_def in self.cards:
            for _ in range(card_def.quantity):
                deck.append({
                    "card_id": f"card_{card_id}",
                    "name": card_def.name,
                    "cost": card_def.cost,
                    "effect": card_def.effect,
                    "effect_value": card_def.effect_value,
                    "effect_target": card_def.effect_target,
                    "properties": card_def.properties,
                })
                card_id += 1
        return deck
