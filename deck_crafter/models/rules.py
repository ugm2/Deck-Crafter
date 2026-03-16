from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator

class TurnPhase(BaseModel):
    """
    Represents a single, distinct phase within a player's turn.
    """
    phase_name: str = Field(..., description="The name of the phase, e.g., 'Draw Phase', 'Main Phase', 'End Phase'.")
    phase_description: str = Field(..., description="A clear description of what actions can be taken during this phase.")


class Rules(BaseModel):
    """
    A model representing the highly structured rules for a card-only game.
    The model's structure guides the AI to produce clear, unambiguous, and well-organized content.
    """

    deck_preparation: str = Field(
        ...,
        description="Step-by-step instructions on how to prepare the main deck and any other card piles before starting the game."
    )
    initial_hands: str = Field(
        ...,
        description="Instructions on how many cards each player is dealt initially and any rules for mulligans or redrawing."
    )

    turn_structure: List[TurnPhase] = Field(
        ...,
        description="A list of the sequential phases that make up a single player's turn. This must be a step-by-step breakdown."
    )

    win_conditions: str = Field(
        ...,
        description="The clear and unequivocal conditions that a player must meet to win the game."
    )
    
    resource_mechanics: Optional[str] = Field(
        None,
        description="Describes how players accumulate and use resources or currency (e.g., Mana, Energy, Gold) if they exist in the game."
    )
    reaction_phase: Optional[str] = Field(
        None,
        description="Describes if and how players can react or respond during opponents' turns (e.g., playing counter-spells, traps)."
    )

    glossary: Optional[Dict[str, str]] = Field(
        None,
        description="A dictionary of game-specific keywords and their precise definitions. Key: Term, Value: Definition. Essential for clarity."
    )

    @field_validator("glossary", mode="before")
    @classmethod
    def coerce_glossary(cls, v):
        if isinstance(v, str):
            return None
        return v

    examples_of_play: Optional[List[str]] = Field(
        None,
        description="A list of concrete examples illustrating complex interactions or a typical sequence of play to resolve ambiguity."
    )

    additional_rules: Optional[List[str]] = Field(
        None,
        description="Any other miscellaneous rules that do not fit into the other categories, such as tie-breakers, hand size limits, etc."
    )
    
    end_of_round: Optional[str] = Field(
        None,
        description="Describes what happens at the end of a full round of turns, if applicable."
    )
    turn_limit: Optional[int] = Field(
        None,
        description="The maximum number of turns or rounds before the game ends, if any."
    )
    scoring_system: Optional[str] = Field(
        None,
        description="Describes how points are calculated, if the game is not won by a binary condition."
    )