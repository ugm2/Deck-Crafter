from typing import List, Optional
from pydantic import BaseModel, Field


class Rules(BaseModel):
    """
    A model representing the structure of the rules for a card-only game, without a board.
    The model defines required fields for setup, turn structure, and win conditions,
    with optional fields for additional rules, reaction phases, and more advanced mechanics.
    """

    initial_hands: str = Field(
        ...,
        description=(
            "Instructions on how many cards each player is dealt initially. "
            "Examples: 'Each player is dealt 5 cards from the main deck', "
            "'Players draw 3 ability cards and 2 weapon cards to start'."
        ),
    )
    deck_preparation: str = Field(
        ...,
        description=(
            "Instructions on how to prepare the main deck or any other card piles before starting. "
            "Examples: 'Shuffle the main deck and place it in the center', "
            "'Separate the action and character cards into two decks'."
        ),
    )
    turn_structure: str = Field(
        ...,
        description=(
            "The order and structure of each player's turn (required). "
            "Examples: 'On your turn, draw 1 card and play 1 action card. You can attack an opponent or perform a special action', "
            "'Players take turns drawing 2 cards from the deck and playing as many cards as they like from their hand'."
        ),
    )
    reaction_phase: Optional[str] = Field(
        None,
        description=(
            "Describes any opportunities players have to react or respond during opponents' turns. "
            "Examples: 'Players may counter an attack by discarding a defense card', "
            "'You may play a trap card in response to an opponent’s attack'."
        ),
    )
    win_conditions: str = Field(
        ...,
        description=(
            "The conditions required to win the game (required). "
            "Examples: 'The first player to collect 10 treasure cards wins', "
            "'The last player standing wins after eliminating all opponents', "
            "'The player with the most points at the end of 10 rounds wins'."
        ),
    )
    additional_rules: Optional[List[str]] = Field(
        None,
        description=(
            "Any additional rules or mechanics that modify the core gameplay. "
            "This can include advanced rules, sudden death conditions, tiebreakers, or game-altering effects. "
            "Examples: ['Players can discard 2 cards to gain 1 extra action', "
            "'In the event of a tie, the player with the most resource cards wins']."
        ),
    )
    end_of_round: Optional[str] = Field(
        None,
        description=(
            "What happens at the end of a round, if the game has distinct rounds. "
            "Examples: 'All players discard down to 3 cards', 'Each player may draw 2 cards and reset their health'."
        ),
    )
    turn_limit: Optional[int] = Field(
        None,
        description=(
            "The maximum number of turns or rounds in the game before it ends. "
            "Examples: 'The game lasts 10 rounds', 'After each player takes 5 turns, the game ends'."
        ),
    )
    scoring_system: Optional[str] = Field(
        None,
        description=(
            "How points are earned or calculated in the game, if applicable. "
            "Examples: 'Players earn 1 point for each treasure collected', "
            "'Each monster defeated gives 2 points', 'Players lose 1 point for every defense card discarded'."
        ),
    )
    resource_mechanics: Optional[str] = Field(
        None,
        description=(
            "Describes how players accumulate and use resources or currency. "
            "Examples: 'Players earn gold by defeating monsters and can spend it to buy new cards', "
            "'Each resource card allows the player to perform an additional action'."
        ),
    )
