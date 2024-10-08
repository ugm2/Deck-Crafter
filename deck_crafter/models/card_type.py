from pydantic import BaseModel, Field


class CardType(BaseModel):
    """
    Represents a category or type of cards within the game.
    Each card type includes details that help generate the individual cards.
    """

    name: str = Field(
        ...,
        description=(
            "The name of the card type or category (**required**). "
            "Examples: 'Organ', 'Virus', 'Medicine', 'Attack', 'Defense', 'Spell', 'Resource', 'Trap', 'Wild'."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "A description of the role or function of this card type in the game (**required**). "
            "Explains how these cards are used during gameplay. "
            "Examples: 'Cards that represent body organs players need to collect', "
            "'Cards used to attack opponents', 'Defense cards that block attacks', "
            "'Special action cards with unique effects'."
        ),
    )
    quantity: int = Field(
        ...,
        description=(
            "The total number of cards of this type in the game (**required**). "
            "Examples: 4, 16, 20, 9."
        ),
    )
    unique_cards: int = Field(
        ...,
        description=(
            "The number of unique cards within this card type (**required**). "
            "Defines how many different cards exist within this type. "
            "Examples: 1, 4, 5."
        ),
    )
