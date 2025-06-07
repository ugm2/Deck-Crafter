from deck_crafter.models.state import CardGameState
from langgraph.graph import END


def should_continue(state: CardGameState) -> str:
    """
    Determines whether card generation should continue or end based on the number of cards generated so far.

    :param state: The current state of the card game.
    :return: The next step to take ("generate_cards" to continue generating or END to finish).
    """
    current_cards_count = len(state.cards) if state.cards is not None else 0

    # Continue generating cards if not enough unique cards are generated
    # Also, ensure concept and number_of_unique_cards are not None before comparison
    if state.concept is not None and state.concept.number_of_unique_cards is not None:
        if current_cards_count < state.concept.number_of_unique_cards:
            return "generate_cards"

    return END  # Stop when enough cards are generated or if concept/count is missing
