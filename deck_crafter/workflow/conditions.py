from deck_crafter.models.state import CardGameState
from langgraph.graph import END


def should_continue(state: CardGameState) -> str:
    """
    Determines whether card generation should continue or end based on the number of cards generated so far.

    :param state: The current state of the card game.
    :return: The next step to take ("generate_cards" to continue generating or END to finish).
    """
    # Continue generating cards if not enough
    if len(state["cards"]) < state["game_concept"].number_of_unique_cards:
        return "generate_cards"
    return END  # Stop when enough cards are generated
