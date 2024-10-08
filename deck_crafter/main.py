import json
import sys
from typing import Optional
from deck_crafter.utils.logger import LoggerWriter
from deck_crafter.workflow.game_workflow import create_game_workflow
from deck_crafter.services.llm_service import VertexAILLM
from deck_crafter.models.state import CardGameState
from deck_crafter.models.user_preferences import UserPreferences
from deck_crafter.utils.config import Config
from pydantic.json import pydantic_encoder


def setup_logging():
    """
    Set up logging to output both to stdout and to a log file.
    """
    log_file = open(Config.LOG_FILE_PATH, "w")
    sys.stdout = LoggerWriter(sys.stdout, log_file)


def get_user_preferences() -> UserPreferences:
    """
    Prompt the user to optionally override the default preferences for the card game.
    If no input is provided, the defaults from the UserPreferences model are used.

    :return: A UserPreferences instance with the user's choices.
    """
    print(
        "Please provide your preferences for the card game (press Enter to skip, defaults in parentheses):"
    )

    # Capture inputs with defaults applied from the Pydantic model
    language = (
        input(f"Preferred language (default: {UserPreferences().language}): ").strip()
        or UserPreferences().language
    )
    theme = (
        input(f"Theme (default: {UserPreferences().theme}): ").strip()
        or UserPreferences().theme
    )
    game_style = (
        input(f"Style preference (default: {UserPreferences().game_style}): ").strip()
        or UserPreferences().game_style
    )
    number_of_players = (
        input(
            f"Number of players (default: {UserPreferences().number_of_players}): "
        ).strip()
        or UserPreferences().number_of_players
    )
    max_unique_cards_input = input(
        f"Maximum number of unique cards (default: {UserPreferences().max_unique_cards}): "
    ).strip()
    max_unique_cards = (
        int(max_unique_cards_input)
        if max_unique_cards_input.isdigit()
        else UserPreferences().max_unique_cards
    )
    target_audience = (
        input(
            f"Target audience (default: {UserPreferences().target_audience}): "
        ).strip()
        or UserPreferences().target_audience
    )
    rule_complexity = (
        input(
            f"Rule complexity (default: {UserPreferences().rule_complexity}): "
        ).strip()
        or UserPreferences().rule_complexity
    )

    # Create the UserPreferences instance with the provided values
    return UserPreferences(
        language=language,
        theme=theme,
        game_style=game_style,
        number_of_players=number_of_players,
        max_unique_cards=max_unique_cards,
        target_audience=target_audience,
        rule_complexity=rule_complexity,
    )


def main():
    """
    Main entry point to generate a card game using the game workflow.
    """
    setup_logging()

    # Initialize the LLM service
    llm_service = VertexAILLM(
        model_name=Config.LLM_MODEL_NAME,
        temperature=Config.LLM_TEMPERATURE,
        max_output_tokens=Config.LLM_MAX_OUTPUT_TOKENS,
    )

    # Create the game generation workflow
    workflow = create_game_workflow(llm_service)

    # Get user preferences, with default values already set in UserPreferences model
    user_preferences = get_user_preferences()

    # Initialize the state
    initial_state = CardGameState(
        game_concept=None,
        cards=[],
        rules=None,
        current_step="generate_concept",
        user_preferences=user_preferences,
    )

    # Run the workflow
    result = workflow.invoke(
        initial_state, config={"recursion_limit": 150, "configurable": {"thread_id": 1}}
    )

    print("\n\nGenerated Card Game:\n")
    print(f"**Title:** {result['game_concept'].title}")
    print(f"**Description:** {result['game_concept'].description}")
    print(f"**Theme:** {result['game_concept'].theme}")
    print(f"**Target Audience:** {result['game_concept'].target_audience}")
    print(f"**Gameplay Style:** {result['game_concept'].game_style}")
    print(f"**Number of Players:** {result['game_concept'].number_of_players}")
    print(f"**Rule Complexity:** {result['game_concept'].rule_complexity}\n")

    print(f"\nCards ({len(result['cards'])}):\n")
    for card in result["cards"]:
        print(f"**{card.name}** ({card.rarity}): {card.effect}")
        print(f"  **Type:** {card.type}, **Cost:** {card.cost}")
        print(f"  **Interactions:** {card.interactions}")
        print(f"  **Flavor text:** {card.flavor_text}\n")

    print("\nRules:")
    print_rule_section("Setup", result["rules"].initial_hands, bullet_points=True)
    print_rule_section(
        "Deck Preparation", result["rules"].deck_preparation, bullet_points=True
    )
    print_rule_section(
        "Turn Structure", result["rules"].turn_structure, bullet_points=True
    )
    print_rule_section(
        "Win Conditions", result["rules"].win_conditions, bullet_points=True
    )
    print_rule_section(
        "Reaction Phase", result["rules"].reaction_phase, bullet_points=True
    )
    print_rule_section(
        "Additional Rules", result["rules"].additional_rules, bullet_points=True
    )
    print_rule_section("End of Round", result["rules"].end_of_round, bullet_points=True)
    print_rule_section("Turn Limit", result["rules"].turn_limit, bullet_points=True)
    print_rule_section(
        "Scoring System", result["rules"].scoring_system, bullet_points=True
    )
    print_rule_section(
        "Resource Mechanics", result["rules"].resource_mechanics, bullet_points=True
    )

    with open(Config.OUTPUT_FILE_PATH, "w") as f:
        json.dump(result, f, indent=4, default=pydantic_encoder, ensure_ascii=False)


def print_rule_section(
    section_name: str, section_content: Optional[str], bullet_points: bool = False
):
    """
    Print a section of the rules with a header and indentation.
    """
    if section_content is None:
        return
    print(f"\n{section_name}:")
    if bullet_points:
        for line in section_content.splitlines():
            print(f"  * {line}")
    else:
        for line in section_content.splitlines():
            print(f"  {line}")


if __name__ == "__main__":
    main()
