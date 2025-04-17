import json
from typing import List, Optional
from deck_crafter.models.card import Card
from deck_crafter.models.rules import Rules
from deck_crafter.workflow.game_workflow import create_game_workflow
from deck_crafter.services.llm_service import create_llm_service
from deck_crafter.models.state import CardGameState
from deck_crafter.models.user_preferences import UserPreferences
from deck_crafter.utils.config import Config
from pydantic.json import pydantic_encoder
from dotenv import load_dotenv

load_dotenv()


def get_user_preferences() -> UserPreferences:
    """
    Prompt the user to optionally override the default preferences for the card game.
    If no input is provided, the defaults from the UserPreferences model are used.

    :return: A UserPreferences instance with the user's choices.
    """
    print(
        "Please provide your preferences for the card game (press Enter to skip, defaults in parentheses):"
    )

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

    return UserPreferences(
        language=language,
        theme=theme,
        game_style=game_style,
        number_of_players=number_of_players,
        target_audience=target_audience,
        rule_complexity=rule_complexity,
    )


def main():
    """
    Main entry point to generate a card game using the game workflow.
    """
    if Config.LLM_PROVIDER == "vertexai":
        llm_service = create_llm_service(
            provider="vertexai",
            model_name=Config.VERTEXAI_MODEL_NAME,
            temperature=Config.LLM_TEMPERATURE,
            max_output_tokens=Config.LLM_MAX_OUTPUT_TOKENS,
            location=Config.VERTEXAI_LOCATION,
        )
    elif Config.LLM_PROVIDER == "ollama":
        llm_service = create_llm_service(
            provider="ollama",
            model_name=Config.OLLAMA_MODEL_NAME,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_OUTPUT_TOKENS,
            base_url=Config.OLLAMA_BASE_URL,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {Config.LLM_PROVIDER}")

    workflow = create_game_workflow(llm_service)

    user_preferences = get_user_preferences()

    initial_state = CardGameState(
        game_concept=None,
        cards=[],
        rules=None,
        user_preferences=user_preferences,
    )

    result = workflow.invoke(
        initial_state, config={"recursion_limit": 150, "configurable": {"thread_id": 1}}
    )

    def print_model(model_instance, title=None):
        """
        Dynamically print the fields of a Pydantic model instance.
        """
        if model_instance is None:
            return

        if title:
            print(f"\n{title}:\n")

        for field_name, field in model_instance.model_fields.items():
            value = getattr(model_instance, field_name)
            if value is not None:
                formatted_field_name = field_name.replace("_", " ").title()
                if isinstance(value, dict):
                    print(f"**{formatted_field_name}:**")
                    for k, v in value.items():
                        print(f"  - **{k}**: {v}")
                else:
                    print(f"**{formatted_field_name}:** {value}")
        print()

    def print_cards(cards: List[Card]):
        """
        Dynamically print a list of Card instances.
        """
        if not cards:
            return

        print(f"\nCards ({len(cards)}):\n")
        for card in cards:
            card_details = []
            for field_name, field in card.model_fields.items():
                value = getattr(card, field_name)
                if value is not None:
                    if field_name == "name":
                        card_name = f"**{value}**"
                    elif field_name == "rarity":
                        card_name += f" ({value}):"
                    else:
                        formatted_field_name = field_name.replace("_", " ").title()
                        card_details.append(f"  **{formatted_field_name}:** {value}")
            print(card_name)
            print("\n".join(card_details))
            print()

    def print_rules(rules: Rules):
        """
        Dynamically print the fields of the Rules model.
        """
        if rules is None:
            return

        print("\nRules:")
        for field_name, field in rules.model_fields.items():
            value = getattr(rules, field_name)
            if value is not None:
                print_rule_section(field_name.replace("_", " ").title(), value)
        print()

    def print_rule_section(
        section_name: str, section_content: Optional[str], bullet_points: bool = True
    ):
        """
        Print a section of the rules with a header and indentation.
        """
        if section_content is None:
            return
        print(f"\n{section_name}:")
        if isinstance(section_content, list):
            for item in section_content:
                print(f"  * {item}")
        elif isinstance(section_content, int):
            print(f"  {section_content}")
        elif bullet_points:
            for line in section_content.splitlines():
                print(f"  * {line}")
        else:
            for line in section_content.splitlines():
                print(f"  {line}")

    print("\n\nGenerated Card Game:\n")
    print_model(result.get("game_concept"), title="Game Concept")
    print_cards(result.get("cards"))
    print_rules(result.get("rules"))

    # Save the result to a JSON file
    with open("output.json", "w") as f:
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
