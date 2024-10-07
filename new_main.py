import json
import sys
from typing import Dict, TypedDict, List, Optional
from dataclasses import dataclass
from deck_crafter.core.llm_service import LLMService, VertexAILLM
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel


class LoggerWriter:
    def __init__(self, *writers):
        self.writers = writers

    def write(self, message):
        for w in self.writers:
            w.write(message)
            w.flush()  # Ensures that the message is written to file/output immediately

    def flush(self):
        for w in self.writers:
            w.flush()


# Redirect output to both stdout and a log file
log_file = open("output.log", "w")
sys.stdout = LoggerWriter(sys.stdout, log_file)


# State definitions
class CardGameState(TypedDict):
    game_concept: dict
    cards: List[dict]
    rules: dict
    current_step: str
    user_preferences: dict


@dataclass
class UserPreferences:
    theme: Optional[str] = "Fantasía tierra media"
    game_style: Optional[str] = "Party game"
    number_of_players: Optional[str] = "4-12"
    max_unique_cards: Optional[int] = 20
    target_audience: Optional[str] = "+18"
    rule_complexity: Optional[str] = "Medio"
    language: Optional[str] = "Español"


class GameConcept(BaseModel):
    theme: str
    title: str
    description: str
    game_style: str
    number_of_players: str
    number_of_unique_cards: int
    target_audience: Optional[str] = None
    rule_complexity: Optional[str] = None
    card_distribution: Optional[Dict[str, int]] = None


class Card(BaseModel):
    # Required fields
    name: str
    type: str  # e.g., Action, Character, Resource
    effect: str

    # Optional fields
    cost: Optional[str] = None  # Some games might not use costs
    flavor_text: Optional[str] = None
    rarity: Optional[str] = None  # Not all games use rarity
    interactions: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class Rules(BaseModel):
    # Required fields
    setup: str
    turn_structure: str
    win_conditions: str

    # Optional field
    special_rules: Optional[str] = None


class ConceptGenerationAgent:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_concept(self, state: CardGameState) -> CardGameState:
        user_prefs = state["user_preferences"]

        # Constructing required and optional fields
        required_fields = [
            "theme: The central theme of the game",
            "title: A catchy, thematic title",
            "description: A brief but clear description of the game",
            "game_style: The core gameplay style (e.g., competitive, cooperative)",
            "number_of_players: The recommended number of players",
            "number_of_unique_cards: Total number of unique cards needed for the game",
        ]

        optional_fields = [
            "target_audience: Age group or specific audience",
            "rule_complexity: Complexity level of the rules",
            "card_distribution: Types of cards and their quantities",
        ]

        # Move fields from optional to required based on user preferences
        if user_prefs.theme:
            required_fields[0] = (
                f"theme: Must be related to the user's selected theme: {user_prefs.theme}"
            )
        if user_prefs.game_style:
            required_fields[3] = (
                f"game_style: Must match user's preference: {user_prefs.game_style}"
            )
        if user_prefs.target_audience:
            required_fields.append(
                f"target_audience: Must be: {user_prefs.target_audience}"
            )
            optional_fields.remove("target_audience: Age group or specific audience")
        if user_prefs.rule_complexity:
            required_fields.append(
                f"rule_complexity: Must be: {user_prefs.rule_complexity}"
            )
            optional_fields.remove("rule_complexity: Complexity level of the rules")

        # Constructing the prompt
        prompt = f"""
        Persona: You are an expert and very critic card game designer.
        Create a concept for a unique and engaging card game in {user_prefs.language}.

        Required fields in your response:
        {', '.join(f'- {field}' for field in required_fields)}

        {'Optional fields (include only if relevant to your game concept):' if optional_fields else ''}
        {', '.join(f'- {field}' for field in optional_fields)}

        User preferences to incorporate:
        {f'- Maximum unique cards: {user_prefs.max_unique_cards}' if user_prefs.max_unique_cards else '- Choose an appropriate number of unique cards for the game.'}
        {f'- Number of players: {user_prefs.number_of_players}' if user_prefs.number_of_players else ''}

        Special instructions:
        1. The number_of_unique_cards in your response must not exceed {user_prefs.max_unique_cards if user_prefs.max_unique_cards else 'a reasonable number for the game concept'}
        2. Any user preferences provided above must be strictly followed in your response
        3. Optional fields should only be included if they add value to the game concept
        """

        print("\n[Concept Generation] Generating the game concept...")
        print(f"[Prompt Used]:\n{prompt}\n")

        game_concept = self.llm_service.call_llm(prompt, records=[GameConcept])

        print(game_concept)

        # Override concept values with user preferences if they exist
        if user_prefs.target_audience:
            game_concept.target_audience = user_prefs.target_audience
        if user_prefs.game_style:
            game_concept.game_style = user_prefs.game_style
        if user_prefs.number_of_players:
            game_concept.number_of_players = user_prefs.number_of_players
        if user_prefs.max_unique_cards:
            game_concept.number_of_unique_cards = user_prefs.max_unique_cards
        if user_prefs.rule_complexity:
            game_concept.rule_complexity = user_prefs.rule_complexity

        state["game_concept"] = game_concept
        return state


class RuleGenerationAgent:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_rules(self, state: CardGameState) -> CardGameState:
        game_concept = state["game_concept"]
        user_prefs = state["user_preferences"]

        # Constructing required and optional rule sections
        required_rules = [
            "setup: Detailed setup instructions considering the game's theme and components",
            "turn_structure: Clear explanation of how players take turns and interact with cards",
            "win_conditions: Specific conditions for winning the game",
        ]

        optional_rules = []
        requires_special_rules = hasattr(
            game_concept, "rule_complexity"
        ) and game_concept.rule_complexity.lower() in ["medium", "hard", "complex"]
        if not requires_special_rules:
            optional_rules.append(
                "special_rules: Any unique mechanics or interactions between cards"
            )
        else:
            required_rules.append(
                "special_rules: Required special mechanics based on card interactions"
            )

        # Constructing the prompt
        prompt = f"""
        Persona: You are an expert and very critic card game designer.
        Create comprehensive rules for the card game: {game_concept.title} in {user_prefs.language}
        Game concept: {game_concept.description}
        Theme: {game_concept.theme}
        Gameplay style: {game_concept.game_style}
        Rule complexity: {game_concept.rule_complexity}
        Number of players: {game_concept.number_of_players}
        Number of unique cards: {game_concept.number_of_unique_cards}
        
        Required rule sections:
        {', '.join(f'- {rule}' for rule in required_rules)}

        {'Optional rule section (include only if relevant):' if optional_rules else ''}
        {', '.join(f'- {rule}' for rule in optional_rules)}
        
        Guidelines for rule creation:
        1. Ensure rules match the game's complexity level
        2. Make sure all card types and interactions are covered
        3. Balance depth of strategy with ease of learning
        4. Consider the target audience when explaining concepts
        """

        print("\n[Rule Generation] Generating the rules...")
        print(f"[Prompt Used]:\n{prompt}\n")

        rules = self.llm_service.call_llm(prompt, records=[Rules])

        print("\n[Rule Generation] Generated rules:")
        print(rules)

        state["rules"] = rules
        return state


class CardGenerationAgent:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_card(self, state: CardGameState) -> CardGameState:
        game_concept = state["game_concept"]
        rules = state["rules"]
        user_prefs = state["user_preferences"]
        existing_cards = state.get("cards", [])

        remaining_types = {
            card_type: count
            for card_type, count in game_concept.card_distribution.items()
            if sum(1 for card in existing_cards if card.type == card_type) < count
        }

        next_type = max(remaining_types.items(), key=lambda x: x[1])[0]

        # Constructing required and optional fields
        required_fields = [
            "name: Card name",
            f"type: Must be '{next_type}'",
            "effect: Card's game effect",
        ]

        optional_fields = [
            "cost: Resource cost to play the card",
            "flavor_text: Thematic description",
            "rarity: Card rarity (Common, Uncommon, Rare, etc.)",
            "interactions: Interactions with other cards/mechanics",
        ]

        # If all existing cards have certain fields, make them required
        if existing_cards:
            if all(card.cost for card in existing_cards):
                optional_fields.remove("cost: Resource cost to play the card")
                required_fields.append("cost: Resource cost to play the card")
            if all(card.rarity for card in existing_cards):
                optional_fields.remove(
                    "rarity: Card rarity (Common, Uncommon, Rare, etc.)"
                )
                required_fields.append(
                    "rarity: Card rarity (Common, Uncommon, Rare, etc.)"
                )
            if all(card.flavor_text for card in existing_cards):
                optional_fields.remove("flavor_text: Thematic description")
                required_fields.append("flavor_text: Thematic description")
            if all(card.interactions for card in existing_cards):
                optional_fields.remove(
                    "interactions: Interactions with other cards/mechanics"
                )
                required_fields.append(
                    "interactions: Interactions with other cards/mechanics"
                )

        # Constructing the prompt
        prompt = f"""
        Persona: You are an expert and very critic card game designer.
        Generate a new card for the game: {game_concept.title} in {user_prefs.language}
        Game description: {game_concept.description}
        Theme: {game_concept.theme}
        Game style: {game_concept.game_style}
        Rule complexity: {game_concept.rule_complexity}
        Ruleset: {rules}
        Target audience: {game_concept.target_audience}

        Required fields in your response:
        {', '.join(f'- {field}' for field in required_fields)}

        Optional fields (include if relevant):
        {', '.join(f'- {field}' for field in optional_fields)}

        Current number of cards: {len(existing_cards)}
        List of existing cards:
        {existing_cards}
        
        Card Distribution Plan:
        {', '.join(f'{k}: {v}' for k, v in game_concept.card_distribution.items())}

        Current cards by type:
        {', '.join(f'{type}: {sum(1 for card in existing_cards if card.type == type)}' for type in game_concept.card_distribution)}
        
        '{next_type}' type should be the next card type you generate

        Create a balanced and thematic card considering:
        1. How it fits into the overall game strategy
        2. Its interactions with existing cards
        3. The game's complexity level
        4. The target audience
        """
        print("\n[Card Generation] Generating cards...")
        print(f"[Prompt Used]:\n{prompt}\n")

        new_card = self.llm_service.call_llm(prompt, records=[Card])

        print(f"Generated card: {new_card}")

        if "cards" not in state:
            state["cards"] = []
        state["cards"].append(new_card)

        return state


def get_user_preferences():
    # Temporarily restore standard stdout only for input capture
    original_stdout = sys.stdout
    sys.stdout = sys.__stdout__
    try:
        print(
            "Please provide your preferences for the card game (press Enter to skip):"
        )
        language = input("Preferred language (default: English): ").strip() or "Español"
        theme = (
            input("Theme (e.g., fantasy, sci-fi): ").strip()
            or "Fantasía tierra media, ambientada en Toledo, España"
        )
        style = (
            input("Style preference (competitive, cooperative, party game): ").strip()
            or "Party game. Solo cartas, sin tablero. Ninguna carta tiene coste. Que sea un juego similar a Exploding Kittens"
        )
        players = input("Number of players (e.g., 2-4): ").strip() or "4-12"
        cards_input = input("Maximum number of unique cards: ").strip()
        unique_cards = int(cards_input) if cards_input.isdigit() else None
        audience = input("Target audience (e.g., +10, +18): ").strip() or "+18"
        complexity = input("Rule complexity (Easy, Medium, Hard): ").strip() or "Medio"
    finally:
        # Restore the logger to capture further output
        sys.stdout = original_stdout

    return UserPreferences(
        theme=theme,
        game_style=style,
        number_of_players=players,
        max_unique_cards=unique_cards,
        target_audience=audience,
        rule_complexity=complexity,
        language=language,
    )


def should_continue(state: CardGameState) -> str:
    print("\n[Game Status] Current step:", state["current_step"])
    if len(state["cards"]) < state["game_concept"].number_of_unique_cards:
        return "generate_cards"  # Continue generating cards if not enough
    return END


# Initialize services and agents
llm_service = VertexAILLM(
    model_name="gemini-1.5-pro-002", temperature=0.5, max_output_tokens=8192
)

concept_agent = ConceptGenerationAgent(llm_service)
card_agent = CardGenerationAgent(llm_service)
rules_agent = RuleGenerationAgent(llm_service)

# Create the graph
workflow = StateGraph(CardGameState)

# Adding nodes
workflow.add_node("generate_concept", concept_agent.generate_concept)
workflow.add_node("generate_rules", rules_agent.generate_rules)
workflow.add_node("generate_cards", card_agent.generate_card)

# Define all edges
workflow.add_edge("generate_concept", "generate_rules")
workflow.add_edge("generate_rules", "generate_cards")
workflow.add_conditional_edges("generate_cards", should_continue)

workflow.set_entry_point("generate_concept")

# Compile the graph
app = workflow.compile(checkpointer=MemorySaver())


def generate_card_game() -> CardGameState:
    user_preferences = get_user_preferences()

    initial_state = CardGameState(
        game_concept={},
        cards=[],
        rules={},
        current_step="generate_concept",
        user_preferences=user_preferences,
    )

    result = app.invoke(
        initial_state, config={"recursion_limit": 150, "configurable": {"thread_id": 1}}
    )
    return result


if __name__ == "__main__":
    result = generate_card_game()

    # Print Game Concept
    print("\nGenerated Card Game:")
    print(f"Title: {result['game_concept'].title}")
    print(f"Description: {result['game_concept'].description}")
    print(f"Theme: {result['game_concept'].theme}")
    print(f"Target Audience: {result['game_concept'].target_audience}")
    print(f"Gameplay Style: {result['game_concept'].game_style}")
    print(f"Number of Players: {result['game_concept'].number_of_players}")
    print(f"Rule Complexity: {result['game_concept'].rule_complexity}")

    # Print Cards
    print(f"\nCards ({len(result['cards'])}):")
    for card in result["cards"]:
        print(f"- {card.name} ({card.rarity}): {card.effect}")
        print(f"  Type: {card.type}, Cost: {card.cost}")
        print(f"  Interactions: {card.interactions}")
        print(f"  Flavor text: {card.flavor_text}\n")

    # Print Rules
    print("\nRules:")

    # Handling setup, turn_structure, win_conditions, and special_rules as dicts or strings
    def print_rule_section(section_name, rule_content):
        print(f"{section_name}:")
        if isinstance(rule_content, dict):
            for key, value in rule_content.items():
                print(f"  - {key}: {value}")
        else:
            for step in rule_content.split("\n"):
                print(f"  - {step.strip()}")

    print_rule_section("Setup", result["rules"].setup)
    print_rule_section("Turn Structure", result["rules"].turn_structure)
    print_rule_section("Win Conditions", result["rules"].win_conditions)
    print_rule_section("Special Rules", result["rules"].special_rules)

log_file.close()
