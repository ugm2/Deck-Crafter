import json
import sys
from typing import TypedDict, List, Optional
from dataclasses import dataclass
from deck_crafter.core.llm_service import LLMService, VertexAILLM
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


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
    theme: Optional[str] = None
    game_style: Optional[str] = None
    number_of_players: Optional[str] = None
    max_unique_cards: Optional[int] = None
    target_audience: Optional[str] = None
    rule_complexity: Optional[str] = None
    language: Optional[str] = "English"


@dataclass
class GameConcept:
    # Required fields
    theme: str
    title: str
    description: str
    game_style: str
    number_of_players: str
    number_of_cards: int  # Changed back to required

    # Optional fields
    target_audience: Optional[str] = None
    rule_complexity: Optional[str] = None
    card_distribution: Optional[dict] = None

    def __post_init__(self):
        self.number_of_cards = int(self.number_of_cards)


@dataclass
class Card:
    # Required fields
    name: str
    type: str  # e.g., Action, Character, Resource
    effect: str

    # Optional fields
    cost: Optional[str] = None  # Some games might not use costs
    flavor_text: Optional[str] = None
    rarity: Optional[str] = None  # Not all games use rarity
    interactions: Optional[str] = None

    def __repr__(self) -> str:
        return f"Card(name={self.name}, type={self.type}, effect={self.effect}, cost={self.cost}, flavor_text={self.flavor_text}, rarity={self.rarity}, interactions={self.interactions})"


@dataclass
class Rules:
    # Required fields
    setup: str
    turn_structure: str
    win_conditions: str

    # Optional field
    special_rules: Optional[str] = None


def process_json_response(json_response: str) -> dict:
    json_response = json_response.strip()
    if json_response.startswith("```json"):
        json_response = json_response[7:]
    if json_response.endswith("```"):
        json_response = json_response[:-3]
    json_response = json_response.strip()
    try:
        json_object = json.loads(json_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode LLM response as JSON: {e}\n{json_response}")
    return json_object


class ConceptGenerationAgent:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_concept(self, state: CardGameState) -> CardGameState:
        user_prefs = state["user_preferences"]

        # First, determine which fields are required based on user preferences
        required_fields = {
            "theme": "The central theme of the game",
            "title": "A catchy, thematic title",
            "description": "A brief but clear description of the game",
            "game_style": "The core gameplay style (e.g., competitive, cooperative)",
            "number_of_players": "The recommended number of players",
            "number_of_cards": "Total number of unique cards needed for the game",
        }

        optional_fields = {
            "target_audience": "Age group or specific audience",
            "rule_complexity": "Complexity level of the rules",
            "card_distribution": "Types of cards and their quantities",
        }

        # Move fields from optional to required based on user preferences
        if user_prefs.theme:
            required_fields["theme"] = (
                f"Must be related to the user's selected theme: {user_prefs.theme}"
            )
            optional_fields.pop("theme", None)

        if user_prefs.game_style:
            required_fields["game_style"] = (
                f"Must match user's preference: {user_prefs.game_style}"
            )
            optional_fields.pop("game_style", None)

        if user_prefs.target_audience:
            required_fields["target_audience"] = (
                f"Must be: {user_prefs.target_audience}"
            )
            optional_fields.pop("target_audience", None)

        if user_prefs.rule_complexity:
            required_fields["rule_complexity"] = (
                f"Must be: {user_prefs.rule_complexity}"
            )
            optional_fields.pop("rule_complexity", None)

        required_fields = "\n".join(f"- {k}: {v}" for k, v in required_fields.items())
        optional_fields = "\n".join(f"- {k}: {v}" for k, v in optional_fields.items())

        prompt = f"""
        Create a concept for a unique and engaging card game in {user_prefs.language}.

        Required fields in your response:
        {required_fields}

        {'Optional fields (include only if relevant to your game concept):' if optional_fields else ''}
        {optional_fields}

        User preferences to incorporate:
        {f'- Maximum unique cards: {user_prefs.max_unique_cards}' if user_prefs.max_unique_cards else '- Choose an appropriate number of unique cards for the game.'}
        {f'- Number of players: {user_prefs.number_of_players}' if user_prefs.number_of_players else ''}

        Special instructions:
        1. The number_of_cards in your response must not exceed {user_prefs.max_unique_cards if user_prefs.max_unique_cards else 'a reasonable number for the game concept'}
        2. Any user preferences provided above must be strictly followed in your response
        3. Optional fields should only be included if they add value to the game concept

        Format the response as a JSON object with all required fields and any relevant optional fields. Example:

        {{
            "theme": "The central theme of the game",
            "title": "A catchy, thematic title",
            "description": "A brief but clear description of the game",
            "game_style": "The core gameplay style (e.g., competitive, cooperative)",
            "number_of_players": "The recommended number of players",
            "number_of_cards": "Total number of unique cards needed for the game",
            "target_audience": "Age group or specific audience",
            "rule_complexity": "Complexity level of the rules",
            "card_distribution": {{
                "type1": number,
                "type2": number,
                ...
            }}
        }}
        """

        print("\n[Concept Generation] Generating the game concept...")
        print(f"[Prompt Used]:\n{prompt}\n")

        concept_json = self.llm_service.call_llm(prompt)
        concept_json = process_json_response(concept_json)

        # Override concept values with user preferences if they exist
        if user_prefs.target_audience:
            concept_json["target_audience"] = user_prefs.target_audience
        if user_prefs.game_style:
            concept_json["game_style"] = user_prefs.game_style
        if user_prefs.number_of_players:
            concept_json["number_of_players"] = user_prefs.number_of_players
        if user_prefs.max_unique_cards:
            concept_json["number_of_cards"] = user_prefs.max_unique_cards
        if user_prefs.rule_complexity:
            concept_json["rule_complexity"] = user_prefs.rule_complexity

        # No need to override values here anymore since the LLM will respect user preferences
        state["game_concept"] = GameConcept(**concept_json)
        state["current_step"] = "generate_cards"
        return state


class CardGenerationAgent:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_card(self, state: CardGameState) -> CardGameState:
        game_concept = state["game_concept"]
        user_prefs = state["user_preferences"]
        existing_cards = state.get("cards", [])

        remaining_types = {
            card_type: count
            for card_type, count in game_concept.card_distribution.items()
            if sum(1 for card in existing_cards if card.type == card_type) < count
        }

        if not remaining_types:
            state["current_step"] = "generate_rules"
            return state

        next_type = max(remaining_types.items(), key=lambda x: x[1])[0]

        # Determine required and optional fields
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

        required_fields = "\n".join(f"- {field}" for field in required_fields)
        optional_fields = "\n".join(f"- {field}" for field in optional_fields)

        prompt = f"""
        Generate a new card for the game: {game_concept.title} in {user_prefs.language}
        Game description: {game_concept.description}
        Theme: {game_concept.theme}
        Game style: {game_concept.game_style}
        Rule complexity: {game_concept.rule_complexity}

        Required fields in your response:
        {required_fields}

        Optional fields (include if relevant):
        {optional_fields}

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

        Format the response as a JSON object including all required fields and any relevant optional fields. Example:
        {{
            "name": "Card name",
            "type": "Card type",
            "effect": "Card's game effect",
            "cost": "Resource cost to play the card",
            "flavor_text": "Thematic description",
            "rarity": "Card rarity (Common, Uncommon, Rare, etc.)",
            "interactions": "Interactions with other cards/mechanics"
        }}
        """
        print("\n[Card Generation] Generating cards...")
        print(f"[Prompt Used]:\n{prompt}\n")

        new_card = self.llm_service.call_llm(prompt)
        new_card = process_json_response(new_card)

        if "cards" not in state:
            state["cards"] = []
        state["cards"].append(Card(**new_card))

        if len(state["cards"]) >= state["game_concept"].number_of_cards:
            state["current_step"] = "generate_rules"

        return state


class RuleGenerationAgent:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_rules(self, state: CardGameState) -> CardGameState:
        game_concept = state["game_concept"]
        user_prefs = state["user_preferences"]
        cards = state["cards"]

        # Determine if special rules are required based on game complexity and card interactions
        requires_special_rules = (
            hasattr(game_concept, "rule_complexity")
            and game_concept.rule_complexity.lower() in ["medium", "hard", "complex"]
            or any(card.interactions for card in cards)
        )

        required_rules = [
            "setup: Detailed setup instructions considering the game's theme and components",
            "turn_structure: Clear explanation of how players take turns and interact with cards",
            "win_conditions: Specific conditions for winning the game",
        ]

        optional_rules = []
        if not requires_special_rules:
            optional_rules.append(
                "special_rules: Any unique mechanics or interactions between cards"
            )
        else:
            required_rules.append(
                "special_rules: Required special mechanics based on card interactions"
            )

        prompt_parts = [
            f"Create comprehensive rules for the card game: {game_concept.title} in {user_prefs.language}",
            f"Game concept: {game_concept.description}",
            f"Theme: {game_concept.theme}",
            f"Gameplay style: {game_concept.game_style}",
            f"Rule complexity: {game_concept.rule_complexity}",
            f"Number of players: {game_concept.number_of_players}",
            "\nRequired rule sections:",
            *[f"- {rule}" for rule in required_rules],
        ]

        if optional_rules:
            prompt_parts.extend(
                [
                    "\nOptional rule section (include only if relevant):",
                    *[f"- {rule}" for rule in optional_rules],
                ]
            )

        prompt_parts.extend(
            [
                "\nHere are all the generated cards:",
                "\n".join(str(card) for card in cards),
                "\nGuidelines for rule creation:",
                "1. Ensure rules match the game's complexity level",
                "2. Make sure all card types and interactions are covered",
                "3. Balance depth of strategy with ease of learning",
                "4. Consider the target audience when explaining concepts",
                "\nFormat the response as a JSON object with all required fields and any relevant optional fields.",
                " Example (only these fields, don't include any other):",
                "{",
                '  "setup": "Detailed setup instructions considering the game\'s theme and components",',
                '  "turn_structure": "Clear explanation of how players take turns and interact with cards",',
                '  "win_conditions": "Specific conditions for winning the game",',
                '  "special_rules": "Any unique mechanics or interactions between cards",',
                "}",
            ]
        )

        prompt = "\n".join(prompt_parts)

        print("\n[Rule Generation] Generating the rules...")
        print(f"[Prompt Used]:\n{prompt}\n")

        rules_json = self.llm_service.call_llm(prompt)
        rules = process_json_response(rules_json)

        state["rules"] = Rules(**rules)
        state["current_step"] = "finished"
        return state


def should_continue(state: CardGameState) -> str:
    return state["current_step"]


def get_user_preferences():
    # Temporarily restore standard stdout only for input capture
    original_stdout = sys.stdout
    sys.stdout = sys.__stdout__
    try:
        print(
            "Please provide your preferences for the card game (press Enter to skip):"
        )
        language = input("Preferred language (default: English): ").strip() or "English"
        theme = input("Theme (e.g., fantasy, sci-fi): ").strip() or None
        style = (
            input("Style preference (competitive, cooperative, party game): ").strip()
            or None
        )
        players = input("Number of players (e.g., 2-4): ").strip() or None
        cards_input = input(
            "Maximum number of unique cards: "
        ).strip()  # Updated prompt
        unique_cards = int(cards_input) if cards_input else None
        audience = input("Target audience (e.g., +10, +18): ").strip() or None
        complexity = input("Rule complexity (Easy, Medium, Hard): ").strip() or None
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


# Initialize services and agents
llm_service = VertexAILLM(
    model_name="gemini-1.5-pro-002", temperature=0.5, max_output_tokens=2048
)

concept_agent = ConceptGenerationAgent(llm_service)
card_agent = CardGenerationAgent(llm_service)
rules_agent = RuleGenerationAgent(llm_service)

# Create the graph
workflow = StateGraph(CardGameState)

# Add nodes and edges (unchanged)
workflow.add_node("generate_concept", concept_agent.generate_concept)
workflow.add_node("generate_cards", card_agent.generate_card)
workflow.add_node("generate_rules", rules_agent.generate_rules)

workflow.add_conditional_edges(
    "generate_concept",
    should_continue,
    {
        "generate_cards": "generate_cards",
    },
)

workflow.add_conditional_edges(
    "generate_cards",
    should_continue,
    {
        "generate_cards": "generate_cards",
        "generate_rules": "generate_rules",
    },
)

workflow.add_conditional_edges(
    "generate_rules",
    should_continue,
    {"finished": END},
)

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
        initial_state, config={"recursion_limit": 100, "configurable": {"thread_id": 1}}
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
