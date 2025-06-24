from typing import Optional
from deck_crafter.models.game_concept import GameConcept
from deck_crafter.models.rules import Rules
from deck_crafter.models.state import CardGameState
from deck_crafter.services.llm_service import LLMService


class RuleGenerationAgent:
    """Generates comprehensive and clear rules based on a game concept."""
    DEFAULT_PROMPT = """
    ### ROLE & PERSONA ###
    Act as a world-class game designer and an expert technical writer. You are renowned for creating rulebooks that are elegant, comprehensive, and exceptionally easy to understand. Your goal is to write a rulebook so clear that it leaves no room for ambiguity or player arguments.

    ### CORE PRINCIPLES OF A WORLD-CLASS RULEBOOK ###
    A professional rulebook is built on universal principles. When generating the content, you MUST ensure your output embodies these qualities:
    1.  **Unambiguous Goal & Setup:** A player must immediately understand the ultimate objective of the game and how to start a new game from scratch. The setup instructions must be a clear, step-by-step sequence.
    2.  **Structured Turn Flow:** A player's turn must be broken down into a clear, sequential series of distinct phases. This structured flow is essential for preventing confusion about when actions can be taken.
    3.  **Explicitly Defined Terminology:** Any special term or keyword that is not part of everyday language (e.g., 'Counter', 'Exhaust', 'Synergy') must be clearly and precisely defined. A central glossary is the gold standard for this.
    4.  **Concrete Examples:** The best way to clarify complex rules or interactions is to provide concrete examples of play. Illustrate non-obvious situations to guide the players.
    5.  **Comprehensive Coverage:** The rules must cover all core aspects of gameplay, including how resources (if any) are managed and how players can react or respond to actions outside of their main turn.

    ### TASK ###
    Your task is to write the complete content for a professional rulebook based on the provided `Game Concept`. Embody your role and apply all the core principles listed above to create a clear, complete, and elegant set of rules. You will be provided with a specific data structure to populate; ensure your output conforms perfectly to it.

    ### INPUT DATA ###
    Game Concept: {game_concept}

    ### OUTPUT INSTRUCTIONS ###
    - The entire rulebook must be written in the language specified in the game concept: '{game_concept[language]}'.
    """

    def __init__(
        self, llm_service: LLMService, base_prompt: Optional[str] = None
    ):
        """
        Initialize the agent with an LLMService for generating game rules and a base prompt template.
        If no base_prompt is provided, it defaults to a built-in template.

        :param llm_service: Instance of LLMService for interacting with the language model.
        :param base_prompt: Optional. String template to provide the base structure for the LLM prompt.
        """
        self.llm_service = llm_service
        self.base_prompt = base_prompt or self.DEFAULT_PROMPT

    def generate_rules(self, state: CardGameState) -> CardGameState:
        """
        Generate comprehensive game rules based on the game concept.

        :param state: The current state of the card game, including the game concept.
        :return: Updated state with the generated rules.
        """
        game_concept: GameConcept = state.concept

        context = self._prepare_context(game_concept)

        rules = self._generate_rules(context)

        if rules:
            state.rules = rules

        return state

    def _prepare_context(self, game_concept: GameConcept) -> dict:
        """
        Prepare the context for the LLM prompt, including the game concept details.

        :param game_concept: The game concept of the card game.
        :return: A dictionary representing the context to pass to the LLM prompt.
        """
        return {
            "game_concept": game_concept.model_dump(),
        }

    def _generate_rules(self, context: dict) -> Optional[Rules]:
        """
        Call the LLM to generate game rules based on the provided context.

        :param context: The context containing details about the game concept.
        :return: The newly generated rules, or None if generation failed.
        """
        return self.llm_service.generate(
            output_model=Rules,
            prompt=self.base_prompt,
            **context
        )
