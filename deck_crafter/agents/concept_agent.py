from typing import Optional
from deck_crafter.models.game_concept import GameConcept
from deck_crafter.models.state import CardGameState
from deck_crafter.models.user_preferences import UserPreferences
from deck_crafter.services.llm_service import LLMService


class ConceptGenerationAgent:
    DEFAULT_PROMPT = """
    You are a world-class card game designer.
    Create a concept for a unique and engaging card game based on the following user preferences:

    User preferences: {user_preferences}

    Ensure the game concept aligns with the preferences provided.
    """

    def __init__(
        self, llm_service: LLMService, base_prompt: Optional[str] = None
    ):
        """
        Initialize the agent with an LLMService for generating game concepts and a base prompt template.
        If no base_prompt is provided, it defaults to a built-in template.

        :param llm_service: Instance of LLMService for interacting with the language model.
        :param base_prompt: Optional. String template to provide the base structure for the LLM prompt.
        """
        self.llm_service = llm_service
        self.base_prompt = base_prompt or self.DEFAULT_PROMPT

    def generate_concept(self, state: CardGameState) -> CardGameState:
        """
        Generate a game concept based on the user preferences in the given state.

        :param state: The current state of the card game including user preferences.
        :return: Updated state with the generated game concept.
        """
        user_preferences: UserPreferences = state.preferences

        context = self._prepare_context(user_preferences)

        game_concept = self._generate_concept(context)

        self._override_with_user_preferences(game_concept, user_preferences)

        state.concept = game_concept

        return state

    def _prepare_context(self, user_preferences: UserPreferences) -> dict:
        """
        Prepare the context for the LLM prompt, including the user preferences.

        :param user_preferences: The user preferences for the card game.
        :return: A dictionary representing the context to pass to the LLM prompt.
        """
        return {
            "user_preferences": user_preferences.model_dump(),
        }

    def _generate_concept(self, context: dict) -> Optional[GameConcept]:
        """
        Call the LLM to generate a game concept based on the provided context.

        :param context: The context containing details about the user preferences.
        :return: The newly generated game concept, or None if generation failed.
        """
        return self.llm_service.generate(
            output_model=GameConcept,
            prompt=self.base_prompt,
            **context
        )

    def _override_with_user_preferences(self, game_concept: GameConcept, user_preferences: UserPreferences) -> None:
        """
        Override specific fields in the game concept with user preferences.

        :param game_concept: The generated game concept to modify.
        :param user_preferences: The user preferences to apply.
        """
        if user_preferences.language:
            game_concept.language = user_preferences.language
        if user_preferences.theme:
            game_concept.theme = user_preferences.theme
        if user_preferences.rule_complexity:
            game_concept.rule_complexity = user_preferences.rule_complexity
