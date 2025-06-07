from typing import Optional
from deck_crafter.models.user_preferences import UserPreferences, RequiredUserPreferences
from deck_crafter.services.llm_service import LLMService


class PreferencesGenerationAgent:
    DEFAULT_PROMPT = """
    You are an expert card game designer.
    Complete all user preferences for a card game using the information provided by the user.
    If any preference is already specified, respect it. If any is missing, complete it coherently with the description and other preferences.
    If the language preference is not specified, infer the language from the game description. Write all fields in the language you infer from the game description.
    Never leave a field empty or null.

    Game description: {game_description}
    Partial preferences: {partial_preferences}
    """

    def __init__(
        self, llm_service: LLMService, base_prompt: Optional[str] = None
    ):
        """
        Initialize the agent with an LLMService for generating user preferences and a base prompt template.
        If no base_prompt is provided, it defaults to a built-in template.

        :param llm_service: Instance of LLMService for interacting with the language model.
        :param base_prompt: Optional. String template to provide the base structure for the LLM prompt.
        """
        self.llm_service = llm_service
        self.base_prompt = base_prompt or self.DEFAULT_PROMPT

    def generate_preferences(self, game_description: str, partial_preferences: Optional[UserPreferences] = None) -> UserPreferences:
        """
        Generate user preferences based on the game description and any partial preferences provided by the user.
        """
        context = {
            "game_description": game_description,
            "partial_preferences": partial_preferences.model_dump() if partial_preferences else {}
        }
        return self._generate_preferences(context)

    def _generate_preferences(self, context: dict) -> UserPreferences:
        """
        Call the LLM to generate user preferences based on the provided context.

        :param context: The context containing the game description and partial preferences.
        :return: The generated user preferences.
        """
        # Llama al LLM para obtener RequiredUserPreferences y lo convierte a UserPreferences
        required = self.llm_service.generate(
            output_model=RequiredUserPreferences,
            prompt=self.base_prompt,
            **context
        )
        return UserPreferences.model_validate(required.model_dump()) 