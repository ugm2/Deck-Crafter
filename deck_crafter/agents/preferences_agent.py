from typing import Optional
from deck_crafter.models.user_preferences import UserPreferences, RequiredUserPreferences
from deck_crafter.services.llm_service import LLMService
from deck_crafter.models.state import CardGameState


class PreferencesGenerationAgent:
    DEFAULT_PROMPT = """
    You are an expert card game designer.
    Complete all user preferences for a card game using the information provided by the user.
    If any preference is already specified, respect it. If any is missing, complete it coherently with the description and other preferences.
    If the language preference is not specified, infer the language from the game description. Write all fields in the language you infer from the game description.
    Never leave a field empty or null.

    CRITIQUE TO ADDRESS: An expert reviewed your previous attempt. You MUST address these points.
    Critique: {critique}

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

    def generate_preferences(self, state: CardGameState) -> dict:
        partial_preferences = state.preferences
        game_description = partial_preferences.game_description
        critique = state.critique
        
        context = {
            "game_description": game_description,
            "partial_preferences": partial_preferences.model_dump(),
            "critique": critique or "This is the first attempt, no critique yet."
        }
        
        required = self.llm_service.generate(
            output_model=RequiredUserPreferences, prompt=self.base_prompt, **context
        )
        # We need to preserve the original game_description
        final_prefs = UserPreferences.model_validate(required.model_dump())
        final_prefs.game_description = game_description
        return {"preferences": final_prefs}

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