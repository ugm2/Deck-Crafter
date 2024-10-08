from typing import Optional
from deck_crafter.models.game_concept import GameConcept
from deck_crafter.models.rules import Rules
from deck_crafter.models.state import CardGameState
from deck_crafter.services.llm_service import LLMService
from langchain_core.prompts import ChatPromptTemplate


class RuleGenerationAgent:
    DEFAULT_PROMPT = ChatPromptTemplate.from_template(
        """
        You are a world-class card game designer.
        Create comprehensive rules for the card game based on the following game concept.

        Game Concept: {game_concept}

        Ensure the rules are clear, balanced, and suitable for the game and target audience.
        """
    )

    def __init__(
        self, llm_service: LLMService, base_prompt: Optional[ChatPromptTemplate] = None
    ):
        """
        Initialize the agent with an LLMService for generating game rules and a base prompt template.
        If no base_prompt is provided, it defaults to a built-in template.

        :param llm_service: Instance of LLMService for interacting with the language model.
        :param base_prompt: Optional. ChatPromptTemplate to provide the base structure for the LLM prompt.
        """
        self.llm_service = llm_service
        self.base_prompt = base_prompt or self.DEFAULT_PROMPT

    def generate_rules(self, state: CardGameState) -> CardGameState:
        """
        Generate comprehensive game rules based on the game concept.

        :param state: The current state of the card game, including the game concept.
        :return: Updated state with the generated rules.
        """
        game_concept: GameConcept = state["game_concept"]

        context = self._prepare_context(game_concept)

        rules = self._generate_rules(context)

        if rules:
            state["rules"] = rules

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
        return self.llm_service.call_llm(
            structured_outputs=[Rules],
            prompt_template=self.base_prompt,
            context=context,
        )
