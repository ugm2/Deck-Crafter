from typing import Optional, Dict
from deck_crafter.models.game_concept import GameConcept
from deck_crafter.models.card import Card
from deck_crafter.models.rules import Rules
from deck_crafter.models.state import CardGameState
from deck_crafter.services.llm_service import LLMService
from langchain_core.prompts import ChatPromptTemplate


class CardGenerationAgent:
    DEFAULT_PROMPT = ChatPromptTemplate.from_template(
        """
        You are a world-class card game designer.
        Generate a new card for the game based on the following details:

        Game Concept: {game_concept}

        Current number of cards: {current_num_cards}
        List of existing cards: {existing_cards}
        
        Card Distribution Plan: {card_distribution_plan}
        Current cards by type: {current_cards_by_type}

        The next card type to generate should be: {next_card_type}

        Ensure the new card fits into the overall game strategy, interacts with existing cards, matches the game's complexity level, and aligns with the game concept.
        """
    )

    def __init__(
        self, llm_service: LLMService, base_prompt: Optional[ChatPromptTemplate] = None
    ):
        """
        Initialize the agent with an LLMService for generating cards and a base prompt template.
        If no base_prompt is provided, it defaults to a built-in template.

        :param llm_service: Instance of LLMService for interacting with the language model.
        :param base_prompt: Optional. ChatPromptTemplate to provide the base structure for the LLM prompt.
        """
        self.llm_service = llm_service
        self.base_prompt = base_prompt or self.DEFAULT_PROMPT

    def generate_card(self, state: CardGameState) -> CardGameState:
        """
        Generate a new card based on the game concept and the current state of the card deck.

        :param state: The current state of the card game, including the game concept and cards generated so far.
        :return: Updated state with the newly generated card.
        """
        game_concept: GameConcept = state["game_concept"]
        existing_cards = state.get("cards", [])

        next_card_type = self._get_next_card_type(game_concept, existing_cards)

        context = self._prepare_context(game_concept, existing_cards, next_card_type)

        new_card = self._generate_new_card(context)

        if new_card:
            state["cards"] = state.get("cards", []) + [new_card]

        return state

    def _get_next_card_type(
        self, game_concept: GameConcept, existing_cards: list
    ) -> str:
        """
        Determine the next card type to generate based on the card distribution and current cards.

        :param game_concept: The game concept containing card distribution details.
        :param existing_cards: The list of cards already generated.
        :return: The next card type to generate.
        """
        remaining_types = {
            card_type: count
            for card_type, count in game_concept.card_distribution.items()
            if sum(1 for card in existing_cards if card.type == card_type) < count
        }

        next_card_type = max(remaining_types.items(), key=lambda x: x[1])[0]
        return next_card_type

    def _prepare_context(
        self, game_concept: GameConcept, existing_cards: list, next_card_type: str
    ) -> Dict:
        """
        Prepare the context for the LLM prompt, including the game concept, card distribution, and current cards.

        :param game_concept: The game concept of the card game.
        :param existing_cards: The list of cards already generated.
        :param next_card_type: The next card type to generate.
        :return: A dictionary representing the context to pass to the LLM prompt.
        """
        context = {
            "game_concept": game_concept.model_dump(),
            "current_num_cards": len(existing_cards),
            "existing_cards": [card.model_dump() for card in existing_cards],
            "card_distribution_plan": game_concept.card_distribution,
            "current_cards_by_type": {
                card_type: sum(1 for card in existing_cards if card.type == card_type)
                for card_type in game_concept.card_distribution
            },
            "next_card_type": next_card_type,
        }
        return context

    def _generate_new_card(self, context: Dict) -> Optional[Card]:
        """
        Call the LLM to generate a new card based on the provided context.

        :param context: The context containing details about the game concept and card state.
        :return: The newly generated card, or None if generation failed.
        """
        return self.llm_service.call_llm(
            structured_outputs=[Card],
            prompt_template=self.base_prompt,
            context=context,
        )
