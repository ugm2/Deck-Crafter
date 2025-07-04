from typing import Optional, Dict, List
from deck_crafter.models.game_concept import GameConcept, CardType
from deck_crafter.models.card import Card
from deck_crafter.models.state import CardGameState
from deck_crafter.services.llm_service import LLMService
from langchain_core.prompts import ChatPromptTemplate


class CardGenerationAgent:
    DEFAULT_PROMPT = ChatPromptTemplate.from_template(
        """
        You are a world-class card game designer.
        Based on the game concept and existing cards, generate the full details for the next card.

        Game Concept:
        {game_concept}

        Current number of cards generated: {current_num_cards}
        Total number of unique cards to generate: {total_unique_cards}

        List of existing cards:
        {existing_cards}

        Next card to generate should be of type: {next_card_type}
        Card Type Description: {card_type_description}
        Quantity for this card: {quantity}

        Ensure the new card fits into the overall game strategy, interacts well with existing cards, matches the game's complexity level, and aligns with the game concept.
        All textual fields (name, description, flavor_text, interactions) should be in '{game_concept[language]}'.
        """
    )

    def __init__(
        self, llm_service: LLMService, base_prompt: Optional[ChatPromptTemplate] = None
    ):
        self.llm_service = llm_service
        self.base_prompt = base_prompt or self.DEFAULT_PROMPT

    def generate_card(self, state: CardGameState) -> CardGameState:
        game_concept: GameConcept = state.concept
        if state.cards is None:
            state.cards = []
        existing_cards: List[Card] = state.cards

        if len(existing_cards) >= game_concept.number_of_unique_cards:
            # All cards have been generated
            return state

        next_card_type = self._determine_next_card_type(game_concept, existing_cards)
        if not next_card_type:
            # No more card types to generate
            return state

        num_cards_generated_for_type = self._get_num_cards_generated_for_type(
            next_card_type, existing_cards
        )

        next_card = self._get_next_card_to_generate(
            next_card_type, num_cards_generated_for_type, existing_cards
        )
        context = self._prepare_context(
            game_concept, existing_cards, next_card, next_card_type
        )
        new_card = self._generate_new_card(context)

        if new_card:
            existing_cards.append(new_card)
            state.cards = existing_cards
        else:
            pass

        return state

    def _get_num_cards_generated_for_type(
        self, card_type: CardType, existing_cards: List[Card]
    ) -> int:
        """
        Counts the number of unique cards generated for a specific CardType.
        """
        card_names = set(
            card.name for card in existing_cards if card.type == card_type.name
        )
        return len(card_names)

    def _determine_next_card_type(
        self, game_concept: GameConcept, existing_cards: List[Card]
    ) -> Optional[CardType]:
        """
        Determines the next CardType that needs more unique cards generated.
        """
        card_type_counts = {card_type.name: 0 for card_type in game_concept.card_types}
        for card in existing_cards:
            if card.type in card_type_counts:
                card_type_counts[card.type] += 1

        for card_type in game_concept.card_types:
            if card_type_counts[card_type.name] < card_type.unique_cards:
                return card_type

        return None  # All card types have generated the required unique cards

    def _get_next_card_to_generate(
        self,
        card_type: CardType,
        num_cards_generated_for_type: int,
        existing_cards: List[Card],
    ) -> Card:
        quantity_per_card = self._calculate_quantity_per_card(
            card_type, num_cards_generated_for_type, existing_cards
        )
        return Card(
            name="",
            quantity=quantity_per_card,
            type=card_type.name,
            description="",
        )

    def _calculate_quantity_per_card(
        self,
        card_type: CardType,
        num_cards_generated_for_type: int,
        existing_cards: List[Card],
    ) -> int:
        """
        Calculates the quantity per unique card within a CardType, adjusting for already generated cards.
        """
        total_quantity = card_type.quantity
        total_unique_cards = card_type.unique_cards

        # Quantities assigned so far
        quantities_assigned = sum(
            card.quantity for card in existing_cards if card.type == card_type.name
        )
        # Remaining quantity to assign
        remaining_quantity = total_quantity - quantities_assigned
        # Remaining unique cards to generate
        remaining_unique_cards = total_unique_cards - num_cards_generated_for_type

        if remaining_unique_cards == 0:
            return 0

        # Calculate base quantity
        base_quantity = remaining_quantity // remaining_unique_cards
        remainder = remaining_quantity % remaining_unique_cards

        # If this is among the first 'remainder' cards, add 1
        if num_cards_generated_for_type < remainder:
            return base_quantity + 1
        else:
            return base_quantity

    def base_quantity(self, card_type: CardType) -> int:
        return card_type.quantity // card_type.unique_cards
    def _prepare_context(
        self,
        game_concept: GameConcept,
        existing_cards: List[Card],
        next_card: Card,
        card_type: CardType,
    ) -> Dict:
        context = {
            "game_concept": game_concept.model_dump(),
            "current_num_cards": len(existing_cards),
            "total_unique_cards": game_concept.number_of_unique_cards,
            "existing_cards": [card.model_dump() for card in existing_cards],
            "next_card_type": card_type.name,
            "card_type_description": card_type.description,
            "quantity": next_card.quantity,
        }
        return context

    def _generate_new_card(self, context: Dict) -> Optional[Card]:
        result = self.llm_service.generate(
            output_model=Card,
            prompt=self.base_prompt,
            **context
        )
        if result:
            return result
        else:
            return None  # Handle the failure case appropriately

