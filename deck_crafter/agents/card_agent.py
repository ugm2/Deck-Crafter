import logging
from typing import Optional, Dict, List
from deck_crafter.models.game_concept import GameConcept, CardType
from deck_crafter.models.card import Card, CardBatch
from deck_crafter.models.state import CardGameState
from deck_crafter.services.llm_service import LLMService
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)
CARD_BATCH_SIZE = 5


class CardGenerationAgent:
    DEFAULT_PROMPT = ChatPromptTemplate.from_template(
        """
        You are a game balance expert designing cards for strategic depth.

        ### ⚠️ MANDATORY CRITIQUE - TOP PRIORITY ⚠️ ###
        READ THIS FIRST. These issues MUST be fixed in your card design:

        {critique}

        ### SIMULATION FEEDBACK (from actual gameplay testing) ###
        {simulation_feedback}

        ---

        If critique mentions specific stats (attack, defense, cost), use those EXACT values.
        If critique says "NERF card X", make that card type weaker than before.
        If critique says "BUFF card X", make that card type stronger.
        If critique mentions "overpowered", reduce stats by at least 30%.
        If critique mentions "underpowered", increase stats by at least 30%.

        ### BALANCE CONSTRAINTS ###
        - Attack values: 1-8 range (avoid extreme outliers)
        - Defense values: 1-10 range
        - Costs: Must scale with power level (powerful = expensive)
        - Effects: Should have clear counterplay options

        ### RESOURCE NAMING (CRITICAL FOR SIMULATION) ###
        The "cost" field MUST use standard resource names for simulation compatibility:
        - Use "X Mana" for magical/fantasy resources (NOT Aether, Essence, Arcana, etc.)
        - Use "X Energy" for physical/stamina resources
        - Use "X Gold" for economic resources
        - Example: "3 Mana" is correct, "3 Aether Points" is WRONG

        ### GAME CONTEXT ###
        Game Concept: {game_concept}
        Cards generated so far: {current_num_cards}/{total_unique_cards}
        Existing cards: {existing_cards}

        ### CARD TO GENERATE (SINGLE CARD ONLY) ###
        Generate exactly ONE card with these properties:
        - Type: {next_card_type}
        - Description: {card_type_description}
        - Quantity field should be: {quantity} (copies of this card in the deck)

        ### REQUIREMENTS ###
        - Output a SINGLE card object, NOT an array
        - Card must complement existing cards without power creep
        - Include strategic tradeoffs (high attack = low defense, strong effect = high cost)
        - Language: '{language}'

        ### STRUCTURED EFFECT FIELDS (IMPORTANT) ###
        You MUST populate these simulation fields:
        - effect_type: One of "none", "draw", "damage", "heal", "gain_points", "gain_resource", "win_game"
        - effect_value: The numeric value (e.g., "Draw 2 cards" = 2, "Deal 3 damage" = 3)
        - effect_target: "self" (affects you), "opponent" (affects enemy), or "any" (player chooses)

        Examples:
        - "Deal 3 damage to opponent" → effect_type="damage", effect_value=3, effect_target="opponent"
        - "Draw 2 cards" → effect_type="draw", effect_value=2, effect_target="self"
        - "Gain 2 points" → effect_type="gain_points", effect_value=2, effect_target="self"
        - Passive/complex effects → effect_type="none", effect_value=0, effect_target="self"
        """
    )

    BATCH_PROMPT = ChatPromptTemplate.from_template(
        """
        You are a game balance expert designing cards for strategic depth.

        ### ⚠️ MANDATORY CRITIQUE - TOP PRIORITY ⚠️ ###
        READ THIS FIRST. These issues MUST be fixed in your card design:

        {critique}

        ### SIMULATION FEEDBACK (from actual gameplay testing) ###
        {simulation_feedback}

        ---

        If critique mentions specific stats (attack, defense, cost), use those EXACT values.
        If critique says "NERF card X", make that card type weaker than before.
        If critique says "BUFF card X", make that card type stronger.

        ### BALANCE CONSTRAINTS ###
        - Attack values: 1-8 range (avoid extreme outliers)
        - Defense values: 1-10 range
        - Costs: Must scale with power level (powerful = expensive)
        - Effects: Should have clear counterplay options

        ### RESOURCE NAMING (CRITICAL FOR SIMULATION) ###
        The "cost" field MUST use standard resource names for simulation compatibility:
        - Use "X Mana" for magical/fantasy resources (NOT Aether, Essence, Arcana, etc.)
        - Use "X Energy" for physical/stamina resources
        - Use "X Gold" for economic resources
        - Example: "3 Mana" is correct, "3 Aether Points" is WRONG

        ### GAME CONTEXT ###
        Game Concept: {game_concept}
        Cards generated so far: {current_num_cards}/{total_unique_cards}
        Existing cards: {existing_cards}

        ### CARDS TO GENERATE (BATCH OF {batch_size}) ###
        Generate exactly {batch_size} unique cards with these specifications:

        {cards_to_generate}

        ### REQUIREMENTS ###
        - Output a JSON object with a "cards" array containing exactly {batch_size} cards
        - Each card must have a unique name
        - Cards must complement existing cards without power creep
        - Include strategic tradeoffs (high attack = low defense, strong effect = high cost)
        - Language: '{language}'

        ### STRUCTURED EFFECT FIELDS (IMPORTANT) ###
        For each card, you MUST populate these simulation fields:
        - effect_type: One of "none", "draw", "damage", "heal", "gain_points", "gain_resource", "win_game"
        - effect_value: The numeric value (e.g., "Draw 2 cards" = 2, "Deal 3 damage" = 3)
        - effect_target: "self" (affects you), "opponent" (affects enemy), or "any" (player chooses)

        Examples:
        - "Deal 3 damage to opponent" → effect_type="damage", effect_value=3, effect_target="opponent"
        - "Draw 2 cards" → effect_type="draw", effect_value=2, effect_target="self"
        - "Gain 2 points" → effect_type="gain_points", effect_value=2, effect_target="self"
        - "Heal 4 health" → effect_type="heal", effect_value=4, effect_target="self"
        - Passive/complex effects → effect_type="none", effect_value=0, effect_target="self"
        """
    )

    def __init__(
        self, llm_service: LLMService, base_prompt: Optional[ChatPromptTemplate] = None
    ):
        self.llm_service = llm_service
        self.base_prompt = base_prompt or self.DEFAULT_PROMPT

    def generate_card(self, state: CardGameState) -> dict:
        game_concept: GameConcept = state.concept
        if state.cards is None:
            state.cards = []
        existing_cards: List[Card] = state.cards
        critique = state.critique

        if len(existing_cards) >= game_concept.number_of_unique_cards:
            logger.debug(f"[CardAgent] All cards generated ({len(existing_cards)}/{game_concept.number_of_unique_cards})")
            return {}

        next_card_type = self._determine_next_card_type(game_concept, existing_cards)
        if not next_card_type:
            return {}

        logger.debug(f"[CardAgent] Generating card {len(existing_cards)+1}/{game_concept.number_of_unique_cards} "
                    f"(type: {next_card_type.name})")

        num_cards_generated_for_type = self._get_num_cards_generated_for_type(next_card_type, existing_cards)
        next_card_to_generate_shell = self._get_next_card_to_generate(next_card_type, num_cards_generated_for_type, existing_cards)

        context = self._prepare_context(game_concept, existing_cards, next_card_to_generate_shell, next_card_type)
        context["critique"] = critique or "First attempt, no critique yet."
        context["simulation_feedback"] = self._format_simulation_feedback(state)

        new_card = self._generate_new_card(context)

        if new_card:
            logger.info(f"[CardAgent] Generated card: {new_card.name} ({new_card.type})")
            existing_cards.append(new_card)
            return {"cards": existing_cards}
        logger.warning("[CardAgent] Failed to generate card")
        return {}

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
            image_description=""
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
            "language": game_concept.language,
        }
        return context

    def _format_simulation_feedback(self, state: CardGameState) -> str:
        """Format simulation analysis as feedback for card generation."""
        if not state.simulation_analysis:
            return "No simulation data available yet."

        analysis = state.simulation_analysis
        lines = []

        # Confidence warning
        if analysis.confidence and analysis.confidence.overall == "low":
            lines.append("⚠️ LOW CONFIDENCE: " + "; ".join(analysis.confidence.reasons))

        # Exact balance adjustments (highest priority)
        if analysis.balance_adjustments:
            from deck_crafter.game_simulator.models.metrics import BalanceAdjustment
            parsed = BalanceAdjustment.parse_adjustments(analysis.balance_adjustments)
            if parsed:
                lines.append("⚡ EXACT STAT CHANGES REQUIRED (from simulation data):")
                for adj in parsed:
                    if adj.current_value and adj.target_value:
                        lines.append(f"  - {adj.card_name}: {adj.stat} {adj.current_value} → {adj.target_value} ({adj.reason})")
                    else:
                        lines.append(f"  - {adj.card_name}: {adj.action} {adj.stat} ({adj.reason})")

        # Problematic cards
        if analysis.problematic_cards:
            lines.append("\nPROBLEMATIC CARDS (from gameplay simulation):")
            for pc in analysis.problematic_cards:
                fix = f" - Suggested fix: {pc.suggested_fix}" if pc.suggested_fix else ""
                lines.append(f"  - {pc.card_name}: {pc.issue_type.upper()} - {pc.evidence}{fix}")

        # High priority fixes
        if analysis.high_priority_fixes:
            lines.append("\nHIGH PRIORITY FIXES:")
            for fix in analysis.high_priority_fixes:
                lines.append(f"  - {fix}")

        # Balance insights
        if analysis.pacing_assessment in ["poor", "needs_work"]:
            lines.append(f"\nPACING ISSUE: {analysis.pacing_assessment}")
            if analysis.pacing_issues:
                for issue in analysis.pacing_issues:
                    lines.append(f"  - {issue.issue} ({issue.severity})")

        # Anti-fun indicators
        if analysis.anti_fun_indicators:
            lines.append("\nANTI-FUN PATTERNS DETECTED:")
            for indicator in analysis.anti_fun_indicators:
                lines.append(f"  - {indicator}")

        return "\n".join(lines) if lines else "Simulation found no major issues."

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

    def generate_cards_batch(self, state: CardGameState) -> dict:
        """Generate multiple cards in a single LLM call for efficiency."""
        game_concept: GameConcept = state.concept
        if state.cards is None:
            state.cards = []
        existing_cards: List[Card] = state.cards

        remaining = game_concept.number_of_unique_cards - len(existing_cards)
        if remaining <= 0:
            logger.debug(f"[CardAgent] All cards generated ({len(existing_cards)} cards)")
            return {}

        logger.info(f"[CardAgent] Generating batch of up to {min(CARD_BATCH_SIZE, remaining)} cards "
                   f"({len(existing_cards)}/{game_concept.number_of_unique_cards} done)")

        # Collect card specs for the batch
        cards_specs = []
        temp_existing = list(existing_cards)

        for _ in range(min(CARD_BATCH_SIZE, remaining)):
            next_card_type = self._determine_next_card_type(game_concept, temp_existing)
            if not next_card_type:
                break

            num_generated = self._get_num_cards_generated_for_type(next_card_type, temp_existing)
            next_card = self._get_next_card_to_generate(next_card_type, num_generated, temp_existing)

            cards_specs.append({
                "type": next_card_type.name,
                "description": next_card_type.description,
                "quantity": next_card.quantity,
            })

            # Track placeholder for next iteration
            temp_existing.append(Card(
                name=f"placeholder_{len(temp_existing)}",
                type=next_card_type.name,
                quantity=next_card.quantity,
                description="",
                image_description=""
            ))

        if not cards_specs:
            return {}

        # Format specs for prompt
        specs_text = "\n".join([
            f"{i+1}. Type: {spec['type']}, Description: {spec['description']}, Quantity: {spec['quantity']}"
            for i, spec in enumerate(cards_specs)
        ])

        critique = state.critique
        context = {
            "game_concept": game_concept.model_dump(),
            "current_num_cards": len(existing_cards),
            "total_unique_cards": game_concept.number_of_unique_cards,
            "existing_cards": [card.model_dump() for card in existing_cards],
            "batch_size": len(cards_specs),
            "cards_to_generate": specs_text,
            "language": game_concept.language,
            "critique": critique or "First attempt, no critique yet.",
            "simulation_feedback": self._format_simulation_feedback(state),
        }

        result = self.llm_service.generate(
            output_model=CardBatch,
            prompt=self.BATCH_PROMPT,
            **context
        )

        if result and result.cards:
            logger.info(f"[CardAgent] Batch generated {len(result.cards)} cards: "
                       f"{[c.name for c in result.cards]}")
            existing_cards.extend(result.cards)
            return {"cards": existing_cards}

        logger.warning("[CardAgent] Batch generation failed or returned no cards")
        return {}

