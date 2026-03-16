"""
Rule Compiler: Translates deck_crafter Rules + Cards into GameDefinition.

Strategy:
1. Pattern-based parsing for common rules (fast, no LLM)
2. LLM fallback for ambiguous/complex rules
"""

import re
import logging
from typing import Literal

from pydantic import BaseModel, Field

from deck_crafter.models.rules import Rules
from deck_crafter.models.card import Card
from deck_crafter.game_simulator.models.game_definition import (
    GameDefinition,
    CardDefinition,
    CardEffect,
    RuleSet,
    WinCondition,
)

logger = logging.getLogger(__name__)


class ParsedCardEffect(BaseModel):
    """LLM response for parsing a card effect."""
    effect_type: Literal["none", "draw", "damage", "heal", "gain_points", "gain_resource", "win_game"] = Field(
        description="The primary mechanical effect of the card"
    )
    effect_value: int = Field(
        default=0,
        description="Numeric value for the effect (e.g., draw 3 cards = 3)"
    )
    effect_target: Literal["self", "opponent", "any"] = Field(
        default="self",
        description="Who the effect targets"
    )
    reasoning: str = Field(
        description="Brief explanation of how you interpreted the card"
    )


class ParsedWinCondition(BaseModel):
    """LLM response for parsing win conditions."""
    win_type: Literal["points", "elimination", "last_standing", "empty_deck", "property_threshold"] = Field(
        description="Type of win condition"
    )
    target_value: int = Field(
        default=0,
        description="Target value if applicable (e.g., first to 10 points)"
    )
    property_name: str = Field(
        default="",
        description="Property name if property_threshold type"
    )
    reasoning: str = Field(
        description="Brief explanation of interpretation"
    )


class ParsedResourceSystem(BaseModel):
    """LLM response for parsing resource mechanics."""
    resource_name: str = Field(
        description="Normalized resource name: must be one of 'mana', 'energy', 'gold', 'action'. "
                    "Map fantasy names like 'Aether', 'Essence', 'Spirit' to 'mana'."
    )
    initial_amount: int = Field(
        default=0,
        description="How much resource players start with (0 if not specified)"
    )
    per_turn_gain: int = Field(
        default=1,
        description="How much resource players gain each turn (default 1 if generates but amount unclear)"
    )
    reasoning: str = Field(
        description="Brief explanation of how you interpreted the resource system"
    )


class ParsedCardCost(BaseModel):
    """LLM response for parsing card costs."""
    resource_name: str = Field(
        description="Normalized resource name: 'mana', 'energy', 'gold', or 'action'. "
                    "Map fantasy names (Aether, Essence, etc.) to 'mana'."
    )
    amount: int = Field(
        description="How much of the resource this card costs"
    )


class RuleCompiler:
    """Compiles deck_crafter Rules + Cards into a simulatable GameDefinition."""

    # Patterns for extracting numbers from text
    NUMBER_WORDS = {
        # English
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "a": 1, "an": 1,
        # Spanish
        "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
        "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    }

    def __init__(self, llm_service=None):
        """
        Args:
            llm_service: Optional LLM service for fallback parsing.
                        If None, only pattern-based parsing is used.
        """
        self.llm_service = llm_service
        self.warnings: list[str] = []
        self.has_resources = True  # Default, may be overridden during compile

    def _has_resource_system(self, rules: Rules) -> bool:
        """Check if the game has a resource system (mana, energy, etc.).

        Returns True by default - only returns False if resource_mechanics
        explicitly states there are no resources.
        """
        if not rules.resource_mechanics:
            # No resource info = assume costs should be respected
            return True

        text = rules.resource_mechanics.lower()

        # Explicit "no resources" indicators
        no_resource_phrases = [
            "no external resource",
            "no resource",
            "no mana",
            "no energy",
            "sin recursos",
            "no hay recursos",
            "cards are played directly",
            "played directly from",
        ]

        for phrase in no_resource_phrases:
            if phrase in text:
                return False

        # If resource_mechanics exists but doesn't say "no resources", assume it has resources
        return True

    def compile(
        self,
        rules: Rules,
        cards: list[Card],
        game_name: str = "Compiled Game",
        num_players: int = 2,
    ) -> GameDefinition:
        """
        Compile Rules + Cards into a GameDefinition.

        Returns:
            GameDefinition ready for simulation
        """
        self.warnings = []

        # Check if resource system exists
        self.has_resources = self._has_resource_system(rules)

        # Parse rules
        rule_set = self._parse_rules(rules)

        # Parse cards
        card_defs = [self._parse_card(card) for card in cards]

        # Parse win condition
        win_condition = self._parse_win_condition(rules.win_conditions)

        # Ensure health is set for elimination games
        if win_condition.type == "elimination" and "health" not in rule_set.initial_properties:
            # Default health: try to extract from rules, else use 20
            health = self._extract_health_from_rules(rules)
            rule_set.initial_properties["health"] = health

        return GameDefinition(
            name=game_name,
            description=f"Compiled from: {rules.deck_preparation[:100]}...",
            cards=card_defs,
            rules=rule_set,
            win_condition=win_condition,
            num_players=num_players,
        )

    def _parse_rules(self, rules: Rules) -> RuleSet:
        """Parse Rules model into RuleSet."""
        # Extract initial hand size
        initial_hand_size = self._extract_hand_size(rules.initial_hands)

        # Extract draw per turn from turn structure
        draw_per_turn = self._extract_draw_per_turn(rules.turn_structure)

        # Extract max cards per turn
        max_cards_per_turn = self._extract_max_cards_per_turn(rules.turn_structure)

        # Extract resources
        initial_resources, resource_per_turn = self._extract_resources(rules.resource_mechanics)

        # Extract initial properties (health, etc.)
        initial_properties = self._extract_initial_properties(rules)

        return RuleSet(
            initial_hand_size=initial_hand_size,
            draw_per_turn=draw_per_turn,
            max_cards_per_turn=max_cards_per_turn,
            initial_resources=initial_resources,
            resource_per_turn=resource_per_turn,
            initial_properties=initial_properties,
        )

    def _extract_number(self, text: str) -> int | None:
        """Extract a number from text (digit or word)."""
        text = text.lower()

        # Try digit first
        match = re.search(r"\b(\d+)\b", text)
        if match:
            return int(match.group(1))

        # Try number words
        for word, num in self.NUMBER_WORDS.items():
            if re.search(rf"\b{word}\b", text):
                return num

        return None

    def _extract_hand_size(self, initial_hands: str) -> int:
        """Extract initial hand size from initial_hands text."""
        text = initial_hands.lower()

        # Common patterns (English + Spanish):
        # EN: "Each player draws 5 cards", "Deal 7 cards to each player"
        # ES: "Cada jugador recibe 5 cartas", "Reparte 5 cartas"
        patterns = [
            # English
            r"draws?\s+(\d+)\s+cards?",
            r"deal\s+(\d+)\s+cards?",
            r"hand\s+of\s+(\d+)",
            r"(\d+)\s+cards?\s+(each|to|per)",
            r"start\s+with\s+(\d+)",
            # Spanish
            r"recibe\s+(\d+)\s+cartas?",
            r"reparte?\s+(\d+)\s+cartas?",
            r"(\d+)\s+cartas?\s+(cada|a|por)",
            r"mano\s+de\s+(\d+)",
            r"comienza\s+con\s+(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))

        # Try generic number extraction
        num = self._extract_number(text)
        if num and 1 <= num <= 20:  # Reasonable hand size
            return num

        self.warnings.append(f"Could not parse hand size from: '{initial_hands}'. Defaulting to 5.")
        return 5

    def _extract_draw_per_turn(self, turn_structure: list) -> int:
        """Extract cards drawn per turn from turn structure."""
        for phase in turn_structure:
            text = phase.phase_description.lower()

            # Look for draw phase
            if "draw" in phase.phase_name.lower() or "draw" in text:
                patterns = [
                    r"draws?\s+(\d+)\s+cards?",
                    r"draw\s+a\s+card",  # "draw a card" = 1
                ]

                for pattern in patterns:
                    match = re.search(pattern, text)
                    if match:
                        if "a card" in text:
                            return 1
                        return int(match.group(1))

        # Default: 1 card per turn
        return 1

    def _extract_max_cards_per_turn(self, turn_structure: list) -> int:
        """Extract max cards playable per turn."""
        for phase in turn_structure:
            text = phase.phase_description.lower()

            # Look for play limits
            patterns = [
                r"play\s+up\s+to\s+(\d+)",
                r"play\s+(\d+)\s+cards?",
                r"maximum\s+of\s+(\d+)",
                r"(\d+)\s+cards?\s+per\s+turn",
            ]

            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return int(match.group(1))

            # "play one card" or "play a card"
            if re.search(r"play\s+(one|a|1)\s+card", text):
                return 1

            # "play any number" or "play as many as"
            if re.search(r"(any\s+number|as\s+many|unlimited)", text):
                return 99

        # Default: 1 card per turn
        return 1

    def _extract_resources(self, resource_mechanics: str | None) -> tuple[dict, dict]:
        """Extract initial resources and per-turn gain."""
        if not resource_mechanics:
            return {}, {}

        # Prefer LLM for understanding natural language descriptions
        if self.llm_service:
            return self._parse_resources_llm(resource_mechanics)

        # Fallback to regex for common patterns (when no LLM)
        text = resource_mechanics.lower()
        initial = {}
        per_turn = {}

        resource_names = ["mana", "energy", "gold", "action", "power"]
        for resource in resource_names:
            if resource in text:
                start_match = re.search(rf"start\s+with\s+(\d+)\s+{resource}", text)
                if start_match:
                    initial[resource] = int(start_match.group(1))

                gain_match = re.search(rf"gain\s+(\d+)\s+{resource}", text)
                if gain_match:
                    per_turn[resource] = int(gain_match.group(1))

        return initial, per_turn

    def _extract_initial_properties(self, rules: Rules) -> dict:
        """Extract initial player properties like health."""
        properties = {}

        # Check all rule text for health/life mentions
        all_text = " ".join([
            rules.deck_preparation,
            rules.initial_hands,
            rules.win_conditions,
            rules.resource_mechanics or "",
        ]).lower()

        # Health/Life
        health_patterns = [
            r"(\d+)\s+(health|life|hp|hit\s+points)",
            r"(health|life)\s+of\s+(\d+)",
            r"start\s+with\s+(\d+)\s+(health|life)",
        ]

        for pattern in health_patterns:
            match = re.search(pattern, all_text)
            if match:
                # Extract the number (could be group 1 or 2 depending on pattern)
                for group in match.groups():
                    if group and group.isdigit():
                        properties["health"] = int(group)
                        break
                break

        # Points (starting points)
        if "point" in all_text and "start" in all_text:
            points_match = re.search(r"start\s+with\s+(\d+)\s+points?", all_text)
            if points_match:
                properties["points"] = int(points_match.group(1))
            else:
                properties["points"] = 0
        else:
            properties["points"] = 0  # Default for points-based games

        return properties

    def _extract_health_from_rules(self, rules: Rules) -> int:
        """Extract starting health from rules, or return default."""
        # Combine all rule text
        all_text = " ".join([
            rules.deck_preparation,
            rules.initial_hands,
            rules.win_conditions,
            rules.resource_mechanics or "",
        ]).lower()

        # Look for health/life values (EN + ES)
        health_patterns = [
            r"(\d+)\s+(health|life|hp|hit\s+points)",
            r"(health|life)\s+(?:of|:)?\s*(\d+)",
            r"start\s+with\s+(\d+)\s+(health|life)",
            # Spanish
            r"(\d+)\s+(?:puntos?\s+de\s+)?(vida|salud)",
            r"(vida|salud)\s+(?:de|:)?\s*(\d+)",
            r"comienza\s+con\s+(\d+)\s+(?:de\s+)?(vida|salud)",
        ]

        for pattern in health_patterns:
            match = re.search(pattern, all_text)
            if match:
                for group in match.groups():
                    if group and group.isdigit():
                        return int(group)

        # Default health for elimination games
        return 20

    def _parse_win_condition(self, win_conditions: str) -> WinCondition:
        """Parse win condition text into WinCondition."""
        text = win_conditions.lower()

        # Points-based win (English + Spanish)
        points_patterns = [
            # English
            r"(\d+)\s+points?\s+(to\s+win|wins?|first)",
            r"first\s+(to|with)\s+(\d+)\s+points?",
            r"reach\s+(\d+)\s+points?",
            r"score\s+(\d+)",
            # Spanish - "10 puntos", "llegar a 10 puntos", "acumular 10 puntos"
            r"(\d+)\s+puntos?\s+(de\s+prestigio|para\s+ganar|gana)",
            r"(llegar|alcanzar|acumular)\s+(\d+)\s+puntos?",
            r"primero\s+(en|con)\s+(\d+)\s+puntos?",
        ]

        for pattern in points_patterns:
            match = re.search(pattern, text)
            if match:
                # Find the number in the match
                for group in match.groups():
                    if group and group.isdigit():
                        return WinCondition(type="points", target_value=int(group))

        # Elimination win (last standing) - EN + ES
        if any(word in text for word in [
            "eliminate", "last", "standing", "defeat all", "only player",
            "eliminar", "último", "en pie", "derrotar a todos", "único jugador"
        ]):
            return WinCondition(type="last_standing")

        # Health-based elimination - EN + ES
        health_words = [
            "health", "life", "hp", "salud", "vida", "puntos de vida",
            "hull integrity", "hull", "shields", "armor",  # Sci-fi variants
        ]
        zero_words = ["0", "zero", "deplete", "cero", "agotar", "reduce"]
        if any(word in text for word in health_words) and any(word in text for word in zero_words):
            return WinCondition(type="elimination")

        # Empty deck - EN + ES
        if ("empty" in text and "deck" in text) or ("vacío" in text and "mazo" in text):
            return WinCondition(type="empty_deck")

        # Try LLM fallback if available
        if self.llm_service:
            return self._parse_win_condition_llm(win_conditions)

        # Default: points to 10
        self.warnings.append(f"Could not parse win condition: '{win_conditions}'. Defaulting to 10 points.")
        return WinCondition(type="points", target_value=10)

    def _parse_card(self, card: Card) -> CardDefinition:
        """Parse a Card into CardDefinition.

        Uses structured effect fields if present, otherwise falls back to
        parsing the description text.
        """
        # Check if card has structured effect fields
        if card.effect_type is not None:
            # Use structured fields directly
            effect_map = {
                "none": CardEffect.NONE,
                "draw": CardEffect.DRAW,
                "damage": CardEffect.DAMAGE,
                "heal": CardEffect.HEAL,
                "gain_points": CardEffect.GAIN_POINTS,
                "gain_resource": CardEffect.GAIN_RESOURCE,
                "win_game": CardEffect.WIN_GAME,
            }
            effect = effect_map.get(card.effect_type, CardEffect.NONE)
            effect_value = card.effect_value or 0
            effect_target = card.effect_target or "self"
        else:
            # Fall back to parsing description
            effect, effect_value, effect_target = self._parse_card_effect(card.description)

        cost = self._parse_card_cost(card.cost)

        return CardDefinition(
            name=card.name,
            quantity=card.quantity,
            cost=cost,
            effect=effect,
            effect_value=effect_value,
            effect_target=effect_target,
            properties={
                "type": card.type,
                "rarity": card.rarity,
                "description": card.description,
            },
        )

    def _parse_card_effect(self, description: str) -> tuple[CardEffect, int, str]:
        """Parse card description into effect type, value, and target."""
        text = description.lower()

        # Win game instantly (EN + ES)
        if any(phrase in text for phrase in [
            "win the game", "wins the game", "you win", "instant win",
            "gana la partida", "ganar el juego", "victoria instantánea"
        ]):
            return CardEffect.WIN_GAME, 0, "self"

        # Draw cards (EN + ES)
        draw_patterns = [
            # English
            r"draw\s+(\d+)\s+cards?",
            r"draw\s+a\s+card",
            r"draw\s+an\s+extra\s+card",
            # Spanish - "roba 2 cartas", "robar una carta", "permite robar"
            r"rob(?:a|ar|e)\s+(\d+)\s+cartas?",
            r"rob(?:a|ar|e)\s+una?\s+carta",
            r"permite\s+robar\s+(\d+)?\s*cartas?",
        ]
        for pattern in draw_patterns:
            match = re.search(pattern, text)
            if match:
                if match.lastindex and match.group(1):
                    value = int(match.group(1))
                else:
                    value = 1
                return CardEffect.DRAW, value, "self"

        # Damage (EN + ES)
        damage_patterns = [
            # English
            r"deal(?:s)?\s+(\d+)\s+damage",
            r"(\d+)\s+damage\s+to",
            r"deals?\s+(\d+)",
            r"attack\s+for\s+(\d+)",
            # Spanish - "inflige 3 daño", "causa 2 de daño"
            r"inflige\s+(\d+)\s+(?:de\s+)?daño",
            r"causa\s+(\d+)\s+(?:de\s+)?daño",
            r"(\d+)\s+(?:de\s+)?daño\s+(?:a|al)",
        ]
        for pattern in damage_patterns:
            match = re.search(pattern, text)
            if match:
                value = int(match.group(1))
                target = "opponent" if any(w in text for w in ["opponent", "oponente", "rival", "enemigo"]) else "any"
                return CardEffect.DAMAGE, value, target

        # Heal (EN + ES)
        heal_patterns = [
            # English
            r"heal\s+(\d+)",
            r"restore\s+(\d+)\s+(health|life|hp)",
            r"gain\s+(\d+)\s+(health|life|hp)",
            r"\+(\d+)\s+(health|life)",
            # Spanish - "cura 3", "restaura 2 de vida"
            r"cura\s+(\d+)",
            r"restaura\s+(\d+)\s+(?:de\s+)?(vida|salud)",
            r"recupera\s+(\d+)\s+(?:de\s+)?(vida|salud)",
        ]
        for pattern in heal_patterns:
            match = re.search(pattern, text)
            if match:
                value = int(match.group(1))
                return CardEffect.HEAL, value, "self"

        # Gain points (EN + ES)
        points_patterns = [
            # English - includes "Provides X base point(s)" format
            r"provides?\s+(\d+)\s+(?:base\s+)?points?",
            r"gain\s+(\d+)\s+points?",
            r"score\s+(\d+)\s+points?",
            r"\+(\d+)\s+points?",
            r"worth\s+(\d+)\s+points?",
            r"(\d+)\s+points?(?:\s+to|\s+when|$)",  # "2 points" at end
            # Spanish - "otorga 3 puntos", "gana 2 puntos", "vale 5 puntos"
            r"otorga\s+(\d+)\s+puntos?",
            r"gana\s+(\d+)\s+puntos?",
            r"proporciona\s+(\d+)\s+puntos?",
            r"vale\s+(\d+)\s+puntos?",
            r"suma\s+(\d+)\s+puntos?",
            r"(\d+)\s+puntos?\s+(?:de\s+prestigio|de\s+victoria)?",  # "3 puntos de prestigio"
        ]
        for pattern in points_patterns:
            match = re.search(pattern, text)
            if match:
                value = int(match.group(1))
                return CardEffect.GAIN_POINTS, value, "self"

        # Gain resource (EN + ES)
        resource_patterns = [
            # English
            r"gain\s+(\d+)\s+(mana|energy|gold|power)",
            r"\+(\d+)\s+(mana|energy|gold|power)",
            # Spanish
            r"gana\s+(\d+)\s+(maná|energía|oro|poder)",
            r"obtiene\s+(\d+)\s+(maná|energía|oro|poder)",
        ]
        for pattern in resource_patterns:
            match = re.search(pattern, text)
            if match:
                value = int(match.group(1))
                return CardEffect.GAIN_RESOURCE, value, "self"

        # Default: try LLM fallback if available
        if self.llm_service:
            return self._parse_card_effect_llm(description)

        return CardEffect.NONE, 0, "self"

    def _parse_card_effect_llm(self, description: str) -> tuple[CardEffect, int, str]:
        """Use LLM to parse card effect when pattern matching fails."""
        prompt = """
        ### TASK ###
        Parse this card game card description and extract its mechanical effect.

        ### CARD DESCRIPTION ###
        {description}

        ### INSTRUCTIONS ###
        Identify the PRIMARY mechanical effect:
        - "draw": Player draws cards
        - "damage": Deals damage to a target
        - "heal": Restores health/life
        - "gain_points": Gains victory points
        - "gain_resource": Gains resources (mana, energy, etc.)
        - "win_game": Instant win condition
        - "none": No clear mechanical effect (flavor only, or too complex)

        Extract the numeric value if present (e.g., "draw 3 cards" = 3).
        Identify the target: self, opponent, or any.
        """

        try:
            result = self.llm_service.generate(
                output_model=ParsedCardEffect,
                prompt=prompt,
                description=description,
            )

            effect_map = {
                "none": CardEffect.NONE,
                "draw": CardEffect.DRAW,
                "damage": CardEffect.DAMAGE,
                "heal": CardEffect.HEAL,
                "gain_points": CardEffect.GAIN_POINTS,
                "gain_resource": CardEffect.GAIN_RESOURCE,
                "win_game": CardEffect.WIN_GAME,
            }

            effect = effect_map.get(result.effect_type, CardEffect.NONE)
            return effect, result.effect_value, result.effect_target

        except Exception as e:
            logger.warning(f"LLM parsing failed for card: {description[:50]}... Error: {e}")
            return CardEffect.NONE, 0, "self"

    def _parse_win_condition_llm(self, win_conditions: str) -> WinCondition:
        """Use LLM to parse win condition when pattern matching fails."""
        prompt = """
        ### TASK ###
        Parse this card game's win conditions.

        ### WIN CONDITIONS TEXT ###
        {win_conditions}

        ### INSTRUCTIONS ###
        Identify the PRIMARY win condition type:
        - "points": First to reach N points wins
        - "elimination": Reduce opponent's health/life to 0
        - "last_standing": Last player remaining wins (multiplayer)
        - "empty_deck": Win when deck is empty
        - "property_threshold": Win when a specific property reaches a value

        Extract the target value if it's a numeric threshold.
        """

        try:
            result = self.llm_service.generate(
                output_model=ParsedWinCondition,
                prompt=prompt,
                win_conditions=win_conditions,
            )

            return WinCondition(
                type=result.win_type,
                target_value=result.target_value,
                property_name=result.property_name,
            )

        except Exception as e:
            logger.warning(f"LLM parsing failed for win condition: {win_conditions[:50]}... Error: {e}")
            return WinCondition(type="points", target_value=10)

    def _parse_resources_llm(self, resource_mechanics: str) -> tuple[dict, dict]:
        """Use LLM to parse resource mechanics when pattern matching fails."""
        prompt = """
        ### TASK ###
        Parse this card game's resource system.

        ### RESOURCE MECHANICS TEXT ###
        {resource_mechanics}

        ### INSTRUCTIONS ###
        1. Identify the primary resource used to play cards
        2. Normalize the name to one of: 'mana', 'energy', 'gold', 'action'
           - Fantasy names like 'Aether', 'Essence', 'Arcane Power' -> 'mana'
           - Physical names like 'Stamina', 'Spirit' -> 'energy'
        3. Extract how much players START with (0 if not mentioned)
        4. Extract how much players GAIN per turn (1 if "generates" but amount unclear)
        """

        try:
            result = self.llm_service.generate(
                output_model=ParsedResourceSystem,
                prompt=prompt,
                resource_mechanics=resource_mechanics,
            )

            per_turn = {result.resource_name: result.per_turn_gain} if result.per_turn_gain > 0 else {}

            # If initial is 0 but per_turn exists, default to per_turn * 3
            # This ensures players can afford cards on turn 1
            if result.initial_amount > 0:
                initial = {result.resource_name: result.initial_amount}
            elif per_turn:
                default_initial = result.per_turn_gain * 3
                initial = {result.resource_name: default_initial}
                logger.info(f"Defaulting initial {result.resource_name} to {default_initial} (per_turn * 3)")
            else:
                initial = {}

            return initial, per_turn

        except Exception as e:
            logger.warning(f"LLM parsing failed for resources: {resource_mechanics[:50]}... Error: {e}")
            return {}, {}

    def _parse_card_cost(self, cost: str | None) -> dict[str, int]:
        """Parse card cost into resource requirements."""
        if not cost:
            return {}

        # If the game has no resource system, ignore card costs
        if not self.has_resources:
            return {}

        text = cost.lower()

        # "No cost" or "Free"
        if any(word in text for word in ["no cost", "free", "none", "sin coste", "gratis"]):
            return {}

        result = {}

        # Resource costs: "2 Mana", "3 Energy", "4 Aether Points", etc.
        resource_pattern = r"(\d+)\s*(mana\s*points?|aether\s*points?|energy\s*points?|action\s*points?|mana|energy|gold|power|action|crystal|aether|essence|spirit|arcana|flux|charge|maná|energía|oro|éter)"
        for match in re.finditer(resource_pattern, text):
            amount = int(match.group(1))
            resource = match.group(2).strip()
            result[resource] = amount

        # Discard cost: "Discard a card" or "Discard 2 cards"
        discard_match = re.search(r"discard\s+(\d+|a|an)\s+cards?", text)
        if discard_match:
            value = discard_match.group(1)
            result["discard"] = 1 if value in ["a", "an"] else int(value)

        # Normalize resource names (same as _extract_resources)
        resource_aliases = {
            "aether": "mana", "aether point": "mana", "aether points": "mana",
            "mana point": "mana", "mana points": "mana",
            "energy point": "energy", "energy points": "energy",
            "action point": "action", "action points": "action",
            "essence": "mana", "spirit": "energy", "arcana": "mana",
        }
        normalized = {}
        for res, val in result.items():
            normalized[resource_aliases.get(res, res)] = val

        # LLM fallback: if regex didn't find anything, use LLM
        if not normalized and self.llm_service:
            normalized = self._parse_card_cost_llm(cost)

        # Last resort fallback: if nothing found but there's a number, assume "mana"
        if not normalized:
            num_match = re.search(r"(\d+)", text)
            if num_match:
                normalized["mana"] = int(num_match.group(1))

        return normalized

    def _parse_card_cost_llm(self, cost: str) -> dict[str, int]:
        """Use LLM to parse card cost when pattern matching fails."""
        prompt = """
        ### TASK ###
        Parse this card's resource cost.

        ### COST TEXT ###
        {cost}

        ### INSTRUCTIONS ###
        1. Extract the numeric cost amount
        2. Normalize the resource name to: 'mana', 'energy', 'gold', or 'action'
           - Fantasy names (Aether, Essence, Arcane, Spirit, etc.) → 'mana'
           - Physical resources (Stamina, Power, etc.) → 'energy'
        """

        try:
            result = self.llm_service.generate(
                output_model=ParsedCardCost,
                prompt=prompt,
                cost=cost,
            )
            return {result.resource_name: result.amount}
        except Exception as e:
            logger.warning(f"LLM parsing failed for card cost: {cost[:30]}... Error: {e}")
            return {}


def compile_game(
    rules: Rules,
    cards: list[Card],
    game_name: str = "Compiled Game",
    num_players: int = 2,
    llm_service=None,
) -> tuple[GameDefinition, list[str]]:
    """
    Convenience function to compile a game.

    Returns:
        Tuple of (GameDefinition, warnings)
    """
    compiler = RuleCompiler(llm_service=llm_service)
    game_def = compiler.compile(rules, cards, game_name, num_players)
    return game_def, compiler.warnings
