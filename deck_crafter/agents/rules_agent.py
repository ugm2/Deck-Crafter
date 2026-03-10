from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from deck_crafter.models.game_concept import GameConcept
from deck_crafter.models.rules import Rules
from deck_crafter.models.state import CardGameState
from deck_crafter.services.llm_service import LLMService


from deck_crafter.models.rules import TurnPhase


class SectionRewrite(BaseModel):
    """Rewritten content for a specific rules section."""
    content: str = Field(..., description="The rewritten section content")


class TurnStructureRewrite(BaseModel):
    """Rewritten turn structure with phases."""
    turn_structure: List[TurnPhase] = Field(
        ...,
        description="The complete rewritten turn structure as a list of phases"
    )


class RulesEnhancement(BaseModel):
    """Additive enhancements to existing rules - NEVER removes or rewrites existing content."""

    new_glossary_entries: Dict[str, str] = Field(
        default_factory=dict,
        description="NEW terms to ADD to glossary. Key: term, Value: definition. "
                    "Only include terms not already defined."
    )
    new_examples: List[str] = Field(
        default_factory=list,
        description="NEW examples of play to ADD. Each should clarify a confusing interaction. "
                    "Do not repeat existing examples."
    )
    new_additional_rules: List[str] = Field(
        default_factory=list,
        description="NEW rules to ADD for edge cases, clarifications, or FAQ items. "
                    "These supplement, not replace, existing rules."
    )
    turn_structure_clarifications: Optional[List[str]] = Field(
        None,
        description="Clarifications to APPEND to turn phase descriptions. "
                    "Format: 'Phase Name: additional clarification text'"
    )
    resource_mechanics_addendum: Optional[str] = Field(
        None,
        description="Additional text to APPEND to resource_mechanics. "
                    "Only if needed to clarify resource generation/spending."
    )
    win_conditions_addendum: Optional[str] = Field(
        None,
        description="Additional text to APPEND to win_conditions. "
                    "Only if there's ambiguity about victory."
    )


class RuleGenerationAgent:
    """Generates comprehensive and clear rules based on a game concept."""
    DEFAULT_PROMPT = """
    ### ROLE & PERSONA ###
    Act as a world-class game designer and technical writer creating balanced, clear rulebooks.

    ### ⚠️ MANDATORY CRITIQUE - TOP PRIORITY ⚠️ ###
    READ THIS FIRST. The following critique MUST be addressed. Your output will be REJECTED if these issues remain:

    {critique}

    ---

    If the critique mentions specific rules, numbers, or mechanics, you MUST incorporate them EXACTLY.
    If the critique says "ADD RULE: X", your output MUST contain rule X verbatim.
    If the critique says "CHANGE X to Y", your output MUST reflect that change.

    ### CORE PRINCIPLES ###
    1. **Clear Objectives**: Players must understand win/lose conditions immediately
    2. **Structured Turn Flow**: Break turns into distinct sequential phases
    3. **Explicit Terminology**: Define all game-specific terms in a glossary
    4. **Concrete Examples**: Provide play examples for complex interactions
    5. **Balance Constraints**: Include limits on actions per turn, hand sizes, resource caps

    ### BALANCE REQUIREMENTS ###
    Every rulebook MUST include:
    - Maximum cards playable per turn (default: 2)
    - Maximum hand size (default: 7)
    - Turn time limit or turn counter for game length
    - Clear resource generation/spending limits

    ### TASK ###
    Generate a complete rulebook based on the Game Concept below.
    REMEMBER: The critique above takes ABSOLUTE PRIORITY over generic principles.

    ### INPUT DATA ###
    Game Concept: {game_concept}

    ### OUTPUT LANGUAGE ###
    Write in: '{game_concept[language]}'
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

    def generate_rules(self, state: CardGameState) -> dict:
        """
        Generate comprehensive game rules based on the game concept.

        :param state: The current state of the card game, including the game concept.
        :return: Updated state with the generated rules.
        """
        game_concept: GameConcept = state.concept
        critique = state.critique

        context = {
            "game_concept": game_concept.model_dump(),
            "critique": critique or "This is the first attempt, no critique yet."
        }

        rules = self.llm_service.generate(
            output_model=Rules, prompt=self.base_prompt, **context
        )

        if rules:
            return {"rules": rules}
        return {}

    ENHANCE_PROMPT = """
    ### ROLE ###
    You are a Technical Editor improving an EXISTING rulebook. Your job is to ADD clarifications
    WITHOUT changing the existing content. The game has great thematic coherence - preserve it!

    ### ⚠️ CRITICAL CONSTRAINTS ⚠️
    - You MUST NOT rewrite existing rules
    - You MUST NOT change the game's theme, flavor, or core mechanics
    - You can ONLY ADD: glossary entries, examples, clarifications, FAQ items
    - If something is unclear, ADD a definition or example - don't rewrite the rule

    ### EXISTING RULES (DO NOT MODIFY) ###
    {existing_rules}

    ### ISSUES TO ADDRESS ###
    {critique}

    ### YOUR TASK ###
    Generate ADDITIVE content to address the issues above:

    1. **new_glossary_entries**: Define any unclear terms (combat, resources, phases)
    2. **new_examples**: Add concrete play examples that show how rules interact
    3. **new_additional_rules**: Add FAQ items or edge case clarifications
    4. **turn_structure_clarifications**: Add detail to specific phases if unclear
    5. **resource_mechanics_addendum**: Clarify resource rules if needed
    6. **win_conditions_addendum**: Clarify victory conditions if ambiguous

    REMEMBER: You are ENHANCING, not REWRITING. The existing text stays exactly as-is.

    ### OUTPUT LANGUAGE ###
    Write in: '{language}'
    """

    def enhance_rules(self, state: CardGameState) -> dict:
        """
        Enhance existing rules with additive content (glossary, examples, clarifications).
        Does NOT rewrite or remove existing content - only adds to it.

        Use this for 'tweak' or 'rewrite_section' rules_action.
        Use generate_rules() for 'overhaul' when you need a complete rewrite.
        """
        if not state.rules:
            return self.generate_rules(state)

        existing_rules = state.rules
        critique = state.critique or "Add more clarity and examples."
        language = state.concept.language if state.concept else "English"

        # Format existing rules for the prompt
        rules_text = f"""
Deck Preparation: {existing_rules.deck_preparation}
Initial Hands: {existing_rules.initial_hands}
Turn Structure: {[f"{p.phase_name}: {p.phase_description}" for p in existing_rules.turn_structure]}
Win Conditions: {existing_rules.win_conditions}
Resource Mechanics: {existing_rules.resource_mechanics or "Not defined"}
Reaction Phase: {existing_rules.reaction_phase or "Not defined"}
Glossary: {existing_rules.glossary or {}}
Examples: {existing_rules.examples_of_play or []}
Additional Rules: {existing_rules.additional_rules or []}
"""

        enhancement = self.llm_service.generate(
            output_model=RulesEnhancement,
            prompt=self.ENHANCE_PROMPT,
            existing_rules=rules_text,
            critique=critique,
            language=language
        )

        if not enhancement:
            return {}

        # Merge enhancements into existing rules (additive only)
        updated_rules = existing_rules.model_copy()

        # Merge glossary
        if enhancement.new_glossary_entries:
            current_glossary = updated_rules.glossary or {}
            updated_rules.glossary = {**current_glossary, **enhancement.new_glossary_entries}

        # Merge examples
        if enhancement.new_examples:
            current_examples = updated_rules.examples_of_play or []
            updated_rules.examples_of_play = current_examples + enhancement.new_examples

        # Merge additional rules
        if enhancement.new_additional_rules:
            current_additional = updated_rules.additional_rules or []
            updated_rules.additional_rules = current_additional + enhancement.new_additional_rules

        # Append to resource mechanics if provided
        if enhancement.resource_mechanics_addendum and updated_rules.resource_mechanics:
            updated_rules.resource_mechanics += f"\n\n{enhancement.resource_mechanics_addendum}"
        elif enhancement.resource_mechanics_addendum:
            updated_rules.resource_mechanics = enhancement.resource_mechanics_addendum

        # Append to win conditions if provided
        if enhancement.win_conditions_addendum and updated_rules.win_conditions:
            updated_rules.win_conditions += f"\n\n{enhancement.win_conditions_addendum}"

        # Apply turn structure clarifications
        if enhancement.turn_structure_clarifications:
            phase_clarifications = {}
            for clarification in enhancement.turn_structure_clarifications:
                if ":" in clarification:
                    phase_name, text = clarification.split(":", 1)
                    phase_clarifications[phase_name.strip().lower()] = text.strip()

            for phase in updated_rules.turn_structure:
                phase_key = phase.phase_name.lower()
                if phase_key in phase_clarifications:
                    phase.phase_description += f"\n\n{phase_clarifications[phase_key]}"

        return {"rules": updated_rules}

    SECTION_REWRITE_PROMPT = """
    ### ROLE ###
    You are a Game Design Surgeon. You must REWRITE ONLY ONE SECTION of the rulebook
    while preserving the game's theme and all other sections EXACTLY as they are.

    ### GAME CONTEXT ###
    Theme: {theme}
    Game Name: {game_name}

    ### THE SECTION TO REWRITE: {section_name} ###
    Current content:
    {current_content}

    ### OTHER SECTIONS (DO NOT TOUCH - for context only) ###
    {other_sections}

    ### ISSUES TO FIX ###
    {critique}

    ### YOUR TASK ###
    Rewrite ONLY the {section_name} section to fix the issues above.

    CRITICAL CONSTRAINTS:
    - PRESERVE the game's theme and flavor
    - PRESERVE terminology used in other sections
    - Fix ONLY the specific issues mentioned
    - Make it clear, unambiguous, and playable
    - Keep the same style/tone as the rest of the rulebook

    ### OUTPUT LANGUAGE ###
    Write in: '{language}'
    """

    TURN_STRUCTURE_REWRITE_PROMPT = """
    ### ROLE ###
    You are a Game Design Surgeon. You must REWRITE the turn structure
    while preserving the game's theme and all other rules EXACTLY as they are.

    ### GAME CONTEXT ###
    Theme: {theme}
    Game Name: {game_name}

    ### CURRENT TURN STRUCTURE (TO REWRITE) ###
    {current_content}

    ### OTHER RULES (DO NOT TOUCH - for context only) ###
    Win Conditions: {win_conditions}
    Resource Mechanics: {resource_mechanics}
    Deck Preparation: {deck_preparation}

    ### ISSUES TO FIX ###
    {critique}

    ### YOUR TASK ###
    Rewrite the turn structure to fix the issues above.

    REQUIREMENTS:
    - Each phase must have a clear name and detailed description
    - Specify what actions are allowed/required in each phase
    - Define the order of phases explicitly
    - Clarify any timing or priority rules
    - PRESERVE the game's thematic flavor
    - PRESERVE terminology from other sections

    ### OUTPUT LANGUAGE ###
    Write in: '{language}'
    """

    # Map section names to Rules model fields (with aliases)
    SECTION_FIELDS = {
        "turn_structure": "turn_structure",
        "turns": "turn_structure",
        "phases": "turn_structure",
        "win_conditions": "win_conditions",
        "victory": "win_conditions",
        "winning": "win_conditions",
        "resource_mechanics": "resource_mechanics",
        "resources": "resource_mechanics",
        "mana": "resource_mechanics",
        "energy": "resource_mechanics",
        "reaction_phase": "reaction_phase",
        "reactions": "reaction_phase",
        "counter": "reaction_phase",
        "deck_preparation": "deck_preparation",
        "deck": "deck_preparation",
        "setup": "deck_preparation",  # "setup" maps to deck_preparation
        "initial_hands": "initial_hands",
        "hands": "initial_hands",
        "starting_hand": "initial_hands",
    }

    def rewrite_section(self, state: CardGameState, target_section: str) -> dict:
        """
        Rewrite ONLY a specific section of the rules while preserving everything else.

        Use this for 'rewrite_section' rules_action with a specific rules_target.
        """
        if not state.rules:
            return self.generate_rules(state)

        existing_rules = state.rules
        critique = state.critique or "Improve clarity and remove ambiguity."
        language = state.concept.language if state.concept else "English"
        theme = state.concept.theme if state.concept else "card game"
        game_name = state.concept.title if state.concept else "the game"

        # Normalize section name
        target_section = target_section.lower().replace(" ", "_")

        # Validate section exists
        if target_section not in self.SECTION_FIELDS:
            # Fall back to enhance if section not recognized
            return self.enhance_rules(state)

        field_name = self.SECTION_FIELDS[target_section]

        # Special handling for turn_structure (it's a list of TurnPhase)
        if target_section == "turn_structure":
            return self._rewrite_turn_structure(state, critique, language, theme, game_name)

        # For simple string sections
        current_content = getattr(existing_rules, field_name) or "Not defined"

        # Build context of other sections
        other_sections = []
        for section, field in self.SECTION_FIELDS.items():
            if section != target_section:
                value = getattr(existing_rules, field, None)
                if value:
                    if isinstance(value, list):
                        value = str(value)
                    other_sections.append(f"{section}: {value}")

        rewrite = self.llm_service.generate(
            output_model=SectionRewrite,
            prompt=self.SECTION_REWRITE_PROMPT,
            section_name=target_section,
            current_content=current_content,
            other_sections="\n".join(other_sections),
            critique=critique,
            language=language,
            theme=theme,
            game_name=game_name,
        )

        if not rewrite:
            return {}

        # Copy existing rules and update only the target section
        updated_rules = existing_rules.model_copy(deep=True)
        setattr(updated_rules, field_name, rewrite.content)

        return {"rules": updated_rules}

    def _rewrite_turn_structure(
        self, state: CardGameState, critique: str, language: str, theme: str, game_name: str
    ) -> dict:
        """Rewrite turn structure specifically (requires TurnPhase list)."""
        existing_rules = state.rules

        current_phases = "\n".join([
            f"- {p.phase_name}: {p.phase_description}"
            for p in existing_rules.turn_structure
        ])

        rewrite = self.llm_service.generate(
            output_model=TurnStructureRewrite,
            prompt=self.TURN_STRUCTURE_REWRITE_PROMPT,
            current_content=current_phases,
            win_conditions=existing_rules.win_conditions,
            resource_mechanics=existing_rules.resource_mechanics or "Not defined",
            deck_preparation=existing_rules.deck_preparation,
            critique=critique,
            language=language,
            theme=theme,
            game_name=game_name,
        )

        if not rewrite or not rewrite.turn_structure:
            return {}

        updated_rules = existing_rules.model_copy(deep=True)
        updated_rules.turn_structure = rewrite.turn_structure

        return {"rules": updated_rules}

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
