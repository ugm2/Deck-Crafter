from pydantic import BaseModel, Field
from typing import List, Optional, TYPE_CHECKING
from deck_crafter.services.llm_service import LLMService
from deck_crafter.models.evaluation import GameEvaluation
from deck_crafter.models.card import Card

if TYPE_CHECKING:
    from deck_crafter.agents.director_agent import RefinementStrategy


class RefinementFeedback(BaseModel):
    """Structured feedback for regeneration agents."""
    rules_critique: str = Field(
        ...,
        description="MANDATORY critique for rules. Must include: (1) What specific rules are broken/missing, "
                    "(2) Exact numbers/values that need changing, (3) New rules to add verbatim. "
                    "ONLY write 'No changes needed' if rules score >= 7."
    )
    cards_critique: str = Field(
        ...,
        description="MANDATORY critique for cards. Must include: (1) Which specific cards have problems, "
                    "(2) What stats/effects need changing with exact values, (3) New card concepts to create. "
                    "ONLY write 'No changes needed' if cards score >= 7."
    )
    cards_to_regenerate: List[str] = Field(
        default_factory=list,
        description="List of SPECIFIC card names that need regeneration. "
                    "Cards NOT in this list will be PRESERVED. "
                    "Use EXACT card names from the provided list. "
                    "Follow the Director's card_focus list. Leave empty if no cards need regeneration."
    )
    priority_issues: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Top 3-5 critical issues. Each must be a SPECIFIC action, not vague advice. "
                    "Example: 'Change Dragon card attack from 10 to 6' NOT 'Balance the cards better'."
    )


class FeedbackSynthesizerAgent:
    """Converts evaluation results into actionable critique for regeneration."""

    PROMPT_TEMPLATE = """
    ### ROLE ###
    You are an AGGRESSIVE Game Design Critic executing the Director's strategy.
    Your job is to translate strategic decisions into SPECIFIC, actionable fixes.
    Vague advice is USELESS. You must give concrete, implementable instructions.

    ### DIRECTOR'S STRATEGY ###
    {strategy_section}

    ### CURRENT CARDS IN THE GAME ###
    These are the actual cards - use their REAL NAMES in your critique:
    {cards_summary}

    ### CRITICAL SCORING THRESHOLDS ###
    - Score < 3: CATASTROPHIC FAILURE - requires complete overhaul
    - Score 3-5: MAJOR PROBLEMS - multiple specific fixes needed
    - Score 5-7: MODERATE ISSUES - targeted improvements required
    - Score >= 7: ACCEPTABLE - only minor tweaks if any

    ### EVALUATION RESULTS (5 metrics, weighted) ###
    Overall Score: {overall_score}/10 (weighted average)

    **PLAYABILITY** (Score: {playability_score}/10, weight 2.0) - MOST IMPORTANT:
    Analysis: {playability_analysis}
    Suggestions: {playability_suggestions}

    **BALANCE** (Score: {balance_score}/10, weight 1.5):
    Analysis: {balance_analysis}
    Suggestions: {balance_suggestions}

    **CLARITY** (Score: {clarity_score}/10, weight 1.2):
    Analysis: {clarity_analysis}
    Suggestions: {clarity_suggestions}

    **THEME ALIGNMENT** (Score: {theme_alignment_score}/10, weight 1.0):
    Analysis: {theme_alignment_analysis}
    Suggestions: {theme_alignment_suggestions}

    **INNOVATION** (Score: {innovation_score}/10, weight 0.8):
    Analysis: {innovation_analysis}
    Suggestions: {innovation_suggestions}

    ### PRIORITIZED SUGGESTIONS (synthesized from all metrics) ###
    {synthesized_suggestions}

    ### MANDATORY OUTPUT REQUIREMENTS ###

    **FOR RULES_CRITIQUE:**
    If any rules-related score (Balance, Clarity, Playability) is below 6:
    - List EXACT rules that are missing or broken
    - Provide COMPLETE new rule text to add/replace
    - Specify EXACT numbers (turn limits, resource costs, draw amounts)
    - Example: "ADD RULE: 'Players may only play 2 cards per turn' to prevent runaway advantage"
    - Example: "CHANGE win condition from 'collect 20 points' to 'collect 10 points' to shorten games"

    **FOR CARDS_CRITIQUE:**
    If any card-related score (Balance, Coherence) is below 6:
    - Reference ACTUAL card names from the list above
    - Provide EXACT stat changes with numbers
    - Example: "NERF '{{card_name}}': reduce attack from X to Y, increase cost from A to B"
    - Example: "BUFF '{{card_name}}': increase health from X to Y, add effect 'Draw 1 card'"
    - Suggest NEW card types if variety is lacking

    **FOR CARDS_TO_REGENERATE (CRITICAL):**
    - If the Director provided card_focus, use those EXACT cards in cards_to_regenerate
    - List ALL cards that MUST be completely regenerated
    - Cards NOT in this list will be PRESERVED as-is
    - Use EXACT card names from the list above
    - If only minor stat tweaks are needed, do NOT include the card here
    - Leave empty [] if no cards need complete regeneration

    **FOR PRIORITY_ISSUES:**
    Each issue MUST be a concrete action with specifics:
    ✓ GOOD: "Reduce '{{actual_card_name}}' damage from 10 to 4"
    ✗ BAD: "Balance the spell cards better"
    ✓ GOOD: "Add rule: 'Maximum hand size is 7 cards, discard excess at end of turn'"
    ✗ BAD: "Clarify hand limit rules"

    ### OUTPUT LANGUAGE ###
    Respond in: {language}

    ### FINAL CHECK ###
    Before outputting, verify:
    1. Every card reference uses an ACTUAL card name from the list above
    2. Every critique includes SPECIFIC names, numbers, or exact rule text
    3. No critique uses vague words like "consider", "maybe", "perhaps", "some"
    4. Priority issues are ACTIONS, not observations
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def synthesize(
        self,
        evaluation: GameEvaluation,
        cards: Optional[List[Card]] = None,
        language: str = "English",
        strategy: Optional["RefinementStrategy"] = None
    ) -> RefinementFeedback:
        """Convert evaluation into actionable feedback, guided by Director's strategy."""
        # Format cards summary for the prompt
        if cards:
            cards_summary = "\n".join([
                f"- {c.name} (Type: {c.type}, Qty: {c.quantity}): {c.description[:100]}..."
                if len(c.description) > 100 else f"- {c.name} (Type: {c.type}, Qty: {c.quantity}): {c.description}"
                for c in cards
            ])
        else:
            cards_summary = "No cards available"

        # Format strategy section
        if strategy:
            # Use new granular fields
            rules_action = getattr(strategy, 'rules_action', 'none')
            cards_action = getattr(strategy, 'cards_action', 'none')
            cards_to_modify = getattr(strategy, 'cards_to_modify', [])
            rules_instruction = getattr(strategy, 'rules_instruction', None)
            cards_instruction = getattr(strategy, 'cards_instruction', None)
            hypothesis = getattr(strategy, 'hypothesis', strategy.reasoning)
            target_metric = getattr(strategy, 'target_metric', 'balance')
            intervention_type = getattr(strategy, 'intervention_type', 'moderate')

            strategy_section = f"""
The Director has designed an EXPERIMENT for this iteration:

## HYPOTHESIS
"{hypothesis}"
Target metric: {target_metric}
Intervention type: {intervention_type.upper()}

## RULES ACTION: {rules_action.upper()}
{f"Target: {strategy.rules_target}" if getattr(strategy, 'rules_target', None) else ""}
{f"Instruction: {rules_instruction}" if rules_instruction else ""}

## CARDS ACTION: {cards_action.upper()}
{f"Cards to modify: {', '.join(cards_to_modify)} (USE THESE EXACT NAMES in cards_to_regenerate)" if cards_to_modify else ""}
{f"Instruction: {cards_instruction}" if cards_instruction else ""}

Director's reasoning: {strategy.reasoning}
{f"Why not alternatives: {strategy.why_not_alternatives}" if getattr(strategy, 'why_not_alternatives', None) else ""}

YOU MUST FOLLOW THE DIRECTOR'S STRATEGY:
- If rules_action is "none", write "No changes needed" for rules_critique
- If cards_action is "none", write "No changes needed" for cards_critique
- If specific cards are listed in cards_to_modify, USE THOSE EXACT NAMES in cards_to_regenerate
- Focus your critique on achieving the Director's hypothesis
"""
        else:
            strategy_section = "No strategic direction provided. Analyze all areas independently."

        # Format synthesized suggestions if available
        synthesized_text = "Not available"
        if evaluation.synthesized_suggestions:
            ss = evaluation.synthesized_suggestions
            lines = []
            if ss.high_priority:
                lines.append("HIGH PRIORITY:")
                for s in ss.high_priority[:3]:  # Top 3
                    lines.append(f"  - [{s.target}] {s.suggestion} (impact: +{s.estimated_impact})")
            if ss.medium_priority:
                lines.append("MEDIUM PRIORITY:")
                for s in ss.medium_priority[:2]:  # Top 2
                    lines.append(f"  - [{s.target}] {s.suggestion}")
            if ss.conflicts:
                lines.append(f"CONFLICTS: {len(ss.conflicts)} pairs of contradicting suggestions")
            synthesized_text = "\n".join(lines) if lines else "No synthesized suggestions"

        return self.llm_service.generate(
            output_model=RefinementFeedback,
            prompt=self.PROMPT_TEMPLATE,
            strategy_section=strategy_section,
            cards_summary=cards_summary,
            overall_score=evaluation.overall_score,
            playability_score=evaluation.playability.score,
            playability_analysis=evaluation.playability.analysis,
            playability_suggestions=evaluation.playability.suggestions or [],
            balance_score=evaluation.balance.score,
            balance_analysis=evaluation.balance.analysis,
            balance_suggestions=evaluation.balance.suggestions or [],
            clarity_score=evaluation.clarity.score,
            clarity_analysis=evaluation.clarity.analysis,
            clarity_suggestions=evaluation.clarity.suggestions or [],
            theme_alignment_score=evaluation.theme_alignment.score,
            theme_alignment_analysis=evaluation.theme_alignment.analysis,
            theme_alignment_suggestions=evaluation.theme_alignment.suggestions or [],
            innovation_score=evaluation.innovation.score,
            innovation_analysis=evaluation.innovation.analysis,
            innovation_suggestions=evaluation.innovation.suggestions or [],
            synthesized_suggestions=synthesized_text,
            language=language,
        )
