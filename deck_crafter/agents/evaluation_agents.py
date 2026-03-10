from typing import Type
from deck_crafter.services.llm_service import LLMService
from deck_crafter.models.game_concept import GameConcept
from deck_crafter.models.rules import Rules
from deck_crafter.models.card import Card
from deck_crafter.models.evaluation import (
    BalanceEvaluation,
    ClarityEvaluation,
    PlayabilityEvaluation,
    ThemeAlignmentEvaluation,
    InnovationEvaluation,
    GameEvaluation,
    EvaluationSummary,
    ValidationResult,
    calculate_weighted_score,
)
from deck_crafter.models.user_preferences import UserPreferences
from pydantic import BaseModel

class BalanceAgent:
    """Evaluates exclusively the game's balance."""
    PROMPT_TEMPLATE = """
    ### ROLE & PERSONA ###
    Act as a legendary 'Metagame Breaker' and a quantitative game theorist. Your job is not to enjoy the game, but to mathematically dismantle it. You live for spreadsheets, probability, and finding the single most dominant strategy (the 'meta'). You have zero tolerance for lazy design or untuned numbers. Completely ignore the theme and narrative; you only care about mechanics and numbers.

    ### TASK & PROCESS ###
    Your sole mission is to provide a brutally honest analysis of this card game's balance. Follow these steps rigorously:
    1.  **Cost-Benefit Analysis:** Evaluate each card. Is its cost (in resources, turns, etc.) proportional to its effect? Identify cards that provide value far above or below their cost.
    2.  **Search for Synergies and Broken Combos:** Actively look for combinations of 2 or more cards that could create an infinite loop, a total opponent lockdown, or an overwhelming and insurmountable advantage.
    3.  **Identification of Useless Cards (Trap Picks):** Point out cards that are strictly worse than others, or whose utility is so situational that they would never be included in an optimized deck.
    4.  **Game Economy Evaluation:** Analyze the rules regarding resource acquisition and spending. Does it favor one playstyle (aggro, control) over others? Is it possible to be easily resource-starved or, conversely, to hoard resources without limit?
    5.  **Score Assignment:** Based on your analysis and the scoring rubric below, assign a numerical score. Be extremely strict.
    6.  **Analysis Write-up:** Write a detailed analysis justifying your score, citing specific examples of cards and rules.
    7.  **Adjustment Suggestions (Nerfs & Buffs):** Propose concrete, numerical changes to improve balance. For example: "Change the cost of 'Spell X' from 3 to 5" or "Reduce the damage of 'Attack Y' from 10 to 7".

    ### SCORING RUBRIC (BALANCE) ###
    Use this 1-10 scale with granular sub-levels for refinement tracking:
    - **10 (Flawless Balance):** The holy grail. Multiple top-tier strategies are viable. Rich, diverse metagame.
    - **9 (Exceptional):** Near-perfect. One strategy marginally superior but requires immense skill to exploit.
    - **8-8.5 (Very Good):** Well-tuned. Handful of competitive archetypes with clear counter-strategies.
    - **7-7.9 (Good):** Solid and functional. Some "auto-include" cards but overall fair.
    - **6.5-6.9 (Near-Good):** Close to solid. Most issues identified, 1-2 problem cards remain.
    - **6-6.4 (Acceptable):** Playable but narrow metagame. Obvious "correct" and "trap" choices.
    - **5-5.9 (Mediocre):** Significant card pool unviable. 1-2 superior strategies dominate.
    - **4 (Poor):** Heavily skewed. One strategy provides massive, unfair advantage.
    - **3 (Severely Unbalanced):** Single strategy or small card set dominates completely.
    - **1-2 (Broken):** Near-infinite combos, "I win" buttons, or fundamentally broken math.

    ### INPUT DATA ###
    Game Concept: {concept}
    Game Rules: {rules}
    All Cards: {cards}

    ### OUTPUT LANGUAGE ###
    Language for response: {language}
    """
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def evaluate(self, concept: GameConcept, rules: Rules, cards: list[Card], language: str) -> BalanceEvaluation:
        return self.llm_service.generate(
            output_model=BalanceEvaluation,
            prompt=self.PROMPT_TEMPLATE,
            concept=concept.model_dump_json(indent=2),
            rules=rules.model_dump_json(indent=2),
            cards=[card.model_dump() for card in cards],
            language=language,
        )

class ThemeAlignmentAgent:
    """
    Evaluates how well the game fulfills its intended vision.
    Merges two perspectives:
    - Coherence: Do mechanics match the theme?
    - Fidelity: Does the game match the user's original request?
    """
    PROMPT_TEMPLATE = """
    ### ROLE & PERSONA ###
    You are a demanding Creative Director and QA Lead combined. You care about TWO things equally:
    1. **Internal Coherence**: Does every element reinforce the theme? Mechanics should feel like natural extensions of the world, not generic systems with a "skin."
    2. **External Fidelity**: Does this game deliver what the user asked for? Every specification in the original request is a contract.

    ### TASK & PROCESS ###
    Evaluate this game's Theme Alignment by analyzing BOTH coherence AND fidelity:

    **COHERENCE ANALYSIS:**
    1. Compare the `GameConcept` with the `Rules` and `Cards`. Do mechanics reinforce the fantasy?
    2. Review cards: Do names, art descriptions, and effects make sense in the game's universe?
    3. Does the game create a believable, immersive atmosphere?

    **FIDELITY ANALYSIS:**
    4. Compare each field in `User Preferences` with the final output. Are all specifications met?
    5. Does the game capture the essence of the user's `game_description`?
    6. Identify any deviations from the original request.

    **SYNTHESIS:**
    7. Assign a SINGLE score that balances both aspects. A thematically coherent game that ignores the user's request fails. A faithful but incoherent game also fails.
    8. Write an analysis covering both coherence and fidelity.
    9. Suggest improvements for BOTH aspects.

    ### SCORING RUBRIC (THEME ALIGNMENT) ###
    Use this 1-10 scale with granular sub-levels:
    - **10 (Perfect Vision):** Flawless coherence AND perfect adherence to user request. A rare masterpiece.
    - **9 (Exceptional):** Near-perfect on both fronts. Minor, negligible deviations.
    - **8-8.5 (Very Good):** Strong coherence and fidelity. World feels alive, request is satisfied.
    - **7-7.9 (Good):** Well-integrated theme, most user requirements met. 1-2 minor issues.
    - **6.5-6.9 (Near-Good):** Solid foundation, some gaps in either coherence or fidelity.
    - **6-6.4 (Acceptable):** Theme feels like a "skin" OR some user specs are missed. Needs work.
    - **5-5.9 (Mediocre):** Multiple coherence breaks OR significant fidelity issues.
    - **4 (Poor):** Frequent theme-mechanic clashes AND major deviations from request.
    - **3 (Severely Misaligned):** Little thematic sense AND bears little resemblance to request.
    - **1-2 (Failure):** Total incoherence AND complete misunderstanding of user's vision.

    ### INPUT DATA ###
    User Request: {game_description}
    User Preferences: {preferences}
    Game Concept: {concept}
    Game Rules: {rules}
    All Cards: {cards}

    ### OUTPUT LANGUAGE ###
    Language for response: {language}
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def evaluate(
        self,
        preferences: UserPreferences,
        concept: GameConcept,
        rules: Rules,
        cards: list[Card],
        language: str
    ) -> ThemeAlignmentEvaluation:
        return self.llm_service.generate(
            output_model=ThemeAlignmentEvaluation,
            prompt=self.PROMPT_TEMPLATE,
            game_description=preferences.game_description,
            preferences=preferences.model_dump_json(indent=2),
            concept=concept.model_dump_json(indent=2),
            rules=rules.model_dump_json(indent=2),
            cards=[card.model_dump() for card in cards],
            language=language,
        )

class ClarityAgent:
    """Evaluates exclusively the clarity of the rules and card text."""
    PROMPT_TEMPLATE = """
    ### ROLE & PERSONA ###
    Act as a world-champion "Rules Lawyer." You are known for your ability to read a rulebook so literally and adversarially that you can find any logical loophole, ambiguity, or undefined term to exploit for victory. You do not make assumptions or infer designer intent; you only follow the text *exactly* as written. If a term is not defined, it is meaningless. If an interaction is not explicit, it is not allowed. Your goal is to break the game through poor wording.

    ### TASK & PROCESS ###
    Your sole mission is to provide a ruthless critique of the clarity of the game's rules and card text. Follow these steps:
    1.  **Literal Interpretation:** Read every rule with extreme literalism. If a rule says "you may," check if it's clear when. If it says "target an opponent," check if it's clear which one when multiple are available.
    2.  **Ambiguity Hunt:** Actively search for vague, subjective, or undefined terms (e.g., "nearby," "soon," "strongest," "most damage").
    3.  **Edge Case Exploration:** Imagine complex or rare game states. Do the rules explicitly state how to resolve them? (e.g., What if two "at the start of the turn" effects trigger simultaneously? What if a card is discarded and played at the same time?).
    4.  **Consistency Check:** Is a keyword (e.g., "Discard," "Destroy," "Exile") used consistently with one single meaning across all rules and cards?
    5.  **Score Assignment:** Based on your findings and the rubric below, assign a numerical score. Be unforgiving.
    6.  **Analysis Write-up:** Justify your score with *specific examples* of ambiguous phrases, contradictory rules, or missing definitions you found.
    7.  **Improvement Suggestions:** Propose concrete rewrites for the problematic text. Suggest adding a Glossary for keywords or an FAQ for complex interactions.

    ### SCORING RUBRIC (CLARITY) ###
    Use this 1-10 scale with granular sub-levels for refinement tracking:
    - **10 (Crystal Clear):** Masterpiece of technical writing. Impossible to misinterpret. All edge cases anticipated.
    - **9 (Exceptional):** Gold-standard rulebook. Perfectly clear with helpful examples.
    - **8-8.5 (Very Good):** Clear and concise. Most player questions anticipated and answered.
    - **7-7.9 (Good):** Well-written and easy to follow. Only minor clarifications needed for rare situations.
    - **6.5-6.9 (Near-Good):** Almost there. Most rules clear, 1-2 interactions need FAQ.
    - **6-6.4 (Acceptable):** Mostly clear, but some key interactions require second read or consensus.
    - **5-5.9 (Mediocre):** Core rules understandable but many edge cases uncovered. Frequent rule debates.
    - **4 (Poor):** Learnable but players argue about fundamental interpretations constantly.
    - **3 (Severely Unclear):** Key rules missing or unusable. Requires "house rules."
    - **1-2 (Contradictory/Unintelligible):** Rules contradict or impossible to understand.

    ### INPUT DATA ###
    Game Concept: {concept}
    Game Rules: {rules}
    Card Examples: {cards}

    ### OUTPUT LANGUAGE ###
    Language for response: {language}
    """
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def evaluate(self, concept: GameConcept, rules: Rules, cards: list[Card], language: str) -> ClarityEvaluation:
        return self.llm_service.generate(
            output_model=ClarityEvaluation,
            prompt=self.PROMPT_TEMPLATE,
            concept=concept.model_dump_json(indent=2),
            rules=rules.model_dump_json(indent=2),
            cards=[card.model_dump() for card in cards],
            language=language,
        )

class InnovationAgent:
    """Evaluates the game's originality and innovation."""
    PROMPT_TEMPLATE = """
    ### ROLE & PERSONA ###
    Act as a judge for the prestigious "Golden Meeple Award for Innovation in Gaming." You have reviewed thousands of games and have become incredibly jaded and difficult to impress. You can instantly spot a re-skinned mechanic or a tired theme from a mile away. Your sole purpose is to find and reward *true* novelty, not just the competent execution of old ideas. You are looking for a spark of genius in a sea of mediocrity.

    ### TASK & PROCESS ###
    Your sole mission is to deliver a ruthless critique of this game's originality. You must deconstruct its theme and mechanics and compare them against established genres. Follow these steps:
    1.  **Thematic Originality Analysis:** Analyze the `GameConcept`. Is the theme itself a fresh idea, or is it a standard trope (e.g., generic medieval fantasy, standard sci-fi space opera)? Does it combine themes in a novel way?
    2.  **Mechanical Originality Analysis:** This is the most critical step. Analyze the `Rules` and `All Cards`. Are the core game loops, resource systems, and win conditions novel? Or are they standard, well-known mechanics (e.g., deck-building, trick-taking, set collection, worker placement) just with different names? Scrutinize the card effects for unique interactions.
    3.  **Synthesis of Novelty:** How do the theme and mechanics combine? Does a common theme get a unique mechanical twist that makes it feel new? Or does a potentially unique theme get weighed down by generic, uninspired mechanics?
    4.  **Score Assignment:** Based on your findings and the rubric below, assign a numerical score. Be extremely critical. A "good" game is not necessarily an "original" game.
    5.  **Analysis Write-up:** Justify your score with specific comparisons. If the game is derivative, name the existing games or mechanics it borrows from. If it is original, explain exactly which idea or interaction is new.
    6.  **Improvement Suggestions:** Propose concrete, creative ideas to increase novelty. Suggest adding a unique resource, a surprising win condition, a new form of player interaction, or a twist on a core mechanic.

    ### SCORING RUBRIC (INNOVATION) ###
    Use this 1-10 scale with granular sub-levels:
    - **10 (Groundbreaking):** A genre-defining masterpiece. Introduces mechanics that will be copied for years.
    - **9 (Exceptional):** Truly novel. Pushes boundaries with a unique, well-executed core idea.
    - **8-8.5 (Very Good):** Innovative. Several clever mechanics or a fresh genre take.
    - **7-7.9 (Good):** Fresh. At least one core mechanic feels new and interesting.
    - **6.5-6.9 (Near-Good):** Some interesting ideas emerging, needs more development.
    - **6-6.4 (Acceptable):** Minor interesting twist, but core gameplay is conventional.
    - **5-5.9 (Mediocre):** Formulaic combination of well-worn tropes. "By-the-numbers" design.
    - **4 (Familiar):** Competent but nothing new. Feels like you've played this before.
    - **3 (Highly Derivative):** Borrows heavily from one or two popular games.
    - **1-2 (Clone):** Direct copy with only theme changed.

    ### INPUT DATA ###
    Game Concept: {concept}
    Game Rules: {rules}
    All Cards: {cards}

    ### OUTPUT LANGUAGE ###
    Language for response: {language}
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def evaluate(self, concept: GameConcept, rules: Rules, cards: list[Card], language: str) -> InnovationEvaluation:
        return self.llm_service.generate(
            output_model=InnovationEvaluation,
            prompt=self.PROMPT_TEMPLATE,
            concept=concept.model_dump_json(indent=2),
            rules=rules.model_dump_json(indent=2),
            cards=[card.model_dump() for card in cards],
            language=language,
        )


# Legacy alias for backward compatibility
OriginalityAgent = InnovationAgent

class PlayabilityAgent:
    """Evaluates the game's playability, game flow, and fun factor."""
    PROMPT_TEMPLATE = """
    ### ROLE & PERSONA ###
    Act as a top-tier Player Experience (PX) designer with a PhD in the psychology of play. Your focus is not on numbers or theme, but on the *feeling* of playing the game. You analyze the emotional journey of the player, the quality of their decisions, and the core "fun factor." Your ultimate question is: "Will players want to play this again?"

    ### TASK & PROCESS ###
    Your sole mission is to provide a ruthless critique of the game's playability and potential for fun. Deconstruct the player's moment-to-moment experience. Follow these steps:
    1.  **Decision Quality & Agency:** Analyze the choices players make. Are they meaningful and interesting? Or is there always an obvious "best" move? Does the player feel in control of their destiny, or are they just a victim of random card draws?
    2.  **Game Flow & Pacing:** How does the game feel over time? Is there a satisfying arc with a clear beginning, middle, and end? Does it build to an exciting climax? Does it drag on, or end too abruptly?
    3.  **The "Fun Loop" & Replayability:** What is the core action that players repeat? Is this loop intrinsically satisfying? What is the primary source of fun (e.g., outsmarting others, a lucky draw, building a powerful engine)? Is there enough variability in the cards and setup to make each game feel different?
    4.  **Emotional Arc:** Does the game create moments of tension, excitement, surprise, and relief? Are there comeback mechanics? Does it create memorable stories that players will talk about after the game is over?
    5.  **Score Assignment:** Based on your findings and the rubric below, assign a numerical score. Be brutally honest about whether you would *personally* want to play this game again.
    6.  **Analysis Write-up:** Justify your score by describing the *feeling* of playing the game. Pinpoint which parts are engaging and which are boring, frustrating, or flat.
    7.  **Improvement Suggestions:** Propose concrete changes to inject more fun. Suggest adding more dramatic "swing" cards, creating more direct player interaction, improving the sense of progression, or adding elements of risk/reward.

    ### SCORING RUBRIC (PLAYABILITY / FUN FACTOR) ###
    Use this 1-10 scale with granular sub-levels for refinement tracking:
    - **10 (Masterpiece of Fun):** Pinnacle of interactive entertainment. Perfect "fun engine" with memorable stories.
    - **9 (Exceptional):** Incredibly fun and compelling. Deep engagement, near-infinite replayability.
    - **8-8.5 (Very Good):** Highly engaging with deep decisions and excellent flow. Very replayable.
    - **7-7.9 (Good):** Genuinely fun and solid. Satisfying loop, would recommend and replay.
    - **6.5-6.9 (Near-Good):** Getting fun. Core loop works, some pacing or depth issues remain.
    - **6-6.4 (Acceptable):** Moments of fun hampered by pacing/depth flaws. Might bore after few plays.
    - **5-5.9 (Mediocre):** Functional but not exciting. Trivial decisions, flat experience.
    - **4 (Poor):** Hollow, unrewarding, or frustrating. Bad pacing.
    - **3 (Boring):** Flat line. No interesting decisions or excitement.
    - **1-2 (Anti-fun):** Frustrating, confusing, or tedious. Feels like work.

    ### INPUT DATA ###
    Game Concept: {concept}
    Game Rules: {rules}
    All Cards: {cards}

    ### OUTPUT LANGUAGE ###
    Language for response: {language}
    """
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def evaluate(self, concept: GameConcept, rules: Rules, cards: list[Card], language: str) -> PlayabilityEvaluation:
        return self.llm_service.generate(
            output_model=PlayabilityEvaluation,
            prompt=self.PROMPT_TEMPLATE,
            concept=concept.model_dump_json(indent=2),
            rules=rules.model_dump_json(indent=2),
            cards=[card.model_dump() for card in cards],
            language=language,
        )

class EvaluationSynthesizerAgent:
    """Combines specialist reports into a final evaluation with weighted scoring."""
    PROMPT_TEMPLATE = """
    ### ROLE & PERSONA ###
    Act as the Executive Producer and Lead Designer of a major game studio. You have just received the final evaluation reports from your specialist teams. Your job is to synthesize these expert opinions into a final, high-level executive summary.

    ### TASK ###
    Write an executive summary (3-5 sentences) that includes:
    1. An opening statement with the overall verdict based on the weighted score ({overall_score:.1f}/10).
    2. The game's greatest strength (cite the highest-scoring metric).
    3. The game's most critical weakness (cite the lowest-scoring metric).
    4. A concluding sentence on the game's overall potential.

    ### SPECIALIST SCORES (weighted importance shown) ###
    - Playability: {playability_score}/10 (weight: 2.0 - most important)
    - Balance: {balance_score}/10 (weight: 1.5)
    - Clarity: {clarity_score}/10 (weight: 1.2)
    - Theme Alignment: {theme_alignment_score}/10 (weight: 1.0)
    - Innovation: {innovation_score}/10 (weight: 0.8 - nice to have)

    ### OUTPUT LANGUAGE ###
    Write the summary in: {language}
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def synthesize(
        self,
        balance_eval: BalanceEvaluation,
        clarity_eval: ClarityEvaluation,
        playability_eval: PlayabilityEvaluation,
        theme_alignment_eval: ThemeAlignmentEvaluation,
        innovation_eval: InnovationEvaluation,
        language: str,
    ) -> GameEvaluation:
        # Use final_score (adjusted if available, otherwise original) for weighted calculation
        scores = {
            "balance": balance_eval.final_score,
            "clarity": clarity_eval.final_score,
            "playability": playability_eval.final_score,
            "theme_alignment": theme_alignment_eval.final_score,
            "innovation": innovation_eval.final_score,
        }
        overall_score = calculate_weighted_score(scores)

        # Generate only the summary using LLM (show adjusted scores)
        summary_result = self.llm_service.generate(
            output_model=EvaluationSummary,
            prompt=self.PROMPT_TEMPLATE,
            balance_score=round(balance_eval.final_score, 1),
            clarity_score=round(clarity_eval.final_score, 1),
            playability_score=round(playability_eval.final_score, 1),
            theme_alignment_score=round(theme_alignment_eval.final_score, 1),
            innovation_score=round(innovation_eval.final_score, 1),
            overall_score=overall_score,
            language=language,
        )

        return GameEvaluation(
            overall_score=overall_score,
            summary=summary_result.summary,
            balance=balance_eval,
            clarity=clarity_eval,
            playability=playability_eval,
            theme_alignment=theme_alignment_eval,
            innovation=innovation_eval,
        )

# Legacy aliases for backward compatibility
CoherenceAgent = ThemeAlignmentAgent
FidelityAgent = ThemeAlignmentAgent


class SuggestionSynthesizerAgent:
    """
    Synthesizes suggestions from all evaluation metrics.
    Deduplicates, prioritizes by impact, and identifies conflicts.
    """
    PROMPT_TEMPLATE = """
    ### ROLE ###
    You are a Game Design Project Manager synthesizing feedback from multiple specialist evaluators.
    Your job is to consolidate their suggestions into a prioritized, non-redundant action plan.

    ### RAW SUGGESTIONS FROM EVALUATORS ###
    {all_suggestions}

    ### YOUR TASK ###
    1. **Deduplicate**: Merge similar suggestions (e.g., "nerf Dragon" and "reduce Dragon attack" are the same)
    2. **Prioritize**: Rank by estimated impact on overall score
    3. **Identify Conflicts**: Flag suggestions that contradict (e.g., "buff X" vs "nerf X")
    4. **Categorize**: Group into high/medium/low priority

    ### PRIORITIZATION RULES ###
    - HIGH: Affects Playability or Balance (highest weighted metrics), estimated impact > 0.5
    - MEDIUM: Affects Clarity or Theme Alignment, or high-weighted with lower impact
    - LOW: Affects Innovation, or minor tweaks

    ### OUTPUT FORMAT ###
    For each suggestion:
    - suggestion: The actionable text (merged if duplicated)
    - target: "rules", "cards", or "both"
    - target_card: Specific card name if applicable
    - estimated_impact: Expected score improvement (0-2.0)
    - priority: "high", "medium", or "low"
    - source_metrics: Which evaluators suggested this
    - conflicts_with: IDs of conflicting suggestions (empty if none)

    ### OUTPUT LANGUAGE ###
    Language for response: {language}
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def synthesize(
        self,
        evaluation: GameEvaluation,
        language: str = "English"
    ) -> "SynthesizedSuggestions":
        from deck_crafter.models.evaluation import SynthesizedSuggestions

        # Collect all suggestions from all metrics
        all_suggestions = []
        metrics = [
            ("Playability", evaluation.playability),
            ("Balance", evaluation.balance),
            ("Clarity", evaluation.clarity),
            ("Theme Alignment", evaluation.theme_alignment),
            ("Innovation", evaluation.innovation),
        ]

        for metric_name, metric_eval in metrics:
            if metric_eval.suggestions:
                for suggestion in metric_eval.suggestions:
                    all_suggestions.append(f"[{metric_name}] {suggestion}")

        if not all_suggestions:
            return SynthesizedSuggestions(
                total_suggestions=0,
                deduplicated_count=0,
            )

        # Format for prompt
        suggestions_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(all_suggestions))

        return self.llm_service.generate(
            output_model=SynthesizedSuggestions,
            prompt=self.PROMPT_TEMPLATE,
            all_suggestions=suggestions_text,
            language=language,
        )


class CrossMetricReviewAgent:
    """
    Second-pass review agent that adjusts scores based on cross-metric awareness.
    Each metric can see other metrics' scores and adjust its own by ±0.5.
    """
    PROMPT_TEMPLATE = """
    ### ROLE ###
    You are conducting a SECOND-PASS review of an evaluation. You originally scored {metric_name} as {original_score}/10.
    Now you can see how OTHER metrics scored the same game. Consider whether your score should be adjusted.

    ### YOUR ORIGINAL EVALUATION ###
    Metric: {metric_name}
    Original Score: {original_score}/10
    Your Analysis: {original_analysis}

    ### OTHER METRICS' SCORES ###
    {other_scores_summary}

    ### CROSS-METRIC CONSIDERATIONS ###
    Sometimes metrics interact:
    - A game with poor Balance (low score) but high Playability might still be fun despite imbalance
    - A game with great Innovation but poor Clarity might not deliver on its promise
    - Theme Alignment issues might be forgiven if Playability is excellent
    - Perfect Clarity doesn't matter if the game isn't fun (low Playability)

    ### YOUR TASK ###
    1. Review your original score in light of the other metrics
    2. Decide if an adjustment is warranted (±0.5 MAX)
    3. Only adjust if there's a genuine cross-metric interaction
    4. If no adjustment needed, set adjusted_score = original_score

    ### ADJUSTMENT RULES ###
    - Maximum adjustment: ±0.5 points
    - Only adjust for genuine cross-metric interactions
    - Don't adjust just because other scores are different
    - Justify any adjustment with specific reasoning

    ### OUTPUT LANGUAGE ###
    Language for response: {language}
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def review(
        self,
        metric_name: str,
        original_eval: "BaseMetricEvaluation",
        all_scores: dict[str, int],
        language: str = "English"
    ) -> "ScoreAdjustment":
        from deck_crafter.models.evaluation import ScoreAdjustment

        # Format other scores
        other_scores = {k: v for k, v in all_scores.items() if k != metric_name}
        other_scores_summary = "\n".join([
            f"- {name.replace('_', ' ').title()}: {score}/10"
            for name, score in sorted(other_scores.items(), key=lambda x: -x[1])
        ])

        return self.llm_service.generate(
            output_model=ScoreAdjustment,
            prompt=self.PROMPT_TEMPLATE,
            metric_name=metric_name.replace('_', ' ').title(),
            original_score=original_eval.score,
            original_analysis=original_eval.analysis[:500],  # Truncate for context
            other_scores_summary=other_scores_summary,
            language=language,
        )


class ValidatorAgent:
    """
    Un agente genérico que valida una salida de datos contra su esquema Pydantic
    y un conjunto de criterios de alto nivel.
    """
    PROMPT_TEMPLATE = """
    ### ROLE & PERSONA ###
    You are a meticulous and strict Quality Assurance Analyst. Your role is to assess a given data output and determine if it meets a specific set of requirements.

    ### TASK ###
    Evaluate the `Output to Validate` against the rules defined in `Target Schema` and `High-Level Criteria`. The descriptions within the schema are requirements.

    ### INPUT DATA ###
    1. **Target Schema**: The JSON Schema the output MUST conform to.
    ```json
    {target_schema}
    ```
    2. **Output to Validate**: The data to check.
    ```json
    {output_to_validate}
    ```
    3. **High-Level Criteria**: Extra rules to check.
    ---
    {high_level_criteria}
    ---
    4. **Surrounding Context (for consistency check)**:
    ```json
    {context_data_json}
    ```

    ### INSTRUCTIONS ###
    - Analyze the output strictly.
    - If ALL checks pass, respond with `is_valid: true` and `feedback: "OK"`.
    - If ANY check fails, respond with `is_valid: false` and provide clear, actionable `feedback` explaining what to fix.
    - Respond ONLY with the requested JSON format.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def validate(
        self,
        output_to_validate: BaseModel,
        output_model: Type[BaseModel],
        high_level_criteria: str,
        context_data_json: str  # Ahora espera un string JSON
    ) -> ValidationResult:
        import json
        target_schema_str = json.dumps(output_model.model_json_schema(), indent=2)
        output_str = output_to_validate.model_dump_json(indent=2)
        
        return self.llm_service.generate(
            output_model=ValidationResult,
            prompt=self.PROMPT_TEMPLATE,
            target_schema=target_schema_str,
            output_to_validate=output_str,
            high_level_criteria=high_level_criteria,
            context_data_json=context_data_json
        )