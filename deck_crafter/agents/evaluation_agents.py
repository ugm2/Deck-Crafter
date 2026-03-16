import logging
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
    ModelScore,
    calculate_weighted_score,
    METRIC_WEIGHTS,
)
from deck_crafter.models.user_preferences import UserPreferences
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class BalanceAgent:
    """Evaluates exclusively the game's balance."""
    PROMPT_TEMPLATE = """
    ### ROLE & PERSONA ###
    Act as an experienced game balance consultant. You analyze card games for fairness and strategic diversity. You understand that PERFECT balance is impossible — even published games like Magic: The Gathering have balance issues. Focus on MEANINGFUL imbalances that hurt gameplay, not theoretical edge cases. These are homebrew games for playing with friends, not competitive tournament products.

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
    Use this 1-10 scale with CALIBRATED interpretation (5 = playable with issues, not "mediocre"):
    - **10 (Flawless Balance):** The holy grail. Multiple top-tier strategies are viable. Rich, diverse metagame.
    - **9 (Exceptional):** Near-perfect. One strategy marginally superior but requires immense skill to exploit.
    - **8-8.5 (Very Good):** Well-tuned. Handful of competitive archetypes with clear counter-strategies.
    - **7-7.9 (Good):** Solid and functional. Some "auto-include" cards but overall fair.
    - **6.5-6.9 (Above Average):** Strong foundation. Most issues minor, 1-2 cards need tuning.
    - **6-6.4 (Average):** Functional metagame, some obvious choices but games are fair.
    - **5-5.9 (Below Average):** Has balance issues but still playable and can be fun. 1-2 dominant strategies.
    - **4-4.9 (Poor):** Significant imbalance. One strategy clearly superior.
    - **3-3.9 (Very Poor):** Major balance failures. Game-warping cards exist.
    - **1-2 (Broken):** Near-infinite combos, "I win" buttons, or fundamentally broken math.

    IMPORTANT: A score of 5 means "has issues but is still a game worth playing."
    Reserve scores below 4 for truly broken designs.

    CALIBRATION CHECK before assigning your final score:
    - Score 2-3 means the game is LITERALLY UNPLAYABLE (infinite combos, impossible to win)
    - Score 4-5 means it HAS issues but can still be played and enjoyed casually
    - Score 6-7 means balance is GOOD for a homebrew game
    - If the game has working resource limits, no infinite combos, and multiple viable cards, it is AT LEAST a 5.
    - Ask yourself: "Would friends playing this casually have fun despite the imbalance?" If yes, score >= 5.

    ### INPUT DATA ###
    Game Concept: {concept}
    Game Rules: {rules}
    All Cards: {cards}
    {simulation_section}

    ### OUTPUT LANGUAGE ###
    Language for response: {language}
    """

    SIMULATION_SECTION = """
    ### SIMULATION EVIDENCE (EMPIRICAL DATA) ###
    This game has been playtested via simulation. Use this data to VALIDATE or REFUTE your theoretical analysis:

    **Summary:** {simulation_summary}

    **Key Metrics:**
    - First player win rate: {first_player_analysis}
    - Strategic diversity: {strategic_diversity}
    - Comeback potential: {comeback_potential}

    **Problematic Cards (from playtesting):**
    {problematic_cards}

    **High Priority Fixes (from playtesting):**
    {high_priority_fixes}

    IMPORTANT: When simulation data contradicts your theoretical analysis, TRUST THE SIMULATION.
    The data shows what actually happens when the game is played.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def evaluate(
        self,
        concept: GameConcept,
        rules: Rules,
        cards: list[Card],
        language: str,
        simulation_analysis=None
    ) -> BalanceEvaluation:
        logger.debug(f"[BalanceAgent] Evaluating game with {len(cards)} cards "
                    f"(has simulation: {simulation_analysis is not None})")
        # Build simulation section if data available
        simulation_section = ""
        if simulation_analysis:
            problematic = "\n".join(
                f"- {c.card_name}: {c.issue_type} - {c.evidence}"
                for c in simulation_analysis.problematic_cards
            ) or "None identified"
            fixes = "\n".join(
                f"- {fix}" for fix in simulation_analysis.high_priority_fixes
            ) or "None"

            simulation_section = self.SIMULATION_SECTION.format(
                simulation_summary=simulation_analysis.summary,
                first_player_analysis=simulation_analysis.first_player_analysis,
                strategic_diversity=simulation_analysis.strategic_diversity,
                comeback_potential=simulation_analysis.comeback_potential,
                problematic_cards=problematic,
                high_priority_fixes=fixes,
            )

        return self.llm_service.generate(
            output_model=BalanceEvaluation,
            prompt=self.PROMPT_TEMPLATE,
            concept=concept.model_dump_json(indent=2),
            rules=rules.model_dump_json(indent=2),
            cards=[card.model_dump() for card in cards],
            simulation_section=simulation_section,
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
    Use this 1-10 scale with CALIBRATED interpretation (5 = playable with issues, not "mediocre"):
    - **10 (Perfect Vision):** Flawless coherence AND perfect adherence to user request. A rare masterpiece.
    - **9 (Exceptional):** Near-perfect on both fronts. Minor, negligible deviations.
    - **8-8.5 (Very Good):** Strong coherence and fidelity. World feels alive, request is satisfied.
    - **7-7.9 (Good):** Well-integrated theme, most user requirements met. 1-2 minor issues.
    - **6.5-6.9 (Above Average):** Solid foundation, theme works, minor gaps in coherence or fidelity.
    - **6-6.4 (Average):** Theme is present and functional, some elements feel generic.
    - **5-5.9 (Below Average):** Theme is recognizable but mechanics don't fully reinforce it. Works but thin.
    - **4-4.9 (Poor):** Frequent theme-mechanic clashes OR major deviations from request.
    - **3-3.9 (Very Poor):** Little thematic sense AND bears little resemblance to request.
    - **1-2 (Failure):** Total incoherence AND complete misunderstanding of user's vision.

    IMPORTANT: A score of 5 means "theme is there and recognizable, mechanics need better integration."
    Reserve scores below 4 for games that feel completely off-target.

    ### INPUT DATA ###
    User Request: {game_description}
    User Preferences: {preferences}
    Game Concept: {concept}
    Game Rules: {rules}
    All Cards: {cards}
    {simulation_section}

    ### OUTPUT LANGUAGE ###
    Language for response: {language}
    """

    SIMULATION_SECTION = """
    ### SIMULATION EVIDENCE (EMPIRICAL DATA) ###
    This game has been playtested via simulation. Use this data to inform your theme assessment:

    **Strategic Diversity:** {strategic_diversity}
    **Dominant Strategies:** {dominant_strategies}

    **Theme-Mechanics Coherence Check:**
    If dominant strategies don't match the theme's intended playstyle, this is a COHERENCE issue.
    Example: A "defensive fortress" theme with "rush aggro" as dominant strategy = low coherence.
    Example: A "sneaky thief" theme where brute force always wins = mechanics betray theme.

    Does the actual gameplay experience (shown by simulation) match what the theme promises?
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def evaluate(
        self,
        preferences: UserPreferences,
        concept: GameConcept,
        rules: Rules,
        cards: list[Card],
        language: str,
        simulation_analysis=None
    ) -> ThemeAlignmentEvaluation:
        logger.debug(f"[ThemeAlignmentAgent] Evaluating theme coherence and fidelity")
        simulation_section = ""
        if simulation_analysis:
            dominant = ", ".join(simulation_analysis.dominant_strategies) or "No dominant strategy"
            simulation_section = self.SIMULATION_SECTION.format(
                strategic_diversity=simulation_analysis.strategic_diversity,
                dominant_strategies=dominant,
            )
            logger.debug(f"[ThemeAlignmentAgent] Using simulation: dominant strategies = {dominant}")

        return self.llm_service.generate(
            output_model=ThemeAlignmentEvaluation,
            prompt=self.PROMPT_TEMPLATE,
            game_description=preferences.game_description,
            preferences=preferences.model_dump_json(indent=2),
            concept=concept.model_dump_json(indent=2),
            rules=rules.model_dump_json(indent=2),
            cards=[card.model_dump() for card in cards],
            simulation_section=simulation_section,
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
    Use this 1-10 scale with CALIBRATED interpretation (5 = playable with issues, not "mediocre"):
    - **10 (Crystal Clear):** Masterpiece of technical writing. Impossible to misinterpret. All edge cases anticipated.
    - **9 (Exceptional):** Gold-standard rulebook. Perfectly clear with helpful examples.
    - **8-8.5 (Very Good):** Clear and concise. Most player questions anticipated and answered.
    - **7-7.9 (Good):** Well-written and easy to follow. Only minor clarifications needed for rare situations.
    - **6.5-6.9 (Above Average):** Solid rulebook. Most rules clear, 1-2 interactions could use FAQ.
    - **6-6.4 (Average):** Understandable rules, some interactions require second read but playable.
    - **5-5.9 (Below Average):** Core rules work but edge cases need clarification. Occasional debates.
    - **4-4.9 (Poor):** Learnable but players often argue about interpretations.
    - **3-3.9 (Very Poor):** Key rules missing or unusable. Requires house rules.
    - **1-2 (Contradictory/Unintelligible):** Rules contradict or impossible to understand.

    IMPORTANT: A score of 5 means "rules work for normal gameplay, edge cases need work."
    Reserve scores below 4 for rules that prevent normal play.

    ### INPUT DATA ###
    Game Concept: {concept}
    Game Rules: {rules}
    Card Examples: {cards}
    {simulation_section}
    {compilation_warnings_section}

    ### OUTPUT LANGUAGE ###
    Language for response: {language}
    """

    SIMULATION_SECTION = """
    ### SIMULATION EVIDENCE (EMPIRICAL DATA) ###
    This game has been playtested via simulation. Use this data to inform your clarity assessment:

    **Completion Rate:** {completion_rate:.0%} of games finished naturally
    **Rule Clarity Issues (from gameplay):** {rule_clarity_issues}

    IMPORTANT: If games frequently fail to complete (<70% completion), this indicates CLARITY problems
    even if rules look clear on paper. The simulation reveals what actually confuses players.
    """

    COMPILATION_WARNINGS_SECTION = """
    ### RULE COMPILATION WARNINGS ###
    When translating rules to simulation, these ambiguities were detected:
    {compilation_warnings}

    Each warning indicates an AMBIGUOUS or MISSING rule that the simulator had to guess about.
    These should LOWER your clarity score: severe warnings (-1.0), minor warnings (-0.3).
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def evaluate(
        self,
        concept: GameConcept,
        rules: Rules,
        cards: list[Card],
        language: str,
        simulation_analysis=None,
        compilation_warnings: list[str] = None
    ) -> ClarityEvaluation:
        logger.debug(f"[ClarityAgent] Evaluating rules clarity "
                    f"(compilation warnings: {len(compilation_warnings) if compilation_warnings else 0})")
        simulation_section = ""
        if simulation_analysis:
            rule_issues = "\n".join(
                f"- {issue}" for issue in simulation_analysis.rule_clarity_issues
            ) if simulation_analysis.rule_clarity_issues else "None detected"

            # Get completion rate from confidence assessment
            completion_rate = 1.0  # Default
            if simulation_analysis.confidence:
                completion_rate = 0.5 if not simulation_analysis.confidence.completion_rate_adequate else 0.85

            simulation_section = self.SIMULATION_SECTION.format(
                completion_rate=completion_rate,
                rule_clarity_issues=rule_issues,
            )
            logger.debug(f"[ClarityAgent] Using simulation: completion rate = {completion_rate:.0%}")

        compilation_warnings_section = ""
        if compilation_warnings:
            warnings_text = "\n".join(f"- {w}" for w in compilation_warnings)
            compilation_warnings_section = self.COMPILATION_WARNINGS_SECTION.format(
                compilation_warnings=warnings_text
            )
            logger.debug(f"[ClarityAgent] {len(compilation_warnings)} compilation warnings will affect score")

        return self.llm_service.generate(
            output_model=ClarityEvaluation,
            prompt=self.PROMPT_TEMPLATE,
            concept=concept.model_dump_json(indent=2),
            rules=rules.model_dump_json(indent=2),
            cards=[card.model_dump() for card in cards],
            simulation_section=simulation_section,
            compilation_warnings_section=compilation_warnings_section,
            language=language,
        )

class InnovationAgent:
    """Evaluates the game's originality and innovation."""
    PROMPT_TEMPLATE = """
    ### ROLE & PERSONA ###
    Act as a judge for the "Golden Meeple Award for Innovation in Gaming." You have deep knowledge of game mechanics across all genres. Your purpose is to evaluate MECHANICAL innovation — novel resource systems, win conditions, card interactions, and player dynamics. You appreciate when familiar themes inspire creative mechanics.

    ### IMPORTANT: THEME vs MECHANICAL INNOVATION ###
    These games are for PERSONAL/HOME USE. Using licensed IPs (Star Wars, Marvel, etc.) or familiar settings is perfectly acceptable.
    Innovation is measured EXCLUSIVELY by MECHANICAL originality — not by theme novelty.
    A Star Wars game with a novel deckbuilding twist is MORE innovative than an original-theme game copying Magic: The Gathering's mechanics.

    ### TASK & PROCESS ###
    Your sole mission is to evaluate this game's mechanical originality. You must deconstruct its mechanics and compare them against established genres. Follow these steps:
    1.  **Theme-Mechanic Integration:** Analyze the `GameConcept`. Does the theme inspire unique MECHANICS? A known IP that drives novel card interactions scores higher than an original theme with generic mechanics. Do NOT penalize familiar themes.
    2.  **Mechanical Originality Analysis:** This is the most critical step. Analyze the `Rules` and `All Cards`. Are the core game loops, resource systems, and win conditions novel? Or are they standard, well-known mechanics (e.g., deck-building, trick-taking, set collection, worker placement) just with different names? Scrutinize the card effects for unique interactions.
    3.  **Synthesis of Novelty:** How do the theme and mechanics combine? Does a common theme get a unique mechanical twist that makes it feel new? Or does a potentially unique theme get weighed down by generic, uninspired mechanics?
    4.  **Score Assignment:** Based on your findings and the rubric below, assign a numerical score. Be extremely critical. A "good" game is not necessarily an "original" game.
    5.  **Analysis Write-up:** Justify your score with specific comparisons. If the game is derivative, name the existing games or mechanics it borrows from. If it is original, explain exactly which idea or interaction is new.
    6.  **Improvement Suggestions:** Propose concrete, creative ideas to increase novelty. Suggest adding a unique resource, a surprising win condition, a new form of player interaction, or a twist on a core mechanic.

    ### SCORING RUBRIC (INNOVATION) ###
    Use this 1-10 scale with CALIBRATED interpretation (5 = standard execution, not "mediocre"):
    - **10 (Groundbreaking):** A genre-defining masterpiece. Introduces mechanics that will be copied for years.
    - **9 (Exceptional):** Truly novel. Pushes boundaries with a unique, well-executed core idea.
    - **8-8.5 (Very Good):** Innovative. Several clever mechanics or a fresh genre take.
    - **7-7.9 (Good):** Fresh. At least one core mechanic feels new and interesting.
    - **6.5-6.9 (Above Average):** Some interesting ideas, combines familiar mechanics in a fresh way.
    - **6-6.4 (Average):** Conventional with a minor twist. Solid execution of known patterns.
    - **5-5.9 (Below Average):** Standard genre execution. Not innovative but not a problem either.
    - **4-4.9 (Familiar):** Very conventional. Feels like you've played this before.
    - **3-3.9 (Highly Derivative):** Mechanics are near-identical copies of existing games. Note: using a known IP is NOT derivative — only copying MECHANICS counts.
    - **1-2 (Clone):** Direct copy with only theme changed.

    IMPORTANT: A score of 5 means "standard, competent design." Innovation is the lowest-weighted metric.
    Don't penalize games harshly for being conventional if they're well-executed.

    ### INPUT DATA ###
    Game Concept: {concept}
    Game Rules: {rules}
    All Cards: {cards}
    {simulation_section}

    ### OUTPUT LANGUAGE ###
    Language for response: {language}
    """

    SIMULATION_SECTION = """
    ### SIMULATION EVIDENCE (EMPIRICAL DATA) ###
    This game has been playtested via simulation. Use this to assess if innovation translates to gameplay:

    **Strategic Diversity:** {strategic_diversity}
    **Fun Indicators:** {fun_indicators}
    **Anti-Fun Indicators:** {anti_fun_indicators}

    **Innovation Reality Check:**
    Innovative mechanics that don't produce varied gameplay aren't truly innovative.
    - LOW strategic diversity suggests mechanics, however novel on paper, collapse to one playstyle.
    - HIGH anti-fun indicators suggest innovation is creating frustration, not engagement.
    - Games can be "innovative" on paper but play identically to standard games.

    Does the claimed innovation actually make the game play DIFFERENTLY?
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def evaluate(
        self,
        concept: GameConcept,
        rules: Rules,
        cards: list[Card],
        language: str,
        simulation_analysis=None
    ) -> InnovationEvaluation:
        logger.debug(f"[InnovationAgent] Evaluating originality and innovation")
        simulation_section = ""
        if simulation_analysis:
            fun_indicators = "\n".join(
                f"- {f}" for f in simulation_analysis.fun_indicators
            ) or "None identified"
            anti_fun = "\n".join(
                f"- {a}" for a in simulation_analysis.anti_fun_indicators
            ) or "None identified"
            simulation_section = self.SIMULATION_SECTION.format(
                strategic_diversity=simulation_analysis.strategic_diversity,
                fun_indicators=fun_indicators,
                anti_fun_indicators=anti_fun,
            )
            logger.debug(f"[InnovationAgent] Using simulation: strategic diversity = {simulation_analysis.strategic_diversity}")

        return self.llm_service.generate(
            output_model=InnovationEvaluation,
            prompt=self.PROMPT_TEMPLATE,
            concept=concept.model_dump_json(indent=2),
            rules=rules.model_dump_json(indent=2),
            cards=[card.model_dump() for card in cards],
            simulation_section=simulation_section,
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
    Use this 1-10 scale with CALIBRATED interpretation (5 = playable with issues, not "mediocre"):
    - **10 (Masterpiece of Fun):** Pinnacle of interactive entertainment. Perfect "fun engine" with memorable stories.
    - **9 (Exceptional):** Incredibly fun and compelling. Deep engagement, near-infinite replayability.
    - **8-8.5 (Very Good):** Highly engaging with deep decisions and excellent flow. Very replayable.
    - **7-7.9 (Good):** Genuinely fun and solid. Satisfying loop, would recommend and replay.
    - **6.5-6.9 (Above Average):** Core loop works well, some pacing or depth improvements possible.
    - **6-6.4 (Average):** Fun with room for improvement. Enjoyable but won't be memorable.
    - **5-5.9 (Below Average):** Has moments of fun but lacks depth or polish. Worth playing once.
    - **4-4.9 (Poor):** Hollow, unrewarding, or frustrating. Bad pacing.
    - **3-3.9 (Very Poor):** Flat line. No interesting decisions or excitement.
    - **1-2 (Anti-fun):** Frustrating, confusing, or tedious. Feels like work.

    IMPORTANT: A score of 5 means "you could have fun playing this, but it needs work."
    Reserve scores below 4 for games that are actively unfun.

    ### INPUT DATA ###
    Game Concept: {concept}
    Game Rules: {rules}
    All Cards: {cards}
    {simulation_section}

    ### OUTPUT LANGUAGE ###
    Language for response: {language}
    """

    SIMULATION_SECTION = """
    ### SIMULATION EVIDENCE (EMPIRICAL DATA) ###
    This game has been playtested via simulation. Use this data to inform your analysis of FUN and FLOW:

    **Summary:** {simulation_summary}

    **Pacing Assessment:** {pacing_assessment}
    **Pacing Issues:**
    {pacing_issues}

    **Fun Indicators (observed in gameplay):**
    {fun_indicators}

    **Anti-Fun Indicators (observed in gameplay):**
    {anti_fun_indicators}

    **Dominant Strategies:** {dominant_strategies}

    IMPORTANT: This data shows what actually happens during play.
    If games are too short/long or repetitive, this affects fun regardless of how good the design looks on paper.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def evaluate(
        self,
        concept: GameConcept,
        rules: Rules,
        cards: list[Card],
        language: str,
        simulation_analysis=None
    ) -> PlayabilityEvaluation:
        logger.debug(f"[PlayabilityAgent] Evaluating fun factor and playability")
        # Build simulation section if data available
        simulation_section = ""
        if simulation_analysis:
            pacing_issues = "\n".join(
                f"- [{p.severity}] {p.issue}"
                for p in simulation_analysis.pacing_issues
            ) or "None identified"
            fun_indicators = "\n".join(
                f"- {f}" for f in simulation_analysis.fun_indicators
            ) or "None identified"
            anti_fun = "\n".join(
                f"- {a}" for a in simulation_analysis.anti_fun_indicators
            ) or "None identified"
            strategies = ", ".join(simulation_analysis.dominant_strategies) or "No dominant strategy"

            simulation_section = self.SIMULATION_SECTION.format(
                simulation_summary=simulation_analysis.summary,
                pacing_assessment=simulation_analysis.pacing_assessment,
                pacing_issues=pacing_issues,
                fun_indicators=fun_indicators,
                anti_fun_indicators=anti_fun,
                dominant_strategies=strategies,
            )
            logger.debug(f"[PlayabilityAgent] Using simulation: pacing = {simulation_analysis.pacing_assessment}")

        return self.llm_service.generate(
            output_model=PlayabilityEvaluation,
            prompt=self.PROMPT_TEMPLATE,
            concept=concept.model_dump_json(indent=2),
            rules=rules.model_dump_json(indent=2),
            cards=[card.model_dump() for card in cards],
            simulation_section=simulation_section,
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
        logger.debug("[EvaluationSynthesizer] Combining 5 metric evaluations...")
        # Use final_score (adjusted if available, otherwise original) for weighted calculation
        scores = {
            "balance": balance_eval.final_score,
            "clarity": clarity_eval.final_score,
            "playability": playability_eval.final_score,
            "theme_alignment": theme_alignment_eval.final_score,
            "innovation": innovation_eval.final_score,
        }
        overall_score = calculate_weighted_score(scores)
        logger.info(f"[EvaluationSynthesizer] Weighted overall score: {overall_score:.2f}/10")

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
    Each metric can see other metrics' scores and adjust its own by ±1.0.
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
    CRITICAL RULE: Your job is to VALIDATE or SLIGHTLY INCREASE scores, NOT to penalize metrics that improved.

    DO NOT lower a metric just because other metrics scored low. Each metric is INDEPENDENT.
    - A high Playability score is CORRECT even if Balance is low — fun games CAN be unbalanced.
    - A high Clarity score is CORRECT even if Innovation is low — clear rules are always good.
    - A high Theme Alignment score is CORRECT regardless of other metrics — theme is independent.

    You MAY increase a score if other metrics reveal your analysis missed a strength.
    You MAY decrease a score ONLY with CONCRETE EVIDENCE that YOUR metric's analysis was objectively wrong (not just "other scores are low").

    ### YOUR TASK ###
    1. Review your original score in light of the other metrics
    2. Decide if an adjustment is warranted
    3. Only adjust if there's a genuine cross-metric interaction
    4. If no adjustment needed, set adjusted_score = original_score

    ### ADJUSTMENT RULES ###
    - Upward adjustments: up to +1.0 (when cross-metric evidence supports a higher score)
    - Downward adjustments: up to -0.25 ONLY (requires concrete evidence, not low-score contagion)
    - Default: NO adjustment. Only adjust with a specific, articulable reason.
    - "Other metrics scored low" is NEVER a valid reason to lower a score.

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


class EvaluationMergeAgent:
    """
    Merges GameEvaluation results from multiple models into one final evaluation.
    Uses median scores (robust to outliers) and LLM-generated summary.
    """

    SUMMARY_PROMPT = """
    ### ROLE ###
    You are a chief game design evaluator consolidating feedback from {num_models} independent reviewers.

    ### INDIVIDUAL SUMMARIES ###
    {model_summaries}

    ### MERGED SCORES ###
    {merged_scores}

    ### TASK ###
    Write a single executive summary (2-4 sentences) that captures the consensus view.
    Where reviewers disagree significantly, note the disagreement.
    Focus on the most actionable insights.

    ### OUTPUT LANGUAGE ###
    Language: {language}
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def merge(
        self,
        evaluations: list[tuple[str, GameEvaluation]],
        language: str = "English",
    ) -> GameEvaluation:
        """Merge multiple GameEvaluation results into one.

        Args:
            evaluations: List of (model_id, GameEvaluation) tuples
            language: Output language for the merged summary
        """
        import statistics

        metrics = list(METRIC_WEIGHTS.keys())

        # Collect per-model scores
        model_scores_list = []
        per_metric_scores: dict[str, list[tuple[str, float, object]]] = {m: [] for m in metrics}

        for model_id, evaluation in evaluations:
            scores_dict = evaluation.get_scores_dict()
            model_scores_list.append(ModelScore(
                model_id=model_id,
                scores=scores_dict,
                overall_score=evaluation.overall_score,
            ))
            for metric in metrics:
                eval_obj = getattr(evaluation, metric)
                per_metric_scores[metric].append((model_id, scores_dict[metric], eval_obj))

        # Merge each metric: median score, pick analysis closest to median
        merged_metrics = {}
        for metric in metrics:
            entries = per_metric_scores[metric]
            scores = [s for _, s, _ in entries]
            median = statistics.median(scores)

            # Pick the evaluation object whose score is closest to the median
            closest = min(entries, key=lambda e: abs(e[1] - median))
            eval_obj = closest[2]

            # Clone with median as the adjusted score
            merged_eval = eval_obj.model_copy()
            merged_eval.adjusted_score = round(median, 2)
            merged_eval.adjustment_reason = (
                f"Panel median from {len(entries)} models "
                f"(scores: {', '.join(f'{s:.1f}' for _, s, _ in entries)})"
            )
            merged_metrics[metric] = merged_eval

        # Compute weighted overall score from medians
        merged_scores_dict = {m: merged_metrics[m].final_score for m in metrics}
        overall = calculate_weighted_score(merged_scores_dict)

        # Generate merged summary via LLM
        model_summaries = "\n\n".join(
            f"**{model_id}** (overall: {ev.overall_score:.1f}/10):\n{ev.summary}"
            for model_id, ev in evaluations
        )
        merged_scores_text = "\n".join(
            f"- {m}: {merged_scores_dict[m]:.1f}/10" for m in metrics
        )

        try:
            summary_result = self.llm_service.generate(
                output_model=EvaluationSummary,
                prompt=self.SUMMARY_PROMPT,
                num_models=len(evaluations),
                model_summaries=model_summaries,
                merged_scores=merged_scores_text,
                language=language,
            )
            summary = summary_result.summary
        except Exception as e:
            logger.warning(f"[EvalMerge] Summary generation failed: {e}")
            summary = evaluations[0][1].summary  # Fallback to first model's summary

        # Merge suggestions from all evaluations
        all_suggestions = []
        for _, ev in evaluations:
            if ev.synthesized_suggestions:
                all_suggestions.extend(ev.synthesized_suggestions.high_priority)
                all_suggestions.extend(ev.synthesized_suggestions.medium_priority)

        merged = GameEvaluation(
            overall_score=round(overall, 2),
            summary=summary,
            balance=merged_metrics["balance"],
            clarity=merged_metrics["clarity"],
            playability=merged_metrics["playability"],
            theme_alignment=merged_metrics["theme_alignment"],
            innovation=merged_metrics["innovation"],
            model_scores=model_scores_list,
        )

        logger.info(
            f"[EvalMerge] Merged {len(evaluations)} evaluations → {overall:.2f}/10 "
            f"(per-model: {', '.join(f'{ms.model_id}={ms.overall_score:.1f}' for ms in model_scores_list)})"
        )
        return merged