from typing import Dict, Type
from deck_crafter.services.llm_service import LLMService
from deck_crafter.models.game_concept import GameConcept
from deck_crafter.models.rules import Rules
from deck_crafter.models.card import Card
from deck_crafter.models.evaluation import (
    BalanceEvaluation,
    CoherenceEvaluation,
    ClarityEvaluation,
    OriginalityEvaluation,
    PlayabilityEvaluation,
    GameEvaluation,
    FidelityEvaluation,
    ValidationResult,
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
    Use this 1-10 scale with extreme rigor:
    - **10 (Flawless Balance):** The holy grail. Multiple top-tier strategies are viable. A rich and diverse metagame that rewards skill.
    - **9 (Exceptional):** Near-perfect. Perhaps one strategy is marginally superior, but the difference is minimal and requires immense skill to exploit.
    - **8 (Very Good):** Well-tuned. A handful of competitive archetypes exist with clear counter-strategies.
    - **7 (Good):** Solid. The game is clearly functional and balanced, but there might be some "auto-include" or clearly suboptimal cards.
    - **6 (Acceptable):** Playable, but the metagame is narrow. There are obvious "correct" cards and many "trap" or useless choices.
    - **5 (Mediocre):** A significant portion of the card pool is not viable. One or two strategies are clearly superior to the rest.
    - **4 (Poor):** The game is heavily skewed. Following a specific strategy provides a massive, unfair advantage.
    - **3 (Severely Unbalanced):** A single strategy or a small set of cards completely dominates the game.
    - **2 (Broken):** Contains easily discoverable near-infinite combos or "I win" buttons.
    - **1 (Catastrophic Failure):** The core math of the game is fundamentally broken. Unplayable from a competitive standpoint.

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

class CoherenceAgent:
    """Evaluates exclusively the game's thematic and mechanical coherence."""
    PROMPT_TEMPLATE = """
    ### ROLE & PERSONA ###
    Act as a demanding and purist Narrative Director, famous for creating deeply immersive game worlds. Your philosophy is that every single element, from the smallest rule to a card's flavor text, must serve and reinforce the central theme. You have zero tolerance for anything that breaks immersion or feels generic and out of place. For you, "the theme is law."

    ### TASK & PROCESS ###
    Your sole mission is to provide a ruthless critique of this game's coherence. Follow these steps:
    1.  **Concept vs. Mechanics Analysis:** Compare the general `GameConcept` with the `Rules` and `Cards`. Do the gameplay mechanics (drawing cards, using resources, win conditions) reinforce the fantasy of the concept, or do they feel like generic mechanics with a thematic "skin"?
    2.  **Card Coherence Analysis:** Review the cards one by one. Do the name, art description, and, most importantly, the card's EFFECT make sense for its role in the game's universe? (Example of incoherence: A card named "Peaceful Farmer" having the ability "Deal 5 damage").
    3.  **Overall Immersion Evaluation:** As a whole, does the game succeed in creating a believable atmosphere? Or are there elements that constantly remind you that you're just playing a game, breaking the "magic"?
    4.  **Score Assignment:** Use the rubric below with maximum rigor to assign a score.
    5.  **Analysis Write-up:** Write a detailed analysis justifying your score, citing specific examples of rules or cards that are particularly good or bad in terms of coherence.
    6.  **Improvement Suggestions:** Propose concrete changes to strengthen the thematic links. For example: "Change the ability of 'Peacemaker Droid' from 'Deal 5 damage' to 'Cancel an opponent's action' to be more consistent with its name."

    ### SCORING RUBRIC (COHERENCE) ###
    Use this 1-10 scale with extreme rigor:
    - **10 (Perfect Symbiosis):** The game transcends its components. The theme and mechanics are indistinguishable, creating a purely immersive experience.
    - **9 (Exceptional):** A masterclass in integration. Every card and rule feels like a natural extension of the game's world.
    - **8 (Very Good):** The theme and mechanics are tightly interwoven. The world feels alive, consistent, and believable.
    - **7 (Good):** The theme is well-integrated and clearly enhances the gameplay experience. Most elements feel right.
    - **6 (Acceptable):** The theme is consistently applied, but it often feels like a superficial "skin" over standard mechanics.
    - **5 (Mediocre):** The theme is present, but there are several noticeable moments where the mechanics contradict it, breaking immersion.
    - **4 (Poor):** The theme and mechanics frequently clash. Many cards or rules feel completely out of place.
    - **3 (Severely Disjointed):** The theme feels like a cheap coat of paint slapped onto a generic mechanical skeleton.
    - **2 (Chaotic):** A theme is mentioned, but almost nothing in the gameplay supports or reflects it.
    - **1 (Thematic Anarchy):** Total chaos. The elements feel completely random and disconnected from the stated theme.

    ### INPUT DATA ###
    Game Concept: {concept}
    Game Rules: {rules}
    All Cards: {cards}

    ### OUTPUT LANGUAGE ###
    Language for response: {language}
    """
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def evaluate(self, concept: GameConcept, rules: Rules, cards: list[Card], language: str) -> CoherenceEvaluation:
        return self.llm_service.generate(
            output_model=CoherenceEvaluation,
            prompt=self.PROMPT_TEMPLATE,
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
    Use this 1-10 scale with extreme rigor:
    - **10 (Crystal Clear):** A masterpiece of technical writing. So clear, concise, and well-structured that it's impossible to misinterpret. It anticipates all edge cases.
    - **9 (Exceptional):** A gold-standard rulebook. Perfectly clear, with helpful examples for complex interactions.
    - **8 (Very Good):** The text is clear and concise. Most player questions are anticipated and answered within the text.
    - **7 (Good):** The rules are well-written and easy to follow for the most part. Only minor clarifications are needed for rare situations.
    - **6 (Acceptable):** Mostly clear, but a few key interactions or rules require a second read or a group consensus/FAQ to resolve.
    - **5 (Mediocre):** The core rules are understandable, but many edge cases are not covered, and several terms are used ambiguously. Leads to frequent pauses to debate rules.
    - **4 (Poor):** The game is learnable, but players will argue about fundamental rule interpretations constantly. The text is vague and confusing.
    - **3 (Severely Unclear):** Key rules are missing or so poorly written they are almost unusable. Requires players to invent "house rules" to function.
    - **2 (Contradictory):** The rulebook contains rules that directly contradict each other.
    - **1 (Unintelligible):** Complete gibberish. It's impossible to learn how to play the game from reading this text.

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

class OriginalityAgent:
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

    ### SCORING RUBRIC (ORIGINALITY) ###
    Use this 1-10 scale with extreme rigor:
    - **10 (Groundbreaking):** A genre-defining masterpiece. Introduces mechanics or concepts that will be copied for years to come.
    - **9 (Exceptional):** Truly novel. Pushes the boundaries of its genre with a unique and well-executed core idea.
    - **8 (Very Good):** Innovative. Features several clever mechanics or a very fresh take on a genre that makes it feel distinct.
    - **7 (Good):** Fresh. Has at least one core mechanic or a combination of ideas that feels new and interesting.
    - **6 (Acceptable):** Contains a minor, interesting twist, but the core gameplay is largely conventional.
    - **5 (Mediocre):** A formulaic combination of well-worn tropes and mechanics. A "by-the-numbers" design.
    - **4 (Familiar):** Competently executed, but brings nothing new to the table. You feel like you've played this exact game before with a different theme.
    - **3 (Highly Derivative):** Borrows heavily and obviously from one or two popular games.
    - **2 (Re-skin):** Essentially a direct copy of an existing game with only the theme changed.
    - **1 (Blatant Clone):** A shameless, near-verbatim copy of a well-known game.

    ### INPUT DATA ###
    Game Concept: {concept}
    Game Rules: {rules}
    All Cards: {cards}

    ### OUTPUT LANGUAGE ###
    Language for response: {language}
    """
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def evaluate(self, concept: GameConcept, rules: Rules, cards: list[Card], language: str) -> OriginalityEvaluation:
        return self.llm_service.generate(
            output_model=OriginalityEvaluation,
            prompt=self.PROMPT_TEMPLATE,
            concept=concept.model_dump_json(indent=2),
            rules=rules.model_dump_json(indent=2),
            cards=[card.model_dump() for card in cards],
            language=language,
        )

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
    Use this 1-10 scale with extreme rigor:
    - **10 (Masterpiece of Fun):** The pinnacle of interactive entertainment. A perfect "fun engine" that is intensely addictive and creates memorable stories in every session.
    - **9 (Exceptional):** Incredibly fun and compelling. Deeply engaging with near-infinite replayability. You can't wait for your next game.
    - **8 (Very Good):** A highly engaging game with deep, interesting decisions and excellent game flow. Very high replayability.
    - **7 (Good):** Genuinely fun and solid. The game loop is satisfying, and you would happily recommend it and play it again.
    - **6 (Acceptable):** Offers moments of fun, but these are hampered by flaws in pacing, a lack of decision depth, or repetitive gameplay. Might get boring after a few plays.
    - **5 (Mediocre):** It's functional, but not particularly exciting or memorable. The decisions feel trivial, and the experience is largely "flat." You wouldn't ask to play it again.
    - **4 (Poor):** The game is functional but the experience is hollow, unrewarding, or frustrating. The pacing is off.
    - **3 (Boring):** The game loop is a flat line with no interesting decisions or moments of excitement.
    - **2 (Frustrating):** The gameplay is actively frustrating, confusing, or feels like a pointless exercise.
    - **1 (A Tedious Chore):** Actively anti-fun. Playing the game feels like work.

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
    """Combina los informes de los especialistas en una evaluación final."""
    PROMPT_TEMPLATE = """
    ### ROLE & PERSONA ###
    Act as the Executive Producer and Lead Designer of a major game studio. You have just received the final evaluation reports from your specialist teams (Balance, Coherence, Clarity, Originality, Playability, and Fidelity). Your job is to synthesize these expert opinions into a final, high-level executive summary and a clear verdict. You are making the final "go/no-go" recommendation for this project.

    ### TASK & PROCESS ###
    Your task is to generate the final, consolidated `GameEvaluation`. Follow these steps:
    1.  **Internalize the Reports:** Read and understand all the provided specialist reports.
    2.  **Structure the Executive Summary:** Write a balanced, insightful executive summary that must include:
        - An opening statement with the overall verdict.
        - The game's greatest strength (citing the highest-scoring report).
        - The game's most critical weakness (citing the lowest-scoring report).
        - A concluding sentence on the game's overall potential.
    3.  **Assemble the Final Object:** Use the provided reports and the pre-calculated `overall_score` to assemble the final `GameEvaluation` object. Do NOT recalculate the score; use the one provided.

    ### PRE-CALCULATED OVERALL SCORE ###
    Overall Score: {overall_score:.2f}/10

    ### SPECIALIST REPORTS ###
    - Balance Report: {balance_report}
    - Coherence Report: {coherence_report}
    - Clarity Report: {clarity_report}
    - Originality Report: {originality_report}
    - Playability Report: {playability_report}
    - Fidelity Report: {fidelity_report}

    ### OUTPUT LANGUAGE ###
    Language for the summary: {language}
    """
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def synthesize(
        self,
        balance_eval: BalanceEvaluation,
        coherence_eval: CoherenceEvaluation,
        clarity_eval: ClarityEvaluation,
        originality_eval: OriginalityEvaluation,
        playability_eval: PlayabilityEvaluation,
        fidelity_eval: FidelityEvaluation,
        language: str,
    ) -> GameEvaluation:
        scores = [
            balance_eval.score,
            coherence_eval.score,
            clarity_eval.score,
            originality_eval.score,
            playability_eval.score,
            fidelity_eval.score,
        ]
        overall_score = sum(scores) / len(scores)

        return self.llm_service.generate(
            output_model=GameEvaluation,
            prompt=self.PROMPT_TEMPLATE,
            balance_report=balance_eval,
            coherence_report=coherence_eval,
            clarity_report=clarity_eval,
            originality_report=originality_eval,
            playability_report=playability_eval,
            fidelity_report=fidelity_eval,
            overall_score=overall_score,
            language=language,
        )

class FidelityAgent:
    """Evalúa exclusivamente la fidelidad del juego generado con respecto a la petición inicial del usuario."""
    PROMPT_TEMPLATE = """
    ### ROLE & PERSONA ###
    Act as a meticulous Project Manager and Quality Assurance lead. Your primary responsibility is to ensure that the final product strictly adheres to the initial client specifications. You are not concerned with whether the game is fun or balanced in the abstract, but only with whether it fulfills every single point of the user's request. A deviation from the request is a failure.

    ### TASK & PROCESS ###
    Your sole mission is to provide a rigorous audit of the game's fidelity to the user's original preferences. Follow these steps meticulously:
    1.  **Direct Comparison:** For each field in the `User Preferences` (language, theme, game_style, number_of_players, target_audience, rule_complexity), compare the requested value with the corresponding value in the final `Game Concept`.
    2.  **Description Analysis:** Read the initial `game_description` and compare its core ideas and sentiment with the final `Game Concept` description and `Rules`. Did the system capture the essence of what the user asked for?
    3.  **Identify Deviations:** Explicitly point out any discrepancies. For example: "User requested 'simple' complexity, but the generated rules include three different resource types and a multi-phase turn structure, which corresponds to 'medium' or 'complex' complexity." or "User requested a 'sci-fi' theme, but the generated cards include 'dragons' and 'magic spells'."
    4.  **Score Assignment:** Based on the degree of adherence and the rubric below, assign a numerical score. Be extremely literal in your judgment.
    5.  **Analysis Write-up:** Write a detailed analysis justifying your score, citing specific examples of successful adherence and failures to meet the requirements.
    6.  **Improvement Suggestions:** Propose concrete changes to the generated concept or rules to better align them with the original user request.

    ### SCORING RUBRIC (FIDELITY) ###
    Use this 1-10 scale with extreme rigor:
    - **10 (Perfect Adherence):** The output is a perfect reflection of the user's request in every aspect.
    - **9 (Exceptional):** Minor, almost negligible deviations in interpretation, but the core spirit and all key requirements are met.
    - **8 (Very Good):** All major requirements are met. There might be small discrepancies in secondary aspects.
    - **7 (Good):** The game is clearly based on the user's request, but one or two key aspects deviate noticeably.
    - **6 (Acceptable):** The output is recognizable, but there are significant deviations that alter the feel of the requested game.
    - **5 (Mediocre):** The system seems to have ignored several key preferences, resulting in a game that is only tangentially related to the request.
    - **4 (Poor):** Major, fundamental aspects of the request (like theme or complexity) were incorrectly generated.
    - **3 (Severely Deviant):** The output bears little resemblance to the initial request.
    - **2 (Incorrect):** The system appears to have completely misunderstood or ignored the core of the user's prompt.
    - **1 (Total Failure):** The generated game is the opposite of or completely unrelated to what was requested.

    ### INPUT DATA ###
    User Prompt: {game_description}
    User Preferences: {preferences}
    Game Concept: {concept}
    Game Rules: {rules}

    ### OUTPUT LANGUAGE ###
    Language for response: {language}
    """
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def evaluate(self, preferences: UserPreferences, concept: GameConcept, rules: Rules, language: str) -> FidelityEvaluation:
        return self.llm_service.generate(
            output_model=FidelityEvaluation,
            prompt=self.PROMPT_TEMPLATE,
            game_description=preferences.game_description,
            preferences=preferences.model_dump_json(indent=2),
            concept=concept.model_dump_json(indent=2),
            rules=rules.model_dump_json(indent=2),
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