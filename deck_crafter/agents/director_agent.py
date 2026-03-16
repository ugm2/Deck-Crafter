import logging
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from deck_crafter.services.llm_service import LLMService
from deck_crafter.models.evaluation import GameEvaluation
from deck_crafter.models.state import RefinementMemory, RefinementExperiment, FailedPattern

logger = logging.getLogger(__name__)


class RefinementStrategy(BaseModel):
    """Strategic decisions with granular control and hypothesis-driven approach."""

    # Hypothesis (scientific method)
    hypothesis: str = Field(
        ...,
        description="Testable prediction: 'If I change X, metric Y will improve by Z'. "
                    "Must be SPECIFIC and MEASURABLE, not vague like 'improve balance'."
    )
    target_metric: str = Field(
        ...,
        description="Primary metric this experiment targets: 'playability', 'balance', "
                    "'clarity', 'theme_alignment', or 'innovation'."
    )
    expected_improvement: float = Field(
        ...,
        ge=0.1,
        le=3.0,
        description="Expected score improvement (0.1-3.0). Be realistic: "
                    "surgical=0.3-0.5, moderate=0.5-1.0, nuclear=1.0-2.0"
    )
    confidence: Literal["low", "medium", "high"] = Field(
        ...,
        description="How confident are you? low=untested idea, medium=based on patterns, "
                    "high=similar approach worked before"
    )

    # Intervention type
    intervention_type: Literal["nuclear", "moderate", "surgical"] = Field(
        ...,
        description="nuclear: significant redesign (score < 3 or 2+ moderate failures). "
                    "moderate: 3-5 coordinated changes (score 3-5). "
                    "surgical: 1-2 minimal changes to test one hypothesis (score > 5)."
    )

    # Granular rules control
    rules_action: Literal["overhaul", "rewrite_section", "tweak", "none"] = Field(
        ...,
        description="overhaul: major rules restructure (nuclear interventions). "
                    "rewrite_section: rewrite one section (e.g., win_conditions). "
                    "tweak: minor wording/number changes. "
                    "none: don't touch rules."
    )
    rules_target: Optional[str] = Field(
        default=None,
        description="Which rules section to modify: 'win_conditions', 'turn_structure', "
                    "'resource_system', 'card_interactions', 'setup', or 'game_flow'. "
                    "Required if rules_action != 'none'."
    )
    rules_instruction: Optional[str] = Field(
        default=None,
        description="Specific instruction for rules changes. Be EXACT: "
                    "'Change win condition from 20 points to 15 points' not 'make shorter'."
    )

    # Granular cards control
    cards_action: Literal["regenerate_many", "regenerate_few", "stat_adjust", "none"] = Field(
        ...,
        description="regenerate_many: regenerate 4+ cards (moderate/nuclear). "
                    "regenerate_few: regenerate 1-3 specific cards. "
                    "stat_adjust: tweak stats without regeneration. "
                    "none: don't touch cards."
    )
    cards_to_modify: List[str] = Field(
        default_factory=list,
        description="EXACT card names to modify. Use real names from the game. "
                    "Required if cards_action != 'none'."
    )
    cards_instruction: Optional[str] = Field(
        default=None,
        description="Specific instruction for card changes: 'Reduce attack by 2, increase cost by 1' "
                    "not 'make weaker'."
    )

    # Reasoning
    reasoning: str = Field(
        ...,
        description="2-3 sentences explaining WHY this strategy. Include expected outcome."
    )
    why_not_alternatives: str = Field(
        ...,
        description="Why NOT other approaches? If a pattern failed before, mention it. "
                    "Shows you considered alternatives."
    )


class ExperimentReflection(BaseModel):
    """Post-hoc analysis of an experiment."""
    hypothesis_confirmed: bool = Field(
        ...,
        description="Did the hypothesis prove correct? True if improvement >= 50% of expected."
    )
    actual_vs_expected: str = Field(
        ...,
        description="Compare actual improvement to expected. "
                    "Example: 'Expected +0.5 balance, got +0.3 (partial success)'"
    )
    lesson_learned: str = Field(
        ...,
        description="One concrete lesson. Be SPECIFIC: "
                    "'Reducing Dragon costs helped but Wizard cards also need adjustment' "
                    "not 'balance is hard'."
    )
    should_continue_pattern: bool = Field(
        ...,
        description="Should we continue this approach next iteration? "
                    "True if positive progress, False if stagnant or negative."
    )
    pattern_to_avoid: Optional[str] = Field(
        default=None,
        description="If this failed, describe the pattern to AVOID in future. "
                    "Example: 'Reducing all costs simultaneously causes resource inflation'"
    )


class DirectorAgent:
    """Reflexive strategic decision-maker using scientific method for refinement."""

    DESIGN_PROMPT = """
### ROLE ###
You are a Senior Game Design Director running SCIENTIFIC EXPERIMENTS on a card game.
Each refinement iteration is an EXPERIMENT with a testable HYPOTHESIS.

### CURRENT STATE ###
Overall Score: {overall_score}/10 (weighted average)
Target Threshold: {threshold}/10
Gap to Close: {gap:.1f} points
Iteration: {iteration}/{max_iterations} ({remaining_iterations} remaining)
{trend_info}

### EVALUATION BREAKDOWN (5 metrics, weighted) ###
**PLAYABILITY** ({playability_score}/10, weight 2.0): {playability_analysis}
**BALANCE** ({balance_score}/10, weight 1.5): {balance_analysis}
**CLARITY** ({clarity_score}/10, weight 1.2): {clarity_analysis}
**THEME ALIGNMENT** ({theme_alignment_score}/10, weight 1.0): {theme_alignment_analysis}
**INNOVATION** ({innovation_score}/10, weight 0.8): {innovation_analysis}

### CURRENT CARDS ###
{cards_summary}

### EXPERIMENT HISTORY ###
{experiment_history}

### LESSONS LEARNED (USE THESE!) ###
{lessons_learned}

### PATTERNS THAT WORKED (REPEAT THESE) ###
{successful_patterns}

### PATTERNS THAT FAILED - CRITICAL: DO NOT REPEAT THESE! ###
{failed_patterns}

⚠️ IMPORTANT: If you see a failed pattern above with the same target_metric + rules_action + rules_target combination you're considering, you MUST choose a DIFFERENT approach. Repeating a failed pattern is FORBIDDEN.

{blocked_metrics_section}
{forced_intervention_section}

### PROBLEMATIC CARDS (RECURRING ISSUES) ###
{problematic_cards}

{simulation_section}

{compilation_warnings_section}

### YOUR TASK: DESIGN THE NEXT EXPERIMENT ###

Follow the SCIENTIFIC METHOD:

1. **HYPOTHESIS**: State a TESTABLE prediction
   - BAD: "Improve balance" (vague, unmeasurable)
   - GOOD: "Reducing Dragon card attack from 8→5 will improve balance by ~0.5"

2. **INTERVENTION TYPE**: Choose based on history
   - SURGICAL: First attempts, or after nuclear failed
   - MODERATE: If 2+ surgical attempts plateaued
   - NUCLEAR: ONLY if stuck after 2+ failed moderate attempts

3. **SPECIFIC ACTIONS**: Not just "change cards" but EXACTLY which and how
   - Rules: Which section? What EXACT change?
   - Cards: Which EXACT cards? What stat adjustments?

4. **CONFIDENCE**: How sure are you this will work?
   - LOW: Untested idea or similar approaches failed
   - MEDIUM: Based on evaluation suggestions
   - HIGH: Similar approach worked before

5. **WHY NOT ALTERNATIVES**: Explain why you're NOT doing other things
   - If a pattern is in FAILED_PATTERNS, you MUST NOT use it again

### CRITICAL RULES ###
- ⛔ FORBIDDEN: If failed_patterns contains an entry with the same (target_metric + rules_action + rules_target) you're considering, you MUST:
  1. Choose a DIFFERENT target_metric, OR
  2. Choose a DIFFERENT rules_action, OR
  3. Choose a DIFFERENT rules_target, OR
  4. Try cards_action instead of rules_action
- Each experiment should test ONE main hypothesis
- Use EXACT card names from the list above
- Be SPECIFIC with numbers and targets
- After 2+ rollbacks: STOP targeting the same metric, try something else entirely

### DECISION GUIDELINES (MANDATORY) ###
- Score < 3.5: You MUST use intervention_type="nuclear". rules_action MUST be "overhaul". cards_action MUST be "regenerate_many". The game is fundamentally broken — tweaking will NOT fix it.
- Score 3.5-5: You MUST use intervention_type="moderate" or "nuclear". rules_action MUST be "rewrite_section" or "overhaul". cards_action MUST NOT be "none".
- Score 5-6.5: Moderate or surgical. Both rules AND cards likely need work.
- Score 6.5-7: Surgical only, fine-tuning
- If simulation completion < 10%: The game CANNOT BE PLAYED. rules_action MUST be "rewrite_section" or "overhaul" targeting the broken mechanic. This takes priority over all other guidelines.
- Failed 2+ surgical: Escalate to moderate
- Failed 2+ moderate: Escalate to nuclear
- Just had rollback: Try COMPLETELY different approach:
  * If you were targeting "clarity", try "balance" or "playability" instead
  * If you were modifying rules, try cards instead (or vice versa)
  * If rules_action was "rewrite_section", try "tweak" or change the target
- Multiple rollbacks on same metric: That metric may be at its ceiling, pivot to another
"""

    REFLECT_PROMPT = """
### ROLE ###
You are analyzing the results of a game design experiment.

### EXPERIMENT DETAILS ###
Iteration: {iteration}
Hypothesis: "{hypothesis}"
Target Metric: {target_metric}
Expected Improvement: +{expected_improvement}

Intervention: {intervention_type}
Rules Action: {rules_action} (target: {rules_target})
Cards Action: {cards_action} (cards: {cards_modified})

### RESULTS ###
Score Before: {score_before}/10
Score After: {score_after}/10
Actual Change: {actual_change:+.1f}

{metric_changes}

### TASK ###
Analyze what happened and extract a lesson for future iterations.
Be SPECIFIC and ACTIONABLE, not vague.

If the experiment failed:
- Identify WHY it failed
- Describe the pattern to AVOID in future

If the experiment succeeded:
- Identify WHY it worked
- Describe the pattern to REPEAT
"""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def design_experiment(
        self,
        evaluation: GameEvaluation,
        threshold: float,
        iteration: int,
        max_iterations: int,
        cards_summary: str = "",
        memory: Optional[RefinementMemory] = None,
        previous_evaluations: Optional[List[GameEvaluation]] = None,
        simulation_analysis=None,
        forced_intervention: Optional[str] = None,
        blocked_metrics: Optional[List[str]] = None,
        compilation_warnings: Optional[List[str]] = None,
    ) -> RefinementStrategy:
        """Design the next refinement experiment using scientific method."""
        logger.info(f"[DirectorAgent] Designing experiment for iteration {iteration}/{max_iterations}")
        logger.info(f"[DirectorAgent] Current score: {evaluation.overall_score:.2f}, threshold: {threshold}")

        if forced_intervention:
            logger.warning(f"[DirectorAgent] FORCED INTERVENTION: {forced_intervention.upper()}")
        if blocked_metrics:
            logger.info(f"[DirectorAgent] Blocked metrics (ceiling): {blocked_metrics}")
        if memory:
            logger.info(f"[DirectorAgent] Memory: {len(memory.failed_patterns)} failed patterns, "
                       f"{len(memory.experiments)} experiments, "
                       f"total failures: {memory.total_failed_iterations}")

        gap = threshold - evaluation.overall_score
        remaining = max_iterations - iteration
        logger.debug(f"[DirectorAgent] Gap to close: {gap:.2f}, iterations remaining: {remaining}")

        # Build trend info
        trend_info = ""
        if previous_evaluations and len(previous_evaluations) > 0:
            scores = [e.overall_score for e in previous_evaluations] + [evaluation.overall_score]
            if len(scores) >= 2:
                last_delta = scores[-1] - scores[-2]
                direction = "improved" if last_delta > 0 else "dropped" if last_delta < 0 else "unchanged"
                trend_info = f"Previous Score: {scores[-2]:.1f}/10 (score {direction} by {abs(last_delta):.1f})"

        # Format memory sections
        if memory:
            experiment_history = self._format_experiments(memory.experiments[-5:])  # Last 5
            lessons_learned = "\n".join(f"- {l}" for l in memory.lessons_learned[-5:]) or "None yet"
            successful_patterns = "\n".join(f"- {p}" for p in memory.successful_patterns[-5:]) or "None yet"
            # Format structured FailedPattern objects
            failed_patterns = self._format_failed_patterns(memory.failed_patterns[-5:])
            problematic_cards = "\n".join(
                f"- {card}: failed {count} times"
                for card, count in sorted(memory.problematic_cards.items(), key=lambda x: -x[1])[:5]
            ) or "None identified"
        else:
            experiment_history = "No previous experiments"
            lessons_learned = "None yet"
            successful_patterns = "None yet"
            failed_patterns = "None yet"
            problematic_cards = "None identified"

        # Build blocked metrics section
        blocked_metrics_section = ""
        if blocked_metrics:
            blocked_metrics_section = f"""
### ⛔ BLOCKED METRICS (DO NOT TARGET THESE) ###
The following metrics have hit their improvement ceiling after multiple failed attempts:
{chr(10).join(f'- {m}: HIT CEILING - stop targeting this metric' for m in blocked_metrics)}

You MUST choose a target_metric that is NOT in this list. Focus on metrics with more improvement potential.
"""

        # Build forced intervention section
        forced_intervention_section = ""
        if forced_intervention:
            forced_intervention_section = f"""
### 🚨 ESCALATION REQUIRED ###
Due to repeated failures at low scores, the system is forcing intervention level: **{forced_intervention.upper()}**

You MUST set intervention_type = "{forced_intervention}". This is NOT optional.
{"Use moderate: 3-5 coordinated changes targeting multiple aspects." if forced_intervention == "moderate" else ""}
{"Use nuclear: Significant redesign. Rethink core mechanics that aren't working." if forced_intervention == "nuclear" else ""}
"""

        # Build compilation warnings section
        compilation_warnings_section = ""
        if compilation_warnings:
            warnings_list = "\n".join(f"- {w}" for w in compilation_warnings)
            compilation_warnings_section = f"""
### GAME-BREAKING STRUCTURAL ISSUES (from rule compiler) ###
The rule compiler detected problems that prevent the game from functioning:
{warnings_list}

These are NOT design opinions — they are HARD FACTS about broken mechanics.
- "resource mismatch": Cards reference a resource that rules don't provide (or vice versa). Standardize to ONE name.
- "win condition unreachable": No card can advance toward the win condition. Either add cards that can, or change the win condition type.

YOU MUST fix these BEFORE any design improvements. A game that can't be played scores 0 on Playability.
"""
            logger.info(f"[DirectorAgent] Including {len(compilation_warnings)} compilation warnings in prompt")

        # Build simulation section if data available
        simulation_section = ""
        if simulation_analysis:
            sim_problems = "\n".join(
                f"- {c.card_name}: {c.issue_type} ({c.evidence})"
                for c in simulation_analysis.problematic_cards
            ) or "None detected"

            sim_fixes = "\n".join(
                f"- {fix}" for fix in simulation_analysis.high_priority_fixes
            ) or "None"

            # Parse balance adjustments for exact instructions
            balance_adj_section = ""
            if simulation_analysis.balance_adjustments:
                from deck_crafter.game_simulator.models.metrics import BalanceAdjustment
                parsed = BalanceAdjustment.parse_adjustments(simulation_analysis.balance_adjustments)
                if parsed:
                    adj_lines = []
                    for adj in parsed:
                        if adj.current_value and adj.target_value:
                            adj_lines.append(f"- {adj.card_name}: {adj.stat} {adj.current_value} → {adj.target_value} ({adj.reason})")
                        else:
                            adj_lines.append(f"- {adj.card_name}: {adj.action} {adj.stat} ({adj.reason})")
                    balance_adj_section = f"""
### ⚡ EXACT BALANCE FIXES (apply these precisely) ###
These are DATA-DRIVEN recommendations. Use cards_action=stat_adjust and apply exactly:
{chr(10).join(adj_lines)}
"""

            # Confidence warning
            confidence_warning = ""
            if simulation_analysis.confidence and simulation_analysis.confidence.overall == "low":
                confidence_warning = f"\n⚠️ LOW CONFIDENCE ANALYSIS: {'; '.join(simulation_analysis.confidence.reasons)}\n"

            simulation_section = f"""
### SIMULATION EVIDENCE (EMPIRICAL DATA FROM PLAYTESTING) ###
⚠️ This data is from ACTUAL SIMULATED GAMES - trust it over theoretical analysis!
{confidence_warning}
**Summary:** {simulation_analysis.summary}

**Key Metrics:**
- Strategic Diversity: {simulation_analysis.strategic_diversity}
- Pacing: {simulation_analysis.pacing_assessment}
- Comeback Potential: {simulation_analysis.comeback_potential}
- First Player Analysis: {simulation_analysis.first_player_analysis}
{balance_adj_section}
**Problematic Cards (from gameplay data):**
{sim_problems}

**High Priority Fixes (from simulation):**
{sim_fixes}

**Fun Indicators:** {', '.join(simulation_analysis.fun_indicators) or 'None'}
**Anti-Fun Indicators:** {', '.join(simulation_analysis.anti_fun_indicators) or 'None'}

⚠️ PRIORITIZE: Address issues identified by simulation data - these are PROVEN problems, not guesses!
"""

        logger.debug("[DirectorAgent] Calling LLM to generate strategy...")
        strategy = self.llm_service.generate(
            output_model=RefinementStrategy,
            prompt=self.DESIGN_PROMPT,
            overall_score=evaluation.overall_score,
            threshold=threshold,
            gap=gap,
            iteration=iteration,
            max_iterations=max_iterations,
            remaining_iterations=remaining,
            trend_info=trend_info,
            playability_score=evaluation.playability.score,
            playability_analysis=evaluation.playability.analysis,
            balance_score=evaluation.balance.score,
            balance_analysis=evaluation.balance.analysis,
            clarity_score=evaluation.clarity.score,
            clarity_analysis=evaluation.clarity.analysis,
            theme_alignment_score=evaluation.theme_alignment.score,
            theme_alignment_analysis=evaluation.theme_alignment.analysis,
            innovation_score=evaluation.innovation.score,
            innovation_analysis=evaluation.innovation.analysis,
            cards_summary=cards_summary or "No cards available",
            experiment_history=experiment_history,
            lessons_learned=lessons_learned,
            successful_patterns=successful_patterns,
            failed_patterns=failed_patterns,
            problematic_cards=problematic_cards,
            simulation_section=simulation_section,
            blocked_metrics_section=blocked_metrics_section,
            forced_intervention_section=forced_intervention_section,
            compilation_warnings_section=compilation_warnings_section,
        )

        logger.info(f"[DirectorAgent] Strategy generated: {strategy.intervention_type.upper()} "
                   f"targeting {strategy.target_metric}")
        logger.info(f"[DirectorAgent] Hypothesis: {strategy.hypothesis[:80]}...")
        logger.info(f"[DirectorAgent] Expected improvement: +{strategy.expected_improvement} "
                   f"(confidence: {strategy.confidence})")
        logger.info(f"[DirectorAgent] Actions - Rules: {strategy.rules_action}, Cards: {strategy.cards_action}")
        if strategy.cards_to_modify:
            logger.debug(f"[DirectorAgent] Cards to modify: {strategy.cards_to_modify}")

        return strategy

    def reflect(
        self,
        experiment: RefinementExperiment,
        evaluation_before: GameEvaluation,
        evaluation_after: GameEvaluation,
    ) -> ExperimentReflection:
        """Reflect on experiment results and extract lessons."""
        logger.info(f"[DirectorAgent] Reflecting on experiment {experiment.iteration}")
        actual_change = (experiment.score_after or 0) - experiment.score_before
        logger.info(f"[DirectorAgent] Score change: {experiment.score_before:.2f} → "
                   f"{experiment.score_after:.2f} ({actual_change:+.2f})")

        # Build metric changes summary
        metric_changes = []
        metrics = ['playability', 'balance', 'clarity', 'theme_alignment', 'innovation']
        for metric in metrics:
            before = getattr(evaluation_before, metric).score
            after = getattr(evaluation_after, metric).score
            delta = after - before
            if abs(delta) >= 0.1:
                direction = "↑" if delta > 0 else "↓"
                metric_changes.append(f"- {metric.capitalize()}: {before:.1f} → {after:.1f} ({direction}{abs(delta):.1f})")

        return self.llm_service.generate(
            output_model=ExperimentReflection,
            prompt=self.REFLECT_PROMPT,
            iteration=experiment.iteration,
            hypothesis=experiment.hypothesis,
            target_metric=experiment.target_metric,
            expected_improvement=experiment.expected_improvement,
            intervention_type=experiment.intervention_type,
            rules_action=experiment.rules_changes or "none",
            rules_target=experiment.rules_changes or "N/A",
            cards_action="modified" if experiment.cards_changed else "none",
            cards_modified=", ".join(experiment.cards_changed) if experiment.cards_changed else "N/A",
            score_before=experiment.score_before,
            score_after=experiment.score_after or 0,
            actual_change=actual_change,
            metric_changes="\n".join(metric_changes) if metric_changes else "No significant metric changes",
        )

    def _format_experiments(self, experiments: List[RefinementExperiment]) -> str:
        """Format experiment history for prompt."""
        if not experiments:
            return "No previous experiments"

        lines = []
        for exp in experiments:
            status = "✓" if exp.hypothesis_confirmed else "✗" if exp.hypothesis_confirmed is False else "?"
            improvement = f"+{exp.actual_improvement:.1f}" if exp.actual_improvement and exp.actual_improvement > 0 else f"{exp.actual_improvement:.1f}" if exp.actual_improvement else "N/A"
            lines.append(
                f"[{status}] Iter {exp.iteration}: {exp.intervention_type.upper()} - "
                f"'{exp.hypothesis[:60]}...' → {improvement}"
            )
            if exp.reflection:
                lines.append(f"    Reflection: {exp.reflection[:80]}...")

        return "\n".join(lines)

    def _format_failed_patterns(self, patterns: List[FailedPattern]) -> str:
        """Format structured FailedPattern objects for prompt."""
        if not patterns:
            return "None yet"

        lines = []
        for fp in patterns:
            # Format as actionable blocked combinations
            target = f"({fp.rules_target})" if fp.rules_target else ""
            cards_info = f", cards={fp.cards_action}" if fp.cards_action != "none" else ""
            lines.append(
                f"- ⛔ BLOCKED: target_metric={fp.target_metric}, "
                f"rules_action={fp.rules_action}{target}{cards_info} "
                f"[Iter {fp.iteration}: {fp.score_before:.1f} → {fp.score_after:.1f}, "
                f"regression={fp.regression:.1f}]"
            )

        return "\n".join(lines)

    # Backward compatibility
    def analyze_and_decide(
        self,
        evaluation: GameEvaluation,
        threshold: float,
        iteration: int,
        max_iterations: int,
        cards_summary: str = "",
        previous_evaluations: Optional[List[GameEvaluation]] = None,
    ) -> RefinementStrategy:
        """Legacy method for backward compatibility."""
        return self.design_experiment(
            evaluation=evaluation,
            threshold=threshold,
            iteration=iteration,
            max_iterations=max_iterations,
            cards_summary=cards_summary,
            memory=None,
            previous_evaluations=previous_evaluations,
        )
