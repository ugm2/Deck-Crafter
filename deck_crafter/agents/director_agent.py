from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from deck_crafter.services.llm_service import LLMService
from deck_crafter.models.evaluation import GameEvaluation
from deck_crafter.models.state import RefinementMemory, RefinementExperiment


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
    intervention_type: Literal["surgical", "moderate", "nuclear"] = Field(
        ...,
        description="surgical: 1-2 minimal changes to test one hypothesis. "
                    "moderate: 3-5 coordinated changes. "
                    "nuclear: significant redesign (ONLY if 2+ moderate attempts failed)."
    )

    # Granular rules control
    rules_action: Literal["none", "tweak", "rewrite_section", "overhaul"] = Field(
        ...,
        description="none: don't touch rules. tweak: minor wording/number changes. "
                    "rewrite_section: rewrite one section (e.g., win_conditions). "
                    "overhaul: major rules restructure (nuclear only)."
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
    cards_action: Literal["none", "stat_adjust", "regenerate_few", "regenerate_many"] = Field(
        ...,
        description="none: don't touch cards. stat_adjust: tweak stats without regeneration. "
                    "regenerate_few: regenerate 1-3 specific cards. "
                    "regenerate_many: regenerate 4+ cards (moderate/nuclear only)."
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

### PROBLEMATIC CARDS (RECURRING ISSUES) ###
{problematic_cards}

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
- Surgical interventions first, escalate only when needed
- Use EXACT card names from the list above
- Be SPECIFIC with numbers and targets
- After 2+ rollbacks: STOP targeting the same metric, try something else entirely

### DECISION GUIDELINES ###
- Score < 4: Consider moderate/nuclear intervention
- Score 4-6: Surgical or moderate based on history
- Score 6-7: Surgical only, fine-tuning
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
    ) -> RefinementStrategy:
        """Design the next refinement experiment using scientific method."""
        gap = threshold - evaluation.overall_score
        remaining = max_iterations - iteration

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
            failed_patterns = "\n".join(f"- {p}" for p in memory.failed_patterns[-5:]) or "None yet"
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

        return self.llm_service.generate(
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
        )

    def reflect(
        self,
        experiment: RefinementExperiment,
        evaluation_before: GameEvaluation,
        evaluation_after: GameEvaluation,
    ) -> ExperimentReflection:
        """Reflect on experiment results and extract lessons."""
        actual_change = (experiment.score_after or 0) - experiment.score_before

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
