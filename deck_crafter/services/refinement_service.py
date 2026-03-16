import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from deck_crafter.agents.director_agent import DirectorAgent
from deck_crafter.agents.feedback_agent import FeedbackSynthesizerAgent, RefinementFeedback
from deck_crafter.agents.rules_agent import RuleGenerationAgent
from deck_crafter.agents.card_agent import CardGenerationAgent
from deck_crafter.models.state import (
    CardGameState, GameStatus, FailedPattern,
    RefinementMemory, RefinementExperiment,
)
from deck_crafter.services.llm_service import LLMService
from game_simulator.integration import run_simulation_for_game
from game_simulator.analysis_agent import GameplayAnalysisAgent

logger = logging.getLogger(__name__)


# --- Helpers (relocated from game.py) ---

def determine_forced_intervention(
    state: CardGameState,
    memory: RefinementMemory,
) -> str | None:
    """Determine if we need to force escalation based on score, simulation, or failure history."""
    if not state.evaluation:
        return None

    score = state.evaluation.overall_score
    total_failed = memory.total_failed_iterations if memory else 0

    # Immediate escalation for catastrophically broken games (no failure history needed)
    if score < 3.5:
        logger.warning(f"[Escalation] Forcing NUCLEAR: catastrophic score={score:.2f}")
        return "nuclear"

    if score < 4.5:
        logger.warning(f"[Escalation] Forcing MODERATE: very low score={score:.2f}")
        return "moderate"

    # Simulation-aware escalation: game can't even be played
    if state.simulation_report and hasattr(state.simulation_report, 'completion_rate'):
        if state.simulation_report.completion_rate < 0.1:
            logger.warning(f"[Escalation] Forcing NUCLEAR: simulation completion {state.simulation_report.completion_rate:.0%}")
            return "nuclear"

    # Existing failure-count escalation
    if score < 5.0 and total_failed >= 3:
        logger.warning(f"[Escalation] Forcing NUCLEAR: score={score:.2f} < 5.0, failures={total_failed}")
        return "nuclear"

    if score < 5.0 and total_failed >= 2:
        logger.warning(f"[Escalation] Forcing MODERATE: score={score:.2f} < 5.0, failures={total_failed}")
        return "moderate"

    if memory:
        for metric, count in memory.metric_failures.items():
            if count >= 3:
                logger.warning(f"[Escalation] Forcing MODERATE: metric '{metric}' failed {count} times")
                return "moderate"

    return None


def should_stop_refinement(
    new_score: float,
    state: CardGameState,
    threshold: float,
) -> tuple[bool, str | None]:
    """
    Determine if refinement should stop.
    Returns (should_stop, reason).
    """
    if new_score >= threshold:
        logger.info(f"[StopCheck] STOP: threshold met ({new_score:.2f} >= {threshold})")
        return True, "threshold_met"

    max_iterations = state.max_evaluation_iterations or 5
    if state.evaluation_iteration >= max_iterations:
        logger.info(f"[StopCheck] STOP: max iterations reached ({state.evaluation_iteration}/{max_iterations})")
        return True, "max_iterations"

    if state.previous_evaluations and len(state.previous_evaluations) >= 2:
        last_scores = [e.overall_score for e in state.previous_evaluations[-2:]]
        if all(s >= new_score for s in last_scores):
            if new_score >= 5.5:
                logger.info(f"[StopCheck] STOP: plateau at acceptable score ({new_score:.2f} >= 5.5)")
                return True, "plateau_at_acceptable_score"
            logger.info(f"[StopCheck] CONTINUE: plateau but score too low ({new_score:.2f} < 5.5)")

    logger.debug(f"[StopCheck] CONTINUE: score={new_score:.2f}, iter={state.evaluation_iteration}/{max_iterations}")
    return False, None


# --- Core refinement step ---

@dataclass
class RefinementResult:
    state: CardGameState
    memory: RefinementMemory
    new_score: float
    previous_score: float
    status: str  # "refined" | "reverted" | "stopped" | "threshold_met"
    stop_reason: str | None
    improved: bool
    actual_improvement: float
    experiment: RefinementExperiment | None = None
    strategy: object | None = None  # RefinementStrategy (avoid circular import)
    feedback: RefinementFeedback | None = None
    rules_changed: bool = False
    cards_changed: list[str] = field(default_factory=list)


def execute_refinement_step(
    state: CardGameState,
    threshold: float,
    llm_service: LLMService,
    eval_workflow,
    num_simulation_games: int = 30,
    use_batch_cards: bool = True,
) -> RefinementResult:
    """
    Execute a single refinement iteration using the Director agent's scientific method.

    This is the core logic extracted from /refine-step. It handles:
    - Director-guided experiment design
    - Granular rules regeneration (tweak/rewrite_section/overhaul)
    - Selective card regeneration (with batch support)
    - Re-simulation and re-evaluation
    - Rollback on score regression with FailedPattern recording
    - Reflection and memory updates

    Does NOT save to DB — caller decides when to persist.
    """
    # Initialize agents
    director_agent = DirectorAgent(llm_service)
    feedback_agent = FeedbackSynthesizerAgent(llm_service)
    rules_agent = RuleGenerationAgent(llm_service)
    card_agent = CardGenerationAgent(llm_service)

    # Load or initialize refinement memory
    memory = state.refinement_memory or RefinementMemory()

    current_score = state.evaluation.overall_score
    previous_score = current_score

    # --- DEBUG: Evaluation breakdown before refinement ---
    logger.debug("=" * 80)
    logger.debug(f" REFINEMENT STEP START — Current score: {current_score:.2f}")
    scores = state.evaluation.get_scores_dict()
    for metric, score in scores.items():
        eval_obj = getattr(state.evaluation, metric, None)
        analysis = eval_obj.analysis[:120] if eval_obj and eval_obj.analysis else "no analysis"
        logger.debug(f"   {metric}: {score}/10 — {analysis}...")
    if state.evaluation.synthesized_suggestions:
        for s in (state.evaluation.synthesized_suggestions.high_priority or [])[:3]:
            logger.debug(f"   HIGH PRIORITY: {s.suggestion}")
    if state.simulation_analysis:
        sa = state.simulation_analysis
        logger.debug(f"   Simulation summary: {sa.summary[:120] if sa.summary else 'N/A'}")
        if sa.high_priority_fixes:
            for fix in sa.high_priority_fixes[:3]:
                logger.debug(f"   SIM FIX: {fix}")
        if sa.rule_clarity_issues:
            for issue in sa.rule_clarity_issues[:3]:
                logger.debug(f"   SIM RULE ISSUE: {issue}")
    logger.debug("=" * 80)

    # Store previous evaluation
    if state.previous_evaluations is None:
        state.previous_evaluations = []
    state.previous_evaluations.append(state.evaluation)
    evaluation_before = state.evaluation

    # Snapshot for rollback
    previous_state_json = state.model_dump_json()

    # --- Director designs experiment ---
    max_iterations = state.max_evaluation_iterations or 5
    cards_summary = ""
    if state.cards:
        cards_summary = "\n".join([
            f"- {c.name} (Type: {c.type}, Qty: {c.quantity}): {c.description[:80]}..."
            if len(c.description) > 80 else f"- {c.name} (Type: {c.type}, Qty: {c.quantity}): {c.description}"
            for c in state.cards
        ])

    if memory.failed_patterns:
        logger.info(f"Memory has {len(memory.failed_patterns)} structured failed patterns")

    forced_intervention = determine_forced_intervention(state, memory)
    if forced_intervention:
        logger.warning(f"ESCALATION: Forcing {forced_intervention.upper()} intervention "
                      f"(score={current_score:.1f}, failures={memory.total_failed_iterations})")

    blocked_metrics = memory.get_blocked_metrics()
    if blocked_metrics:
        logger.info(f"Blocked metrics (ceiling detected): {blocked_metrics}")

    strategy = director_agent.design_experiment(
        evaluation=state.evaluation,
        threshold=threshold,
        iteration=state.evaluation_iteration + 1,
        max_iterations=max_iterations,
        cards_summary=cards_summary,
        memory=memory,
        previous_evaluations=state.previous_evaluations[:-1] if len(state.previous_evaluations) > 1 else None,
        simulation_analysis=state.simulation_analysis,
        forced_intervention=forced_intervention,
        blocked_metrics=blocked_metrics,
    )

    if memory.check_pattern_blocked(strategy.target_metric, strategy.rules_action, strategy.rules_target):
        logger.warning("Strategy matches failed pattern - Director should have avoided this!")

    # Code-level override: enforce minimum action levels for forced interventions
    if forced_intervention == "nuclear":
        if strategy.rules_action in ("tweak", "none"):
            logger.warning(f"Overriding rules_action '{strategy.rules_action}' → 'overhaul' (forced NUCLEAR)")
            strategy.rules_action = "overhaul"
        if strategy.cards_action in ("stat_adjust", "none"):
            logger.warning(f"Overriding cards_action '{strategy.cards_action}' → 'regenerate_many' (forced NUCLEAR)")
            strategy.cards_action = "regenerate_many"
        if strategy.intervention_type == "surgical":
            strategy.intervention_type = "nuclear"
    elif forced_intervention == "moderate":
        if strategy.rules_action in ("tweak", "none"):
            logger.warning(f"Overriding rules_action '{strategy.rules_action}' → 'rewrite_section' (forced MODERATE)")
            strategy.rules_action = "rewrite_section"
        if strategy.cards_action == "none":
            logger.warning(f"Overriding cards_action 'none' → 'regenerate_few' (forced MODERATE)")
            strategy.cards_action = "regenerate_few"
        if strategy.intervention_type == "surgical":
            strategy.intervention_type = "moderate"

    logger.info(f"Director experiment: {strategy.intervention_type.upper()} targeting {strategy.target_metric}")
    logger.info(f"Hypothesis: {strategy.hypothesis}")
    logger.info(f"Expected improvement: +{strategy.expected_improvement} (confidence: {strategy.confidence})")
    logger.info(f"Rules: {strategy.rules_action} | Cards: {strategy.cards_action}")

    # --- Synthesize feedback ---
    language = state.concept.language if state.concept else "English"
    feedback = feedback_agent.synthesize(state.evaluation, state.cards, language, strategy=strategy)

    logger.debug(f" FEEDBACK — Priority issues: {feedback.priority_issues}")
    logger.debug(f" FEEDBACK — Rules critique ({len(feedback.rules_critique)} chars): {feedback.rules_critique[:200]}")
    logger.debug(f" FEEDBACK — Cards critique ({len(feedback.cards_critique)} chars): {feedback.cards_critique[:200]}")
    logger.debug(f" FEEDBACK — Cards to regenerate: {feedback.cards_to_regenerate}")

    rules_changed = False
    cards_changed = []

    # --- Regenerate rules (granular) ---
    should_refine_rules = strategy.rules_action != "none"
    director_commands_rules = strategy.rules_action in ("rewrite_section", "overhaul")
    feedback_approves_rules = feedback.rules_critique.lower() != "no changes needed"
    if should_refine_rules and (director_commands_rules or feedback_approves_rules):
        # If feedback said "no changes" but Director explicitly commands rewrite/overhaul, override
        if director_commands_rules and not feedback_approves_rules:
            logger.warning(f"Feedback said 'no changes needed' but Director commands {strategy.rules_action} — overriding")
            state.critique = strategy.rules_instruction or f"Director mandates {strategy.rules_action} on {strategy.rules_target}"
        else:
            state.critique = feedback.rules_critique
        logger.info(f"Refining rules: {strategy.rules_action} on {strategy.rules_target}")
        if strategy.rules_instruction:
            state.critique += f"\n\nDIRECTOR INSTRUCTION: {strategy.rules_instruction}"

        logger.debug(f" RULES CRITIQUE sent to agent ({len(state.critique)} chars):\n{state.critique[:500]}")

        # Snapshot rules before
        rules_before_wc = state.rules.win_conditions if state.rules else []
        rules_before_phases = len(state.rules.turn_structure) if state.rules else 0
        if strategy.rules_action == "tweak":
            logger.info("Using ADDITIVE rules enhancement (glossary, examples, FAQ)")
            result = rules_agent.enhance_rules(state)
        elif strategy.rules_action == "rewrite_section" and strategy.rules_target:
            logger.info(f"Using SECTION-SPECIFIC rewrite for: {strategy.rules_target}")
            result = rules_agent.rewrite_section(state, strategy.rules_target)
        else:
            logger.info("Using FULL rules regeneration (overhaul)")
            result = rules_agent.generate_rules(state)

        if "rules" in result:
            state.rules = result["rules"]
            rules_changed = True
            # Log what changed
            rules_after_wc = state.rules.win_conditions if state.rules else []
            rules_after_phases = len(state.rules.turn_structure) if state.rules else 0
            logger.debug(f" RULES CHANGED — Phases: {rules_before_phases} → {rules_after_phases}")
            logger.debug(f" RULES CHANGED — Win conditions before: {rules_before_wc}")
            logger.debug(f" RULES CHANGED — Win conditions after: {rules_after_wc}")
            if state.rules.resource_mechanics:
                rm = state.rules.resource_mechanics
                if isinstance(rm, str):
                    logger.debug(f" RULES CHANGED — Resources: {rm[:150]}")
                else:
                    logger.debug(f" RULES CHANGED — Resources: start={rm.starting_resources}, per_turn={rm.per_turn_gain}")
            if state.rules.turn_structure:
                for phase in state.rules.turn_structure:
                    desc = phase.description[:100] if hasattr(phase, 'description') else str(phase)[:100]
                    name = phase.phase_name if hasattr(phase, 'phase_name') else "?"
                    logger.debug(f" RULES PHASE: {name} — {desc}")
        else:
            logger.warning("Rules agent returned NO rules!")
    elif strategy.rules_action == "none":
        logger.info("Skipping rules refinement (Director: rules_action=none)")

    # --- Selective card regeneration ---
    should_refine_cards = strategy.cards_action != "none"
    director_commands_cards = strategy.cards_action in ("regenerate_few", "regenerate_many")
    feedback_approves_cards = feedback.cards_critique.lower() != "no changes needed"
    if should_refine_cards and (director_commands_cards or feedback_approves_cards):
        # If feedback said "no changes" but Director explicitly commands regeneration, override
        if director_commands_cards and not feedback_approves_cards:
            logger.warning(f"Feedback said 'no changes needed' but Director commands {strategy.cards_action} — overriding")
            state.critique = strategy.cards_instruction or f"Director mandates {strategy.cards_action}"
        else:
            state.critique = feedback.cards_critique
        logger.info(f"Refining cards: {strategy.cards_action}")
        if strategy.cards_instruction:
            state.critique += f"\n\nDIRECTOR INSTRUCTION: {strategy.cards_instruction}"

        logger.debug(f" CARDS CRITIQUE sent to agent ({len(state.critique)} chars):\n{state.critique[:500]}")

        cards_to_remove = set(strategy.cards_to_modify) if strategy.cards_to_modify else set(feedback.cards_to_regenerate or [])

        if cards_to_remove:
            original_count = len(state.cards) if state.cards else 0
            state.cards = [c for c in (state.cards or []) if c.name not in cards_to_remove]
            cards_removed = original_count - len(state.cards)
            cards_changed = list(cards_to_remove)
            logger.info(f"Keeping {len(state.cards)} cards, regenerating {cards_removed}: {cards_to_remove}")

            for card_name in cards_to_remove:
                memory.problematic_cards[card_name] = memory.problematic_cards.get(card_name, 0) + 1
        else:
            if strategy.cards_action == "regenerate_many":
                logger.warning("No specific cards listed with regenerate_many, regenerating all")
                cards_changed = [c.name for c in (state.cards or [])]
                state.cards = []
            else:
                logger.info("No specific cards to modify")

        # Regenerate missing cards
        total_cards = state.concept.number_of_unique_cards if state.concept else 0
        if use_batch_cards:
            while len(state.cards) < total_cards:
                result = card_agent.generate_cards_batch(state)
                if not result or "cards" not in result:
                    break
                state.cards = result["cards"]
        else:
            while len(state.cards) < total_cards:
                result = card_agent.generate_card(state)
                if "cards" in result:
                    state.cards = result["cards"]
                else:
                    break
        # Log regenerated cards
        if cards_changed:
            logger.debug(f" CARDS — Removed: {cards_changed}")
            logger.debug(f" CARDS — Total after regen: {len(state.cards or [])}")
            # Log a sample of card stats for balance check
            for c in (state.cards or [])[-5:]:
                logger.debug(f" CARD SAMPLE: {c.name} | cost={c.cost} | effect={c.effect_type}:{c.effect_value} | desc={c.description[:80]}")
    elif strategy.cards_action == "none":
        logger.info("Skipping cards refinement (Director: cards_action=none)")

    # --- Re-simulate if needed ---
    if state.simulation_analysis is not None and (rules_changed or cards_changed):
        logger.info("Re-running simulation to validate refinement changes...")
        try:
            report, warnings = run_simulation_for_game(
                rules=state.rules,
                cards=state.cards,
                game_name=state.concept.title if state.concept else "Game",
                num_games=num_simulation_games,
                seed=42,
                llm_service=llm_service,
                use_cache=False,
            )

            analysis_agent = GameplayAnalysisAgent(llm_service)
            new_analysis = analysis_agent.analyze(report, language=language)

            prev_issues = len(state.simulation_analysis.high_priority_fixes) if state.simulation_analysis.high_priority_fixes else 0
            new_issues = len(new_analysis.high_priority_fixes) if new_analysis.high_priority_fixes else 0
            logger.info(f"Simulation validated: {prev_issues} issues before → {new_issues} after")

            state.simulation_analysis = new_analysis
            state.simulation_report = report
            state.compilation_warnings = warnings
            # Debug simulation results
            if report:
                logger.debug(f" SIMULATION — Games: {report.games_run}, "
                           f"Completion: {report.completion_rate:.0%}, "
                           f"Avg turns: {report.avg_turns:.1f}")
                if hasattr(report, 'common_errors') and report.common_errors:
                    for err, count in list(report.common_errors.items())[:5]:
                        logger.debug(f" SIM ERROR: {err} (x{count})")
            if warnings:
                logger.debug(f" COMPILATION WARNINGS: {warnings[:3]}")
        except Exception as e:
            logger.warning(f"Validation simulation failed: {e}, proceeding with cached analysis")

    # --- Re-evaluate ---
    logger.info("Re-evaluating game after refinement...")
    eval_state = {"game_state": state}
    eval_result = eval_workflow.invoke(
        eval_state,
        config={"configurable": {"thread_id": f"refine-eval-{state.game_id}"}}
    )
    state = eval_result['game_state']
    state.evaluation_iteration += 1
    state.evaluation_threshold = threshold
    state.status = GameStatus.EVALUATED
    state.updated_at = datetime.now(timezone.utc)

    new_score = state.evaluation.overall_score
    improved = new_score > current_score
    actual_improvement = new_score - current_score

    logger.info(f"Evaluation complete: {current_score:.2f} → {new_score:.2f} "
               f"({actual_improvement:+.2f}, {'improved' if improved else 'regressed'})")
    # Debug: per-metric comparison
    scores_before = evaluation_before.get_scores_dict()
    scores_after = state.evaluation.get_scores_dict()
    logger.debug("METRIC COMPARISON (before → after):")
    for metric in scores_after:
        b = scores_before.get(metric, 0)
        a = scores_after[metric]
        logger.debug(f"   {metric}: {b:.1f} → {a:.1f} ({a - b:+.1f})")

    # Update best score
    if state.best_score_achieved is None or new_score > state.best_score_achieved:
        state.best_score_achieved = new_score
        logger.info(f"New best score achieved: {new_score:.2f}")

    # --- Create experiment record ---
    experiment = RefinementExperiment(
        iteration=state.evaluation_iteration,
        hypothesis=strategy.hypothesis,
        target_metric=strategy.target_metric,
        expected_improvement=strategy.expected_improvement,
        intervention_type=strategy.intervention_type,
        rules_changes=strategy.rules_target if rules_changed else None,
        cards_changed=cards_changed,
        score_before=previous_score,
        score_after=new_score,
        actual_improvement=actual_improvement,
    )

    # --- Reflect and update memory ---
    if actual_improvement >= strategy.expected_improvement * 0.5:
        experiment.hypothesis_confirmed = True
    else:
        experiment.hypothesis_confirmed = False

    try:
        reflection = director_agent.reflect(experiment, evaluation_before, state.evaluation)
        experiment.reflection = reflection.lesson_learned

        memory.lessons_learned.append(reflection.lesson_learned)
        if reflection.should_continue_pattern:
            pattern = f"{strategy.intervention_type}: {strategy.hypothesis[:50]}"
            if pattern not in memory.successful_patterns:
                memory.successful_patterns.append(pattern)
            memory.record_success(strategy.target_metric, new_score, state.evaluation_iteration)
        if reflection.pattern_to_avoid:
            memory.failed_pattern_strings.append(reflection.pattern_to_avoid)
    except Exception as e:
        logger.warning(f"Reflection failed: {e}")
        experiment.reflection = f"Improvement: {actual_improvement:+.1f} (expected {strategy.expected_improvement:+.1f})"

    memory.experiments.append(experiment)

    # --- Rollback if score degraded ---
    status = "refined"
    stop_reason = None

    if new_score < previous_score - 0.1:
        logger.warning(f"ROLLBACK: Score degraded ({previous_score:.1f} → {new_score:.1f})")

        failed_pattern = FailedPattern(
            iteration=state.evaluation_iteration,
            target_metric=strategy.target_metric,
            intervention_type=strategy.intervention_type,
            rules_action=strategy.rules_action,
            rules_target=strategy.rules_target,
            cards_action=strategy.cards_action,
            cards_affected=cards_changed if isinstance(cards_changed, list) else [],
            score_before=previous_score,
            score_after=new_score,
            regression=new_score - previous_score,
        )
        memory.add_failed_pattern(failed_pattern)
        memory.record_failure(strategy.target_metric)

        experiment.hypothesis_confirmed = False
        experiment.reflection = f"FAILED: Caused score regression from {previous_score:.1f} to {new_score:.1f}"

        # Rollback state but preserve memory
        state = CardGameState.model_validate_json(previous_state_json)
        state.evaluation_iteration += 1
        state.refinement_memory = memory
        new_score = previous_score
        actual_improvement = 0.0
        improved = False
        status = "reverted"
    else:
        # Check stop conditions
        stop, reason = should_stop_refinement(new_score, state, threshold)
        if stop:
            status = "threshold_met" if reason == "threshold_met" else "stopped"
            stop_reason = reason

    state.refinement_memory = memory

    return RefinementResult(
        state=state,
        memory=memory,
        new_score=new_score,
        previous_score=previous_score,
        status=status,
        stop_reason=stop_reason,
        improved=improved,
        actual_improvement=actual_improvement,
        experiment=experiment,
        strategy=strategy,
        feedback=feedback,
        rules_changed=rules_changed,
        cards_changed=cards_changed,
    )
