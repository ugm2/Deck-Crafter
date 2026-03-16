import logging
from typing import TypedDict
from langgraph.graph import StateGraph, END

from deck_crafter.workflow.checkpointer import create_checkpointer

logger = logging.getLogger(__name__)

from deck_crafter.models.state import CardGameState, GameStatus
from deck_crafter.models.evaluation import (
    BalanceEvaluation,
    ClarityEvaluation,
    PlayabilityEvaluation,
    ThemeAlignmentEvaluation,
    InnovationEvaluation,
    GameEvaluation
)
from deck_crafter.services.llm_service import LLMService
from deck_crafter.agents.evaluation_agents import (
    BalanceAgent,
    ClarityAgent,
    PlayabilityAgent,
    ThemeAlignmentAgent,
    InnovationAgent,
    EvaluationSynthesizerAgent,
    CrossMetricReviewAgent,
    SuggestionSynthesizerAgent,
)


class EvaluationState(TypedDict):
    """State for the evaluation workflow with 5 metrics."""
    game_state: CardGameState
    balance_eval: BalanceEvaluation
    clarity_eval: ClarityEvaluation
    playability_eval: PlayabilityEvaluation
    theme_alignment_eval: ThemeAlignmentEvaluation
    innovation_eval: InnovationEvaluation
    final_evaluation: GameEvaluation

def create_multi_agent_evaluation_workflow(llm_service: LLMService) -> StateGraph:
    """
    Creates a two-pass evaluation workflow:
    Pass 1: 5 evaluation agents work in parallel
    Pass 2: Cross-metric review where agents can adjust scores ±0.5
    Finally: Synthesizer combines results with weighted scoring
    """
    # Instantiate all evaluation agents (5 metrics)
    balance_agent = BalanceAgent(llm_service)
    clarity_agent = ClarityAgent(llm_service)
    playability_agent = PlayabilityAgent(llm_service)
    theme_alignment_agent = ThemeAlignmentAgent(llm_service)
    innovation_agent = InnovationAgent(llm_service)
    cross_metric_review_agent = CrossMetricReviewAgent(llm_service)
    synthesizer_agent = EvaluationSynthesizerAgent(llm_service)
    suggestion_synthesizer = SuggestionSynthesizerAgent(llm_service)

    # --- PASS 1: Parallel Evaluation Nodes ---
    def run_balance_eval(state: EvaluationState):
        game = state['game_state']
        logger.info(f"[EvalWorkflow] Running BALANCE evaluation (has simulation: {game.simulation_analysis is not None})")
        result = balance_agent.evaluate(
            game.concept, game.rules, game.cards, game.concept.language,
            simulation_analysis=game.simulation_analysis
        )
        logger.info(f"[EvalWorkflow] BALANCE score: {result.score:.2f}")
        return {"balance_eval": result}

    def run_clarity_eval(state: EvaluationState):
        game = state['game_state']
        logger.info(f"[EvalWorkflow] Running CLARITY evaluation (warnings: {len(game.compilation_warnings)})")
        result = clarity_agent.evaluate(
            game.concept, game.rules, game.cards, game.concept.language,
            simulation_analysis=game.simulation_analysis,
            compilation_warnings=game.compilation_warnings
        )
        logger.info(f"[EvalWorkflow] CLARITY score: {result.score:.2f}")
        return {"clarity_eval": result}

    def run_playability_eval(state: EvaluationState):
        game = state['game_state']
        logger.info(f"[EvalWorkflow] Running PLAYABILITY evaluation (has simulation: {game.simulation_analysis is not None})")
        result = playability_agent.evaluate(
            game.concept, game.rules, game.cards, game.concept.language,
            simulation_analysis=game.simulation_analysis
        )
        logger.info(f"[EvalWorkflow] PLAYABILITY score: {result.score:.2f}")
        return {"playability_eval": result}

    def run_theme_alignment_eval(state: EvaluationState):
        game = state['game_state']
        logger.info(f"[EvalWorkflow] Running THEME_ALIGNMENT evaluation")
        result = theme_alignment_agent.evaluate(
            game.preferences, game.concept, game.rules, game.cards, game.concept.language,
            simulation_analysis=game.simulation_analysis
        )
        logger.info(f"[EvalWorkflow] THEME_ALIGNMENT score: {result.score:.2f}")
        return {"theme_alignment_eval": result}

    def run_innovation_eval(state: EvaluationState):
        game = state['game_state']
        logger.info(f"[EvalWorkflow] Running INNOVATION evaluation")
        result = innovation_agent.evaluate(
            game.concept, game.rules, game.cards, game.concept.language,
            simulation_analysis=game.simulation_analysis
        )
        logger.info(f"[EvalWorkflow] INNOVATION score: {result.score:.2f}")
        return {"innovation_eval": result}

    # --- PASS 2: Cross-Metric Review Node ---
    def run_cross_metric_review(state: EvaluationState):
        """Second pass: each metric sees others' scores and can adjust ±1.0"""
        logger.info("[EvalWorkflow] Starting PASS 2: Cross-metric review")
        language = state['game_state'].concept.language

        # Collect all pass 1 scores
        all_scores = {
            "balance": state['balance_eval'].score,
            "clarity": state['clarity_eval'].score,
            "playability": state['playability_eval'].score,
            "theme_alignment": state['theme_alignment_eval'].score,
            "innovation": state['innovation_eval'].score,
        }
        logger.info(f"[EvalWorkflow] Pass 1 scores: {all_scores}")

        # Review each metric (run in sequence to avoid race conditions)
        metrics = [
            ("balance", state['balance_eval']),
            ("clarity", state['clarity_eval']),
            ("playability", state['playability_eval']),
            ("theme_alignment", state['theme_alignment_eval']),
            ("innovation", state['innovation_eval']),
        ]

        for metric_name, eval_obj in metrics:
            try:
                adjustment = cross_metric_review_agent.review(
                    metric_name=metric_name,
                    original_eval=eval_obj,
                    all_scores=all_scores,
                    language=language,
                )
                # Apply adjustment (asymmetric: up to +1.0, down to -0.25 only)
                delta = adjustment.adjusted_score - adjustment.original_score
                clamped_delta = max(-0.25, min(1.0, delta))
                eval_obj.adjusted_score = eval_obj.score + clamped_delta
                eval_obj.adjustment_reason = adjustment.adjustment_reason
                if abs(clamped_delta) > 0.01:
                    logger.info(f"[EvalWorkflow] {metric_name.upper()} adjusted: "
                               f"{eval_obj.score:.2f} → {eval_obj.adjusted_score:.2f} "
                               f"({clamped_delta:+.2f}): {adjustment.adjustment_reason[:50]}...")
            except Exception as e:
                # If review fails, keep original score
                logger.warning(f"[EvalWorkflow] Cross-metric review failed for {metric_name}: {e}")
                eval_obj.adjusted_score = float(eval_obj.score)
                eval_obj.adjustment_reason = "Review skipped"

        return {
            "balance_eval": state['balance_eval'],
            "clarity_eval": state['clarity_eval'],
            "playability_eval": state['playability_eval'],
            "theme_alignment_eval": state['theme_alignment_eval'],
            "innovation_eval": state['innovation_eval'],
        }

    # --- Final Synthesis Node ---
    def run_synthesis(state: EvaluationState):
        logger.info("[EvalWorkflow] Running final synthesis")
        language = state['game_state'].concept.language

        # First, create the evaluation
        final_evaluation = synthesizer_agent.synthesize(
            balance_eval=state['balance_eval'],
            clarity_eval=state['clarity_eval'],
            playability_eval=state['playability_eval'],
            theme_alignment_eval=state['theme_alignment_eval'],
            innovation_eval=state['innovation_eval'],
            language=language,
        )
        logger.info(f"[EvalWorkflow] Final weighted score: {final_evaluation.overall_score:.2f}")

        # Then, synthesize and deduplicate suggestions
        try:
            synthesized = suggestion_synthesizer.synthesize(
                evaluation=final_evaluation,
                language=language,
            )
            final_evaluation.synthesized_suggestions = synthesized
            total = len(synthesized.high_priority) + len(synthesized.medium_priority) + len(synthesized.low_priority)
            logger.info(f"[EvalWorkflow] Synthesized {total} suggestions (high: {len(synthesized.high_priority)}, "
                       f"med: {len(synthesized.medium_priority)}, low: {len(synthesized.low_priority)})")
        except Exception as e:
            # If synthesis fails, continue without it
            logger.warning(f"[EvalWorkflow] Suggestion synthesis failed: {e}")

        game_state = state['game_state']
        game_state.evaluation = final_evaluation
        game_state.status = GameStatus.EVALUATED
        logger.info(f"[EvalWorkflow] Evaluation complete. Overall: {final_evaluation.overall_score:.2f}/10")
        return {"final_evaluation": final_evaluation, "game_state": game_state}

    # --- Assemble the Graph ---
    workflow = StateGraph(EvaluationState)

    # Pass 1: Add all evaluation nodes (5 metrics in parallel)
    workflow.add_node("balance", run_balance_eval)
    workflow.add_node("clarity", run_clarity_eval)
    workflow.add_node("playability", run_playability_eval)
    workflow.add_node("theme_alignment", run_theme_alignment_eval)
    workflow.add_node("innovation", run_innovation_eval)

    # Pass 2: Cross-metric review (after all pass 1 complete)
    workflow.add_node("cross_metric_review", run_cross_metric_review)

    # Final synthesis
    workflow.add_node("synthesize", run_synthesis)

    # Entry point
    workflow.add_node("start_evaluation", lambda state: state)
    workflow.set_entry_point("start_evaluation")

    # Connect entry to all parallel pass 1 nodes
    workflow.add_edge("start_evaluation", "balance")
    workflow.add_edge("start_evaluation", "clarity")
    workflow.add_edge("start_evaluation", "playability")
    workflow.add_edge("start_evaluation", "theme_alignment")
    workflow.add_edge("start_evaluation", "innovation")

    # Connect all pass 1 nodes to cross-metric review (pass 2)
    workflow.add_edge(["balance", "clarity", "playability", "theme_alignment", "innovation"], "cross_metric_review")

    # Connect pass 2 to synthesis
    workflow.add_edge("cross_metric_review", "synthesize")

    # End after synthesis
    workflow.add_edge("synthesize", END)

    return workflow.compile(checkpointer=create_checkpointer()) 