from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END

from deck_crafter.workflow.checkpointer import create_checkpointer

from deck_crafter.models.state import CardGameState, GameStatus
from deck_crafter.models.evaluation import GameEvaluation
from deck_crafter.services.llm_service import LLMService
from deck_crafter.agents.rules_agent import RuleGenerationAgent
from deck_crafter.agents.card_agent import CardGenerationAgent
from deck_crafter.agents.feedback_agent import FeedbackSynthesizerAgent, RefinementFeedback
from .evaluation_workflow import create_multi_agent_evaluation_workflow


class RefinementState(TypedDict):
    """State for the refinement workflow loop."""
    game_state: CardGameState
    feedback: Optional[RefinementFeedback]
    should_stop: bool


def create_refinement_workflow(llm_service: LLMService) -> StateGraph:
    """
    Creates a refinement workflow that iteratively improves game rules and cards
    based on evaluation feedback until quality threshold is met or max iterations reached.

    Loop:
      1. Check: score >= threshold OR iterations >= max? → END
      2. Synthesize feedback from evaluation
      3. Regenerate rules with critique
      4. Regenerate cards with critique
      5. Re-evaluate
      6. → Loop back to Check
    """
    # Initialize agents
    feedback_agent = FeedbackSynthesizerAgent(llm_service)
    rules_agent = RuleGenerationAgent(llm_service)
    card_agent = CardGenerationAgent(llm_service)

    # Get evaluation workflow (compiled)
    eval_workflow = create_multi_agent_evaluation_workflow(llm_service)

    def check_should_stop(state: RefinementState) -> dict:
        """Check if we should stop the refinement loop."""
        game_state = state['game_state']

        # No evaluation yet? Can't refine
        if not game_state.evaluation:
            print("--- REFINEMENT: No evaluation found, stopping ---")
            return {"should_stop": True}

        current_score = game_state.evaluation.overall_score
        threshold = game_state.evaluation_threshold
        iteration = game_state.evaluation_iteration
        max_iterations = game_state.max_evaluation_iterations

        print(f"--- REFINEMENT CHECK: Score={current_score:.1f}, Threshold={threshold}, Iteration={iteration}/{max_iterations} ---")

        # Stop if score meets threshold
        if current_score >= threshold:
            print(f"--- REFINEMENT: Score {current_score:.1f} >= {threshold}, stopping ---")
            return {"should_stop": True}

        # Stop if max iterations reached
        if iteration >= max_iterations:
            print(f"--- REFINEMENT: Max iterations ({max_iterations}) reached, stopping ---")
            return {"should_stop": True}

        return {"should_stop": False}

    def synthesize_feedback(state: RefinementState) -> dict:
        """Convert evaluation into actionable feedback for regeneration."""
        game_state = state['game_state']
        language = game_state.concept.language if game_state.concept else "English"

        print("--- REFINEMENT: Synthesizing feedback from evaluation ---")
        feedback = feedback_agent.synthesize(game_state.evaluation, game_state.cards, language)

        print(f"--- FEEDBACK PRIORITIES: {feedback.priority_issues} ---")
        return {"feedback": feedback}

    def regenerate_rules(state: RefinementState) -> dict:
        """Regenerate rules using feedback critique."""
        game_state = state['game_state']
        feedback = state['feedback']

        # Skip if rules are fine
        if feedback.rules_critique.lower() == "no changes needed":
            print("--- REFINEMENT: Rules OK, skipping regeneration ---")
            return {}

        print("--- REFINEMENT: Regenerating rules with critique ---")

        # Set critique for rules agent
        game_state.critique = feedback.rules_critique

        # Generate new rules
        result = rules_agent.generate_rules(game_state)

        if "rules" in result:
            game_state.rules = result["rules"]
            print("--- REFINEMENT: Rules regenerated ---")

        return {"game_state": game_state}

    def regenerate_cards(state: RefinementState) -> dict:
        """Regenerate all cards using feedback critique."""
        game_state = state['game_state']
        feedback = state['feedback']

        # Skip if cards are fine
        if feedback.cards_critique.lower() == "no changes needed":
            print("--- REFINEMENT: Cards OK, skipping regeneration ---")
            return {}

        print("--- REFINEMENT: Regenerating cards with critique ---")

        # Set critique and reset cards for full regeneration
        game_state.critique = feedback.cards_critique
        game_state.cards = []  # Clear to regenerate all

        # Generate cards one by one until complete
        total_cards = game_state.concept.number_of_unique_cards if game_state.concept else 0
        while len(game_state.cards) < total_cards:
            result = card_agent.generate_card(game_state)
            if "cards" in result:
                game_state.cards = result["cards"]
            else:
                break  # No more cards to generate

        print(f"--- REFINEMENT: {len(game_state.cards)} cards regenerated ---")
        return {"game_state": game_state}

    def run_evaluation(state: RefinementState) -> dict:
        """Re-evaluate the game after regeneration."""
        game_state = state['game_state']

        print("--- REFINEMENT: Re-evaluating game ---")

        # Store previous evaluation
        if game_state.previous_evaluations is None:
            game_state.previous_evaluations = []
        game_state.previous_evaluations.append(game_state.evaluation)

        # Run evaluation workflow
        eval_state = {"game_state": game_state}
        eval_result = eval_workflow.invoke(eval_state, config={"configurable": {"thread_id": "refine-eval"}})

        # Update game state with new evaluation
        game_state = eval_result['game_state']
        game_state.evaluation_iteration += 1
        game_state.status = GameStatus.EVALUATED

        new_score = game_state.evaluation.overall_score
        print(f"--- REFINEMENT: New score = {new_score:.1f} (iteration {game_state.evaluation_iteration}) ---")

        return {"game_state": game_state}

    def route_after_check(state: RefinementState) -> str:
        """Route based on should_stop flag."""
        if state.get('should_stop', False):
            return END
        return "synthesize_feedback"

    # Build the graph
    workflow = StateGraph(RefinementState)

    # Add nodes
    workflow.add_node("check_stop", check_should_stop)
    workflow.add_node("synthesize_feedback", synthesize_feedback)
    workflow.add_node("regenerate_rules", regenerate_rules)
    workflow.add_node("regenerate_cards", regenerate_cards)
    workflow.add_node("evaluate", run_evaluation)

    # Set entry point
    workflow.set_entry_point("check_stop")

    # Add edges
    workflow.add_conditional_edges(
        "check_stop",
        route_after_check,
        {
            END: END,
            "synthesize_feedback": "synthesize_feedback"
        }
    )
    workflow.add_edge("synthesize_feedback", "regenerate_rules")
    workflow.add_edge("regenerate_rules", "regenerate_cards")
    workflow.add_edge("regenerate_cards", "evaluate")
    workflow.add_edge("evaluate", "check_stop")  # Loop back

    return workflow.compile(checkpointer=create_checkpointer())
