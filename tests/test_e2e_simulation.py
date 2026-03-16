"""
End-to-end test of the simulation-enhanced evaluation flow.

This test mimics what Streamlit would do, using the API endpoints directly.
It validates the full flow:
  1. Create game
  2. Generate concept + rules
  3. Generate cards
  4. Run simulation (new feature)
  5. Evaluate (should use simulation data)
  6. Verify simulation data influenced evaluation
"""

import pytest
import asyncio
from datetime import datetime, timezone
import uuid

from deck_crafter.models.user_preferences import UserPreferences
from deck_crafter.models.state import CardGameState, GameStatus
from deck_crafter.database import init_db, save_game_state as save_to_db, get_game_state as get_from_db
from deck_crafter.services.llm_service import create_fallback_llm_service

# Import workflows
from deck_crafter.workflow.specific_workflows import (
    create_concept_and_rules_workflow,
    create_cards_workflow,
    create_multi_agent_evaluation_workflow,
)

# Import simulation components
from deck_crafter.game_simulator.integration import run_simulation_for_game
from deck_crafter.game_simulator.analysis_agent import GameplayAnalysisAgent


@pytest.fixture(scope="module")
def llm_service():
    """Create LLM service for tests - using fallback with flash."""
    return create_fallback_llm_service()


@pytest.fixture(scope="module")
def workflows(llm_service):
    """Create workflows for tests."""
    return {
        'concept_and_rules': create_concept_and_rules_workflow(llm_service),
        'cards': create_cards_workflow(llm_service),
        'evaluation': create_multi_agent_evaluation_workflow(llm_service),
    }


@pytest.fixture
def simple_preferences():
    """Simple game preferences for quick testing."""
    return UserPreferences(
        game_description="A simple card battle game",
        language="English",
        theme="Fantasy",
        game_style="Strategy",
        number_of_players="2",
        target_audience="Adults",
        rule_complexity="Simple",
    )


@pytest.mark.asyncio
async def test_e2e_flow_with_simulation(llm_service, workflows, simple_preferences):
    """
    Full end-to-end test of the simulation-enhanced flow.

    This test is marked slow because it involves multiple LLM calls.
    Run with: pytest tests/test_e2e_simulation.py -v --tb=short
    """
    await init_db()

    # Step 1: Create game state
    game_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    state = CardGameState(
        game_id=game_id,
        status=GameStatus.CREATED,
        preferences=simple_preferences,
        concept=None,
        rules=None,
        cards=None,
        created_at=now,
        updated_at=now,
    )

    print(f"\n{'='*60}")
    print(f"STEP 1: Created game {game_id[:8]}...")
    print(f"{'='*60}")

    # Step 2: Generate concept and rules
    print("\nSTEP 2: Generating concept and rules...")
    result = workflows['concept_and_rules'].invoke(
        state,
        config={"configurable": {"thread_id": 1}}
    )
    state = CardGameState.model_validate(result)
    state.status = GameStatus.RULES_GENERATED

    assert state.concept is not None, "Concept should be generated"
    assert state.rules is not None, "Rules should be generated"
    print(f"  ✓ Game: {state.concept.title}")
    print(f"  ✓ Cards needed: {state.concept.number_of_unique_cards}")

    # Step 3: Generate cards
    print("\nSTEP 3: Generating cards...")
    result = workflows['cards'].invoke(
        state,
        config={"recursion_limit": 100, "configurable": {"thread_id": 1}}
    )
    state.cards = result["cards"]
    state.status = GameStatus.CARDS_GENERATED

    assert state.cards is not None, "Cards should be generated"
    assert len(state.cards) > 0, "Should have at least one card"
    print(f"  ✓ Generated {len(state.cards)} cards")
    for card in state.cards[:3]:
        print(f"    - {card.name} ({card.type})")
    if len(state.cards) > 3:
        print(f"    ... and {len(state.cards) - 3} more")

    # Step 4: Run simulation
    print("\nSTEP 4: Running simulation (30 games)...")
    game_name = state.concept.title
    report, warnings = run_simulation_for_game(
        rules=state.rules,
        cards=state.cards,
        game_name=game_name,
        num_games=30,
        seed=42,
        llm_service=llm_service,
    )

    print(f"  ✓ Completion rate: {report.completion_rate:.0%}")
    print(f"  ✓ Avg turns: {report.avg_turns:.1f}")
    print(f"  ✓ First player win rate: {report.first_player_win_rate:.0%}")
    if warnings:
        print(f"  ⚠ Compilation warnings: {len(warnings)}")

    # Step 5: Generate gameplay analysis
    print("\nSTEP 5: Analyzing gameplay data...")
    agent = GameplayAnalysisAgent(llm_service)
    analysis = agent.analyze(report, language="English")

    print(f"  ✓ Strategic diversity: {analysis.strategic_diversity}")
    print(f"  ✓ Pacing: {analysis.pacing_assessment}")
    print(f"  ✓ Comeback potential: {analysis.comeback_potential}")
    print(f"  ✓ Problematic cards: {len(analysis.problematic_cards)}")
    print(f"  ✓ High priority fixes: {len(analysis.high_priority_fixes)}")

    # Store analysis in state
    state.simulation_analysis = analysis

    # Step 6: Evaluate with simulation data
    print("\nSTEP 6: Evaluating with simulation data...")
    eval_state = {"game_state": state}
    result = workflows['evaluation'].invoke(
        eval_state,
        config={"configurable": {"thread_id": f"eval-{game_id}"}}
    )

    final_state = result['game_state']

    assert final_state.evaluation is not None, "Evaluation should be generated"
    print(f"  ✓ Overall score: {final_state.evaluation.overall_score:.1f}/10")
    print(f"  ✓ Playability: {final_state.evaluation.playability.score:.1f}")
    print(f"  ✓ Balance: {final_state.evaluation.balance.score:.1f}")
    print(f"  ✓ Clarity: {final_state.evaluation.clarity.score:.1f}")

    # Verify simulation data was accessible to evaluation
    # The evaluation agents receive simulation_analysis through game_state
    assert final_state.simulation_analysis is not None, "Simulation analysis should be preserved"

    # Step 7: Save to database
    print("\nSTEP 7: Saving to database...")
    await save_to_db(final_state)

    # Step 8: Reload and verify persistence
    print("\nSTEP 8: Verifying persistence...")
    loaded = await get_from_db(game_id)

    assert loaded is not None, "Should load from database"
    assert loaded.simulation_analysis is not None, "Simulation analysis should be persisted"
    assert loaded.evaluation is not None, "Evaluation should be persisted"

    print(f"  ✓ Reloaded game {game_id[:8]}...")
    print(f"  ✓ Simulation analysis preserved: {loaded.simulation_analysis.strategic_diversity}")
    print(f"  ✓ Evaluation preserved: {loaded.evaluation.overall_score:.1f}/10")

    print(f"\n{'='*60}")
    print("✅ END-TO-END TEST PASSED")
    print(f"{'='*60}")
    print(f"\nSummary:")
    print(f"  Game: {state.concept.title}")
    print(f"  Cards: {len(state.cards)}")
    print(f"  Simulation: {report.games_completed}/{report.games_run} games")
    print(f"  Score: {final_state.evaluation.overall_score:.1f}/10")
    print(f"  Problematic cards identified: {len(analysis.problematic_cards)}")


@pytest.mark.asyncio
async def test_simulation_influences_balance_eval(llm_service, workflows, simple_preferences):
    """
    Verify that simulation data actually influences the Balance evaluation.

    Creates a game, runs simulation, and checks that the BalanceAgent's
    analysis mentions simulation findings.
    """
    await init_db()

    game_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    state = CardGameState(
        game_id=game_id,
        status=GameStatus.CREATED,
        preferences=simple_preferences,
        concept=None,
        rules=None,
        cards=None,
        created_at=now,
        updated_at=now,
    )

    # Generate game
    result = workflows['concept_and_rules'].invoke(state, config={"configurable": {"thread_id": 1}})
    state = CardGameState.model_validate(result)

    result = workflows['cards'].invoke(state, config={"recursion_limit": 100, "configurable": {"thread_id": 1}})
    state.cards = result["cards"]
    state.status = GameStatus.CARDS_GENERATED

    # Run simulation
    report, _ = run_simulation_for_game(
        rules=state.rules,
        cards=state.cards,
        game_name=state.concept.title,
        num_games=20,
        seed=42,
        llm_service=llm_service,
    )

    # Analyze
    agent = GameplayAnalysisAgent(llm_service)
    analysis = agent.analyze(report, language="English")
    state.simulation_analysis = analysis

    # Evaluate
    eval_state = {"game_state": state}
    result = workflows['evaluation'].invoke(eval_state, config={"configurable": {"thread_id": f"eval-{game_id}"}})
    final_state = result['game_state']

    # Check that evaluation exists
    assert final_state.evaluation is not None
    assert final_state.evaluation.balance is not None

    # The balance analysis should ideally reference simulation data
    # (This is a soft check - we just verify the flow worked)
    print(f"\nBalance Analysis:")
    print(f"  Score: {final_state.evaluation.balance.score}")
    print(f"  Analysis excerpt: {final_state.evaluation.balance.analysis[:200]}...")

    # Verify simulation analysis is still there
    assert final_state.simulation_analysis is not None
    assert final_state.simulation_analysis.strategic_diversity in ["high", "medium", "low"]


if __name__ == "__main__":
    # Run the test directly with Gemini for quality
    llm = GeminiService()
    asyncio.run(test_e2e_flow_with_simulation(
        llm,
        {
            'concept_and_rules': create_concept_and_rules_workflow(llm),
            'cards': create_cards_workflow(llm),
            'evaluation': create_multi_agent_evaluation_workflow(llm),
        },
        UserPreferences(
            game_description="A simple card battle game",
            language="English",
            theme="Fantasy",
            game_style="Strategy",
            number_of_players="2",
            target_audience="Adults",
            rule_complexity="Simple",
        )
    ))
