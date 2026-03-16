"""
Generate simple test games for simulator testing.
Run with: python -m game_simulator.generate_test_games
"""

import asyncio
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from deck_crafter.models.user_preferences import UserPreferences
from deck_crafter.models.state import CardGameState, GameStatus
from deck_crafter.services.llm_service import GeminiService
from deck_crafter.workflow.specific_workflows import (
    create_concept_and_rules_workflow,
    create_cards_workflow,
)
from deck_crafter.database import init_db, save_game_state
from deck_crafter.utils.config import Config


# Simple game preferences for testing
TEST_GAMES = [
    {
        "name": "Simple Points Game (EN)",
        "preferences": UserPreferences(
            game_description="A simple card game where players collect points by playing cards. First to 10 points wins.",
            language="English",
            theme="Fantasy",
            game_style="Competitive",
            number_of_players="2",
            target_audience="Casual",
            rule_complexity="Simple",
            art_style="Cartoon",
        ),
    },
    {
        "name": "Juego de Puntos Simple (ES)",
        "preferences": UserPreferences(
            game_description="Un juego de cartas simple donde los jugadores coleccionan puntos jugando cartas. El primero en llegar a 10 puntos gana.",
            language="Spanish",
            theme="Medieval",
            game_style="Competitive",
            number_of_players="2",
            target_audience="Casual",
            rule_complexity="Simple",
            art_style="Cartoon",
        ),
    },
    {
        "name": "Quick Battle Game (EN)",
        "preferences": UserPreferences(
            game_description="A fast combat card game. Players attack each other with cards that deal damage. First to reduce opponent to 0 health wins.",
            language="English",
            theme="Sci-Fi",
            game_style="Competitive",
            number_of_players="2",
            target_audience="Casual",
            rule_complexity="Simple",
            art_style="Cartoon",
        ),
    },
]


async def generate_game(preferences: UserPreferences, name: str, llm_service) -> CardGameState:
    """Generate a complete game (concept, rules, cards)."""
    game_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # Create initial state
    state = CardGameState(
        game_id=game_id,
        status=GameStatus.CREATED,
        preferences=preferences,
        created_at=now,
        updated_at=now,
    )

    print(f"\n{'='*60}")
    print(f"Generating: {name}")
    print(f"Game ID: {game_id}")
    print("="*60)

    # Create workflows
    concept_rules_workflow = create_concept_and_rules_workflow(llm_service)
    cards_workflow = create_cards_workflow(llm_service)

    # Generate concept and rules
    print("Generating concept and rules...")
    result = concept_rules_workflow.invoke(state, config={"configurable": {"thread_id": 1}})
    result_state = CardGameState.model_validate(result)
    state.concept = result_state.concept
    state.rules = result_state.rules
    state.status = GameStatus.RULES_GENERATED
    state.updated_at = datetime.now(timezone.utc)

    print(f"  Title: {state.concept.title if state.concept else 'N/A'}")
    print(f"  Win condition: {state.rules.win_conditions[:80] if state.rules else 'N/A'}...")

    # Generate cards
    print("Generating cards...")
    result = cards_workflow.invoke(state, config={"recursion_limit": 150, "configurable": {"thread_id": 1}})
    state.cards = result["cards"]
    state.status = GameStatus.CARDS_GENERATED
    state.updated_at = datetime.now(timezone.utc)

    print(f"  Cards generated: {len(state.cards)}")

    # Save to database
    await save_game_state(state)
    print(f"  Saved to database!")

    return state


async def main():
    # Initialize database
    await init_db()

    # Create LLM service (Gemini)
    print(f"Initializing Gemini LLM service with model: {Config.GEMINI_MODEL}...")
    llm_service = GeminiService(model=Config.GEMINI_MODEL)

    # Generate each test game
    generated_games = []
    for game_config in TEST_GAMES:
        try:
            state = await generate_game(
                preferences=game_config["preferences"],
                name=game_config["name"],
                llm_service=llm_service,
            )
            generated_games.append(state)
        except Exception as e:
            print(f"ERROR generating {game_config['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    for state in generated_games:
        print(f"  - {state.concept.title if state.concept else 'Unknown'}: {state.game_id}")
        if state.rules:
            print(f"    Win: {state.rules.win_conditions[:60]}...")
        if state.cards:
            print(f"    Cards: {len(state.cards)}")
            # Show first few card descriptions
            for card in state.cards[:3]:
                print(f"      - {card.name}: {card.description[:50]}...")


if __name__ == "__main__":
    asyncio.run(main())
