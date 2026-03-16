"""
Test the rule compiler and simulator with a real generated game.
Run with: python -m game_simulator.test_real_game [--llm]
"""

import sqlite3
import json
import os
from game_simulator.rule_compiler import compile_game
from game_simulator.statistics import run_quick_simulation
from deck_crafter.models.rules import Rules
from deck_crafter.models.card import Card


def get_llm_service():
    """Get LLM service if available."""
    try:
        from deck_crafter.services.llm_service import GroqService
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            print("Using Groq LLM for fallback parsing...")
            return GroqService(model="llama-3.3-70b-versatile", api_key=api_key)
        else:
            print("No GROQ_API_KEY found, using pattern-based parsing only")
    except Exception as e:
        print(f"Could not initialize LLM service: {e}")
    return None


def load_game_from_db(game_id: str) -> tuple[Rules, list[Card], str] | None:
    """Load a game's rules and cards from the database."""
    conn = sqlite3.connect("deck_crafter.db")
    cursor = conn.execute(
        "SELECT rules, cards, json_extract(concept, '$.title') FROM games WHERE game_id = ?",
        (game_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row or not row[0] or not row[1]:
        return None

    rules_data = json.loads(row[0])
    cards_data = json.loads(row[1])
    title = row[2] or "Unknown Game"

    rules = Rules.model_validate(rules_data)
    cards = [Card.model_validate(c) for c in cards_data]

    return rules, cards, title


def list_games_with_rules():
    """List all games that have rules and cards."""
    conn = sqlite3.connect("deck_crafter.db")
    cursor = conn.execute("""
        SELECT game_id, status, json_extract(concept, '$.title')
        FROM games
        WHERE rules IS NOT NULL AND cards IS NOT NULL
        ORDER BY updated_at DESC
        LIMIT 10
    """)
    games = cursor.fetchall()
    conn.close()
    return games


def test_game(game_id: str, num_games: int = 20, llm_service=None):
    """Test a specific game."""
    print(f"\n{'='*60}")
    print(f"Loading game: {game_id}")
    print("="*60)

    result = load_game_from_db(game_id)
    if not result:
        print("ERROR: Could not load game (missing rules or cards)")
        return None

    rules, cards, title = result
    print(f"Title: {title}")
    print(f"Cards: {len(cards)}")

    # Show rules summary
    print(f"\nRules Summary:")
    print(f"  - Initial hands: {rules.initial_hands[:100]}...")
    print(f"  - Win conditions: {rules.win_conditions[:100]}...")
    print(f"  - Turn phases: {len(rules.turn_structure)}")

    # Compile
    print(f"\nCompiling rules {'(with LLM fallback)' if llm_service else '(pattern-only)'}...")
    game_def, warnings = compile_game(rules, cards, game_name=title, llm_service=llm_service)

    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    print(f"\nCompiled GameDefinition:")
    print(f"  - Initial hand size: {game_def.rules.initial_hand_size}")
    print(f"  - Draw per turn: {game_def.rules.draw_per_turn}")
    print(f"  - Max cards per turn: {game_def.rules.max_cards_per_turn}")
    print(f"  - Win condition: {game_def.win_condition.type} ({game_def.win_condition.target_value})")
    print(f"  - Initial resources: {game_def.rules.initial_resources}")
    print(f"  - Initial properties: {game_def.rules.initial_properties}")

    # Show some card effects
    print(f"\nCard effects (first 5):")
    for card_def in game_def.cards[:5]:
        print(f"  - {card_def.name}: {card_def.effect.value} ({card_def.effect_value})")

    # Run simulation
    print(f"\nRunning simulation ({num_games} games)...")
    report = run_quick_simulation(game_def, num_games=num_games, seed=42)

    print(f"\nSimulation Report:")
    print(f"  - Completion rate: {report.completion_rate:.1%}")
    print(f"  - First player win rate: {report.first_player_win_rate:.1%}")
    print(f"  - Avg turns: {report.avg_turns:.1f}")
    print(f"  - Min/Max turns: {report.min_turns}/{report.max_turns}")

    if report.issues:
        print(f"\nIssues detected:")
        for issue in report.issues:
            print(f"  - {issue}")

    if report.cards_never_played:
        print(f"\nCards never played: {report.cards_never_played[:5]}")

    return report


if __name__ == "__main__":
    # Check if --llm flag is passed
    use_llm = "--llm" in os.sys.argv

    # Get LLM service if requested
    llm_service = get_llm_service() if use_llm else None

    # List available games
    print("\nAvailable games with rules:")
    games = list_games_with_rules()
    for game_id, status, title in games:
        print(f"  [{status}] {title} ({game_id[:8]}...)")

    # Test each game
    for game_id, status, title in games[:3]:
        test_game(game_id, llm_service=llm_service)
