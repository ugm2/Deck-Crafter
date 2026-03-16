"""
Synthetic test games with known properties.
These games are designed to validate that the simulator correctly detects:
- Card power imbalances
- First player advantages
- Infinite loops / stuck states
- Dead cards
- Balanced gameplay
"""

from game_simulator.models.game_definition import (
    GameDefinition,
    CardDefinition,
    CardEffect,
    RuleSet,
    WinCondition,
)


def create_i_win_game() -> GameDefinition:
    """
    A game with one massively overpowered card that wins instantly.

    Expected findings:
    - The "I Win" card should have ~95%+ win correlation
    - Games where it's played should end quickly
    """
    return GameDefinition(
        name="I Win Test Game",
        description="Tests detection of game-winning OP cards",
        cards=[
            CardDefinition(
                name="I Win",
                quantity=4,  # 4 copies - whoever draws it first wins
                effect=CardEffect.WIN_GAME,
                effect_value=0,
            ),
            CardDefinition(
                name="Normal Card",
                quantity=20,
                effect=CardEffect.GAIN_POINTS,
                effect_value=1,
            ),
        ],
        rules=RuleSet(
            initial_hand_size=5,
            draw_per_turn=1,
            max_cards_per_turn=1,
        ),
        win_condition=WinCondition(
            type="points",
            target_value=10,
        ),
        num_players=2,
    )


def create_first_player_op_game() -> GameDefinition:
    """
    A game where first player has a massive advantage (draws more cards).

    Expected findings:
    - First player win rate should be >70%
    - Issue detected for first player advantage
    """
    return GameDefinition(
        name="First Player OP Test",
        description="Tests detection of turn order imbalance",
        cards=[
            # First player starts with more draws
            CardDefinition(
                name="Point Card",
                quantity=30,
                effect=CardEffect.GAIN_POINTS,
                effect_value=1,
            ),
        ],
        rules=RuleSet(
            initial_hand_size=10,  # First player gets full hand before second player
            draw_per_turn=1,
            max_cards_per_turn=3,  # Can play many cards = whoever has more wins
        ),
        win_condition=WinCondition(
            type="points",
            target_value=15,  # First to 15 points wins
        ),
        num_players=2,
    )


def create_dead_cards_game() -> GameDefinition:
    """
    A game with cards that do nothing useful.

    Expected findings:
    - "Useless Card" should appear in cards_never_played or very low play rate
    - Should not affect win rates
    """
    return GameDefinition(
        name="Dead Cards Test",
        description="Tests detection of useless cards",
        cards=[
            CardDefinition(
                name="Good Card",
                quantity=20,
                effect=CardEffect.GAIN_POINTS,
                effect_value=2,
            ),
            CardDefinition(
                name="Useless Card",
                quantity=10,
                effect=CardEffect.NONE,  # Does absolutely nothing
                effect_value=0,
            ),
        ],
        rules=RuleSet(
            initial_hand_size=5,
            draw_per_turn=1,
            max_cards_per_turn=1,
        ),
        win_condition=WinCondition(
            type="points",
            target_value=10,
        ),
        num_players=2,
    )


def create_perfect_balance_game() -> GameDefinition:
    """
    A game with variable card values so luck matters more than turn order.
    The variance in card values creates situations where either player can win.

    Expected findings:
    - Win rate should be approximately 50/50 (within 35-65% range)
    - No critical issues should be detected
    - Completion rate should be high
    """
    return GameDefinition(
        name="Perfect Balance Test",
        description="Tests baseline calibration - should show ~50% win rate",
        cards=[
            # Mix of card values - luck of the draw determines winner
            CardDefinition(
                name="Small Point",
                quantity=20,
                effect=CardEffect.GAIN_POINTS,
                effect_value=1,
            ),
            CardDefinition(
                name="Medium Point",
                quantity=15,
                effect=CardEffect.GAIN_POINTS,
                effect_value=2,
            ),
            CardDefinition(
                name="Big Point",
                quantity=5,
                effect=CardEffect.GAIN_POINTS,
                effect_value=3,
            ),
        ],
        rules=RuleSet(
            initial_hand_size=5,
            draw_per_turn=1,
            max_cards_per_turn=1,
        ),
        win_condition=WinCondition(
            type="points",
            target_value=15,  # Higher threshold = more variance
        ),
        num_players=2,
    )


def create_long_game() -> GameDefinition:
    """
    A game that takes many turns to complete.
    Used to test game length detection.

    Expected findings:
    - High turn counts
    - Should still complete (high completion rate)
    """
    return GameDefinition(
        name="Long Game Test",
        description="Tests games that take many turns",
        cards=[
            CardDefinition(
                name="Tiny Point",
                quantity=100,
                effect=CardEffect.GAIN_POINTS,
                effect_value=1,  # Only 1 point per card
            ),
        ],
        rules=RuleSet(
            initial_hand_size=3,
            draw_per_turn=1,
            max_cards_per_turn=1,
        ),
        win_condition=WinCondition(
            type="points",
            target_value=30,  # Need 30 points = 30 turns minimum
        ),
        num_players=2,
    )


def create_quick_game() -> GameDefinition:
    """
    A game that ends very quickly.

    Expected findings:
    - Low turn counts (< 5 turns)
    - High completion rate
    """
    return GameDefinition(
        name="Quick Game Test",
        description="Tests games that end quickly",
        cards=[
            CardDefinition(
                name="Big Points",
                quantity=20,
                effect=CardEffect.GAIN_POINTS,
                effect_value=5,  # 5 points per card
            ),
        ],
        rules=RuleSet(
            initial_hand_size=5,
            draw_per_turn=1,
            max_cards_per_turn=2,  # Can play 2 per turn = 10 points/turn
        ),
        win_condition=WinCondition(
            type="points",
            target_value=10,  # Only need 2 cards to win
        ),
        num_players=2,
    )


# Registry of all synthetic games
SYNTHETIC_GAMES = {
    "i_win": create_i_win_game,
    "first_player_op": create_first_player_op_game,
    "dead_cards": create_dead_cards_game,
    "perfect_balance": create_perfect_balance_game,
    "long_game": create_long_game,
    "quick_game": create_quick_game,
}


def get_synthetic_game(name: str) -> GameDefinition:
    """Get a synthetic game by name."""
    if name not in SYNTHETIC_GAMES:
        raise ValueError(f"Unknown synthetic game: {name}. Available: {list(SYNTHETIC_GAMES.keys())}")
    return SYNTHETIC_GAMES[name]()


def get_all_synthetic_games() -> dict[str, GameDefinition]:
    """Get all synthetic games."""
    return {name: factory() for name, factory in SYNTHETIC_GAMES.items()}
