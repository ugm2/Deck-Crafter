"""
Tests for the rule compiler.
"""

import pytest
from deck_crafter.models.rules import Rules, TurnPhase
from deck_crafter.models.card import Card
from game_simulator.rule_compiler import RuleCompiler, compile_game
from game_simulator.models.game_definition import CardEffect


class TestRuleCompiler:
    """Test pattern-based rule parsing."""

    @pytest.fixture
    def compiler(self):
        return RuleCompiler()

    def test_extract_hand_size_simple(self, compiler):
        """Test extracting hand size from common phrases."""
        cases = [
            ("Each player draws 5 cards", 5),
            ("Deal 7 cards to each player", 7),
            ("Players start with a hand of 6 cards", 6),
            ("Each player is dealt 3 cards from the deck", 3),
        ]
        for text, expected in cases:
            result = compiler._extract_hand_size(text)
            assert result == expected, f"Failed for: {text}"

    def test_extract_draw_per_turn(self, compiler):
        """Test extracting draw per turn."""
        # Turn structure with draw phase
        phases = [
            TurnPhase(phase_name="Draw Phase", phase_description="Draw 2 cards from your deck."),
            TurnPhase(phase_name="Main Phase", phase_description="Play cards from your hand."),
        ]
        result = compiler._extract_draw_per_turn(phases)
        assert result == 2

    def test_extract_max_cards_per_turn(self, compiler):
        """Test extracting max cards playable."""
        cases = [
            ([TurnPhase(phase_name="Main", phase_description="Play up to 2 cards")], 2),
            ([TurnPhase(phase_name="Main", phase_description="Play one card from your hand")], 1),
            ([TurnPhase(phase_name="Main", phase_description="Play any number of cards")], 99),
        ]
        for phases, expected in cases:
            result = compiler._extract_max_cards_per_turn(phases)
            assert result == expected

    def test_parse_win_condition_points(self, compiler):
        """Test parsing points-based win conditions."""
        cases = [
            ("First player to reach 10 points wins", "points", 10),
            ("Score 15 points to win the game", "points", 15),
            ("The player with 20 points wins", "points", 20),
        ]
        for text, expected_type, expected_value in cases:
            result = compiler._parse_win_condition(text)
            assert result.type == expected_type
            assert result.target_value == expected_value

    def test_parse_win_condition_elimination(self, compiler):
        """Test parsing elimination win conditions."""
        cases = [
            ("The last player standing wins", "last_standing"),
            ("Eliminate all opponents to win", "last_standing"),
            ("Reduce opponent health to 0 to win", "elimination"),
        ]
        for text, expected_type in cases:
            result = compiler._parse_win_condition(text)
            assert result.type == expected_type

    def test_parse_card_effect_draw(self, compiler):
        """Test parsing draw effects."""
        cases = [
            ("Draw 3 cards from your deck", CardEffect.DRAW, 3),
            ("Draw a card", CardEffect.DRAW, 1),
        ]
        for desc, expected_effect, expected_value in cases:
            effect, value, _ = compiler._parse_card_effect(desc)
            assert effect == expected_effect
            assert value == expected_value

    def test_parse_card_effect_damage(self, compiler):
        """Test parsing damage effects."""
        cases = [
            ("Deal 5 damage to an opponent", CardEffect.DAMAGE, 5),
            ("Deals 3 damage to target player", CardEffect.DAMAGE, 3),
        ]
        for desc, expected_effect, expected_value in cases:
            effect, value, _ = compiler._parse_card_effect(desc)
            assert effect == expected_effect
            assert value == expected_value

    def test_parse_card_effect_points(self, compiler):
        """Test parsing point gain effects."""
        cases = [
            ("Gain 2 points", CardEffect.GAIN_POINTS, 2),
            ("Score 5 points", CardEffect.GAIN_POINTS, 5),
            ("+3 points", CardEffect.GAIN_POINTS, 3),
            ("This card is worth 4 points", CardEffect.GAIN_POINTS, 4),
        ]
        for desc, expected_effect, expected_value in cases:
            effect, value, _ = compiler._parse_card_effect(desc)
            assert effect == expected_effect, f"Failed for: {desc}"
            assert value == expected_value, f"Failed for: {desc}"

    def test_parse_card_cost(self, compiler):
        """Test parsing card costs."""
        cases = [
            ("2 Mana", {"mana": 2}),
            ("3 Energy", {"energy": 3}),
            ("No cost", {}),
            ("Free", {}),
        ]
        for cost, expected in cases:
            result = compiler._parse_card_cost(cost)
            assert result == expected, f"Failed for: {cost}"


class TestFullCompilation:
    """Test compiling full Rules + Cards."""

    def test_simple_points_game(self):
        """Test compiling a simple points-based game."""
        rules = Rules(
            deck_preparation="Shuffle all cards into a single deck.",
            initial_hands="Each player draws 5 cards.",
            turn_structure=[
                TurnPhase(phase_name="Draw", phase_description="Draw 1 card."),
                TurnPhase(phase_name="Play", phase_description="Play up to 2 cards."),
            ],
            win_conditions="First player to 15 points wins the game.",
        )

        cards = [
            Card(
                name="Point Card",
                quantity=20,
                type="Basic",
                description="Gain 2 points.",
                image_description="A shiny coin.",
            ),
            Card(
                name="Big Score",
                quantity=5,
                type="Rare",
                description="+5 points when played.",
                image_description="A treasure chest.",
            ),
        ]

        game_def, warnings = compile_game(rules, cards, "Test Game")

        assert game_def.rules.initial_hand_size == 5
        assert game_def.rules.draw_per_turn == 1
        assert game_def.rules.max_cards_per_turn == 2
        assert game_def.win_condition.type == "points"
        assert game_def.win_condition.target_value == 15

        # Check card effects were parsed
        point_card = next(c for c in game_def.cards if c.name == "Point Card")
        assert point_card.effect == CardEffect.GAIN_POINTS
        assert point_card.effect_value == 2

        big_score = next(c for c in game_def.cards if c.name == "Big Score")
        assert big_score.effect == CardEffect.GAIN_POINTS
        assert big_score.effect_value == 5

    def test_damage_game(self):
        """Test compiling a damage-based game."""
        rules = Rules(
            deck_preparation="Shuffle the deck.",
            initial_hands="Deal 7 cards to each player.",
            turn_structure=[
                TurnPhase(phase_name="Main", phase_description="Play one card."),
            ],
            win_conditions="Reduce your opponent's health to 0 to win.",
        )

        cards = [
            Card(
                name="Fireball",
                quantity=10,
                type="Attack",
                description="Deal 3 damage to your opponent.",
                cost="2 Mana",
                image_description="A ball of fire.",
            ),
        ]

        game_def, warnings = compile_game(rules, cards, "Damage Game")

        assert game_def.rules.initial_hand_size == 7
        assert game_def.win_condition.type == "elimination"

        fireball = game_def.cards[0]
        assert fireball.effect == CardEffect.DAMAGE
        assert fireball.effect_value == 3
        assert fireball.cost == {"mana": 2}


class TestSimulationWithCompiledGame:
    """Test that compiled games can be simulated."""

    def test_compiled_game_simulates(self):
        """Test that a compiled game runs in the simulator."""
        from game_simulator.statistics import run_quick_simulation

        rules = Rules(
            deck_preparation="Shuffle all cards.",
            initial_hands="Each player draws 5 cards.",
            turn_structure=[
                TurnPhase(phase_name="Draw", phase_description="Draw 1 card."),
                TurnPhase(phase_name="Play", phase_description="Play 1 card."),
            ],
            win_conditions="First to 10 points wins.",
        )

        cards = [
            Card(
                name="Score",
                quantity=40,
                type="Basic",
                description="Gain 1 point.",
                image_description="A point token.",
            ),
        ]

        game_def, _ = compile_game(rules, cards, "Simple Game")

        # Run simulation
        report = run_quick_simulation(game_def, num_games=20, seed=42)

        assert report.completion_rate > 0.8  # Most games should complete
        assert report.avg_turns > 0
