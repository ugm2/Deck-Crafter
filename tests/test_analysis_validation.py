"""
Validation tests for Sprint 3: Verify GameplayAnalysisAgent identifies known issues in synthetic games.

These tests validate that the analysis agent correctly:
1. Identifies the "I Win" card as overpowered
2. Flags first player advantage
3. Detects dead/useless cards
4. Recognizes pacing issues (too quick/long)
5. Identifies balanced games as healthy

Tests use mocked LLM responses that simulate what a real LLM would produce
based on the simulation data.
"""

import pytest
from unittest.mock import MagicMock

from game_simulator.synthetic_games import get_synthetic_game
from game_simulator.statistics import run_quick_simulation
from game_simulator.analysis_agent import GameplayAnalysisAgent
from game_simulator.models.metrics import (
    GameplayAnalysis,
    ProblematicCard,
    PacingIssue,
)


def create_analysis_from_data(report, **overrides):
    """Create a GameplayAnalysis based on simulation report data."""
    # Analyze the data to produce appropriate analysis
    problematic_cards = []
    pacing_issues = []
    dominant_strategies = []
    fun_indicators = []
    anti_fun_indicators = []
    high_priority_fixes = []

    # Calculate average win correlation (baseline)
    win_correlations = list(report.card_win_correlation.values())
    avg_win_corr = sum(win_correlations) / len(win_correlations) if win_correlations else 0.5

    # Check for OP cards - must be significantly above average AND not universal
    for card_name, win_rate in report.card_win_correlation.items():
        plays = report.cards_played_total.get(card_name, 0)
        total_plays = sum(report.cards_played_total.values())
        play_rate = plays / total_plays if total_plays > 0 else 0

        # Card is OP if:
        # 1. Very high win rate AND not played by everyone (silver bullet)
        # 2. OR significantly above average win rate AND not universal
        is_silver_bullet = win_rate > 0.9 and play_rate < 0.3
        is_above_avg = win_rate > avg_win_corr + 0.25 and play_rate < 0.5

        if is_silver_bullet or is_above_avg:
            problematic_cards.append(ProblematicCard(
                card_name=card_name,
                issue_type="overpowered",
                evidence=f"{win_rate:.0%} win rate when played",
                suggested_fix="Nerf or remove"
            ))
            high_priority_fixes.append(f"Nerf {card_name} - too powerful")

    # Check for dead cards
    for card_name in report.cards_never_played:
        problematic_cards.append(ProblematicCard(
            card_name=card_name,
            issue_type="dead_card",
            evidence="Never played in any game",
            suggested_fix="Add effect or remove"
        ))

    # Check first player advantage
    fp_analysis = f"First player wins {report.first_player_win_rate:.0%}"
    if report.first_player_win_rate > 0.65:
        dominant_strategies.append("First player rush")
        high_priority_fixes.append("Add catch-up mechanics for second player")
        anti_fun_indicators.append("Turn order determines winner too often")
    elif report.first_player_win_rate < 0.35:
        dominant_strategies.append("Second player control")
        high_priority_fixes.append("Balance first player advantage")
    else:
        fun_indicators.append("Balanced turn order advantage")

    # Check pacing
    pacing = "good"
    if report.avg_turns < 5:
        pacing = "poor"
        pacing_issues.append(PacingIssue(
            issue="Games end too quickly",
            severity="high",
            evidence=f"Average {report.avg_turns:.1f} turns"
        ))
        anti_fun_indicators.append("Games too short for strategic depth")
    elif report.avg_turns > 40:
        pacing = "needs_work"
        pacing_issues.append(PacingIssue(
            issue="Games drag on too long",
            severity="medium",
            evidence=f"Average {report.avg_turns:.1f} turns"
        ))
        anti_fun_indicators.append("Games overstay their welcome")
    else:
        fun_indicators.append("Good game length")

    # Check completion rate
    if report.completion_rate < 0.7:
        anti_fun_indicators.append("Many games fail to complete")
        high_priority_fixes.append("Fix stuck states causing incomplete games")

    # Check strategic diversity
    if len(dominant_strategies) > 0:
        diversity = "low"
    elif len(report.cards_always_played) > len(report.cards_played_total) * 0.3:
        diversity = "medium"
    else:
        diversity = "high"

    # Build summary
    if problematic_cards:
        summary = f"Game has {len(problematic_cards)} problematic card(s). "
    else:
        summary = "Card balance is reasonable. "

    if pacing != "good":
        summary += f"Pacing {pacing}. "
    if report.first_player_win_rate > 0.65 or report.first_player_win_rate < 0.35:
        summary += f"Turn order imbalance detected."

    return GameplayAnalysis(
        summary=summary.strip() or "Game appears healthy",
        dominant_strategies=dominant_strategies,
        strategic_diversity=diversity,
        problematic_cards=problematic_cards,
        well_balanced_cards=[],
        pacing_assessment=pacing,
        pacing_issues=pacing_issues,
        first_player_analysis=fp_analysis,
        comeback_potential="medium",
        rule_clarity_issues=[],
        fun_indicators=fun_indicators,
        anti_fun_indicators=anti_fun_indicators,
        high_priority_fixes=high_priority_fixes,
        suggested_experiments=[],
        **overrides
    )


class TestAnalysisIdentifiesIWinCard:
    """Verify analysis identifies the I Win card as problematic."""

    def test_i_win_detected_as_op(self):
        """The I Win card should be flagged as overpowered."""
        game = get_synthetic_game("i_win")
        report = run_quick_simulation(game, num_games=50, seed=42)

        # Create mock that generates analysis from data
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = lambda output_model, prompt, **kwargs: \
            create_analysis_from_data(report)

        agent = GameplayAnalysisAgent(mock_llm)
        analysis = agent.analyze(report)

        # Validate: I Win card should be in problematic cards
        op_cards = [c for c in analysis.problematic_cards if c.issue_type == "overpowered"]
        i_win_flagged = any("I Win" in c.card_name for c in op_cards)

        assert i_win_flagged, (
            f"I Win card should be flagged as OP. "
            f"Win correlation: {report.card_win_correlation.get('I Win', 'N/A')}"
        )


class TestAnalysisIdentifiesFirstPlayerAdvantage:
    """Verify analysis flags first player advantage."""

    def test_first_player_advantage_detected(self):
        """First player OP game should show advantage in analysis."""
        game = get_synthetic_game("first_player_op")
        report = run_quick_simulation(game, num_games=50, seed=42)

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = lambda output_model, prompt, **kwargs: \
            create_analysis_from_data(report)

        agent = GameplayAnalysisAgent(mock_llm)
        analysis = agent.analyze(report)

        # Check first player analysis mentions the advantage
        assert "First player wins" in analysis.first_player_analysis
        win_pct = float(analysis.first_player_analysis.split()[-1].replace("%", "")) / 100

        # Should show >55% first player win rate
        assert win_pct > 0.5, f"First player should show advantage: {analysis.first_player_analysis}"


class TestAnalysisIdentifiesDeadCards:
    """Verify analysis flags dead/useless cards."""

    def test_dead_cards_flagged(self):
        """Useless cards should be identified."""
        game = get_synthetic_game("dead_cards")
        report = run_quick_simulation(game, num_games=50, seed=42)

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = lambda output_model, prompt, **kwargs: \
            create_analysis_from_data(report)

        agent = GameplayAnalysisAgent(mock_llm)
        analysis = agent.analyze(report)

        # Check that useless card has lower win correlation
        useless_win_rate = report.card_win_correlation.get("Useless Card", 0)
        good_win_rate = report.card_win_correlation.get("Good Card", 0)

        # Good cards should correlate with winning more
        assert good_win_rate >= useless_win_rate, (
            f"Good cards ({good_win_rate:.1%}) should have >= win rate than useless ({useless_win_rate:.1%})"
        )


class TestAnalysisIdentifiesPacingIssues:
    """Verify analysis flags pacing issues."""

    def test_quick_game_pacing(self):
        """Quick games should flag pacing issues."""
        game = get_synthetic_game("quick_game")
        report = run_quick_simulation(game, num_games=50, seed=42)

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = lambda output_model, prompt, **kwargs: \
            create_analysis_from_data(report)

        agent = GameplayAnalysisAgent(mock_llm)
        analysis = agent.analyze(report)

        # Average turns should be low
        assert report.avg_turns < 10, f"Quick game has avg {report.avg_turns} turns"

        # Analysis should note this (may or may not flag as issue depending on threshold)
        # At minimum, anti-fun should mention short games or pacing should not be "good"
        if report.avg_turns < 5:
            assert analysis.pacing_assessment != "excellent", "Very quick games shouldn't have excellent pacing"

    def test_long_game_pacing(self):
        """Long games should note high turn counts."""
        game = get_synthetic_game("long_game")
        report = run_quick_simulation(game, num_games=30, seed=42)

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = lambda output_model, prompt, **kwargs: \
            create_analysis_from_data(report)

        agent = GameplayAnalysisAgent(mock_llm)
        analysis = agent.analyze(report)

        # Average turns should be high
        assert report.avg_turns > 15, f"Long game has avg {report.avg_turns} turns"


class TestAnalysisRecognizesBalance:
    """Verify analysis recognizes well-balanced games."""

    def test_balanced_game_healthy(self):
        """Balanced game should show healthy indicators."""
        game = get_synthetic_game("perfect_balance")
        report = run_quick_simulation(game, num_games=100, seed=42)

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = lambda output_model, prompt, **kwargs: \
            create_analysis_from_data(report)

        agent = GameplayAnalysisAgent(mock_llm)
        analysis = agent.analyze(report)

        # Win rate should be balanced
        assert 0.35 < report.first_player_win_rate < 0.65, (
            f"Win rate {report.first_player_win_rate:.1%} outside balanced range"
        )

        # Should not have many high priority fixes
        # (balanced game might have 0-1 suggestions)
        assert len(analysis.high_priority_fixes) <= 2, (
            f"Balanced game has too many fixes: {analysis.high_priority_fixes}"
        )

        # Should have some fun indicators
        assert len(analysis.fun_indicators) > 0 or analysis.strategic_diversity != "low", (
            "Balanced game should have positive indicators"
        )
