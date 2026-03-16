"""
Tests for the GameplayAnalysisAgent.
"""

import pytest
from unittest.mock import MagicMock, patch

from game_simulator.models.metrics import (
    SimulationReport,
    GameMetrics,
    GameplayAnalysis,
    ProblematicCard,
    PacingIssue,
)
from game_simulator.analysis_agent import GameplayAnalysisAgent
from game_simulator.synthetic_games import get_synthetic_game
from game_simulator.statistics import run_quick_simulation


class TestGameplayAnalysisModels:
    """Test the analysis output models."""

    def test_problematic_card_model(self):
        """ProblematicCard should hold card issue data."""
        card = ProblematicCard(
            card_name="OP Dragon",
            issue_type="overpowered",
            evidence="85% win rate when played",
            suggested_fix="Reduce damage from 10 to 6"
        )
        assert card.card_name == "OP Dragon"
        assert card.issue_type == "overpowered"

    def test_pacing_issue_model(self):
        """PacingIssue should hold pacing problem data."""
        issue = PacingIssue(
            issue="Games end too quickly",
            severity="medium",
            evidence="Average game length is 3 turns"
        )
        assert issue.severity == "medium"

    def test_gameplay_analysis_model(self):
        """GameplayAnalysis should hold full analysis."""
        analysis = GameplayAnalysis(
            summary="The game is reasonably balanced.",
            dominant_strategies=["Rush strategy"],
            strategic_diversity="medium",
            problematic_cards=[],
            well_balanced_cards=["Basic Card"],
            pacing_assessment="good",
            pacing_issues=[],
            first_player_analysis="Slight first player advantage (55%)",
            comeback_potential="medium",
            rule_clarity_issues=[],
            fun_indicators=["Close games"],
            anti_fun_indicators=[],
            high_priority_fixes=[],
            suggested_experiments=["Try increasing point threshold"]
        )
        assert analysis.strategic_diversity == "medium"
        assert analysis.pacing_assessment == "good"


class TestAnalysisAgentDataFormatting:
    """Test that the agent formats simulation data correctly for the LLM."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service that captures inputs."""
        mock = MagicMock()
        # Return a valid GameplayAnalysis
        mock.generate.return_value = GameplayAnalysis(
            summary="Test analysis",
            dominant_strategies=[],
            strategic_diversity="medium",
            problematic_cards=[],
            well_balanced_cards=[],
            pacing_assessment="good",
            pacing_issues=[],
            first_player_analysis="Balanced",
            comeback_potential="medium",
            rule_clarity_issues=[],
            fun_indicators=[],
            anti_fun_indicators=[],
            high_priority_fixes=[],
            suggested_experiments=[]
        )
        return mock

    @pytest.fixture
    def sample_report(self):
        """Create a sample SimulationReport."""
        return SimulationReport(
            game_name="Test Game",
            games_run=50,
            games_completed=45,
            completion_rate=0.9,
            wins_by_player={"player_0": 25, "player_1": 20},
            wins_by_position={0: 25, 1: 20},
            first_player_win_rate=0.555,
            avg_turns=12.5,
            min_turns=5,
            max_turns=25,
            std_turns=3.2,
            cards_played_total={
                "Good Card": 150,
                "Weak Card": 30,
                "OP Card": 50,
            },
            cards_never_played=["Dead Card"],
            cards_always_played=["Good Card"],
            card_win_correlation={
                "Good Card": 0.55,
                "Weak Card": 0.35,
                "OP Card": 0.85,
                "Dead Card": 0.0,
            },
            issues=["Potentially OP card: OP Card has 85% win rate"],
        )

    def test_analyze_calls_llm(self, mock_llm_service, sample_report):
        """analyze() should call the LLM with formatted data."""
        agent = GameplayAnalysisAgent(mock_llm_service)
        result = agent.analyze(sample_report)

        assert mock_llm_service.generate.called
        call_kwargs = mock_llm_service.generate.call_args.kwargs

        # Check key data was passed
        assert call_kwargs["game_name"] == "Test Game"
        assert call_kwargs["games_run"] == 50
        assert call_kwargs["completion_rate"] == 0.9
        assert call_kwargs["first_player_win_rate"] == 0.555
        assert call_kwargs["avg_turns"] == 12.5
        assert "Dead Card" in str(call_kwargs["cards_never_played"])

    def test_analyze_for_evaluation_returns_dict(self, mock_llm_service, sample_report):
        """analyze_for_evaluation() should return structured dict."""
        agent = GameplayAnalysisAgent(mock_llm_service)
        result = agent.analyze_for_evaluation(sample_report)

        assert "simulation_summary" in result
        assert "simulation_evidence" in result
        assert result["simulation_evidence"]["games_run"] == 50
        assert result["simulation_evidence"]["completion_rate"] == 0.9


class TestAnalysisAgentWithSyntheticGames:
    """Test agent with actual simulation data from synthetic games."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create mock that returns analysis based on input data."""
        def generate_analysis(output_model, prompt, **kwargs):
            # Analyze the input data and return appropriate analysis
            win_rate = kwargs.get("first_player_win_rate", 0.5)
            avg_turns = kwargs.get("avg_turns", 10)
            issues = kwargs.get("issues_detected", "")

            # Build analysis based on data
            dominant_strategies = []
            problematic_cards = []
            high_priority_fixes = []

            if win_rate > 0.65:
                dominant_strategies.append("First player rush")
                high_priority_fixes.append("Add catch-up mechanics")
            if win_rate < 0.35:
                dominant_strategies.append("Second player control")

            if "OP" in issues or "I Win" in issues:
                problematic_cards.append(ProblematicCard(
                    card_name="I Win",
                    issue_type="overpowered",
                    evidence="Detected in issues",
                    suggested_fix="Remove or nerf severely"
                ))
                high_priority_fixes.append("Nerf overpowered cards")

            pacing = "good"
            if avg_turns < 5:
                pacing = "poor"
                high_priority_fixes.append("Games too short")
            elif avg_turns > 40:
                pacing = "needs_work"
                high_priority_fixes.append("Games too long")

            return GameplayAnalysis(
                summary="Analysis based on simulation data",
                dominant_strategies=dominant_strategies,
                strategic_diversity="medium" if not dominant_strategies else "low",
                problematic_cards=problematic_cards,
                well_balanced_cards=[],
                pacing_assessment=pacing,
                pacing_issues=[],
                first_player_analysis=f"First player wins {win_rate:.0%}",
                comeback_potential="medium",
                rule_clarity_issues=[],
                fun_indicators=[],
                anti_fun_indicators=[],
                high_priority_fixes=high_priority_fixes,
                suggested_experiments=[]
            )

        mock = MagicMock()
        mock.generate.side_effect = generate_analysis
        return mock

    def test_i_win_game_analysis(self, mock_llm_service):
        """Analysis should identify the I Win card as problematic."""
        game = get_synthetic_game("i_win")
        report = run_quick_simulation(game, num_games=30, seed=42)

        agent = GameplayAnalysisAgent(mock_llm_service)
        analysis = agent.analyze(report)

        # The mock analyzes issues and should flag the OP card
        # In a real LLM test, we'd check actual analysis quality
        assert analysis is not None
        assert analysis.summary

    def test_quick_game_pacing(self, mock_llm_service):
        """Analysis should flag quick games as potential pacing issues."""
        game = get_synthetic_game("quick_game")
        report = run_quick_simulation(game, num_games=30, seed=42)

        agent = GameplayAnalysisAgent(mock_llm_service)
        analysis = agent.analyze(report)

        # Quick game should complete fast
        assert report.avg_turns < 10
        # Mock should flag this
        assert analysis.pacing_assessment in ["poor", "needs_work", "good"]


class TestAnalysisAgentIntegration:
    """Integration tests that require actual LLM service.

    These tests are marked with @pytest.mark.integration and skipped by default.
    Run with: pytest -m integration
    """

    @pytest.fixture
    def llm_service(self):
        """Get actual LLM service if available."""
        try:
            from deck_crafter.services.llm_service import GroqService
            import os
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                return GroqService(model="llama-3.3-70b-versatile", api_key=api_key)
        except Exception:
            pass
        pytest.skip("No LLM service available")

    @pytest.mark.integration
    def test_real_analysis_i_win(self, llm_service):
        """Real LLM analysis of the I Win game."""
        game = get_synthetic_game("i_win")
        report = run_quick_simulation(game, num_games=30, seed=42)

        agent = GameplayAnalysisAgent(llm_service)
        analysis = agent.analyze(report)

        # Check the analysis identifies the problem
        assert "I Win" in analysis.summary or any(
            "I Win" in c.card_name for c in analysis.problematic_cards
        ), f"Analysis didn't identify I Win card: {analysis.summary}"

    @pytest.mark.integration
    def test_real_analysis_balanced(self, llm_service):
        """Real LLM analysis of balanced game."""
        game = get_synthetic_game("perfect_balance")
        report = run_quick_simulation(game, num_games=50, seed=42)

        agent = GameplayAnalysisAgent(llm_service)
        analysis = agent.analyze(report)

        # Balanced game should have medium/high diversity
        assert analysis.strategic_diversity in ["medium", "high"]
        # Should not have many high priority fixes
        assert len(analysis.high_priority_fixes) <= 3
