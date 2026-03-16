"""
GameplayAnalysisAgent: Transforms simulation statistics into qualitative game design insights.

This agent reads SimulationReport data and produces actionable analysis that can:
1. Feed into the evaluation workflow (providing evidence for static evaluators)
2. Guide the DirectorAgent's refinement decisions
3. Give human designers concrete, data-driven feedback
"""
import logging

from deck_crafter.game_simulator.models.metrics import SimulationReport, GameplayAnalysis, AnalysisConfidence
from deck_crafter.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class GameplayAnalysisAgent:
    """
    Analyzes simulation results and produces qualitative gameplay insights.

    Unlike static evaluation agents that imagine gameplay, this agent has
    actual evidence from simulated games. It translates statistics into
    design-relevant observations.
    """

    PROMPT_TEMPLATE = """
    ### ROLE & PERSONA ###
    You are a Game Analytics Expert who has just observed {games_run} playtests of a card game.
    You have access to detailed statistics from these games. Your job is to translate raw data
    into actionable game design insights.

    Unlike theoretical analysis, you have EVIDENCE. Every observation you make must be grounded
    in the data provided. Don't speculate - report what actually happened in the games.

    ### SIMULATION RESULTS ###

    **Game:** {game_name}
    **Games Simulated:** {games_run} games ({completion_rate:.0%} completed naturally)

    **Win Distribution:**
    - First player win rate: {first_player_win_rate:.1%}
    - Wins by position: {wins_by_position}

    **Game Length:**
    - Average turns: {avg_turns:.1f} (std: {std_turns:.1f})
    - Range: {min_turns} to {max_turns} turns

    **Card Performance:**
    Cards played (total across all games): {cards_played_summary}

    Cards never played in any game: {cards_never_played}
    Cards played in >90% of games: {cards_always_played}

    **Card Win Correlation** (win rate when card was played):
    {card_win_correlation_summary}

    **Automatic Issue Detection:**
    {issues_detected}

    **Root Cause Diagnostics:**
    {failure_diagnostics}

    ### YOUR TASK ###
    Analyze these results and produce insights in these categories:

    1. **Strategic Findings**: What strategies emerged? Is there strategic diversity or one dominant path?

    2. **Card Balance**: Which cards are problematic? Use the win correlation and play frequency data.
       - Overpowered: High win correlation (>70%) + played frequently
       - Underpowered: Low win correlation (<30%) or rarely played despite availability
       - Dead cards: Never played
       - Combo enablers: Cards that when played together produce high win rates

    3. **Pacing Analysis**: Is game length appropriate? Use turn statistics.
       - Too short (<5 turns avg): May lack depth
       - Too long (>40 turns avg): May be tedious
       - High variance: Games inconsistent

    4. **Balance Insights**:
       - First player advantage >60% or <40% is problematic
       - If completion rate <70%, rules may have stuck states

    5. **Fun Indicators**: Based on the data, identify signs of good/bad player experience.
       - Close win distributions = exciting games
       - High variance in outcomes = unpredictable (can be good or bad)
       - One-sided results = unfun for the loser

    6. **Recommendations**: What specific changes would improve the game? Be concrete and cite your evidence.

    7. **Balance Adjustments**: For each problematic card, provide a SPECIFIC numerical fix.
       Format: "CardName: action (reason based on data)"
       Examples:
       - "Dragon: reduce damage from 6 to 4 (85% win correlation when played)"
       - "Healer: reduce heal from 5 to 3 (games never end due to heal spam)"
       - "Weak Goblin: increase attack from 1 to 2 (never played, too weak vs alternatives)"
       - "Shield Wall: reduce defense from 8 to 5 (stalemates in 40% of games)"

       Rules for balance adjustments:
       - Only suggest changes for cards with clear statistical evidence
       - Reduce strong stats by 20-40% for OP cards
       - Increase weak stats by 30-50% for underpowered cards
       - If a card is never played, it needs a significant buff
       - Reference the exact win correlation or play frequency in your reason

    ### IMPORTANT GUIDELINES ###
    - Every claim must reference the data provided
    - Use the card names exactly as given
    - Distinguish between "needs fixing" and "worth testing"
    - If data is inconclusive, say so rather than speculating
    - Focus on the 3-5 most important findings

    ### OUTPUT LANGUAGE ###
    Language for response: {language}
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def analyze(
        self,
        report: SimulationReport,
        language: str = "English"
    ) -> GameplayAnalysis:
        """
        Analyze a simulation report and produce qualitative insights.

        Args:
            report: SimulationReport from running simulations
            language: Output language for the analysis

        Returns:
            GameplayAnalysis with structured insights
        """
        logger.info(f"[GameplayAnalysis] Analyzing {report.games_run} games "
                   f"(completion: {report.completion_rate:.0%}, avg turns: {report.avg_turns:.1f})")

        # Format card performance summary (top 10 most played)
        sorted_cards = sorted(
            report.cards_played_total.items(),
            key=lambda x: -x[1]
        )
        cards_played_summary = ", ".join(
            f"{name}: {count}" for name, count in sorted_cards[:10]
        )
        if len(sorted_cards) > 10:
            cards_played_summary += f" ... ({len(sorted_cards) - 10} more)"

        # Format win correlation (sorted by correlation, show top suspicious)
        sorted_correlation = sorted(
            report.card_win_correlation.items(),
            key=lambda x: -x[1]
        )
        # Show top 5 highest and bottom 5 lowest
        high_corr = sorted_correlation[:5]
        low_corr = sorted_correlation[-5:] if len(sorted_correlation) > 5 else []

        correlation_lines = []
        if high_corr:
            correlation_lines.append("Highest win correlation:")
            for name, corr in high_corr:
                plays = report.cards_played_total.get(name, 0)
                correlation_lines.append(f"  - {name}: {corr:.1%} (played {plays}x)")

        if low_corr:
            correlation_lines.append("\nLowest win correlation:")
            for name, corr in low_corr:
                plays = report.cards_played_total.get(name, 0)
                correlation_lines.append(f"  - {name}: {corr:.1%} (played {plays}x)")

        card_win_correlation_summary = "\n".join(correlation_lines) if correlation_lines else "No data"

        # Format issues
        issues_text = "\n".join(f"- {issue}" for issue in report.issues) if report.issues else "No issues automatically detected"

        # Format failure diagnostics
        if report.failure_reasons:
            failure_diagnostics = "\n".join(f"- {reason}" for reason in report.failure_reasons)
        else:
            failure_diagnostics = "No root cause issues detected"

        # Call LLM
        analysis = self.llm_service.generate(
            output_model=GameplayAnalysis,
            prompt=self.PROMPT_TEMPLATE,
            game_name=report.game_name,
            games_run=report.games_run,
            completion_rate=report.completion_rate,
            first_player_win_rate=report.first_player_win_rate,
            wins_by_position=report.wins_by_position,
            avg_turns=report.avg_turns,
            std_turns=report.std_turns,
            min_turns=report.min_turns,
            max_turns=report.max_turns,
            cards_played_summary=cards_played_summary,
            cards_never_played=report.cards_never_played or "None",
            cards_always_played=report.cards_always_played or "None",
            card_win_correlation_summary=card_win_correlation_summary,
            issues_detected=issues_text,
            failure_diagnostics=failure_diagnostics,
            language=language,
        )

        # Add confidence assessment based on report quality
        analysis.confidence = AnalysisConfidence.from_report(report)

        logger.info(f"[GameplayAnalysis] Complete: strategic_diversity={analysis.strategic_diversity}, "
                   f"pacing={analysis.pacing_assessment}, confidence={analysis.confidence.overall}")
        if analysis.problematic_cards:
            logger.debug(f"[GameplayAnalysis] Problematic cards: "
                        f"{[c.card_name for c in analysis.problematic_cards[:5]]}")
        if analysis.high_priority_fixes:
            logger.debug(f"[GameplayAnalysis] High priority fixes: {analysis.high_priority_fixes[:3]}")

        return analysis

    def analyze_for_evaluation(
        self,
        report: SimulationReport,
        language: str = "English"
    ) -> dict:
        """
        Produce analysis formatted for integration with evaluation agents.

        Returns a dict that can be injected into evaluation prompts to provide
        empirical evidence alongside static analysis.
        """
        analysis = self.analyze(report, language)

        return {
            "simulation_summary": analysis.summary,
            "simulation_evidence": {
                "games_run": report.games_run,
                "completion_rate": report.completion_rate,
                "first_player_win_rate": report.first_player_win_rate,
                "avg_turns": report.avg_turns,
            },
            "problematic_cards": [
                {"name": c.card_name, "issue": c.issue_type, "evidence": c.evidence}
                for c in analysis.problematic_cards
            ],
            "balance_insights": {
                "first_player_analysis": analysis.first_player_analysis,
                "strategic_diversity": analysis.strategic_diversity,
                "comeback_potential": analysis.comeback_potential,
            },
            "pacing": {
                "assessment": analysis.pacing_assessment,
                "issues": [{"issue": p.issue, "severity": p.severity} for p in analysis.pacing_issues],
            },
            "recommendations": analysis.high_priority_fixes,
        }
