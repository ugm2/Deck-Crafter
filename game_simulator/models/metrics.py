from pydantic import BaseModel, Field
from typing import Any
from collections import Counter
from itertools import combinations
import statistics
import re


class GameMetrics(BaseModel):
    """Metrics collected from a single game simulation."""
    game_id: str
    completed: bool  # Did the game reach a natural conclusion?
    turns_played: int
    winner_id: str | None = None
    win_reason: str | None = None

    # Per-player stats
    cards_played: dict[str, int] = Field(default_factory=dict)  # player_id -> count
    cards_played_by_name: dict[str, int] = Field(default_factory=dict)  # card_name -> count

    # Action tracking
    total_actions: int = 0
    actions_by_type: dict[str, int] = Field(default_factory=dict)

    # Game state at end
    final_state_summary: dict[str, Any] = Field(default_factory=dict)


class SimulationReport(BaseModel):
    """Aggregate report from multiple game simulations."""
    game_name: str
    games_run: int
    games_completed: int

    # Completion
    completion_rate: float = 0.0

    # Win distribution
    wins_by_player: dict[str, int] = Field(default_factory=dict)  # player_id -> wins
    wins_by_position: dict[int, int] = Field(default_factory=dict)  # position (0,1,...) -> wins
    first_player_win_rate: float = 0.0

    # Turn statistics
    avg_turns: float = 0.0
    min_turns: int = 0
    max_turns: int = 0
    std_turns: float = 0.0

    # Card statistics
    cards_played_total: dict[str, int] = Field(default_factory=dict)  # card_name -> total plays
    cards_never_played: list[str] = Field(default_factory=list)
    cards_always_played: list[str] = Field(default_factory=list)  # Played in >90% of games

    # Card win correlation
    card_win_correlation: dict[str, float] = Field(default_factory=dict)  # card_name -> win rate when played

    # Individual game metrics (optional)
    game_metrics: list[GameMetrics] = Field(default_factory=list)

    # Issues detected
    issues: list[str] = Field(default_factory=list)

    @classmethod
    def from_metrics(
        cls,
        game_name: str,
        metrics: list[GameMetrics],
        all_card_names: set[str],
    ) -> "SimulationReport":
        """Build a report from a list of game metrics."""
        if not metrics:
            return cls(game_name=game_name, games_run=0, games_completed=0)

        games_run = len(metrics)
        completed_games = [m for m in metrics if m.completed]
        games_completed = len(completed_games)

        # Win distribution
        wins_by_player: Counter = Counter()
        wins_by_position: Counter = Counter()

        for m in completed_games:
            if m.winner_id:
                wins_by_player[m.winner_id] += 1
                # Extract position from player_id (e.g., "player_0" -> 0)
                try:
                    pos = int(m.winner_id.split("_")[-1])
                    wins_by_position[pos] += 1
                except (ValueError, IndexError):
                    pass

        # First player win rate
        first_player_wins = wins_by_position.get(0, 0)
        first_player_win_rate = first_player_wins / games_completed if games_completed else 0.0

        # Turn statistics
        turns = [m.turns_played for m in completed_games]
        avg_turns = statistics.mean(turns) if turns else 0.0
        std_turns = statistics.stdev(turns) if len(turns) > 1 else 0.0
        min_turns = min(turns) if turns else 0
        max_turns = max(turns) if turns else 0

        # Card statistics
        cards_played_total: Counter = Counter()
        card_games_played_in: dict[str, set[str]] = {name: set() for name in all_card_names}
        card_wins: dict[str, int] = {name: 0 for name in all_card_names}

        for m in metrics:
            for card_name, count in m.cards_played_by_name.items():
                cards_played_total[card_name] += count
                card_games_played_in.setdefault(card_name, set()).add(m.game_id)

        # Card win correlation: games where card was played AND player won
        for m in completed_games:
            if m.winner_id:
                for card_name in m.cards_played_by_name.keys():
                    card_wins[card_name] = card_wins.get(card_name, 0) + 1

        card_win_correlation = {}
        for card_name in all_card_names:
            games_with_card = len(card_games_played_in.get(card_name, set()))
            if games_with_card > 0:
                card_win_correlation[card_name] = card_wins.get(card_name, 0) / games_with_card

        # Cards never played
        cards_never_played = [name for name in all_card_names if cards_played_total.get(name, 0) == 0]

        # Cards always played (>90% of games)
        threshold = 0.9 * games_run
        cards_always_played = [
            name for name in all_card_names
            if len(card_games_played_in.get(name, set())) >= threshold
        ]

        # Detect issues
        issues = []
        if games_completed / games_run < 0.5:
            issues.append(f"Low completion rate: {games_completed}/{games_run} games completed")
        if first_player_win_rate > 0.7:
            issues.append(f"First player advantage: {first_player_win_rate:.1%} win rate")
        if first_player_win_rate < 0.3:
            issues.append(f"Second player advantage: first player only wins {first_player_win_rate:.1%}")
        if cards_never_played:
            issues.append(f"Dead cards (never played): {cards_never_played}")

        # OP card detection: only flag if win correlation is significantly above average
        # A card that everyone plays will naturally have ~50% win correlation
        # A truly OP card will have much higher win correlation even when rarely played
        avg_win_correlation = (
            sum(card_win_correlation.values()) / len(card_win_correlation)
            if card_win_correlation else 0.5
        )
        for card_name, win_rate in card_win_correlation.items():
            # Card is suspicious if:
            # 1. Very high win correlation (>90%) AND
            # 2. Not played in most games (< 50% of games) - suggesting it's a "silver bullet"
            # OR
            # 3. Win rate is significantly above average (>30% higher than avg)
            games_with_card = len(card_games_played_in.get(card_name, set()))
            played_rate = games_with_card / games_run if games_run else 0

            is_silver_bullet = win_rate > 0.9 and played_rate < 0.5
            is_above_average = win_rate > avg_win_correlation + 0.3 and played_rate > 0.1

            if is_silver_bullet:
                issues.append(
                    f"Potentially OP card: {card_name} has {win_rate:.1%} win rate "
                    f"(played in only {played_rate:.0%} of games)"
                )
            elif is_above_average:
                issues.append(
                    f"Potentially OP card: {card_name} has {win_rate:.1%} win correlation "
                    f"(avg is {avg_win_correlation:.1%})"
                )

        return cls(
            game_name=game_name,
            games_run=games_run,
            games_completed=games_completed,
            completion_rate=games_completed / games_run if games_run else 0.0,
            wins_by_player=dict(wins_by_player),
            wins_by_position=dict(wins_by_position),
            first_player_win_rate=first_player_win_rate,
            avg_turns=avg_turns,
            min_turns=min_turns,
            max_turns=max_turns,
            std_turns=std_turns,
            cards_played_total=dict(cards_played_total),
            cards_never_played=cards_never_played,
            cards_always_played=cards_always_played,
            card_win_correlation=card_win_correlation,
            game_metrics=metrics,
            issues=issues,
        )


class ProblematicCard(BaseModel):
    """A card identified as problematic during gameplay analysis."""
    card_name: str = Field(..., description="Name of the card")
    issue_type: str = Field(..., description="Type of issue: 'overpowered', 'underpowered', 'combo_enabler', 'dead_card'")
    evidence: str = Field(..., description="Statistical evidence supporting this finding")
    suggested_fix: str | None = Field(None, description="Specific fix recommendation")


class PacingIssue(BaseModel):
    """A pacing issue identified during gameplay analysis."""
    issue: str = Field(..., description="Description of the pacing issue")
    severity: str = Field(..., description="'high', 'medium', or 'low'")
    evidence: str = Field(..., description="Game data supporting this finding")


class GameplayAnalysis(BaseModel):
    """
    Qualitative analysis produced by GameplayAnalysisAgent.
    Transforms raw simulation statistics into actionable game design insights.
    """
    # Summary
    summary: str = Field(..., description="2-3 sentence executive summary of gameplay findings")

    # Strategic findings
    dominant_strategies: list[str] = Field(
        default_factory=list,
        description="Dominant strategies or playstyles observed (e.g., 'rush aggro', 'card hoarding')"
    )
    strategic_diversity: str = Field(
        ...,
        description="Assessment of strategic variety: 'high' (many viable paths), 'medium', or 'low' (one optimal path)"
    )

    # Card-level findings
    problematic_cards: list[ProblematicCard] = Field(
        default_factory=list,
        description="Cards that need balance attention"
    )
    well_balanced_cards: list[str] = Field(
        default_factory=list,
        description="Cards that performed as expected (good reference points)"
    )

    # Pacing and flow
    pacing_assessment: str = Field(
        ...,
        description="Overall pacing verdict: 'excellent', 'good', 'needs_work', 'poor'"
    )
    pacing_issues: list[PacingIssue] = Field(
        default_factory=list,
        description="Specific pacing problems identified"
    )

    # Balance insights
    first_player_analysis: str = Field(
        ...,
        description="Analysis of first-player advantage/disadvantage and its causes"
    )
    comeback_potential: str = Field(
        ...,
        description="Assessment of whether losing players can recover: 'high', 'medium', 'low', 'none'"
    )

    # Clarity (from stuck states)
    rule_clarity_issues: list[str] = Field(
        default_factory=list,
        description="Rule ambiguities that caused games to stall or fail"
    )

    # Fun indicators
    fun_indicators: list[str] = Field(
        default_factory=list,
        description="Positive signs for enjoyment (close games, variety, dramatic moments)"
    )
    anti_fun_indicators: list[str] = Field(
        default_factory=list,
        description="Negative signs (steamrolls, repetitive games, lack of decisions)"
    )

    # Concrete recommendations
    high_priority_fixes: list[str] = Field(
        default_factory=list,
        description="Most important changes to make, based on gameplay evidence"
    )
    suggested_experiments: list[str] = Field(
        default_factory=list,
        description="Balance changes to test in future simulations"
    )

    # Specific balance adjustments
    balance_adjustments: list[str] = Field(
        default_factory=list,
        description="Specific numerical adjustments. Format: 'CardName: action (reason)'. "
                    "Examples: 'Dragon: reduce damage from 6 to 4 (85% win correlation)', "
                    "'Healer: reduce heal from 5 to 3 (games never end due to heal spam)', "
                    "'Weak Goblin: increase attack from 1 to 2 (never played, too weak)'"
    )

    # Confidence assessment
    confidence: "AnalysisConfidence | None" = Field(
        default=None,
        description="Confidence assessment for this analysis"
    )


class AnalysisConfidence(BaseModel):
    """Confidence assessment for gameplay analysis."""
    overall: str = Field(
        ...,
        description="Overall confidence: 'high', 'medium', or 'low'"
    )
    sample_size_adequate: bool = Field(
        default=True,
        description="Whether sample size is adequate (20+ games recommended)"
    )
    completion_rate_adequate: bool = Field(
        default=True,
        description="Whether completion rate is adequate (50%+ recommended)"
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Reasons for confidence level"
    )

    @classmethod
    def from_report(cls, report: "SimulationReport") -> "AnalysisConfidence":
        """Calculate confidence from simulation report."""
        reasons = []
        sample_ok = True
        completion_ok = True

        if report.games_run < 10:
            reasons.append(f"Very small sample: only {report.games_run} games (recommend 20+)")
            sample_ok = False
        elif report.games_run < 20:
            reasons.append(f"Small sample: {report.games_run} games (recommend 20+)")

        if report.completion_rate < 0.3:
            reasons.append(f"Very low completion: {report.completion_rate:.0%} (recommend 50%+)")
            completion_ok = False
        elif report.completion_rate < 0.5:
            reasons.append(f"Low completion: {report.completion_rate:.0%} (recommend 50%+)")

        # Determine overall confidence
        if not sample_ok or not completion_ok:
            overall = "low"
        elif report.games_run < 20 or report.completion_rate < 0.5:
            overall = "medium"
        else:
            overall = "high"

        return cls(
            overall=overall,
            sample_size_adequate=sample_ok,
            completion_rate_adequate=completion_ok,
            reasons=reasons,
        )


class CardCombo(BaseModel):
    """A card combination with synergy information."""
    cards: tuple[str, str] = Field(..., description="The two card names")
    games_together: int = Field(..., description="Games where both cards were played")
    win_rate_together: float = Field(..., description="Win rate when combo was played")
    win_rate_solo_avg: float = Field(..., description="Average solo win rate of cards")
    synergy_score: float = Field(..., description="Difference: together - solo_avg")

    @classmethod
    def find_combos(
        cls,
        metrics: list[GameMetrics],
        card_win_correlation: dict[str, float],
        min_games: int = 3,
    ) -> list["CardCombo"]:
        """Find card combos from game metrics."""
        combo_plays: Counter[tuple[str, str]] = Counter()
        combo_wins: Counter[tuple[str, str]] = Counter()

        for m in metrics:
            if not m.completed:
                continue
            cards_played = set(m.cards_played_by_name.keys())
            for c1, c2 in combinations(sorted(cards_played), 2):
                combo_plays[(c1, c2)] += 1
                if m.winner_id:
                    combo_wins[(c1, c2)] += 1

        combos = []
        for (c1, c2), plays in combo_plays.items():
            if plays < min_games:
                continue
            win_rate = combo_wins[(c1, c2)] / plays
            solo_avg = (
                card_win_correlation.get(c1, 0.5) + card_win_correlation.get(c2, 0.5)
            ) / 2
            synergy = win_rate - solo_avg

            combos.append(cls(
                cards=(c1, c2),
                games_together=plays,
                win_rate_together=win_rate,
                win_rate_solo_avg=solo_avg,
                synergy_score=synergy,
            ))

        # Sort by synergy score (highest first)
        return sorted(combos, key=lambda x: -x.synergy_score)


class BalanceAdjustment(BaseModel):
    """A structured balance adjustment parsed from text."""
    card_name: str = Field(..., description="Name of the card to adjust")
    stat: str = Field(..., description="Stat to change: damage, cost, heal, attack, defense")
    current_value: int | None = Field(None, description="Current stat value")
    target_value: int | None = Field(None, description="Target stat value")
    action: str = Field(..., description="Action: reduce, increase, remove")
    reason: str = Field(..., description="Reason from simulation data")

    @classmethod
    def parse_adjustments(cls, adjustments: list[str]) -> list["BalanceAdjustment"]:
        """Parse balance adjustment strings into structured form."""
        parsed = []
        # Pattern: "CardName: reduce/increase stat from X to Y (reason)"
        pattern = r"^(.+?):\s*(reduce|increase|remove)\s+(\w+)(?:\s+from\s+(\d+)\s+to\s+(\d+))?\s*\((.+?)\)$"

        for adj in adjustments:
            match = re.match(pattern, adj.strip(), re.IGNORECASE)
            if match:
                parsed.append(cls(
                    card_name=match.group(1).strip(),
                    action=match.group(2).lower(),
                    stat=match.group(3).lower(),
                    current_value=int(match.group(4)) if match.group(4) else None,
                    target_value=int(match.group(5)) if match.group(5) else None,
                    reason=match.group(6).strip(),
                ))
            else:
                # Fallback: simpler pattern "CardName: action (reason)"
                simple = re.match(r"^(.+?):\s*(.+?)\s*\((.+?)\)$", adj.strip())
                if simple:
                    parsed.append(cls(
                        card_name=simple.group(1).strip(),
                        action="adjust",
                        stat="unknown",
                        current_value=None,
                        target_value=None,
                        reason=simple.group(3).strip(),
                    ))

        return parsed
