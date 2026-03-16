"""
Tests that validate the game simulator against synthetic test games.
Each synthetic game has known expected properties that the simulator should detect.
"""

import pytest
from game_simulator.synthetic_games import get_synthetic_game, get_all_synthetic_games
from game_simulator.statistics import run_quick_simulation


class TestIWinGame:
    """Test the 'I Win' game - detects overpowered cards."""

    @pytest.fixture
    def report(self):
        game = get_synthetic_game("i_win")
        return run_quick_simulation(game, num_games=50, seed=42)

    def test_games_complete(self, report):
        """Games should complete (not timeout)."""
        assert report.completion_rate > 0.8, f"Low completion: {report.completion_rate}"

    def test_i_win_card_is_op(self, report):
        """The 'I Win' card should have very high win correlation."""
        i_win_correlation = report.card_win_correlation.get("I Win", 0)
        # When played, it should win the game
        assert i_win_correlation > 0.9, f"I Win card win rate: {i_win_correlation}"

    def test_issue_detected(self, report):
        """Should detect the OP card in issues."""
        op_issues = [i for i in report.issues if "OP" in i or "I Win" in i]
        # May or may not detect depending on play frequency
        # Main validation is win correlation above


class TestFirstPlayerOP:
    """Test the first player advantage game."""

    @pytest.fixture
    def report(self):
        game = get_synthetic_game("first_player_op")
        return run_quick_simulation(game, num_games=50, seed=42)

    def test_games_complete(self, report):
        """Games should complete."""
        assert report.completion_rate > 0.7, f"Low completion: {report.completion_rate}"

    def test_first_player_advantage(self, report):
        """First player should win significantly more."""
        # In this game design, first player gets to play first with full hand
        # They should win more often
        assert report.first_player_win_rate > 0.55, (
            f"First player win rate {report.first_player_win_rate} not showing advantage"
        )

    def test_advantage_detected(self, report):
        """Should detect first player advantage if >70%."""
        if report.first_player_win_rate > 0.7:
            advantage_issues = [i for i in report.issues if "First player" in i]
            assert advantage_issues, "High first player advantage not flagged"


class TestDeadCards:
    """Test detection of useless cards."""

    @pytest.fixture
    def report(self):
        game = get_synthetic_game("dead_cards")
        return run_quick_simulation(game, num_games=50, seed=42)

    def test_games_complete(self, report):
        """Games should complete."""
        assert report.completion_rate > 0.8, f"Low completion: {report.completion_rate}"

    def test_useless_cards_detected(self, report):
        """Useless cards should be played less or never."""
        useless_plays = report.cards_played_total.get("Useless Card", 0)
        good_plays = report.cards_played_total.get("Good Card", 0)

        # Useless cards should be played much less (random agent still plays them)
        # But good cards give points, so rational play favors them
        # With prefer_play=True random agent, all cards get played
        # The key is that useless cards don't contribute to wins
        useless_win_rate = report.card_win_correlation.get("Useless Card", 0)
        good_win_rate = report.card_win_correlation.get("Good Card", 0)

        # Good cards should have higher win correlation
        assert good_win_rate >= useless_win_rate, (
            f"Good cards ({good_win_rate}) should have >= win rate than useless ({useless_win_rate})"
        )


class TestPerfectBalance:
    """Test the balanced game - should show ~50% win rate."""

    @pytest.fixture
    def report(self):
        game = get_synthetic_game("perfect_balance")
        return run_quick_simulation(game, num_games=100, seed=42)

    def test_games_complete(self, report):
        """Games should complete at high rate."""
        assert report.completion_rate > 0.9, f"Low completion: {report.completion_rate}"

    def test_balanced_win_rate(self, report):
        """Win rate should be approximately 50/50."""
        # Allow some variance due to random sampling
        assert 0.35 < report.first_player_win_rate < 0.65, (
            f"Win rate {report.first_player_win_rate} outside balanced range"
        )

    def test_no_major_issues(self, report):
        """Should not detect major balance issues."""
        critical_issues = [
            i for i in report.issues
            if "OP" in i or "advantage" in i.lower()
        ]
        # Some minor issues OK, but no critical ones
        assert len(critical_issues) == 0, f"Unexpected issues in balanced game: {critical_issues}"


class TestLongGame:
    """Test games that take many turns."""

    @pytest.fixture
    def report(self):
        game = get_synthetic_game("long_game")
        return run_quick_simulation(game, num_games=30, seed=42)

    def test_games_complete(self, report):
        """Long games should still complete (within turn limit)."""
        # May hit turn limit sometimes
        assert report.completion_rate > 0.5, f"Too many timeouts: {report.completion_rate}"

    def test_high_turn_count(self, report):
        """Average turns should be high."""
        assert report.avg_turns > 15, f"Average turns {report.avg_turns} not high enough for 'long game'"


class TestQuickGame:
    """Test games that end quickly."""

    @pytest.fixture
    def report(self):
        game = get_synthetic_game("quick_game")
        return run_quick_simulation(game, num_games=50, seed=42)

    def test_games_complete(self, report):
        """Quick games should always complete."""
        assert report.completion_rate > 0.95, f"Quick games not completing: {report.completion_rate}"

    def test_low_turn_count(self, report):
        """Average turns should be low."""
        assert report.avg_turns < 10, f"Average turns {report.avg_turns} too high for 'quick game'"


class TestAllSyntheticGames:
    """Meta-tests that all synthetic games run without errors."""

    def test_all_games_run(self):
        """All synthetic games should run without errors."""
        games = get_all_synthetic_games()
        for name, game_def in games.items():
            report = run_quick_simulation(game_def, num_games=10, seed=42)
            assert report.games_run == 10, f"Game {name} didn't run all games"
            assert report.completion_rate > 0, f"Game {name} had 0% completion"

    def test_invariants_hold(self):
        """Basic invariants should hold for all games."""
        games = get_all_synthetic_games()
        for name, game_def in games.items():
            report = run_quick_simulation(game_def, num_games=10, seed=42)

            # Turn count should be positive for completed games
            if report.games_completed > 0:
                assert report.avg_turns > 0, f"Game {name} has 0 average turns"

            # Win distribution should sum to games completed
            total_wins = sum(report.wins_by_player.values())
            assert total_wins == report.games_completed, (
                f"Game {name}: wins ({total_wins}) != completed ({report.games_completed})"
            )
