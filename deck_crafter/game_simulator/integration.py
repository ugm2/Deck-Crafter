"""
Integration helpers for connecting game_simulator with deck_crafter evaluation.

This module provides functions to:
1. Run simulation on a CardGameState
2. Produce GameplayAnalysis that feeds into evaluation agents
"""
import logging

from deck_crafter.models.state import CardGameState

logger = logging.getLogger(__name__)
from deck_crafter.models.rules import Rules
from deck_crafter.models.card import Card
from deck_crafter.services.llm_service import LLMService

from typing import Literal

from deck_crafter.game_simulator.rule_compiler import compile_game
from deck_crafter.game_simulator.statistics import SimulationRunner
from deck_crafter.game_simulator.analysis_agent import GameplayAnalysisAgent
from deck_crafter.game_simulator.models.metrics import SimulationReport, GameplayAnalysis
from deck_crafter.game_simulator.cache import get_cached_report, cache_report
from deck_crafter.game_simulator.agents.strategic_agent import StrategicAgent, HeuristicAgent

AgentType = Literal["random", "heuristic", "strategic"]


def run_simulation_for_game(
    rules: Rules,
    cards: list[Card],
    game_name: str = "Game",
    num_games: int = 30,
    seed: int = 42,
    llm_service: LLMService | None = None,
    use_cache: bool = True,
    agent_type: AgentType = "random",
) -> tuple[SimulationReport, list[str]]:
    """
    Compile rules and run simulation.

    Args:
        rules: Game rules
        cards: List of cards
        game_name: Name for the game
        num_games: Number of games to simulate
        seed: Random seed for reproducibility
        llm_service: Optional LLM service for rule fallback parsing
        use_cache: Whether to use cached results (default True)

    Returns:
        Tuple of (SimulationReport, list of compilation warnings)
    """
    logger.info(f"[Simulation] Starting simulation for '{game_name}' "
               f"(num_games={num_games}, agent_type={agent_type}, use_cache={use_cache})")

    # Check cache first
    if use_cache:
        cached = get_cached_report(rules, cards)
        if cached:
            logger.info(f"[Simulation] Using cached report (completion rate: {cached.completion_rate:.0%})")
            return cached, []

    logger.debug(f"[Simulation] Compiling game rules...")
    # Compile rules to game definition
    game_def, warnings = compile_game(
        rules=rules,
        cards=cards,
        game_name=game_name,
        llm_service=llm_service,
    )
    if warnings:
        logger.warning(f"[Simulation] Compilation warnings ({len(warnings)}): {warnings[:3]}...")

    # Create agents based on type
    agents = None
    if agent_type == "strategic":
        if not llm_service:
            raise ValueError("StrategicAgent requires llm_service")
        agents = [StrategicAgent(llm_service, verbose=False) for _ in range(game_def.num_players)]
        # Set game context for strategic agents
        win_desc = f"{game_def.win_condition.type}: {game_def.win_condition.target_value}"
        for agent in agents:
            agent.set_game_context(win_desc)
    elif agent_type == "heuristic":
        agents = [HeuristicAgent() for _ in range(game_def.num_players)]
    # else: agents=None means RandomAgent (default)

    # Run simulation
    logger.info(f"[Simulation] Running {num_games} games...")
    runner = SimulationRunner(
        game_def=game_def,
        num_games=num_games,
        seed=seed,
    )
    report = runner.run(agents=agents)

    logger.info(f"[Simulation] Completed: {report.games_run} games, "
               f"completion rate: {report.completion_rate:.0%}, "
               f"avg turns: {report.avg_turns:.1f}")
    if report.first_player_win_rate:
        logger.debug(f"[Simulation] First player win rate: {report.first_player_win_rate:.0%}")

    # Cache the result
    if use_cache:
        cache_report(rules, cards, report)
        logger.debug(f"[Simulation] Results cached")

    return report, warnings


def analyze_game(
    state: CardGameState,
    llm_service: LLMService,
    num_games: int = 30,
    seed: int = 42,
) -> GameplayAnalysis:
    """
    Run simulation and produce gameplay analysis for a CardGameState.

    This function:
    1. Compiles the game's rules and cards
    2. Runs N simulated games
    3. Uses GameplayAnalysisAgent to produce qualitative insights

    Args:
        state: The game state to analyze
        llm_service: LLM service for analysis
        num_games: Number of games to simulate
        seed: Random seed

    Returns:
        GameplayAnalysis with insights from simulation
    """
    game_name = state.concept.title if state.concept else "Game"
    logger.info(f"[Simulation] Analyzing game '{game_name}'")

    if not state.rules or not state.cards:
        logger.error("[Simulation] Cannot analyze: missing rules or cards")
        raise ValueError("Game must have rules and cards to simulate")

    # Run simulation
    report, warnings = run_simulation_for_game(
        rules=state.rules,
        cards=state.cards,
        game_name=game_name,
        num_games=num_games,
        seed=seed,
        llm_service=llm_service,
    )

    # Analyze results
    logger.debug("[Simulation] Running GameplayAnalysisAgent...")
    agent = GameplayAnalysisAgent(llm_service)
    language = state.concept.language if state.concept else "English"
    analysis = agent.analyze(report, language=language)

    logger.info(f"[Simulation] Analysis complete: strategic diversity = {analysis.strategic_diversity}, "
               f"pacing = {analysis.pacing_assessment}")
    if analysis.problematic_cards:
        logger.debug(f"[Simulation] Problematic cards: {[c.card_name for c in analysis.problematic_cards]}")

    return analysis


def enrich_state_with_simulation(
    state: CardGameState,
    llm_service: LLMService,
    num_games: int = 30,
    seed: int = 42,
) -> CardGameState:
    """
    Add simulation analysis to a CardGameState.

    This modifies the state in place and also returns it for convenience.

    Args:
        state: The game state to enrich
        llm_service: LLM service for analysis
        num_games: Number of games to simulate
        seed: Random seed

    Returns:
        The same state with simulation_analysis populated
    """
    analysis = analyze_game(state, llm_service, num_games, seed)
    state.simulation_analysis = analysis
    return state


def should_run_simulation(state: CardGameState) -> bool:
    """
    Determine if simulation should be run for this game state.

    Heuristics:
    - Always run if user requested it (future: add flag to state)
    - Run during refinement (after first evaluation)
    - Run if previous evaluation was uncertain/borderline

    Args:
        state: The game state to check

    Returns:
        True if simulation would be beneficial
    """
    # Run if in refinement phase
    if state.evaluation_iteration > 0:
        return True

    # Run if we already have evaluation and it's borderline
    if state.evaluation:
        score = state.evaluation.overall_score
        if 5.5 <= score <= 7.5:  # Borderline range
            return True

    return False
