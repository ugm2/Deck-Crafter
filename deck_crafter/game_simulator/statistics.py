"""
Statistical simulation runner.
Runs multiple games and aggregates metrics.
"""

from deck_crafter.game_simulator.models.game_definition import GameDefinition
from deck_crafter.game_simulator.models.metrics import GameMetrics, SimulationReport
from deck_crafter.game_simulator.agents.base import PlayerAgent
from deck_crafter.game_simulator.agents.random_agent import RandomAgent
from deck_crafter.game_simulator.engine import GameEngine


class SimulationRunner:
    """
    Runs multiple game simulations and produces aggregate statistics.
    """

    def __init__(
        self,
        game_def: GameDefinition,
        num_games: int = 50,
        max_turns: int = 100,
        seed: int | None = None,
    ):
        self.game_def = game_def
        self.num_games = num_games
        self.max_turns = max_turns
        self.base_seed = seed

    def run(
        self,
        agents: list[PlayerAgent] | None = None,
    ) -> SimulationReport:
        """
        Run the simulation campaign.

        Args:
            agents: Player agents to use. Defaults to RandomAgents.

        Returns:
            SimulationReport with aggregate statistics
        """
        # Default to random agents
        if agents is None:
            agents = [
                RandomAgent(seed=self.base_seed + i if self.base_seed else None)
                for i in range(self.game_def.num_players)
            ]

        metrics_list: list[GameMetrics] = []

        for game_num in range(self.num_games):
            # Create engine with deterministic seed for reproducibility
            game_seed = (self.base_seed or 0) + game_num if self.base_seed is not None else None

            engine = GameEngine(
                game_def=self.game_def,
                agents=agents,
                seed=game_seed,
                max_turns=self.max_turns,
            )

            _, metrics = engine.run_game(game_id=f"game_{game_num}")
            metrics_list.append(metrics)

        # Get all card names for statistics
        all_card_names = {card.name for card in self.game_def.cards}

        # Build aggregate report
        report = SimulationReport.from_metrics(
            game_name=self.game_def.name,
            metrics=metrics_list,
            all_card_names=all_card_names,
        )

        return report


def run_quick_simulation(
    game_def: GameDefinition,
    num_games: int = 20,
    seed: int = 42,
) -> SimulationReport:
    """
    Convenience function to quickly run a simulation.
    """
    runner = SimulationRunner(
        game_def=game_def,
        num_games=num_games,
        seed=seed,
    )
    return runner.run()
