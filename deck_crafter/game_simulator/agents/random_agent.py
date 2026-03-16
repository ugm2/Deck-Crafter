import random

from deck_crafter.game_simulator.agents.base import PlayerAgent, Action
from deck_crafter.game_simulator.models.state import GameSimulationState


class RandomAgent(PlayerAgent):
    """
    Baseline agent that picks a random legal action.
    Useful for testing and as a baseline for comparison.
    """

    def __init__(self, seed: int | None = None, prefer_play: bool = True):
        """
        Args:
            seed: Random seed for reproducibility
            prefer_play: If True, prefer playing cards over passing when possible
        """
        self.rng = random.Random(seed)
        self.prefer_play = prefer_play

    def choose_action(
        self,
        state: GameSimulationState,
        legal_actions: list[Action],
    ) -> Action:
        if not legal_actions:
            # Should never happen, but return pass as fallback
            return Action(action_type="pass")

        if self.prefer_play:
            # Prefer playing cards over passing
            play_actions = [a for a in legal_actions if a.action_type == "play_card"]
            if play_actions:
                return self.rng.choice(play_actions)

        return self.rng.choice(legal_actions)
