from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Literal

from game_simulator.models.state import GameSimulationState, CardInstance


class Action(BaseModel):
    """An action a player can take."""
    action_type: Literal["play_card", "pass"]
    card: CardInstance | None = None
    target_player_id: str | None = None


class PlayerAgent(ABC):
    """Base class for AI players."""

    @abstractmethod
    def choose_action(
        self,
        state: GameSimulationState,
        legal_actions: list[Action],
    ) -> Action:
        """
        Select an action from the list of legal actions.

        Args:
            state: Current game state
            legal_actions: List of valid actions to choose from

        Returns:
            The chosen action
        """
        pass

    def on_game_start(self, state: GameSimulationState):
        """Called when a new game starts. Override to reset agent state."""
        pass

    def on_game_end(self, state: GameSimulationState):
        """Called when a game ends. Override for learning or logging."""
        pass
