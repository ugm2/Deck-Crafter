from pydantic import BaseModel, Field
from typing import Any
from copy import deepcopy


class CardInstance(BaseModel):
    """A specific card in play, in hand, or in deck."""
    card_id: str  # Unique ID for this instance
    name: str  # Card name from definition
    properties: dict[str, Any] = Field(default_factory=dict)  # Runtime properties


class PlayerState(BaseModel):
    """State of a single player during simulation."""
    player_id: str
    hand: list[CardInstance] = Field(default_factory=list)
    deck: list[CardInstance] = Field(default_factory=list)
    discard: list[CardInstance] = Field(default_factory=list)
    in_play: list[CardInstance] = Field(default_factory=list)
    resources: dict[str, int] = Field(default_factory=dict)
    properties: dict[str, Any] = Field(default_factory=dict)  # health, points, etc.
    is_eliminated: bool = False

    def draw_cards(self, count: int) -> list[CardInstance]:
        """Draw cards from deck to hand. Returns cards drawn."""
        drawn = self.deck[:count]
        self.deck = self.deck[count:]
        self.hand.extend(drawn)
        return drawn

    def find_card_in_hand(self, card_name: str) -> CardInstance | None:
        """Find a card by name in hand."""
        for card in self.hand:
            if card.name == card_name:
                return card
        return None


class ActionLog(BaseModel):
    """Record of an action taken during simulation."""
    turn: int
    player_id: str
    action_type: str
    card_name: str | None = None
    target: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    result: str = ""


class GameSimulationState(BaseModel):
    """Complete state of a game simulation."""
    game_id: str
    turn_number: int = 1
    current_player_idx: int = 0
    players: list[PlayerState] = Field(default_factory=list)
    action_log: list[ActionLog] = Field(default_factory=list)
    game_over: bool = False
    winner_id: str | None = None
    win_reason: str | None = None
    max_turns: int = 100

    @property
    def current_player(self) -> PlayerState:
        return self.players[self.current_player_idx]

    @property
    def active_players(self) -> list[PlayerState]:
        return [p for p in self.players if not p.is_eliminated]

    def advance_turn(self):
        """Move to next player's turn."""
        # Find next non-eliminated player
        start_idx = self.current_player_idx
        while True:
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
            if not self.players[self.current_player_idx].is_eliminated:
                break
            if self.current_player_idx == start_idx:
                # All players eliminated or back to start
                break

        # Increment turn if we've gone around
        if self.current_player_idx <= start_idx:
            self.turn_number += 1

    def log_action(
        self,
        action_type: str,
        card_name: str | None = None,
        target: str | None = None,
        details: dict | None = None,
        result: str = "",
    ):
        """Log an action to the game history."""
        self.action_log.append(
            ActionLog(
                turn=self.turn_number,
                player_id=self.current_player.player_id,
                action_type=action_type,
                card_name=card_name,
                target=target,
                details=details or {},
                result=result,
            )
        )

    def clone(self) -> "GameSimulationState":
        """Create a deep copy of the state."""
        return GameSimulationState.model_validate(deepcopy(self.model_dump()))
