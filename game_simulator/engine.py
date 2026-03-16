"""
Game simulation engine.
Runs games using a GameDefinition and player agents.
"""

import random
import uuid
from collections import Counter

from game_simulator.models.state import (
    GameSimulationState,
    PlayerState,
    CardInstance,
)
from game_simulator.models.game_definition import GameDefinition, CardEffect
from game_simulator.models.metrics import GameMetrics
from game_simulator.agents.base import PlayerAgent, Action


class GameEngine:
    """
    Executes game simulations.
    Handles game setup, turn loop, action resolution, and win conditions.
    """

    def __init__(
        self,
        game_def: GameDefinition,
        agents: list[PlayerAgent],
        seed: int | None = None,
        max_turns: int = 100,
    ):
        self.game_def = game_def
        self.agents = agents
        self.rng = random.Random(seed)
        self.max_turns = max_turns

        if len(agents) != game_def.num_players:
            raise ValueError(
                f"Expected {game_def.num_players} agents, got {len(agents)}"
            )

    def run_game(self, game_id: str | None = None) -> tuple[GameSimulationState, GameMetrics]:
        """
        Run a single game to completion.

        Returns:
            Tuple of (final_state, metrics)
        """
        game_id = game_id or str(uuid.uuid4())[:8]

        # Initialize state
        state = self._setup_game(game_id)

        # Notify agents
        for agent in self.agents:
            agent.on_game_start(state)

        # Track metrics
        cards_played_by_player: Counter = Counter()
        cards_played_by_name: Counter = Counter()
        actions_by_type: Counter = Counter()

        # Main game loop
        while not state.game_over and state.turn_number <= self.max_turns:
            current_player = state.current_player
            current_agent = self.agents[state.current_player_idx]

            # Start of turn: draw cards and gain resources
            self._start_of_turn(state)

            # Get legal actions
            legal_actions = self._get_legal_actions(state)

            # Track cards played this turn
            cards_played_this_turn = 0
            max_cards = self.game_def.rules.max_cards_per_turn

            # Action loop within turn
            while cards_played_this_turn < max_cards and not state.game_over:
                # Filter to remaining legal actions
                legal_actions = self._get_legal_actions(state)

                if not legal_actions:
                    break

                # Agent chooses action
                action = current_agent.choose_action(state, legal_actions)
                actions_by_type[action.action_type] += 1

                if action.action_type == "pass":
                    state.log_action("pass", result="Turn ended")
                    break

                if action.action_type == "play_card" and action.card:
                    # Execute the card
                    result = self._execute_card(state, action.card, action.target_player_id)

                    # Track metrics
                    cards_played_by_player[current_player.player_id] += 1
                    cards_played_by_name[action.card.name] += 1
                    cards_played_this_turn += 1

                    state.log_action(
                        "play_card",
                        card_name=action.card.name,
                        target=action.target_player_id,
                        result=result,
                    )

                    # Check win condition after each card
                    self._check_win_condition(state)

            # End of turn
            state.advance_turn()

        # Handle timeout
        if state.turn_number > self.max_turns and not state.game_over:
            state.game_over = True
            state.win_reason = "Turn limit reached"

        # Notify agents
        for agent in self.agents:
            agent.on_game_end(state)

        # Build metrics
        metrics = GameMetrics(
            game_id=game_id,
            completed=state.winner_id is not None,
            turns_played=state.turn_number,
            winner_id=state.winner_id,
            win_reason=state.win_reason,
            cards_played=dict(cards_played_by_player),
            cards_played_by_name=dict(cards_played_by_name),
            total_actions=sum(actions_by_type.values()),
            actions_by_type=dict(actions_by_type),
            final_state_summary=self._summarize_final_state(state),
        )

        return state, metrics

    def _setup_game(self, game_id: str) -> GameSimulationState:
        """Initialize game state from definition."""
        # Build and shuffle deck
        deck_data = self.game_def.build_deck()
        self.rng.shuffle(deck_data)

        # Create card instances
        deck = [
            CardInstance(
                card_id=cd["card_id"],
                name=cd["name"],
                properties={
                    "cost": cd["cost"],
                    "effect": cd["effect"],
                    "effect_value": cd["effect_value"],
                    "effect_target": cd["effect_target"],
                    **cd["properties"],
                },
            )
            for cd in deck_data
        ]

        # Create players
        players = []
        for i in range(self.game_def.num_players):
            player = PlayerState(
                player_id=f"player_{i}",
                resources=dict(self.game_def.rules.initial_resources),
                properties=dict(self.game_def.rules.initial_properties),
            )
            players.append(player)

        # Distribute cards to players (deal initial hands)
        cards_per_player = len(deck) // self.game_def.num_players
        for i, player in enumerate(players):
            start = i * cards_per_player
            end = start + cards_per_player
            player.deck = deck[start:end]

        # Draw initial hands
        for player in players:
            player.draw_cards(self.game_def.rules.initial_hand_size)

        return GameSimulationState(
            game_id=game_id,
            players=players,
            max_turns=self.max_turns,
        )

    def _start_of_turn(self, state: GameSimulationState):
        """Handle start of turn effects."""
        player = state.current_player

        # Draw cards
        draw_count = self.game_def.rules.draw_per_turn
        if draw_count > 0 and player.deck:
            drawn = player.draw_cards(draw_count)
            if drawn:
                state.log_action("draw", details={"count": len(drawn)}, result=f"Drew {len(drawn)} cards")

        # Gain resources
        for resource, amount in self.game_def.rules.resource_per_turn.items():
            player.resources[resource] = player.resources.get(resource, 0) + amount

    def _get_legal_actions(self, state: GameSimulationState) -> list[Action]:
        """Get all legal actions for current player."""
        actions = []
        player = state.current_player

        # Can always pass
        actions.append(Action(action_type="pass"))

        # Check each card in hand
        for card in player.hand:
            cost = card.properties.get("cost", {})

            # Check if player can afford the card
            can_afford = all(
                player.resources.get(res, 0) >= amount
                for res, amount in cost.items()
            )

            if can_afford:
                effect_target = card.properties.get("effect_target", "self")

                if effect_target == "self":
                    actions.append(Action(action_type="play_card", card=card))
                elif effect_target == "opponent":
                    # Target each opponent
                    for other in state.players:
                        if other.player_id != player.player_id and not other.is_eliminated:
                            actions.append(
                                Action(
                                    action_type="play_card",
                                    card=card,
                                    target_player_id=other.player_id,
                                )
                            )
                elif effect_target == "any":
                    # Can target any player including self
                    for other in state.players:
                        if not other.is_eliminated:
                            actions.append(
                                Action(
                                    action_type="play_card",
                                    card=card,
                                    target_player_id=other.player_id,
                                )
                            )

        return actions

    def _execute_card(
        self,
        state: GameSimulationState,
        card: CardInstance,
        target_player_id: str | None,
    ) -> str:
        """Execute a card's effect. Returns description of what happened."""
        player = state.current_player

        # Pay cost
        cost = card.properties.get("cost", {})
        for res, amount in cost.items():
            player.resources[res] = player.resources.get(res, 0) - amount

        # Remove card from hand and add to discard
        player.hand = [c for c in player.hand if c.card_id != card.card_id]
        player.discard.append(card)

        # Get effect details
        effect = card.properties.get("effect", CardEffect.NONE)
        if isinstance(effect, str):
            effect = CardEffect(effect)
        effect_value = card.properties.get("effect_value", 0)

        # Resolve target
        target = player  # Default to self
        if target_player_id:
            for p in state.players:
                if p.player_id == target_player_id:
                    target = p
                    break

        # Apply effect
        result = f"Played {card.name}"

        if effect == CardEffect.NONE:
            result = f"Played {card.name} (no effect)"

        elif effect == CardEffect.DRAW:
            drawn = player.draw_cards(effect_value)
            result = f"Drew {len(drawn)} cards"

        elif effect == CardEffect.GAIN_POINTS:
            player.properties["points"] = player.properties.get("points", 0) + effect_value
            result = f"Gained {effect_value} points (now {player.properties['points']})"

        elif effect == CardEffect.DAMAGE:
            target.properties["health"] = target.properties.get("health", 0) - effect_value
            result = f"Dealt {effect_value} damage to {target.player_id}"
            if target.properties["health"] <= 0:
                target.is_eliminated = True
                result += f" - {target.player_id} eliminated!"

        elif effect == CardEffect.HEAL:
            target.properties["health"] = target.properties.get("health", 0) + effect_value
            result = f"Healed {effect_value} to {target.player_id}"

        elif effect == CardEffect.GAIN_RESOURCE:
            resource_type = card.properties.get("resource_type", "generic")
            player.resources[resource_type] = player.resources.get(resource_type, 0) + effect_value
            result = f"Gained {effect_value} {resource_type}"

        elif effect == CardEffect.WIN_GAME:
            state.game_over = True
            state.winner_id = player.player_id
            state.win_reason = f"{card.name} played - instant win"
            result = f"GAME OVER - {player.player_id} wins!"

        return result

    def _check_win_condition(self, state: GameSimulationState):
        """Check if any player has won."""
        if state.game_over:
            return

        win_cond = self.game_def.win_condition

        if win_cond.type == "points":
            for player in state.players:
                points = player.properties.get("points", 0)
                if points >= win_cond.target_value:
                    state.game_over = True
                    state.winner_id = player.player_id
                    state.win_reason = f"Reached {points} points"
                    return

        elif win_cond.type == "elimination":
            alive = [p for p in state.players if not p.is_eliminated]
            if len(alive) <= 1:
                state.game_over = True
                if alive:
                    state.winner_id = alive[0].player_id
                    state.win_reason = "Last player standing"
                else:
                    state.win_reason = "All players eliminated"

        elif win_cond.type == "last_standing":
            alive = [p for p in state.players if not p.is_eliminated]
            if len(alive) == 1:
                state.game_over = True
                state.winner_id = alive[0].player_id
                state.win_reason = "Last player standing"

        elif win_cond.type == "property_threshold":
            prop_name = win_cond.property_name
            for player in state.players:
                value = player.properties.get(prop_name, 0)
                if value >= win_cond.target_value:
                    state.game_over = True
                    state.winner_id = player.player_id
                    state.win_reason = f"Reached {value} {prop_name}"
                    return

    def _summarize_final_state(self, state: GameSimulationState) -> dict:
        """Create a summary of the final game state."""
        return {
            "turns": state.turn_number,
            "winner": state.winner_id,
            "reason": state.win_reason,
            "players": {
                p.player_id: {
                    "points": p.properties.get("points", 0),
                    "health": p.properties.get("health", 0),
                    "cards_in_hand": len(p.hand),
                    "cards_in_deck": len(p.deck),
                    "eliminated": p.is_eliminated,
                }
                for p in state.players
            },
        }
