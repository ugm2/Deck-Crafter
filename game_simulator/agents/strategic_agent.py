"""
StrategicAgent: LLM-powered player that makes strategic decisions.

Unlike RandomAgent, this agent uses an LLM to evaluate the game state
and choose actions strategically. Useful for:
- Testing games with realistic playstyles
- Detecting strategies that are too obvious or too subtle
- Validating that decision-making is meaningful
"""

from pydantic import BaseModel, Field
from game_simulator.agents.base import PlayerAgent, Action
from game_simulator.models.state import GameSimulationState, CardInstance
from deck_crafter.services.llm_service import LLMService


class ActionChoice(BaseModel):
    """LLM's chosen action with reasoning."""
    action: str = Field(..., description="Either 'play_card' or 'pass'")
    card_name: str | None = Field(None, description="Name of the card to play (if action is play_card)")
    reasoning: str = Field(..., description="Brief explanation of why this action was chosen")


class StrategicAgent(PlayerAgent):
    """
    LLM-powered agent that makes strategic decisions.

    Uses the LLM to evaluate:
    - Current hand and available plays
    - Resources and health/points
    - Opponent's state (visible information)
    - Progress toward win condition

    Good for testing if games have meaningful decisions.
    """

    PROMPT_TEMPLATE = """
    You are playing a card game. Analyze the current situation and choose the BEST action.

    ### GAME STATE ###
    Turn: {turn_number}
    You are: {player_id}

    **Your Status:**
    - Hand: {hand_cards}
    - Resources: {resources}
    - Points/Health: {properties}

    **Opponent Status:**
    - Points/Health: {opponent_properties}

    **Win Condition:** {win_condition}

    ### LEGAL ACTIONS ###
    You can take ONE of these actions:
    {legal_actions_text}

    ### STRATEGY HINTS ###
    - If you're behind, prioritize aggressive plays
    - If you're ahead, consider defensive plays
    - Cards with immediate effect are often better than passive cards
    - Consider the cost vs benefit of each card

    ### YOUR DECISION ###
    Choose the action that best advances your position toward winning.
    If playing a card, specify its exact name from your hand.
    """

    def __init__(self, llm_service: LLMService, verbose: bool = False):
        """
        Args:
            llm_service: LLM service for decision making
            verbose: If True, print reasoning for each decision
        """
        self.llm_service = llm_service
        self.verbose = verbose
        self._game_context: dict = {}

    def set_game_context(self, win_condition: str):
        """Set additional context about the game being played."""
        self._game_context["win_condition"] = win_condition

    def choose_action(
        self,
        state: GameSimulationState,
        legal_actions: list[Action],
    ) -> Action:
        if not legal_actions:
            return Action(action_type="pass")

        # Only one option? Take it
        if len(legal_actions) == 1:
            return legal_actions[0]

        # All pass actions? Just pass
        if all(a.action_type == "pass" for a in legal_actions):
            return legal_actions[0]

        try:
            # Build context for LLM
            player = state.current_player
            opponent = next(
                (p for p in state.players if p.player_id != player.player_id),
                None
            )

            # Format hand
            hand_cards = ", ".join(
                f"{c.name}" for c in player.hand
            ) or "Empty"

            # Format resources
            resources = ", ".join(
                f"{k}: {v}" for k, v in player.resources.items()
            ) or "None"

            # Format properties
            properties = ", ".join(
                f"{k}: {v}" for k, v in player.properties.items()
            ) or "None"

            opponent_properties = "Unknown"
            if opponent:
                opponent_properties = ", ".join(
                    f"{k}: {v}" for k, v in opponent.properties.items()
                ) or "None"

            # Format legal actions
            actions_text = []
            for i, action in enumerate(legal_actions):
                if action.action_type == "play_card" and action.card:
                    actions_text.append(f"{i+1}. Play '{action.card.name}'")
                else:
                    actions_text.append(f"{i+1}. Pass (end turn)")
            legal_actions_text = "\n".join(actions_text)

            # Get win condition from context
            win_condition = self._game_context.get("win_condition", "Unknown")

            # Call LLM
            result = self.llm_service.generate(
                output_model=ActionChoice,
                prompt=self.PROMPT_TEMPLATE,
                turn_number=state.turn_number,
                player_id=player.player_id,
                hand_cards=hand_cards,
                resources=resources,
                properties=properties,
                opponent_properties=opponent_properties,
                legal_actions_text=legal_actions_text,
                win_condition=win_condition,
            )

            if self.verbose:
                print(f"[{player.player_id}] {result.action}: {result.reasoning}")

            # Find matching action
            if result.action == "pass":
                return next(
                    (a for a in legal_actions if a.action_type == "pass"),
                    legal_actions[0]
                )

            if result.action == "play_card" and result.card_name:
                # Find the action that plays this card
                for action in legal_actions:
                    if action.action_type == "play_card" and action.card:
                        if action.card.name.lower() == result.card_name.lower():
                            return action

                # Card name didn't match exactly, try partial match
                for action in legal_actions:
                    if action.action_type == "play_card" and action.card:
                        if result.card_name.lower() in action.card.name.lower():
                            return action

            # Fallback: pick first play action, or pass
            play_actions = [a for a in legal_actions if a.action_type == "play_card"]
            if play_actions:
                return play_actions[0]
            return legal_actions[0]

        except Exception as e:
            if self.verbose:
                print(f"LLM decision failed: {e}, falling back to random")
            # Fallback to simple heuristic
            play_actions = [a for a in legal_actions if a.action_type == "play_card"]
            if play_actions:
                return play_actions[0]
            return legal_actions[0]

    def on_game_start(self, state: GameSimulationState):
        """Reset per-game state."""
        pass

    def on_game_end(self, state: GameSimulationState):
        """Could be used for learning in future."""
        pass


class HeuristicAgent(PlayerAgent):
    """
    Rule-based agent that uses simple heuristics.

    Faster than StrategicAgent (no LLM calls) but smarter than RandomAgent.
    Good for quick testing with reasonable play patterns.
    """

    def __init__(self, aggression: float = 0.7):
        """
        Args:
            aggression: 0.0 = very defensive, 1.0 = very aggressive
        """
        self.aggression = aggression

    def choose_action(
        self,
        state: GameSimulationState,
        legal_actions: list[Action],
    ) -> Action:
        if not legal_actions:
            return Action(action_type="pass")

        play_actions = [a for a in legal_actions if a.action_type == "play_card"]
        pass_actions = [a for a in legal_actions if a.action_type == "pass"]

        if not play_actions:
            return pass_actions[0] if pass_actions else legal_actions[0]

        # Sort cards by estimated value
        def card_value(action: Action) -> float:
            if not action.card:
                return 0
            props = action.card.properties

            # Higher value = better to play
            value = 0

            # Points/damage are valuable
            effect_value = props.get("effect_value", 0)
            effect = props.get("effect", "none")

            if effect in ["gain_points", "damage"]:
                value += effect_value * self.aggression
            elif effect == "draw":
                value += effect_value * 0.5  # Draw is valuable but less urgent
            elif effect == "heal":
                value += effect_value * (1 - self.aggression)  # More valuable when defensive
            elif effect == "win_game":
                value += 100  # Always play this

            return value

        # Sort by value descending
        sorted_plays = sorted(play_actions, key=card_value, reverse=True)

        # Play the best card if it has positive value
        best = sorted_plays[0]
        if card_value(best) > 0:
            return best

        # Otherwise pass if we can
        return pass_actions[0] if pass_actions else best
