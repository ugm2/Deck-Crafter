import difflib
import json
import logging
import re
from datetime import datetime, timezone

from deck_crafter.models.chat import ActionPlan, ChatAction, ChatMessage, PlannedAction
from deck_crafter.models.state import CardGameState
from deck_crafter.services.llm_service import LLMService

logger = logging.getLogger(__name__)

REFINEMENT_PROMPT = """You are an expert game designer improving a card game through targeted edits.
You receive the current game state, evaluation scores, and a history of what worked and failed.
Your job is to plan concrete actions that will improve the game's weakest areas.

## Available Actions

| intent | params | description |
|--------|--------|-------------|
| edit_card | card_name, field, value | Change a card's field (name, description, cost, quantity, effect_type, effect_value, rarity, type) |
| edit_rule | section, value | Change a rules section (turn_structure, win_conditions, resource_mechanics, deck_preparation, initial_hands, turn_limit, scoring_system, additional_rules, glossary) |
| add_card | description | Generate a new card from a natural language description |
| remove_card | card_name | Remove a card from the game |
| regenerate_cards | filter, instruction | Regenerate cards matching filter (e.g. type, rarity) with instruction |
| regenerate_rules | section, instruction | Rewrite a rules section with instruction |

## Current Game State

{state_summary}

## Evaluation Results

{evaluation_details}

## Refinement Memory

{refinement_memory}

## Instructions

Analyze the evaluation and plan targeted actions to improve the game.

RULES:
- Focus on the weakest metrics first (weighted by importance: playability 2.0, balance 1.5, clarity 1.2, theme_alignment 1.0, innovation 0.8)
- Make SMALL, SURGICAL changes. Do not overhaul everything at once.
- If a metric is blocked (ceiling detected), skip it and target something else.
- NEVER repeat a failed pattern from the refinement memory.
- Each edit_card action changes ONE field. Use multiple actions for multiple fields.
- Prefer edit_card/edit_rule over regenerate_cards/regenerate_rules — targeted fixes beat rewrites.
- If evaluation suggestions mention specific cards or rules, address those directly.
- Set response_language to match the game's language.
- Set understanding to explain your reasoning: what you're targeting and why.
"""

ORCHESTRATOR_PROMPT = """You are the game editing assistant for Deck-Crafter.
You receive a user message about a card game and must decide what actions to take.

## Available Actions

| intent | params | description |
|--------|--------|-------------|
| edit_card | card_name, field, value | Change a card's field (name, description, cost, quantity, effect_type, effect_value, rarity, type) |
| edit_rule | section, value | Change a rules section (turn_structure, win_conditions, resource_mechanics, deck_preparation, initial_hands, turn_limit, scoring_system, additional_rules, glossary) |
| add_card | description | Generate a new card from a natural language description |
| remove_card | card_name | Remove a card from the game |
| regenerate_cards | filter, instruction | Regenerate cards matching filter (e.g. type, rarity) with instruction |
| regenerate_rules | section, instruction | Rewrite a rules section with instruction |
| improve_metric | metric | Run a refinement iteration targeting a specific metric (playability, balance, clarity, theme_alignment, innovation) |
| improve_general | (none) | Run a refinement iteration targeting the weakest metric |
| evaluate | (none) | Run simulation + full panel evaluation |
| simulate | (none) | Run gameplay simulation only |
| query | question | Answer a question about the game state |
| explain | topic | Explain evaluation scores, simulation results, or game mechanics |
| undo | (none) | Revert the last change |

## Current Game State

{state_summary}

## Conversation History

{chat_history}

## User Message

{message}

## Instructions

Analyze the user's message and output an ActionPlan.
- Set `needs_clarification=true` ONLY if the request is genuinely ambiguous or destructive.
- For card names, use the closest match from the game — don't ask for clarification on minor typos.
- You can plan multiple actions for a single message.
- Set `response_language` to match the game's language (check the state summary).
- Prefer direct edits (edit_card, edit_rule) over regeneration when the change is specific.
- Use improve_metric/improve_general only for vague "make it better" requests.
"""


class OrchestratorAgent:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self._undo_stack: list[str] = []
        self._max_undo = 10

    def process_message(
        self,
        message: str,
        state: CardGameState,
        chat_history: list[ChatMessage] | None = None,
        run_eval: bool = False,
        eval_workflow=None,
        num_sim_games: int = 30,
    ) -> tuple[CardGameState, str, list[ChatAction], bool]:
        """Process a user message and return (updated_state, response_text, actions, eval_ran)."""
        chat_history = chat_history or []
        score_before = state.evaluation.overall_score if state.evaluation else None

        # 1. Plan
        plan = self._plan(message, state, chat_history)

        if plan.needs_clarification:
            return state, plan.clarification_question or "Could you clarify what you'd like to change?", [], False

        # 2. Snapshot once before any state-modifying actions
        modifying_intents = {
            "edit_card", "edit_rule", "add_card", "remove_card",
            "regenerate_cards", "regenerate_rules", "improve_metric", "improve_general",
        }
        has_modifying = any(a.intent in modifying_intents for a in plan.actions)
        if has_modifying:
            self._push_snapshot(state)

        # 3. Execute
        actions = []
        state_changed = False
        for planned in plan.actions:
            action, new_state, changed = self._execute_action(
                planned, state, eval_workflow, num_sim_games
            )
            actions.append(action)
            if changed:
                state = new_state
                state_changed = True

        # 3. Optional evaluation after changes
        eval_ran = False
        if run_eval and state_changed and eval_workflow:
            eval_action, state = self._run_evaluation(state, eval_workflow, num_sim_games)
            actions.append(eval_action)
            eval_ran = True

        # 4. Generate response
        score_after = state.evaluation.overall_score if state.evaluation else None
        response = self._generate_response(plan, actions, state, score_before, score_after)

        return state, response, actions, eval_ran

    def _plan(self, message: str, state: CardGameState, history: list[ChatMessage]) -> ActionPlan:
        summary = self._build_state_summary(state)
        history_text = self._format_history(history[-15:])

        prompt = ORCHESTRATOR_PROMPT.format(
            state_summary=summary,
            chat_history=history_text or "(no previous messages)",
            message=message,
        )

        result = self.llm.generate(ActionPlan, prompt)
        if result is None:
            return ActionPlan(
                understanding="Failed to parse user request",
                actions=[],
                needs_clarification=True,
                clarification_question="I couldn't understand your request. Could you rephrase it?",
            )
        logger.info(f"[Orchestrator] Plan: {result.understanding} | Actions: {[a.intent for a in result.actions]}")
        return result

    def _execute_action(
        self,
        planned: PlannedAction,
        state: CardGameState,
        eval_workflow=None,
        num_sim_games: int = 30,
    ) -> tuple[ChatAction, CardGameState, bool]:
        """Execute a single planned action. Returns (action_result, new_state, state_changed)."""
        intent = planned.intent
        params = planned.params
        executor = getattr(self, f"_exec_{intent}", None)

        if executor is None:
            return ChatAction(
                intent=intent, description=f"Unknown action: {intent}", success=False
            ), state, False

        try:
            if intent in ("improve_metric", "improve_general"):
                return executor(state, params, eval_workflow, num_sim_games)
            return executor(state, params)
        except Exception as e:
            logger.exception(f"[Orchestrator] Action '{intent}' failed: {e}")
            return ChatAction(
                intent=intent, description=f"Failed: {e}", success=False
            ), state, False

    # --- Direct mutation executors ---

    def _exec_edit_card(self, state: CardGameState, params: dict) -> tuple[ChatAction, CardGameState, bool]:
        card_name = params.get("card_name", "")
        field = params.get("field", "")
        value = params.get("value")

        card = self._find_card(state, card_name)
        if not card:
            return ChatAction(
                intent="edit_card",
                description=f"Card '{card_name}' not found",
                target=card_name,
                success=False,
            ), state, False

        if not hasattr(card, field):
            return ChatAction(
                intent="edit_card",
                description=f"Card field '{field}' does not exist",
                target=card.name,
                success=False,
            ), state, False

        old_value = getattr(card, field)
        # Coerce type
        if isinstance(old_value, int) and not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                pass
        setattr(card, field, value)
        return ChatAction(
            intent="edit_card",
            description=f"Changed {card.name}.{field}: {old_value} → {value}",
            target=card.name,
        ), state, True

    def _exec_edit_rule(self, state: CardGameState, params: dict) -> tuple[ChatAction, CardGameState, bool]:
        section = params.get("section", "")
        value = params.get("value")

        if not state.rules:
            return ChatAction(
                intent="edit_rule", description="No rules to edit", success=False,
            ), state, False

        if not hasattr(state.rules, section):
            return ChatAction(
                intent="edit_rule",
                description=f"Rules section '{section}' does not exist",
                success=False,
            ), state, False

        old_value = getattr(state.rules, section)
        setattr(state.rules, section, value)
        old_display = str(old_value)[:80] if old_value else "None"
        new_display = str(value)[:80] if value else "None"
        return ChatAction(
            intent="edit_rule",
            description=f"Changed rules.{section}: {old_display} → {new_display}",
            target=section,
        ), state, True

    def _exec_remove_card(self, state: CardGameState, params: dict) -> tuple[ChatAction, CardGameState, bool]:
        card_name = params.get("card_name", "")
        card = self._find_card(state, card_name)
        if not card:
            return ChatAction(
                intent="remove_card",
                description=f"Card '{card_name}' not found",
                success=False,
            ), state, False

        state.cards.remove(card)
        return ChatAction(
            intent="remove_card",
            description=f"Removed card: {card.name}",
            target=card.name,
        ), state, True

    def _exec_undo(self, state: CardGameState, params: dict) -> tuple[ChatAction, CardGameState, bool]:
        if not self._undo_stack:
            return ChatAction(
                intent="undo", description="Nothing to undo", success=False,
            ), state, False

        snapshot_json = self._undo_stack.pop()
        restored = CardGameState.model_validate_json(snapshot_json)
        return ChatAction(
            intent="undo", description="Reverted to previous state",
        ), restored, True

    # --- LLM-powered executors ---

    def _exec_add_card(self, state: CardGameState, params: dict) -> tuple[ChatAction, CardGameState, bool]:
        from deck_crafter.models.card import Card

        description = params.get("description", "")
        if not description:
            return ChatAction(
                intent="add_card", description="No card description provided", success=False,
            ), state, False

        game_context = ""
        if state.concept:
            game_context = f"Game: {state.concept.title}, Theme: {state.concept.theme}, Language: {state.concept.language}"
        if state.rules and state.rules.resource_mechanics:
            game_context += f"\nResource system: {state.rules.resource_mechanics[:200]}"

        existing_types = set()
        if state.cards:
            existing_types = {c.type for c in state.cards}

        prompt = f"""Generate a single card for this game.

{game_context}

Existing card types: {', '.join(existing_types) if existing_types else 'None'}

Card request: {description}

Generate the card with all required fields: name, quantity, type, description, cost, image_description.
Match the game's language and theme."""

        card = self.llm.generate(Card, prompt)
        if card is None:
            return ChatAction(
                intent="add_card", description="Failed to generate card", success=False,
            ), state, False

        if state.cards is None:
            state.cards = []
        state.cards.append(card)
        return ChatAction(
            intent="add_card",
            description=f"Added card: {card.name} ({card.type}, cost: {card.cost})",
            target=card.name,
        ), state, True

    def _exec_regenerate_rules(self, state: CardGameState, params: dict) -> tuple[ChatAction, CardGameState, bool]:
        from deck_crafter.agents.rules_agent import RuleGenerationAgent

        section = params.get("section", "")
        instruction = params.get("instruction", "Improve this section")

        if not state.rules or not state.concept:
            return ChatAction(
                intent="regenerate_rules", description="No rules/concept to regenerate from", success=False,
            ), state, False

        agent = RuleGenerationAgent(self.llm)
        new_rules = agent.generate(state.concept, existing_rules=state.rules)
        if new_rules is None:
            return ChatAction(
                intent="regenerate_rules", description="Failed to regenerate rules", success=False,
            ), state, False

        state.rules = new_rules
        return ChatAction(
            intent="regenerate_rules",
            description=f"Regenerated rules (section: {section or 'all'})",
            target=section or "all",
        ), state, True

    def _exec_regenerate_cards(self, state: CardGameState, params: dict) -> tuple[ChatAction, CardGameState, bool]:
        from deck_crafter.agents.card_agent import CardGenerationAgent

        filter_criteria = params.get("filter", "")
        instruction = params.get("instruction", "")

        if not state.cards or not state.concept:
            return ChatAction(
                intent="regenerate_cards", description="No cards/concept to regenerate from", success=False,
            ), state, False

        # Filter cards to regenerate
        cards_to_regen = []
        cards_to_keep = []
        for card in state.cards:
            match = False
            if filter_criteria:
                filter_lower = filter_criteria.lower()
                if filter_lower in (card.type or "").lower():
                    match = True
                elif filter_lower in (card.rarity or "").lower():
                    match = True
                elif filter_lower in card.name.lower():
                    match = True
            if match:
                cards_to_regen.append(card)
            else:
                cards_to_keep.append(card)

        if not cards_to_regen:
            return ChatAction(
                intent="regenerate_cards",
                description=f"No cards matched filter: '{filter_criteria}'",
                success=False,
            ), state, False

        agent = CardGenerationAgent(self.llm)
        new_cards = agent.generate_batch(
            concept=state.concept,
            rules=state.rules,
            num_cards=len(cards_to_regen),
            existing_cards=cards_to_keep,
        )
        if not new_cards:
            return ChatAction(
                intent="regenerate_cards", description="Failed to regenerate cards", success=False,
            ), state, False

        state.cards = cards_to_keep + new_cards
        return ChatAction(
            intent="regenerate_cards",
            description=f"Regenerated {len(new_cards)} cards (filter: {filter_criteria})",
            target=filter_criteria,
        ), state, True

    def _exec_improve_metric(
        self, state: CardGameState, params: dict,
        eval_workflow=None, num_sim_games: int = 30,
    ) -> tuple[ChatAction, CardGameState, bool]:
        from deck_crafter.services.refinement_service import execute_refinement_step

        metric = params.get("metric", "playability")
        if not eval_workflow:
            return ChatAction(
                intent="improve_metric",
                description="No evaluation workflow available",
                success=False,
            ), state, False

        threshold = state.evaluation_threshold or 7.5
        result = execute_refinement_step(
            state=state,
            threshold=threshold,
            llm_service=self.llm,
            eval_workflow=eval_workflow,
            num_simulation_games=num_sim_games,
            use_batch_cards=True,
        )
        desc = f"Refinement iteration ({result.status}): {result.previous_score:.2f} → {result.new_score:.2f}"
        return ChatAction(
            intent="improve_metric",
            description=desc,
            target=metric,
            success=result.status != "reverted",
        ), result.state, True

    def _exec_improve_general(
        self, state: CardGameState, params: dict,
        eval_workflow=None, num_sim_games: int = 30,
    ) -> tuple[ChatAction, CardGameState, bool]:
        return self._exec_improve_metric(state, params, eval_workflow, num_sim_games)

    # --- Read-only executors ---

    def _exec_query(self, state: CardGameState, params: dict) -> tuple[ChatAction, CardGameState, bool]:
        question = params.get("question", "")
        # Build answer from state data directly
        answer = self._answer_query(state, question)
        return ChatAction(
            intent="query", description=answer,
        ), state, False

    def _exec_explain(self, state: CardGameState, params: dict) -> tuple[ChatAction, CardGameState, bool]:
        topic = params.get("topic", "")
        explanation = self._build_explanation(state, topic)
        return ChatAction(
            intent="explain", description=explanation,
        ), state, False

    def _exec_evaluate(self, state: CardGameState, params: dict) -> tuple[ChatAction, CardGameState, bool]:
        # This is a no-op here; actual evaluation runs via run_eval flag or explicitly
        return ChatAction(
            intent="evaluate",
            description="Evaluation requested — will run simulation + panel evaluation",
        ), state, False

    def _exec_simulate(self, state: CardGameState, params: dict) -> tuple[ChatAction, CardGameState, bool]:
        return ChatAction(
            intent="simulate",
            description="Simulation requested — will run gameplay simulation",
        ), state, False

    # --- Evaluation pipeline ---

    def _run_evaluation(self, state: CardGameState, eval_workflow, num_sim_games: int) -> tuple[ChatAction, CardGameState]:
        from deck_crafter.game_simulator.integration import run_simulation_for_game
        from deck_crafter.game_simulator.rule_compiler import normalize_card_resources

        # Normalize resources
        if state.rules and state.cards:
            normalize_card_resources(state.rules, state.cards)

        # Simulation
        try:
            game_name = state.concept.title if state.concept else "Game"
            report, warnings = run_simulation_for_game(
                rules=state.rules, cards=state.cards,
                game_name=game_name, num_games=num_sim_games,
            )
            if report:
                state.simulation_report = report
                state.compilation_warnings = warnings or []
        except Exception as e:
            logger.warning(f"[Orchestrator] Simulation failed: {e}")

        # Panel evaluation
        try:
            import uuid
            game_state_dict = {
                "game_state": state,
            }
            eval_result = eval_workflow.invoke(
                game_state_dict,
                config={"configurable": {"thread_id": f"chat-eval-{uuid.uuid4().hex[:8]}"}},
            )
            if eval_result and "final_evaluation" in eval_result:
                state.evaluation = eval_result["final_evaluation"]
        except Exception as e:
            logger.warning(f"[Orchestrator] Evaluation failed: {e}")

        score = state.evaluation.overall_score if state.evaluation else 0
        return ChatAction(
            intent="evaluate",
            description=f"Evaluation complete. Score: {score:.2f}",
        ), state

    # --- Response generation ---

    def _generate_response(
        self, plan: ActionPlan, actions: list[ChatAction],
        state: CardGameState, score_before: float | None, score_after: float | None,
    ) -> str:
        """Generate a natural language response summarizing what was done."""
        parts = []

        for action in actions:
            status = "" if action.success else " (failed)"
            parts.append(f"- {action.description}{status}")

        if score_before is not None and score_after is not None and score_before != score_after:
            delta = score_after - score_before
            parts.append(f"- Score: {score_before:.2f} → {score_after:.2f} ({delta:+.2f})")

        if not parts:
            return plan.understanding

        return "\n".join(parts)

    # --- Helpers ---

    def _find_card(self, state: CardGameState, name: str):
        if not state.cards:
            return None
        # Exact match first
        for card in state.cards:
            if card.name.lower() == name.lower():
                return card
        # Fuzzy match
        names = [c.name for c in state.cards]
        matches = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
        if matches:
            for card in state.cards:
                if card.name == matches[0]:
                    return card
        return None

    def _push_snapshot(self, state: CardGameState):
        snapshot = state.model_dump_json()
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    def _build_state_summary(self, state: CardGameState) -> str:
        parts = []
        if state.concept:
            parts.append(f"Title: {state.concept.title}")
            parts.append(f"Theme: {state.concept.theme}")
            parts.append(f"Language: {state.concept.language}")
            parts.append(f"Players: {state.concept.number_of_players}")

        if state.evaluation:
            ev = state.evaluation
            parts.append(f"Score: {ev.overall_score:.2f}")
            for metric in ["playability", "balance", "clarity", "theme_alignment", "innovation"]:
                m = getattr(ev, metric, None)
                if m and hasattr(m, "score"):
                    parts.append(f"  {metric}: {m.score}/10")

        if state.rules:
            if state.rules.resource_mechanics:
                parts.append(f"Resource: {state.rules.resource_mechanics[:100]}")
            if state.rules.win_conditions:
                parts.append(f"Win conditions: {state.rules.win_conditions[:100]}")
            if state.rules.turn_limit:
                parts.append(f"Turn limit: {state.rules.turn_limit}")

        if state.cards:
            parts.append(f"Cards: {len(state.cards)} unique")
            types = {}
            for c in state.cards:
                types[c.type] = types.get(c.type, 0) + 1
            parts.append(f"Types: {', '.join(f'{t}({n})' for t, n in types.items())}")
            parts.append("Card list:")
            for c in state.cards:
                effect = ""
                if c.effect_type and c.effect_type != "none":
                    effect = f", effect: {c.effect_type}"
                    if c.effect_value:
                        effect += f" {c.effect_value}"
                parts.append(f"  - {c.name} ({c.type}, cost: {c.cost or 'free'}, qty: {c.quantity}{effect})")

        if state.simulation_report:
            sr = state.simulation_report
            parts.append(f"Simulation: {sr.completion_rate*100:.0f}% completion, avg {sr.avg_turns:.0f} turns")
            if sr.issues:
                parts.append(f"Issues: {'; '.join(sr.issues[:3])}")

        return "\n".join(parts)

    def _format_history(self, history: list[ChatMessage]) -> str:
        if not history:
            return ""
        lines = []
        for msg in history:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)

    def _answer_query(self, state: CardGameState, question: str) -> str:
        """Answer a query about game state without LLM — just formatted data."""
        q = question.lower()

        if "cost" in q or "costo" in q or "expensive" in q or "caro" in q:
            if state.cards:
                cards_with_cost = [
                    (c.name, c.cost, self._parse_cost_number(c.cost))
                    for c in state.cards if c.cost
                ]
                cards_with_cost.sort(key=lambda x: x[2])
                # Extract numeric filter from question (e.g. "more than 3", "greater than 3", "más de 3")
                filtered = self._filter_by_number(cards_with_cost, q, key=lambda x: x[2])
                if filtered is not None:
                    if not filtered:
                        return "No cards match that filter."
                    return "Matching cards:\n" + "\n".join(f"  {name}: {cost}" for name, cost, _ in filtered)
                return "Cards by cost:\n" + "\n".join(f"  {name}: {cost}" for name, cost, _ in cards_with_cost)

        if "type" in q or "tipo" in q:
            if state.cards:
                by_type = {}
                for c in state.cards:
                    by_type.setdefault(c.type, []).append(c.name)
                return "Cards by type:\n" + "\n".join(
                    f"  {t}: {', '.join(names)}" for t, names in by_type.items()
                )

        if "score" in q or "puntuación" in q or "evaluation" in q:
            if state.evaluation:
                ev = state.evaluation
                lines = [f"Overall: {ev.overall_score:.2f}"]
                for m in ["playability", "balance", "clarity", "theme_alignment", "innovation"]:
                    metric = getattr(ev, m, None)
                    if metric and hasattr(metric, "score"):
                        lines.append(f"  {m}: {metric.score}/10")
                return "\n".join(lines)

        if "card" in q or "carta" in q or "how many" in q or "cuántas" in q or "cuantas" in q:
            if state.cards:
                return f"Total cards: {len(state.cards)}\n" + "\n".join(
                    f"  {c.name} ({c.type}, cost: {c.cost})" for c in state.cards
                )

        # Fallback: return summary
        return self._build_state_summary(state)

    @staticmethod
    def _filter_by_number(items: list, question: str, key=None) -> list | None:
        """Extract a numeric comparison from the question and filter items.

        Supports: "more/greater/higher than N", "less/lower/fewer than N",
        "equal to N", "exactly N", and Spanish equivalents.
        Returns None if no comparison found (caller falls back to listing all).
        """
        patterns = [
            # more than / greater than / higher than / above / más de / mayor que
            (r"(?:more|greater|higher|above)\s+than\s+(\d+)", "gt"),
            (r"(?:más|mas)\s+de\s+(\d+)", "gt"),
            (r"(?:mayor|mayores)\s+(?:que|de|a)\s+(\d+)", "gt"),
            (r">\s*(\d+)", "gt"),
            # less than / lower than / below / fewer than / menos de / menor que
            (r"(?:less|lower|fewer|below)\s+than\s+(\d+)", "lt"),
            (r"(?:menos)\s+de\s+(\d+)", "lt"),
            (r"(?:menor|menores)\s+(?:que|de|a)\s+(\d+)", "lt"),
            (r"<\s*(\d+)", "lt"),
            # at least / al menos / >=
            (r"(?:at\s+least|>=)\s*(\d+)", "gte"),
            (r"(?:al\s+menos)\s+(\d+)", "gte"),
            # at most / como mucho / <=
            (r"(?:at\s+most|<=)\s*(\d+)", "lte"),
            (r"(?:como\s+mucho|como\s+máximo)\s+(\d+)", "lte"),
            # equal / exactly / igual a / exactamente
            (r"(?:equal\s+to|exactly|=)\s*(\d+)", "eq"),
            (r"(?:igual\s+a|exactamente)\s+(\d+)", "eq"),
        ]

        q = question.lower()
        for pattern, op in patterns:
            m = re.search(pattern, q)
            if m:
                threshold = int(m.group(1))
                ops = {
                    "gt": lambda v: v > threshold,
                    "lt": lambda v: v < threshold,
                    "gte": lambda v: v >= threshold,
                    "lte": lambda v: v <= threshold,
                    "eq": lambda v: v == threshold,
                }
                extract = key or (lambda x: x)
                return [item for item in items if ops[op](extract(item))]

        return None

    @staticmethod
    def _parse_cost_number(cost: str) -> float:
        """Extract the leading number from a cost string like '3 Crédito'."""
        m = re.match(r"(\d+(?:\.\d+)?)", cost.strip())
        return float(m.group(1)) if m else 0.0

    def _build_explanation(self, state: CardGameState, topic: str) -> str:
        """Build an explanation about a specific topic."""
        t = topic.lower()

        if state.evaluation:
            ev = state.evaluation
            for metric in ["playability", "balance", "clarity", "theme_alignment", "innovation"]:
                if metric in t or metric.replace("_", " ") in t:
                    m = getattr(ev, metric, None)
                    if m:
                        lines = [f"{metric}: {m.score}/10"]
                        if hasattr(m, "analysis") and m.analysis:
                            lines.append(f"Analysis: {m.analysis}")
                        if hasattr(m, "suggestions") and m.suggestions:
                            lines.append("Suggestions:")
                            for s in m.suggestions:
                                lines.append(f"  - {s}")
                        return "\n".join(lines)

            # General evaluation explanation
            lines = [f"Overall score: {ev.overall_score:.2f}"]
            if ev.summary:
                lines.append(ev.summary)
            return "\n".join(lines)

        return "No evaluation data available. Run an evaluation first."

    # === Chat-driven refinement loop ===

    def run_refinement_loop(
        self,
        state: CardGameState,
        eval_workflow,
        num_iterations: int = 5,
        num_sim_games: int = 30,
        on_iteration=None,
        save_state=None,
    ) -> CardGameState:
        """Run a chat-driven refinement loop.

        Each iteration: plan edits → execute → evaluate → compare → keep or rollback.

        Args:
            state: Current game state (must have evaluation).
            eval_workflow: Panel evaluation workflow.
            num_iterations: Max iterations to run.
            num_sim_games: Number of simulation games per evaluation.
            on_iteration: Optional callback(iteration, state, score_before, score_after, actions, kept).
        """
        from deck_crafter.models.state import RefinementMemory

        if not state.refinement_memory:
            state.refinement_memory = RefinementMemory()
        memory = state.refinement_memory

        for i in range(1, num_iterations + 1):
            iteration_num = state.evaluation_iteration + 1
            score_before = state.evaluation.overall_score if state.evaluation else 0.0

            logger.info(f"[Refinement] === Iteration {iteration_num} (loop {i}/{num_iterations}) | Score: {score_before:.2f} ===")

            # 1. Build refinement prompt
            plan = self._plan_refinement(state, memory)
            if not plan.actions:
                logger.warning(f"[Refinement] No actions planned, stopping.")
                if on_iteration:
                    on_iteration(iteration_num, state, score_before, score_before, [], True)
                break

            logger.info(f"[Refinement] Plan: {plan.understanding}")
            logger.info(f"[Refinement] Actions: {[a.intent for a in plan.actions]}")

            # 2. Snapshot before changes
            snapshot = state.model_dump_json()

            try:
                # 3. Execute actions
                actions = []
                for planned in plan.actions:
                    action, new_state, changed = self._execute_action(planned, state, eval_workflow, num_sim_games)
                    actions.append(action)
                    if changed:
                        state = new_state

                # 4. Evaluate
                eval_action, state = self._run_evaluation(state, eval_workflow, num_sim_games)
                actions.append(eval_action)
                score_after = state.evaluation.overall_score if state.evaluation else 0.0
            except Exception as e:
                logger.exception(f"[Refinement] Iteration {iteration_num} crashed: {e}")
                # Restore from snapshot
                try:
                    state = CardGameState.model_validate_json(snapshot)
                except Exception:
                    pass
                memory.total_failed_iterations += 1
                memory.lessons_learned.append(f"Iteration {iteration_num}: CRASHED ({e})")
                actions = []
                if on_iteration:
                    on_iteration(iteration_num, state, score_before, score_before, actions, False)
                continue

            # 5. Compare & decide
            delta = score_after - score_before
            kept = delta >= -0.1  # Keep if improved or within tolerance

            if kept:
                state.evaluation_iteration = iteration_num
                if delta > 0.05:
                    # Record success
                    memory.consecutive_rollbacks = 0
                    memory.record_success("overall", score_after, iteration_num)
                    memory.successful_patterns.append(
                        f"Iteration {iteration_num}: {plan.understanding} | {score_before:.2f} → {score_after:.2f}"
                    )
                    logger.info(f"[Refinement] KEPT (improved): {score_before:.2f} → {score_after:.2f} (+{delta:.2f})")
                else:
                    logger.info(f"[Refinement] KEPT (within tolerance): {score_before:.2f} → {score_after:.2f} ({delta:+.2f})")

                # Update best score
                if state.best_score_achieved is None or score_after > state.best_score_achieved:
                    state.best_score_achieved = score_after
            else:
                # Rollback
                try:
                    state = CardGameState.model_validate_json(snapshot)
                except Exception as e:
                    logger.error(f"[Refinement] Rollback deserialization failed: {e}")
                memory.consecutive_rollbacks += 1
                memory.total_failed_iterations += 1
                memory.lessons_learned.append(
                    f"Iteration {iteration_num}: REVERTED ({score_before:.2f} → {score_after:.2f}). "
                    f"Plan was: {plan.understanding}"
                )
                memory.failed_pattern_strings.append(
                    f"FAILED iteration {iteration_num}: {plan.understanding} | "
                    f"Actions: {', '.join(a.intent for a in actions)} | "
                    f"{score_before:.2f} → {score_after:.2f}"
                )
                logger.warning(f"[Refinement] REVERTED: {score_before:.2f} → {score_after:.2f} ({delta:+.2f})")

            state.refinement_memory = memory

            if on_iteration:
                on_iteration(iteration_num, state, score_before, score_after if kept else score_before, actions, kept)

            # Save after every iteration
            if save_state:
                try:
                    save_state(state)
                except Exception as e:
                    logger.warning(f"[Refinement] Failed to save state: {e}")

        return state

    def _plan_refinement(self, state: CardGameState, memory: "RefinementMemory") -> ActionPlan:
        """Build a refinement-specific prompt and get an action plan."""
        summary = self._build_state_summary(state)
        eval_details = self._build_evaluation_details(state)
        memory_text = self._format_refinement_memory(memory)

        prompt = REFINEMENT_PROMPT.format(
            state_summary=summary,
            evaluation_details=eval_details,
            refinement_memory=memory_text,
        )

        result = self.llm.generate(ActionPlan, prompt)
        if result is None:
            return ActionPlan(understanding="Failed to generate refinement plan", actions=[])
        return result

    def _build_evaluation_details(self, state: CardGameState) -> str:
        """Build detailed evaluation text for the refinement prompt."""
        if not state.evaluation:
            return "No evaluation available."

        ev = state.evaluation
        lines = [f"Overall score: {ev.overall_score:.2f}"]
        weights = {"playability": 2.0, "balance": 1.5, "clarity": 1.2, "theme_alignment": 1.0, "innovation": 0.8}

        for metric in ["playability", "balance", "clarity", "theme_alignment", "innovation"]:
            m = getattr(ev, metric, None)
            if not m:
                continue
            score = m.score if hasattr(m, "score") else "?"
            weight = weights.get(metric, 1.0)
            lines.append(f"\n### {metric} ({score}/10, weight: {weight})")
            if hasattr(m, "analysis") and m.analysis:
                lines.append(f"Analysis: {m.analysis[:500]}")
            if hasattr(m, "suggestions") and m.suggestions:
                for s in m.suggestions[:3]:
                    lines.append(f"  - {s}")

        if ev.summary:
            lines.append(f"\nSummary: {ev.summary[:300]}")

        # Simulation data
        if state.simulation_report:
            sr = state.simulation_report
            lines.append(f"\nSimulation: {sr.completion_rate*100:.0f}% completion, avg {sr.avg_turns:.0f} turns")
            if sr.issues:
                for issue in sr.issues[:5]:
                    lines.append(f"  Issue: {issue}")

        return "\n".join(lines)

    def _format_refinement_memory(self, memory: "RefinementMemory") -> str:
        """Format the RefinementMemory object as text for the prompt."""
        lines = []

        # Score history from experiments
        if memory.experiments:
            lines.append("## Score History")
            for exp in memory.experiments[-10:]:  # Last 10
                result = ""
                if exp.score_after is not None:
                    delta = exp.score_after - exp.score_before
                    confirmed = "confirmed" if exp.hypothesis_confirmed else "rejected"
                    result = f" → {exp.score_after:.2f} ({delta:+.2f}, {confirmed})"
                lines.append(
                    f"  Iter {exp.iteration}: {exp.score_before:.2f}{result} "
                    f"| target={exp.target_metric}, type={exp.intervention_type}"
                )

        # Failed patterns — critical to avoid repeating
        if memory.failed_pattern_strings:
            lines.append("\n## FAILED PATTERNS (do NOT repeat)")
            for fp in memory.failed_pattern_strings[-10:]:
                lines.append(f"  - {fp}")

        # Successful patterns
        if memory.successful_patterns:
            lines.append("\n## Successful Patterns (these worked)")
            for sp in memory.successful_patterns[-5:]:
                lines.append(f"  - {sp}")

        # Blocked metrics
        blocked = memory.get_blocked_metrics()
        if blocked:
            lines.append(f"\n## BLOCKED METRICS (ceiling detected): {', '.join(blocked)}")
            lines.append("Do not target these metrics — focus on others.")

        # Lessons learned
        if memory.lessons_learned:
            lines.append("\n## Lessons Learned")
            for lesson in memory.lessons_learned[-5:]:
                lines.append(f"  - {lesson}")

        # Problematic cards
        if memory.problematic_cards:
            prob = sorted(memory.problematic_cards.items(), key=lambda x: -x[1])[:5]
            lines.append("\n## Problematic Cards (flagged multiple times)")
            for card, count in prob:
                lines.append(f"  - {card}: flagged {count} times")

        # Stats
        lines.append(f"\nTotal failed iterations: {memory.total_failed_iterations}")
        lines.append(f"Consecutive rollbacks: {memory.consecutive_rollbacks}")

        if not lines:
            return "No refinement history yet. This is the first iteration."

        return "\n".join(lines)
