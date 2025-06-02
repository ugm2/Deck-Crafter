from langgraph.graph import StateGraph, END
from deck_crafter.agents.concept_agent import ConceptGenerationAgent
from deck_crafter.agents.rules_agent import RuleGenerationAgent
from deck_crafter.agents.card_agent import CardGenerationAgent
from deck_crafter.models.state import CardGameState
from deck_crafter.services.llm_service import LLMService
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any
from deck_crafter.agents.preferences_agent import PreferencesGenerationAgent
from deck_crafter.workflow.conditions import should_continue


def create_preferences_workflow(llm_service: LLMService) -> StateGraph:
    """Create a workflow for generating user preferences from a game description and partial preferences."""
    preferences_agent = PreferencesGenerationAgent(llm_service)
    
    def generate_preferences(state: CardGameState) -> CardGameState:
        # Pasar tanto la descripción como las preferencias parciales
        game_description = None
        partial_preferences = None
        if hasattr(state, 'preferences') and state.preferences:
            game_description = getattr(state.preferences, 'game_description', None)
            partial_preferences = state.preferences
        # Si no hay preferencias, buscar si el propio state tiene game_description
        if not game_description and hasattr(state, 'game_description'):
            game_description = state.game_description
        # Generar preferencias completas
        generated_preferences = preferences_agent.generate_preferences(
            game_description=game_description,
            partial_preferences=partial_preferences
        )
        state.preferences = generated_preferences
        return state
    
    workflow = StateGraph(CardGameState)
    workflow.add_node("generate_preferences", generate_preferences)
    workflow.set_entry_point("generate_preferences")
    return workflow.compile(checkpointer=MemorySaver())


def create_concept_workflow(llm_service: LLMService) -> StateGraph:
    """Create a workflow for generating a game concept."""
    concept_agent = ConceptGenerationAgent(llm_service)
    
    def generate_concept(state: CardGameState) -> CardGameState:
        return concept_agent.generate_concept(state)
    
    workflow = StateGraph(CardGameState)
    workflow.add_node("generate_concept", generate_concept)
    workflow.set_entry_point("generate_concept")
    return workflow.compile(checkpointer=MemorySaver())


def create_rules_workflow(llm_service: LLMService) -> StateGraph:
    """Create a workflow specifically for rules generation."""
    rule_agent = RuleGenerationAgent(llm_service)
    workflow = StateGraph(CardGameState)
    workflow.add_node("generate_rules", rule_agent.generate_rules)
    workflow.set_entry_point("generate_rules")
    return workflow.compile(checkpointer=MemorySaver())


def create_cards_workflow(llm_service: LLMService) -> StateGraph:
    """Create a workflow specifically for card generation."""
    card_agent = CardGenerationAgent(llm_service)
    workflow = StateGraph(CardGameState)
    workflow.add_node("generate_cards", card_agent.generate_card)

    # Add conditional logic to decide when to stop generating cards
    workflow.add_conditional_edges("generate_cards", should_continue)

    workflow.set_entry_point("generate_cards")
    return workflow.compile(checkpointer=MemorySaver()) 