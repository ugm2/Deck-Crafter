from langgraph.graph import StateGraph, END
from deck_crafter.agents.concept_agent import ConceptGenerationAgent
from deck_crafter.agents.rules_agent import RuleGenerationAgent
from deck_crafter.agents.card_agent import CardGenerationAgent
from deck_crafter.models.state import CardGameState
from deck_crafter.services.llm_service import LLMService
from langgraph.checkpoint.memory import MemorySaver
from deck_crafter.agents.preferences_agent import PreferencesGenerationAgent
from deck_crafter.workflow.conditions import should_continue


def create_preferences_workflow(llm_service: LLMService) -> StateGraph:
    """Create a workflow for generating user preferences from a game description and partial preferences."""
    preferences_agent = PreferencesGenerationAgent(llm_service)
    
    def generate_preferences(state: CardGameState) -> CardGameState:
        game_description = None
        partial_preferences = None
        if hasattr(state, 'preferences') and state.preferences:
            game_description = getattr(state.preferences, 'game_description', None)
            partial_preferences = state.preferences
        if not game_description and hasattr(state, 'game_description'):
            game_description = state.game_description
        generated_preferences = preferences_agent.generate_preferences(
            game_description=game_description,
            partial_preferences=partial_preferences
        )
        state.preferences = generated_preferences
        return state
    
    workflow = StateGraph(CardGameState)
    
    metadata = {
        "model": llm_service.model_name,
        "provider": llm_service.__class__.__name__,
        "temperature": getattr(llm_service, "config", {}).get("temperature", getattr(getattr(llm_service, "config", {}).get("options", {}), "temperature", None)),
        "max_tokens": getattr(llm_service, "config", {}).get("max_tokens", getattr(getattr(llm_service, "config", {}).get("options", {}), "num_predict", None)),
        "workflow": "preferences_generation"
    }
    
    workflow.add_node("generate_preferences", generate_preferences, metadata=metadata)
    workflow.set_entry_point("generate_preferences")

    return workflow.compile(checkpointer=MemorySaver())


def create_concept_workflow(llm_service: LLMService) -> StateGraph:
    """Create a workflow for generating a game concept."""
    concept_agent = ConceptGenerationAgent(llm_service)
    
    def generate_concept(state: CardGameState) -> CardGameState:
        return concept_agent.generate_concept(state)
    
    workflow = StateGraph(CardGameState)
    
    metadata = {
        "model": llm_service.model_name,
        "provider": llm_service.__class__.__name__,
        "temperature": getattr(llm_service, "config", {}).get("temperature", getattr(getattr(llm_service, "config", {}).get("options", {}), "temperature", None)),
        "max_tokens": getattr(llm_service, "config", {}).get("max_tokens", getattr(getattr(llm_service, "config", {}).get("options", {}), "num_predict", None)),
        "workflow": "concept_generation"
    }
    
    workflow.add_node("generate_concept", generate_concept, metadata=metadata)
    workflow.set_entry_point("generate_concept")

    return workflow.compile(checkpointer=MemorySaver())


def create_rules_workflow(llm_service: LLMService) -> StateGraph:
    """Create a workflow specifically for rules generation."""
    rule_agent = RuleGenerationAgent(llm_service)
    workflow = StateGraph(CardGameState)
    
    metadata = {
        "model": llm_service.model_name,
        "provider": llm_service.__class__.__name__,
        "temperature": getattr(llm_service, "config", {}).get("temperature", getattr(getattr(llm_service, "config", {}).get("options", {}), "temperature", None)),
        "max_tokens": getattr(llm_service, "config", {}).get("max_tokens", getattr(getattr(llm_service, "config", {}).get("options", {}), "num_predict", None)),
        "workflow": "rules_generation"
    }
    
    workflow.add_node("generate_rules", rule_agent.generate_rules, metadata=metadata)
    workflow.set_entry_point("generate_rules")

    return workflow.compile(checkpointer=MemorySaver())


def create_cards_workflow(llm_service: LLMService) -> StateGraph:
    """Create a workflow specifically for card generation."""
    card_agent = CardGenerationAgent(llm_service)
    workflow = StateGraph(CardGameState)
    
    metadata = {
        "model": llm_service.model_name,
        "provider": llm_service.__class__.__name__,
        "temperature": getattr(llm_service, "config", {}).get("temperature", getattr(getattr(llm_service, "config", {}).get("options", {}), "temperature", None)),
        "max_tokens": getattr(llm_service, "config", {}).get("max_tokens", getattr(getattr(llm_service, "config", {}).get("options", {}), "num_predict", None)),
        "workflow": "cards_generation"
    }
    
    workflow.add_node("generate_cards", card_agent.generate_card, metadata=metadata)

    workflow.add_conditional_edges("generate_cards", should_continue)

    workflow.set_entry_point("generate_cards")

    return workflow.compile(checkpointer=MemorySaver())


def create_concept_and_rules_workflow(llm_service: LLMService) -> StateGraph:
    """Create a workflow that combines concept and rules generation in sequence."""
    concept_agent = ConceptGenerationAgent(llm_service)
    rule_agent = RuleGenerationAgent(llm_service)
    
    workflow = StateGraph(CardGameState)
    
    metadata = {
        "model": llm_service.model_name,
        "provider": llm_service.__class__.__name__,
        "temperature": getattr(llm_service, "config", {}).get("temperature", getattr(getattr(llm_service, "config", {}).get("options", {}), "temperature", None)),
        "max_tokens": getattr(llm_service, "config", {}).get("max_tokens", getattr(getattr(llm_service, "config", {}).get("options", {}), "num_predict", None)),
        "workflow": "concept_and_rules_generation"
    }
    
    workflow.add_node("generate_concept", concept_agent.generate_concept, metadata=metadata)
    workflow.add_node("generate_rules", rule_agent.generate_rules, metadata=metadata)
    
    workflow.add_edge("generate_concept", "generate_rules")
    workflow.set_entry_point("generate_concept")
    
    return workflow.compile(checkpointer=MemorySaver()) 