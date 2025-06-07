from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from deck_crafter.models.state import CardGameState
from deck_crafter.services.llm_service import LLMService
from deck_crafter.workflow.specific_workflows import (
    create_concept_workflow,
    create_rules_workflow,
    create_cards_workflow,
    create_preferences_workflow
)

def create_game_workflow(llm_service: LLMService) -> StateGraph:
    """Create the main workflow for generating a card game."""
    workflow = StateGraph(CardGameState)
    
    workflow.add_node("generate_concept", create_concept_workflow(llm_service))
    workflow.add_node("generate_rules", create_rules_workflow(llm_service))
    workflow.add_node("generate_cards", create_cards_workflow(llm_service))
    
    workflow.add_edge("generate_concept", "generate_rules")
    workflow.add_edge("generate_rules", "generate_cards")
    
    workflow.set_entry_point("generate_concept")
    
    return workflow.compile(checkpointer=MemorySaver())
