from langgraph.graph import StateGraph
from deck_crafter.agents.concept_agent import ConceptGenerationAgent
from deck_crafter.agents.rules_agent import RuleGenerationAgent
from deck_crafter.agents.card_agent import CardGenerationAgent
from deck_crafter.workflow.conditions import should_continue
from deck_crafter.models.state import CardGameState
from deck_crafter.services.llm_service import LLMService
from langgraph.checkpoint.memory import MemorySaver


def create_game_workflow(llm_service: LLMService) -> StateGraph:
    """
    Create and configure the game workflow, including concept generation, rule generation, and card generation.

    :param llm_service: The LLM service used by agents to generate the game components.
    :return: A configured StateGraph representing the card game generation process.
    """

    concept_agent = ConceptGenerationAgent(llm_service)
    rule_agent = RuleGenerationAgent(llm_service)
    card_agent = CardGenerationAgent(llm_service)

    workflow = StateGraph(CardGameState)

    # Adding nodes for each stage of the game generation process
    workflow.add_node("generate_concept", concept_agent.generate_concept)
    workflow.add_node("generate_rules", rule_agent.generate_rules)
    workflow.add_node("generate_cards", card_agent.generate_card)

    # Define transitions between stages
    workflow.add_edge("generate_concept", "generate_rules")
    workflow.add_edge("generate_rules", "generate_cards")

    # Add conditional logic to decide when to stop generating cards
    workflow.add_conditional_edges("generate_cards", should_continue)

    workflow.set_entry_point("generate_concept")

    return workflow.compile(checkpointer=MemorySaver())
