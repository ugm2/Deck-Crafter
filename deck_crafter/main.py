import logging
from core.llm_service import LLMService
from agents.card_generation_agent import CardGenerationAgent
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict


class CardGameState(TypedDict):
    theme: str
    cards: list[str]


# Initialize the service and agents
llm_service = LLMService(
    model_name="gemini-1.5-pro-002", temperature=0.5, max_output_tokens=1024
)

card_gen_agent = CardGenerationAgent(llm_service)

# Create the graph
graph = StateGraph(CardGameState)

# Add node for the card generation agent
graph.add_node("CardGeneration", card_gen_agent.generate_card)

# Add the START node
graph.set_entry_point("CardGeneration")
graph.set_finish_point("CardGeneration")

# Compile the graph
app = graph.compile(checkpointer=MemorySaver())

# Run the graph with the initial context (theme)
state = CardGameState()
state["theme"] = "fantasy"

# Invoke the graph with the required 'thread_id' or other configurable keys
logging.debug("Invoking the graph...")
result = app.invoke(state, config={"configurable": {"thread_id": 1}})

logging.debug(f"Final result: {result}")
