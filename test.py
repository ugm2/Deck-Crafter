from langchain_google_vertexai import ChatVertexAI, create_structured_runnable
from langchain_core.prompts import ChatPromptTemplate
from deck_crafter.models import GameConcept, Rules

# Step 1: Generate the game concept first
llm = ChatVertexAI(model_name="gemini-1.5-pro-002", location="us-east1")
prompt_concept = ChatPromptTemplate.from_template(
    """
    You are a world-class card game designer. 
    Your task is to design a structured game concept based on user preferences in {input}.
    Make sure to adhere to the required format and ensure the game is engaging.
    """
)

# Create chain for generating game concept
chain_concept = create_structured_runnable([GameConcept], llm, prompt=prompt_concept)

# Step 2: Invoke the concept creation
concept_result: GameConcept = chain_concept.invoke(
    {"input": "User wants a game for 4-6 players."}
)
print("Game Concept Generated:", concept_result)
concept_result.target_audience = None
print(concept_result.model_dump())

# # Step 3: Use the game concept as context for generating the rules
# prompt_rules = ChatPromptTemplate.from_template(
#     """
#     You are a world-class card game designer. Based on the following game concept: {game_concept},
#     your task is to design structured rules for this card game. Please provide the setup, turn structure, win conditions,
#     and any additional or special rules that make the game unique.
#     """
# )

# # Create chain for generating rules
# chain_rules = create_structured_runnable([Rules], llm, prompt=prompt_rules)

# # Step 4: Invoke the rules creation based on the generated concept
# rules_result = chain_rules.invoke({"game_concept": concept_result})
# print("Game Rules Generated:", rules_result)
