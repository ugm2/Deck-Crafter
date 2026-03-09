from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from deck_crafter.models.state import CardGameState
from deck_crafter.models.game_concept import GameConcept
from deck_crafter.models.rules import Rules
from deck_crafter.models.card import Card
from deck_crafter.models.user_preferences import UserPreferences

from deck_crafter.services.llm_service import LLMService
from deck_crafter.agents.preferences_agent import PreferencesGenerationAgent
from deck_crafter.agents.concept_agent import ConceptGenerationAgent
from deck_crafter.agents.rules_agent import RuleGenerationAgent
from deck_crafter.agents.card_agent import CardGenerationAgent
from deck_crafter.agents.image_agent import ImageGenerationAgent
from deck_crafter.agents.evaluation_agents import ValidatorAgent
from .reflective_step import ReflectiveStep
from .conditions import should_continue
from .evaluation_workflow import create_multi_agent_evaluation_workflow


def create_preferences_workflow(llm_service: LLMService) -> StateGraph:
    """Crea un workflow para generar preferencias con un ciclo de reflexión exigente."""
    preferences_agent = PreferencesGenerationAgent(llm_service)
    validator = ValidatorAgent(llm_service)

    # --- NUEVOS CRITERIOS EXIGENTES PARA PREFERENCIAS ---
    preferences_criteria = """
    Review against this strict checklist. A failure in any point means the output is invalid.
    1.  **Completeness**: All fields (language, theme, game_style, number_of_players, target_audience, rule_complexity) MUST be filled with non-empty, meaningful strings.
    2.  **Logical Consistency**: Check for contradictions. For example, if 'target_audience' is 'Niños' (Kids), 'rule_complexity' cannot be 'Hard' or 'Complex'. If 'game_style' is 'Party Game', 'number_of_players' should not be '2'.
    3.  **Language Validity**: The 'language' field must be a single, real-world language name (e.g., 'Español', 'English'), not a mix or a sentence.
    """

    preferences_step = ReflectiveStep(
        agent_method=preferences_agent.generate_preferences,
        validator=validator,
        model_class=UserPreferences,
        state_key='preferences',
        criteria=preferences_criteria,
        max_attempts=3
    )
    
    workflow = StateGraph(CardGameState)
    def start_node(state: CardGameState):
        state.refinement_count = 0
        state.critique = None
        return state
        
    workflow.add_node("start", start_node)
    workflow.set_entry_point("start")
    preferences_step.add_to_graph(workflow, "start", END)
    
    return workflow.compile(checkpointer=MemorySaver())


def create_concept_and_rules_workflow(llm_service: LLMService) -> StateGraph:
    """Crea un workflow de concepto y reglas con ciclos de reflexión exigentes."""
    concept_agent = ConceptGenerationAgent(llm_service)
    rule_agent = RuleGenerationAgent(llm_service)
    validator = ValidatorAgent(llm_service)

    # --- NUEVOS CRITERIOS EXIGENTES PARA CONCEPTO ---
    concept_criteria = """
    Review against this strict checklist. A failure in any point means the output is invalid.
    1.  **Structural Integrity**: There MUST be at least 3 distinct `card_types`.
    2.  **Mathematical Coherence**: The `number_of_unique_cards` field MUST be equal to the sum of all `unique_cards` values within the `card_types` list.
    3.  **Title Quality**: The `title` must not be generic. It cannot contain the phrases 'Card Game', 'Juego de Cartas', or 'Deck'.
    4.  **Description Depth**: The `description` must be at least 20 words long.
    """

    concept_step = ReflectiveStep(
        agent_method=concept_agent.generate_concept,
        validator=validator,
        model_class=GameConcept,
        state_key='concept',
        criteria=concept_criteria,
        max_attempts=3
    )

    # --- NUEVOS CRITERIOS EXIGENTES PARA REGLAS ---
    rules_criteria = """
    Review against this strict checklist. A failure in any single point means the entire output is invalid.
    - **Completeness Check**: 
        - Does the output explicitly define a `turn_limit`? If null, it is invalid.
        - Does the output include a `glossary` with at least 4 defined terms? If not, it is invalid.
        - Does the output provide at least 2 distinct `examples_of_play`? If not, it is invalid.
    - **Clarity Check**: 
        - Scan the `win_conditions`. Is there ANY word that could be considered ambiguous (e.g., 'about', 'approximately', 'might', 'could', 'usually')? If so, it is invalid.
        - The `turn_structure` must contain at least 3 distinct phases. If not, it is invalid.
    - **Coherence Check**:
        - The `win_conditions` must directly and obviously relate to the core `description` of the game concept. The thematic link cannot be subtle. If it is, it is invalid.
    """
    
    rules_step = ReflectiveStep(
        agent_method=rule_agent.generate_rules,
        validator=validator,
        model_class=Rules,
        state_key='rules',
        criteria=rules_criteria,
        max_attempts=3
    )

    workflow = StateGraph(CardGameState)
    def start_node(state: CardGameState):
        state.refinement_count = 0
        state.critique = None
        return state
    
    workflow.add_node("start", start_node)
    workflow.set_entry_point("start")
    
    concept_step.add_to_graph(workflow, "start", "rules_entry_bridge")
    
    workflow.add_node("rules_entry_bridge", start_node)
    rules_step.add_to_graph(workflow, "rules_entry_bridge", END)
    
    return workflow.compile(checkpointer=MemorySaver())


def create_cards_workflow(llm_service: LLMService) -> StateGraph:
    """
    Crea un workflow simple para la generación de cartas en un bucle,
    sin el ciclo de reflexión, pero manteniendo el log de progreso.
    """
    card_agent = CardGenerationAgent(llm_service)
    from .conditions import should_continue

    workflow = StateGraph(CardGameState)

    # Nodo 1: El generador de cartas
    # Llama al método del agente para generar una única carta y actualizar el estado.
    def generate_card_node(state: CardGameState) -> dict:
        print(f"--- ATTEMPTING TO GENERATE CARD ---")
        return card_agent.generate_card(state)

    workflow.add_node("generate_card", generate_card_node)

    # Nodo 2: El nodo que comprueba el progreso y actúa como ancla del bucle
    def check_completion_node(state: CardGameState) -> dict:
        """Este nodo comprueba el progreso y lo muestra en la terminal."""
        current_cards_count = len(state.cards) if state.cards else 0
        total_unique_cards = state.concept.number_of_unique_cards if state.concept else 0
        
        print(f"--- CARD PROGRESS: {current_cards_count} / {total_unique_cards} generated ---")
        return {}
    
    workflow.add_node("check_completion", check_completion_node)

    # El punto de entrada es el nodo de comprobación
    workflow.set_entry_point("check_completion")

    # Desde la comprobación, decidimos si generar o terminar
    workflow.add_conditional_edges(
        "check_completion",
        should_continue,
        {
            "generate_cards": "generate_card",
            END: END
        }
    )

    # Después de generar una carta, volvemos a la comprobación para el siguiente ciclo
    workflow.add_edge("generate_card", "check_completion")

    return workflow.compile(checkpointer=MemorySaver())


def create_image_generation_workflow(llm_service: LLMService) -> StateGraph:
    """Crea un workflow simple para la generación de imágenes (sin reflexión)."""
    image_agent = ImageGenerationAgent(llm_service)
    workflow = StateGraph(CardGameState)
    workflow.add_node("generate_images", image_agent.generate_images)
    workflow.set_entry_point("generate_images")
    return workflow.compile(checkpointer=MemorySaver())
