from langgraph.graph import StateGraph, END
from deck_crafter.agents.concept_agent import ConceptGenerationAgent
from deck_crafter.agents.rules_agent import RuleGenerationAgent
from deck_crafter.agents.card_agent import CardGenerationAgent
from deck_crafter.agents.image_agent import ImageGenerationAgent
from deck_crafter.models.state import CardGameState, GameStatus
from deck_crafter.services.llm_service import LLMService
from langgraph.checkpoint.memory import MemorySaver
from deck_crafter.agents.preferences_agent import PreferencesGenerationAgent
from deck_crafter.workflow.conditions import should_continue
from typing import TypedDict, Optional
from deck_crafter.agents.evaluation_agents import (
    BalanceAgent,
    CoherenceAgent,
    ClarityAgent,
    OriginalityAgent,
    PlayabilityAgent,
    EvaluationSynthesizerAgent,
    FidelityAgent,
)
from deck_crafter.models.evaluation import (
    BalanceEvaluation,
    CoherenceEvaluation,
    ClarityEvaluation,
    OriginalityEvaluation,
    PlayabilityEvaluation,
    FidelityEvaluation,
)


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


def create_image_generation_workflow(llm_service: LLMService) -> StateGraph:
    """Create a workflow specifically for image generation."""
    image_agent = ImageGenerationAgent(llm_service)
    workflow = StateGraph(CardGameState)
    
    metadata = {
        "model": llm_service.model_name,
        "provider": llm_service.__class__.__name__,
        "temperature": getattr(llm_service, "config", {}).get("temperature", getattr(getattr(llm_service, "config", {}).get("options", {}), "temperature", None)),
        "max_tokens": getattr(llm_service, "config", {}).get("max_tokens", getattr(getattr(llm_service, "config", {}).get("options", {}), "num_predict", None)),
        "workflow": "image_generation"
    }
    
    workflow.add_node("generate_images", image_agent.generate_images, metadata=metadata)
    workflow.set_entry_point("generate_images")

    return workflow.compile(checkpointer=MemorySaver())


class EvaluationWorkflowState(TypedDict):
    """Define el estado para el workflow de evaluación, conteniendo el juego y los informes parciales."""
    game_state: CardGameState
    balance_report: Optional[BalanceEvaluation]
    coherence_report: Optional[CoherenceEvaluation]
    clarity_report: Optional[ClarityEvaluation]
    originality_report: Optional[OriginalityEvaluation]
    playability_report: Optional[PlayabilityEvaluation]
    fidelity_report: Optional[FidelityEvaluation]

def create_multi_agent_evaluation_workflow(llm_service: LLMService) -> StateGraph:
    """
    Crea un workflow que utiliza un comité de agentes expertos para evaluar un juego.
    """
    # Instanciar todos los agentes
    balance_agent = BalanceAgent(llm_service)
    coherence_agent = CoherenceAgent(llm_service)
    clarity_agent = ClarityAgent(llm_service)
    originality_agent = OriginalityAgent(llm_service)
    playability_agent = PlayabilityAgent(llm_service)
    fidelity_agent = FidelityAgent(llm_service)
    synthesizer_agent = EvaluationSynthesizerAgent(llm_service)

    # --- Nodos del Grafo ---
    
    def run_balance_analysis(state: EvaluationWorkflowState) -> dict:
        game = state["game_state"]
        report = balance_agent.evaluate(game.concept, game.rules, game.cards, game.concept.language)
        return {"balance_report": report}

    def run_coherence_analysis(state: EvaluationWorkflowState) -> dict:
        game = state["game_state"]
        report = coherence_agent.evaluate(game.concept, game.rules, game.cards, game.concept.language)
        return {"coherence_report": report}

    def run_clarity_analysis(state: EvaluationWorkflowState) -> dict:
        game = state["game_state"]
        report = clarity_agent.evaluate(game.concept, game.rules, game.cards, game.concept.language)
        return {"clarity_report": report}

    def run_originality_analysis(state: EvaluationWorkflowState) -> dict:
        game = state["game_state"]
        report = originality_agent.evaluate(game.concept, game.rules, game.cards, game.concept.language)
        return {"originality_report": report}

    def run_playability_analysis(state: EvaluationWorkflowState) -> dict:
        game = state["game_state"]
        report = playability_agent.evaluate(game.concept, game.rules, game.cards, game.concept.language)
        return {"playability_report": report}

    def run_fidelity_analysis(state: EvaluationWorkflowState) -> dict:
        game = state["game_state"]
        report = fidelity_agent.evaluate(
            preferences=game.preferences,
            concept=game.concept,
            rules=game.rules,
            language=game.concept.language,
        )
        return {"fidelity_report": report}
        
    def run_synthesis(state: EvaluationWorkflowState) -> dict:
        final_evaluation = synthesizer_agent.synthesize(
            balance_eval=state["balance_report"],
            coherence_eval=state["coherence_report"],
            clarity_eval=state["clarity_report"],
            originality_eval=state["originality_report"],
            playability_eval=state["playability_report"],
            fidelity_eval=state["fidelity_report"],
            language=state["game_state"].concept.language,
        )
        updated_game_state = state["game_state"].model_copy(deep=True)
        updated_game_state.evaluation = final_evaluation
        updated_game_state.status = GameStatus.EVALUATED
        return {"game_state": updated_game_state}

    workflow = StateGraph(EvaluationWorkflowState)
    
    workflow.add_node("balance_analyzer", run_balance_analysis)
    workflow.add_node("coherence_analyzer", run_coherence_analysis)
    workflow.add_node("clarity_analyzer", run_clarity_analysis)
    workflow.add_node("originality_analyzer", run_originality_analysis)
    workflow.add_node("playability_analyzer", run_playability_analysis)
    workflow.add_node("fidelity_analyzer", run_fidelity_analysis)
    
    workflow.add_node("synthesizer", run_synthesis)

    workflow.set_entry_point("balance_analyzer")
    workflow.add_edge("balance_analyzer", "coherence_analyzer")
    workflow.add_edge("coherence_analyzer", "clarity_analyzer")
    workflow.add_edge("clarity_analyzer", "originality_analyzer")
    workflow.add_edge("originality_analyzer", "playability_analyzer")
    workflow.add_edge("playability_analyzer", "fidelity_analyzer")
    
    workflow.add_edge("fidelity_analyzer", "synthesizer")
    
    workflow.add_edge("synthesizer", END)

    return workflow.compile(checkpointer=MemorySaver())
