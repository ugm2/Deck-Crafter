from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from deck_crafter.models.state import CardGameState
from deck_crafter.models.evaluation import (
    BalanceEvaluation,
    CoherenceEvaluation,
    ClarityEvaluation,
    OriginalityEvaluation,
    PlayabilityEvaluation,
    FidelityEvaluation,
    GameEvaluation
)
from deck_crafter.services.llm_service import LLMService
from deck_crafter.agents.evaluation_agents import (
    BalanceAgent,
    CoherenceAgent,
    ClarityAgent,
    OriginalityAgent,
    PlayabilityAgent,
    FidelityAgent,
    EvaluationSynthesizerAgent
)

class EvaluationState(TypedDict):
    """El estado específico para el workflow de evaluación."""
    game_state: CardGameState
    balance_eval: BalanceEvaluation
    coherence_eval: CoherenceEvaluation
    clarity_eval: ClarityEvaluation
    originality_eval: OriginalityEvaluation
    playability_eval: PlayabilityEvaluation
    fidelity_eval: FidelityEvaluation
    final_evaluation: GameEvaluation

def create_multi_agent_evaluation_workflow(llm_service: LLMService) -> StateGraph:
    """
    Crea un workflow donde múltiples agentes de evaluación trabajan en paralelo
    y un sintetizador combina sus resultados.
    """
    # Instanciar todos los agentes evaluadores
    balance_agent = BalanceAgent(llm_service)
    coherence_agent = CoherenceAgent(llm_service)
    clarity_agent = ClarityAgent(llm_service)
    originality_agent = OriginalityAgent(llm_service)
    playability_agent = PlayabilityAgent(llm_service)
    fidelity_agent = FidelityAgent(llm_service)
    synthesizer_agent = EvaluationSynthesizerAgent(llm_service)

    # --- Nodos de Evaluación Paralela ---
    def run_balance_eval(state: EvaluationState):
        game = state['game_state']
        return {"balance_eval": balance_agent.evaluate(game.concept, game.rules, game.cards, game.concept.language)}

    def run_coherence_eval(state: EvaluationState):
        game = state['game_state']
        return {"coherence_eval": coherence_agent.evaluate(game.concept, game.rules, game.cards, game.concept.language)}

    def run_clarity_eval(state: EvaluationState):
        game = state['game_state']
        return {"clarity_eval": clarity_agent.evaluate(game.concept, game.rules, game.cards, game.concept.language)}

    def run_originality_eval(state: EvaluationState):
        game = state['game_state']
        return {"originality_eval": originality_agent.evaluate(game.concept, game.rules, game.cards, game.concept.language)}

    def run_playability_eval(state: EvaluationState):
        game = state['game_state']
        return {"playability_eval": playability_agent.evaluate(game.concept, game.rules, game.cards, game.concept.language)}
    
    def run_fidelity_eval(state: EvaluationState):
        game = state['game_state']
        return {"fidelity_eval": fidelity_agent.evaluate(game.preferences, game.concept, game.rules, game.concept.language)}

    # --- Nodo de Síntesis Final ---
    def run_synthesis(state: EvaluationState):
        final_evaluation = synthesizer_agent.synthesize(
            balance_eval=state['balance_eval'],
            coherence_eval=state['coherence_eval'],
            clarity_eval=state['clarity_eval'],
            originality_eval=state['originality_eval'],
            playability_eval=state['playability_eval'],
            fidelity_eval=state['fidelity_eval'],
            language=state['game_state'].concept.language,
        )
        # Actualizamos el estado original del juego
        game_state = state['game_state']
        game_state.evaluation = final_evaluation
        game_state.status = "evaluated"
        return {"final_evaluation": final_evaluation, "game_state": game_state}

    # --- Ensamblar el Grafo ---
    workflow = StateGraph(EvaluationState)

    # Añadimos todos los nodos de evaluación
    workflow.add_node("balance", run_balance_eval)
    workflow.add_node("coherence", run_coherence_eval)
    workflow.add_node("clarity", run_clarity_eval)
    workflow.add_node("originality", run_originality_eval)
    workflow.add_node("playability", run_playability_eval)
    workflow.add_node("fidelity", run_fidelity_eval)
    workflow.add_node("synthesize", run_synthesis)

    # 1. Crear un nodo de entrada simple.
    workflow.add_node("start_evaluation", lambda state: state)

    # 2. Establecerlo como el ÚNICO punto de entrada.
    workflow.set_entry_point("start_evaluation")

    # 3. Conectar la entrada a TODOS los nodos paralelos.
    workflow.add_edge("start_evaluation", "balance")
    workflow.add_edge("start_evaluation", "coherence")
    workflow.add_edge("start_evaluation", "clarity")
    workflow.add_edge("start_evaluation", "originality")
    workflow.add_edge("start_evaluation", "playability")
    workflow.add_edge("start_evaluation", "fidelity")
    
    # 4. Conectar todos los nodos paralelos al nodo de síntesis
    workflow.add_edge(["balance", "coherence", "clarity", "originality", "playability", "fidelity"], "synthesize")
    
    # 5. El final del grafo es después de la síntesis
    workflow.add_edge("synthesize", END)

    return workflow.compile(checkpointer=MemorySaver()) 