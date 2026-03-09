from typing import Type, Callable
from pydantic import BaseModel
from langgraph.graph import StateGraph, END

from deck_crafter.models.state import CardGameState
from deck_crafter.agents.evaluation_agents import ValidatorAgent

class ReflectiveStep:
    """
    Representa un paso de workflow auto-corregible y reutilizable.
    """
    def __init__(
        self,
        agent_method: Callable,
        validator: ValidatorAgent,
        model_class: Type[BaseModel],
        state_key: str,
        criteria: str,
        max_attempts: int = 2,
    ):
        self.agent_method = agent_method
        self.validator = validator
        self.model_class = model_class
        self.state_key = state_key
        self.criteria = criteria
        self.max_attempts = max_attempts

    def generate(self, state: CardGameState) -> dict:
        print(f"--- ATTEMPTING TO GENERATE '{self.state_key}' (Refinement attempt: {state.refinement_count}) ---")
        return self.agent_method(state)

    def critique(self, state: CardGameState) -> dict:
        """Nodo que critica la salida del generador y siempre devuelve un veredicto claro."""
        print(f"--- CRITIQUING '{self.state_key}' ---")
        
        output_to_validate = getattr(state, self.state_key)

        if isinstance(output_to_validate, list) and output_to_validate:
            output_to_validate = output_to_validate[-1]

        if not output_to_validate:
            return {"critique": "Agent failed to produce any output. Please try again.", "refinement_count": state.refinement_count + 1}
            
        context_json_str = state.model_dump_json(exclude={self.state_key})
            
        validation_result = self.validator.validate(
            output_to_validate=output_to_validate,
            output_model=self.model_class,
            high_level_criteria=self.criteria,
            context_data_json=context_json_str
        )
        
        if not validation_result.is_valid:
            print(f"--- CRITIQUE FOR '{self.state_key}': Needs Revision. Feedback: {validation_result.feedback} ---")
            
            output_list = getattr(state, self.state_key)
            if isinstance(output_list, list):
                output_list.pop()

            return {
                "critique": validation_result.feedback,
                "refinement_count": state.refinement_count + 1
            }
        else:
            print(f"--- CRITIQUE FOR '{self.state_key}': Approved. ---")
            return {"critique": None, "refinement_count": 0}

    def decide(self, state: CardGameState) -> str:
        """Nodo de decisión PURO. Solo lee el estado y devuelve una ruta."""
        if state.critique and state.refinement_count < self.max_attempts:
            return f"generate_{self.state_key}"
        return "continue"

    def add_to_graph(self, workflow: StateGraph, entry_node_name: str, success_exit_node_name: str):
        """Añade este ciclo de reflexión completo al grafo principal."""
        generate_node_name = f"generate_{self.state_key}"
        critique_node_name = f"critique_{self.state_key}"

        workflow.add_node(generate_node_name, self.generate)
        workflow.add_node(critique_node_name, self.critique)

        workflow.add_edge(entry_node_name, generate_node_name)
        workflow.add_edge(generate_node_name, critique_node_name)
        workflow.add_conditional_edges(
            critique_node_name,
            self.decide,
            {
                generate_node_name: generate_node_name,
                "continue": success_exit_node_name
            }
        ) 