import json
from deck_crafter.game_simulator.llm_service import llm_service
from deck_crafter.game_simulator.simulation_models import (
    GameState, 
    CoherenceReport, 
    PlayerAction, 
    StructuredStateUpdate, 
    WinConditionStatus
)
import random
from typing import List, Dict, Any

class GameMasterAgent:
    """
    El agente que conoce y aplica las reglas de CUALQUIER juego,
    basándose en la descripción proporcionada en el JSON del juego.
    """
    def __init__(self, game_data: Dict[str, Any]):
        self.game_data = game_data
        # Se asume una estructura de estado por defecto si no se especifica en el juego.
        # Un juego bien generado debería tener su propia sección 'state_definition'.
        self.state_def = game_data.get('state_definition', {
            "player_properties": ["points", "health"],
            "card_zones": ["hand", "deck", "discard_pile", "in_play"]
        })

    def get_initial_setup(self, player_ids: List[str]) -> StructuredStateUpdate:
        """
        Llama al LLM para generar el estado inicial completo de la partida,
        incluyendo la creación y reparto de mazos y manos.
        """
        all_cards = []
        # Asegurarse de que el juego tiene la clave 'cards' y 'all_cards'
        if 'cards' in self.game_data and 'all_cards' in self.game_data['cards']:
            for card in self.game_data['cards']['all_cards']:
                all_cards.extend([card] * card.get('quantity', 1))
        
        random.shuffle(all_cards)
        
        prompt = f"""
        Tu tarea es configurar el estado inicial de una partida de '{self.game_data['concept']['title']}'.
        La estructura del estado de un jugador se define por estas propiedades y zonas: {json.dumps(self.state_def)}
        Reglas de mano inicial: "{self.game_data['rules']['initial_hands']}"
        Jugadores: {player_ids}
        
        TAREA OBLIGATORIA:
        Debes generar un `StructuredStateUpdate` con una única operación `update_property` para la ruta `'players'`.
        El valor de esta operación debe ser un diccionario de objetos `PlayerState` completos y válidos.
        
        INSTRUCCIONES CRÍTICAS:
        1.  Para cada jugador, crea un objeto `PlayerState` con su `id` y los diccionarios `properties` y `zones` inicializados.
        2.  **Reparte TODAS las cartas existentes ({len(all_cards)} en total) equitativamente entre las zonas `deck` de los jugadores.**
        3.  **Después de repartir los mazos, MUEVE el número de cartas indicado por la regla 'mano inicial' del `deck` a la `hand` de cada jugador.**
        4.  El resultado final debe ser que cada jugador tenga un mazo y una mano con **objetos de carta completos**, no vacíos. El no cumplimiento de este paso resultará en un fallo.
        """
        return llm_service.generate(output_model=StructuredStateUpdate, prompt=prompt)

    def interpret_action(self, action: PlayerAction, state: GameState) -> StructuredStateUpdate:
        card_desc = "N/A"
        if action.card_name:
            card_desc = next((c.get('description', '') for c in self.game_data['cards']['all_cards'] if c.get('name') == action.card_name), "No se encontró.")

        prompt = f"""
        Eres el Game Master. El jugador '{state.current_player_id}' ha declarado la intención de realizar la acción '{action.action_type}'.
        - Carta involucrada: '{action.card_name}' (Descripción: {card_desc})
        - Objetivo declarado: '{action.target_id}'
        - Razonamiento del jugador: '{action.reasoning}'
        - Reglas del juego: {json.dumps(self.game_data['rules'])}
        - Estado actual del juego: {state.model_dump_json()}

        TAREA:
        Basado en las reglas, el estado del juego y la descripción de la carta, determina las consecuencias exactas de esta acción.
        Traduce estas consecuencias a una lista de cambios de estado atómicos (`StateChange`).
        
        REGLAS FUNDAMENTALES PARA TI COMO GAME MASTER:
        1.  No puedes inventar cartas. Solo puedes mover o modificar cartas que ya existen en alguna de las zonas del jugador (mano, mazo, etc.).
        2.  Si el jugador no tiene cartas en el mazo, no puede robar. La acción de robar debe fallar o no producir cambios.
        3.  Si una acción declarada por el jugador es imposible (ej: jugar una carta que no tiene), tu `reasoning` debe indicarlo y la lista de `changes` debe estar vacía.
        4.  NO debes generar cambios para el campo `game_log`.
        """
        return llm_service.generate(output_model=StructuredStateUpdate, prompt=prompt)

    def check_win_condition(self, state: GameState) -> WinConditionStatus:
        """
        Llama al LLM para que evalúe si se ha cumplido alguna condición de victoria.
        """
        prompt = f"""
        Eres el Game Master. Evalúa si la partida ha terminado.
        Las condiciones de victoria son: "{self.game_data['rules']['win_conditions']}".
        El estado actual del juego es: {state.model_dump_json()}.
        
        Lee las condiciones de victoria y evalúa si alguna se cumple en el estado actual.
        Responde con el modelo 'WinConditionStatus', indicando si el juego terminó, quién ganó y por qué.
        """
        return llm_service.generate(output_model=WinConditionStatus, prompt=prompt)


class PlayerAgent:
    """
    Agente estratega que decide la intención de juego basándose en un estado genérico.
    """
    def __init__(self, game_rules: Dict[str, Any], state_def: Dict[str, Any]):
        self.rules = game_rules
        self.state_def = state_def

    def choose_action(self, game_state: GameState) -> PlayerAction:
        """
        Llama al LLM para que decida la mejor acción estratégica para el turno actual.
        """
        current_player_id = game_state.current_player_id
        player_state = game_state.players[current_player_id]
        
        prompt_context = f"""
        Eres un jugador de cartas de nivel campeón mundial. Tu objetivo es ganar cumpliendo estas condiciones: "{self.rules['win_conditions']}".
        El estado del juego se define por propiedades y zonas.
        
        --- ANÁLISIS DE LA SITUACIÓN (Turno {game_state.turn_number}) ---
        Juegas tú, {current_player_id}.

        TUS PROPIEDADES:
        """
        for prop, value in player_state.properties.items():
            prompt_context += f"- {prop}: {value}\n"
        
        prompt_context += "\nTUS ZONAS:\n"
        for zone, cards in player_state.zones.items():
            prompt_context += f"- {zone}: {[c.get('name', 'Carta sin nombre') for c in cards]}\n"
        
        # Añadir contexto del oponente
        opponent_id = next(id for id in game_state.players if id != current_player_id)
        opponent_state = game_state.players[opponent_id]
        prompt_context += f"\nESTADO VISIBLE DEL OPONENTE ({opponent_id}):\nPROPIEDADES:\n"
        for prop, value in opponent_state.properties.items():
            prompt_context += f"- {prop}: {value}\n"
        prompt_context += "ZONAS:\n"
        for zone, cards in opponent_state.zones.items():
            if zone not in ['hand', 'deck']: # No mostramos zonas ocultas
                 prompt_context += f"- {zone}: {[c.get('name', 'Carta sin nombre') for c in cards]}\n"

        prompt_final = prompt_context + """
        TAREA:
        Analiza la situación y tu zona 'hand'. Formula un plan estratégico y decide tu acción inmediata.
        Tu respuesta DEBE ser un objeto JSON que siga el esquema de `PlayerAction`.
        """
        return llm_service.generate(output_model=PlayerAction, prompt=prompt_final)

class CoherenceAgent:
    """
    Analiza el log final de la partida para encontrar fallos de diseño y balance.
    """
    def analyze_log(self, game_log: List[str], game_data: Dict[str, Any]) -> CoherenceReport:
        """
        Llama al LLM para generar un informe de coherencia sobre la partida simulada.
        """
        prompt = f"""
        Eres un analista experto en diseño de juegos. Has observado una partida del juego '{game_data['concept']['title']}'.
        - Reglas y Concepto: {json.dumps({'concept': game_data['concept'], 'rules': game_data['rules']})}
        - Log completo de la partida: {''.join(game_log)}

        Analiza todo en conjunto y genera un informe de coherencia y balance usando el modelo `CoherenceReport`. Sé crítico, detallado y busca problemas sutiles en las reglas, el balance y la experiencia de juego.
        """
        return llm_service.generate(output_model=CoherenceReport, prompt=prompt)