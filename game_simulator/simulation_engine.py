import random
from game_simulator.game_state_manager import GameStateManager
from game_simulator.ai_agents import PlayerAgent, CoherenceAgent, GameMasterAgent
from pydantic import BaseModel

def log_model_state(title: str, model_instance: BaseModel):
    """Función auxiliar para imprimir el estado de un modelo Pydantic de forma legible."""
    print("\n" + "="*80)
    print(f"[[[ LOG: {title} ]]]")
    print("="*80)
    try:
        # Usamos model_dump_json para una salida bonita y formateada
        print(model_instance.model_dump_json(indent=2))
    except Exception as e:
        print(f"No se pudo serializar el modelo: {e}")
    print("="*80 + "\n")

class SimulationEngine:
    def __init__(self, game_data: dict, num_players: int = 2):
        self.game_data = game_data
        self.player_ids = [f"Jugador_{i+1}" for i in range(num_players)]
        self.manager = GameStateManager(self.player_ids)
        self.gm_agent = GameMasterAgent(game_data)
        self.player_agent = PlayerAgent(
            game_rules=game_data['rules'], 
            state_def=self.gm_agent.state_def
        )
        self.coherence_agent = CoherenceAgent()

    def _setup_game_state(self):
        """Configura el estado inicial del juego de forma robusta."""
        print("Motor de simulación (Python): Creando y barajando el mazo completo...")

        # 1. Python crea el mazo completo y lo baraja
        all_cards = []
        if 'cards' in self.game_data and 'all_cards' in self.game_data['cards']:
            for card in self.game_data['cards']['all_cards']:
                all_cards.extend([card] * card.get('quantity', 1))
        random.shuffle(all_cards)

        # 2. Python reparte los mazos a los jugadores
        num_players = len(self.player_ids)
        for i, player_id in enumerate(self.player_ids):
            player_state = self.manager.state.players[player_id]
            player_state.zones['deck'] = all_cards[i::num_players]
            player_state.zones['hand'] = []
            player_state.zones['discard_pile'] = []
            player_state.zones['in_play'] = []
            player_state.properties['points'] = 0
            player_state.properties['health'] = 0

        self.manager.state.game_log.append("[ENGINE]: Mazos repartidos por Python.")
        log_model_state("ESTADO TRAS REPARTO DE MAZOS (ANTES DE MANO INICIAL)", self.manager.state)

        # 3. Se llama al GameMaster (IA) para la tarea específica de robar la mano inicial
        print("Llamando al Game Master (IA) para interpretar reglas de mano inicial...")
        for player_id in self.player_ids:
            print(f"Game Master está preparando la mano para {player_id}...")
            hand_draw_update = self.gm_agent.get_initial_hand_draw_changes(player_id, self.manager.state)
            
            log_model_state(f"CAMBIOS DE MANO INICIAL PARA {player_id} (propuesto por GM)", hand_draw_update)
            
            self.manager.apply_changes(hand_draw_update.changes)
            self.manager.state.game_log.append(f"[GAME MASTER]: {hand_draw_update.reasoning}")


    def run(self, max_turns: int = 20):
        # 1. Configuración inicial por el Game Master
        print("Game Master está configurando la partida...")
        setup_update = self.gm_agent.get_initial_setup(self.player_ids)
        
        # LOG: Mostramos la configuración que propone el GM
        log_model_state("CAMBIOS DE ESTADO PARA LA CONFIGURACIÓN INICIAL (propuesto por GM)", setup_update)
        
        self.manager.apply_changes(setup_update.changes)
        self.manager.state.game_log.append(f"[GAME MASTER]: {setup_update.reasoning}")

        # LOG: Mostramos el estado completo después de la configuración
        log_model_state("ESTADO DEL JUEGO TRAS CONFIGURACIÓN INICIAL", self.manager.state)

        # 2. Bucle de juego
        while self.manager.state.turn_number <= max_turns:
            current_p_id = self.manager.state.current_player_id
            
            # LOG: Estado al inicio del turno
            log_model_state(f"INICIO TURNO {self.manager.state.turn_number} - JUGADOR: {current_p_id}", self.manager.state)

            # 2a. El estratega decide su intención
            print(f"Turno {self.manager.state.turn_number} - {current_p_id}: Pensando intención...")
            intent = self.player_agent.choose_action(self.manager.state)
            
            # LOG: La intención exacta que devuelve el PlayerAgent
            log_model_state(f"INTENCIÓN DEL JUGADOR {current_p_id} (PlayerAction)", intent)
            self.manager.state.game_log.append(f"[{current_p_id}]: Mi intención es '{intent.action_type}' con la carta '{intent.card_name}'. Razón: {intent.reasoning}")

            # 2b. El Game Master interpreta la intención y genera los cambios
            print(f"Game Master está interpretando la acción...")
            state_update = self.gm_agent.interpret_action(intent, self.manager.state)

            # LOG: Los cambios de estado exactos que dicta el GM
            log_model_state(f"CAMBIOS DE ESTADO DICTADOS POR GM (StructuredStateUpdate)", state_update)
            
            self.manager.apply_changes(state_update.changes)
            self.manager.state.game_log.append(f"[GAME MASTER]: {state_update.reasoning}")

            # LOG: Estado completo después de aplicar los cambios del turno
            log_model_state(f"ESTADO DEL JUEGO TRAS ACCIÓN DE {current_p_id}", self.manager.state)

            # 2c. El Game Master comprueba si alguien ha ganado
            win_status = self.gm_agent.check_win_condition(self.manager.state)

            # LOG: La decisión del GM sobre las condiciones de victoria
            log_model_state("COMPROBACIÓN DE VICTORIA (WinConditionStatus)", win_status)

            if win_status.is_game_over:
                self.manager.state.game_over = True
                self.manager.state.winner = win_status.winner
                self.manager.state.game_log.append(f"[GAME MASTER]: ¡Fin de la partida! {win_status.reason}")
                break

            self.manager.advance_turn()
            self.manager.state.game_log.append(f"\n--- Turno {self.manager.state.turn_number}, juega {self.manager.state.current_player_id} ---")

        if not self.manager.state.game_over:
            self.manager.state.game_log.append(f"\n[GAME MASTER]: Fin de la simulación por alcanzar el límite de {max_turns} turnos.")
        
        print("\n--- Analizando Coherencia del Juego con la IA ---")
        report = self.coherence_agent.analyze_log(self.manager.state.game_log, self.game_data)
        return report