# game_state_manager.py
from typing import Dict, Any, List
from game_simulator.simulation_models import GameState, PlayerState
import dpath

class GameStateManager:
    """Gestiona el estado del juego. No conoce reglas, solo aplica cambios genéricos."""

    def __init__(self, player_ids: List[str]):
        initial_players = {p_id: PlayerState(id=p_id) for p_id in player_ids}
        self.state: GameState = GameState(
            players=initial_players,
            current_player_id=player_ids[0]
        )

    def apply_changes(self, changes: List[Dict[str, Any]]):
        temp_state_dict = self.state.model_dump()
        
        for change in changes:
            op = change.operation
            path = change.path
            value = change.value
            
            try:
                if op == 'update_property':
                    dpath.set(temp_state_dict, path, value)
                elif op == 'add_to_list':
                    dpath.get(temp_state_dict, path).append(value)
                elif op == 'remove_from_list':
                    target_list = dpath.get(temp_state_dict, path)
                    id_key = change.get('item_id_key')
                    id_value = change.get('item_id_value')
                    
                    if not all([target_list, id_key, id_value]):
                        self.state.game_log.append(f"ADVERTENCIA: Faltan datos para 'remove_from_list'.")
                        continue

                    initial_len = len(target_list)
                    target_list[:] = [item for item in target_list if item.get(id_key) != id_value]
                    if len(target_list) == initial_len:
                        self.state.game_log.append(f"ADVERTENCIA: No se encontró item con {id_key}='{id_value}' para eliminar en la ruta '{path}'.")

            except Exception as e:
                log_msg = f"ERROR al aplicar cambio: op='{op}', path='{path}'. Error: {e}"
                self.state.game_log.append(log_msg)
        
        self.state = GameState.model_validate(temp_state_dict)
    
    def advance_turn(self):
        player_ids = list(self.state.players.keys())
        current_idx = player_ids.index(self.state.current_player_id)
        next_idx = (current_idx + 1) % len(player_ids)
        self.state.current_player_id = player_ids[next_idx]
        if next_idx == 0:
            self.state.turn_number += 1