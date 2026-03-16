# simulation_models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional

class Buff(BaseModel):
    stat: Literal['attack', 'defense']
    value: int
    duration: int

class UnitInPlay(BaseModel):
    id: str # Identificador único para la unidad en juego
    card: Dict[str, Any]
    current_health: int
    buffs: List[Buff] = Field(default_factory=list)

class PlayerState(BaseModel):
    """Modelo de estado de jugador genérico."""
    id: str
    properties: Dict[str, Any] = Field(default_factory=dict, description="Propiedades del jugador, ej: {'vida': 20, 'puntos': 0}")
    zones: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, description="Zonas de cartas del jugador, ej: {'Mano': [...], 'Mazo': [...]}")

class GameState(BaseModel):
    """Contenedor principal del estado."""
    players: Dict[str, PlayerState]
    turn_number: int = 1
    current_player_id: str
    game_log: List[str] = Field(default_factory=list)
    game_over: bool = False
    winner: str | None = None

class PlayerAction(BaseModel):
    action_type: Literal['play_card', 'attack', 'end_turn']
    card_name: Optional[str] = None
    target_id: Optional[str] = Field(None, description="ID del jugador o unidad objetivo.")
    reasoning: str

class StateChange(BaseModel):
    operation: Literal['update_property', 'add_to_list', 'remove_from_list']
    path: str = Field(..., description="Ruta al valor a cambiar, ej: 'players.Jugador_1.points'")
    value: Any = Field(..., description="El nuevo valor o el objeto a añadir.")
    item_id: Optional[str] = Field(None, description="Para remove_from_list, el ID del item a eliminar.")

class StructuredStateUpdate(BaseModel):
    reasoning: str
    changes: List[StateChange]

class WinConditionStatus(BaseModel):
    is_game_over: bool
    winner: Optional[str] = None
    reason: str

class CoherenceReport(BaseModel):
    balance_rating: int
    coherence_issues: List[str]
    identified_loops: List[str]
    useless_cards: List[str]
    overpowered_strategies: List[str]
    suggestions: str
