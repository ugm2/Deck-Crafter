"""Centralized checkpointer configuration for LangGraph workflows.

This module provides a configured MemorySaver that registers all custom
types used in workflow states to avoid serialization warnings.
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Register all custom types used in workflow states (5-metric evaluation system)
ALLOWED_MSGPACK_MODULES = [
    ('deck_crafter.models.state', 'CardGameState'),
    ('deck_crafter.models.state', 'GameStatus'),
    # New 5-metric evaluation classes
    ('deck_crafter.models.evaluation', 'BalanceEvaluation'),
    ('deck_crafter.models.evaluation', 'ClarityEvaluation'),
    ('deck_crafter.models.evaluation', 'PlayabilityEvaluation'),
    ('deck_crafter.models.evaluation', 'ThemeAlignmentEvaluation'),
    ('deck_crafter.models.evaluation', 'InnovationEvaluation'),
    ('deck_crafter.models.evaluation', 'GameEvaluation'),
    # Legacy aliases (backward compatibility)
    ('deck_crafter.models.evaluation', 'CoherenceEvaluation'),
    ('deck_crafter.models.evaluation', 'FidelityEvaluation'),
    ('deck_crafter.models.evaluation', 'OriginalityEvaluation'),
    # Other models
    ('deck_crafter.models.game_concept', 'GameConcept'),
    ('deck_crafter.models.card_type', 'CardType'),
    ('deck_crafter.models.rules', 'Rules'),
    ('deck_crafter.models.card', 'Card'),
    ('deck_crafter.models.user_preferences', 'UserPreferences'),
]


def create_checkpointer() -> MemorySaver:
    """Create a MemorySaver with all custom types registered for serialization."""
    serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
    return MemorySaver(serde=serde)
