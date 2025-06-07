from fastapi import APIRouter, HTTPException
from typing import Dict
import uuid
from datetime import datetime, timezone
from deck_crafter.models.user_preferences import UserPreferences
from deck_crafter.workflow.specific_workflows import (
    create_concept_workflow,
    create_rules_workflow,
    create_cards_workflow,
    create_preferences_workflow
)
from deck_crafter.services.llm_service import create_llm_service
from deck_crafter.models.state import CardGameState, GameStatus
from deck_crafter.utils.config import Config

router = APIRouter()

llm_service = create_llm_service(
    provider=Config.LLM_PROVIDER,
    model=Config.LLM_MODEL,
    api_key=Config.GROQ_API_KEY
)

concept_workflow = create_concept_workflow(llm_service)
rules_workflow = create_rules_workflow(llm_service)
cards_workflow = create_cards_workflow(llm_service)

# Game state storage (in-memory for now)
games: Dict[str, CardGameState] = {}

@router.post("/start")
async def start_game(preferences: UserPreferences) -> Dict[str, str]:
    """Start a new game with the given preferences."""
    game_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    needs_generation = not preferences or not all([
        preferences.language,
        preferences.theme,
        preferences.game_style,
        preferences.number_of_players,
        preferences.target_audience,
        preferences.rule_complexity
    ])
    if needs_generation:
        preferences_workflow = create_preferences_workflow(llm_service)
        initial_state = CardGameState(
            game_id=game_id,
            status=GameStatus.CREATED,
            preferences=preferences,
            concept=None,
            rules=None,
            cards=None,
            created_at=now,
            updated_at=now
        )
        result_state = preferences_workflow.invoke(initial_state, config={"configurable": {"thread_id": 1}})
        preferences = result_state["preferences"]

    initial_state = CardGameState(
        game_id=game_id,
        status=GameStatus.CREATED,
        preferences=preferences,
        concept=None,
        rules=None,
        cards=None,
        created_at=now,
        updated_at=now
    )

    games[game_id] = initial_state
    return {"game_id": game_id}

@router.get("/{game_id}")
async def get_game_state(game_id: str) -> CardGameState:
    """Get the current state of a game."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    return games[game_id]

@router.post("/{game_id}/concept")
async def generate_concept(game_id: str) -> Dict[str, str]:
    """Generate the game concept."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games[game_id]
    if game.status != GameStatus.CREATED:
        raise HTTPException(status_code=400, detail="Invalid game state for concept generation")
    
    state = game
    result = concept_workflow.invoke(state, config={"configurable": {"thread_id": 1}})

    result_state = CardGameState.model_validate(result)
    games[game_id].concept = result_state.concept
    games[game_id].status = GameStatus.CONCEPT_GENERATED
    games[game_id].updated_at = datetime.now(timezone.utc)
    
    return {"status": GameStatus.CONCEPT_GENERATED}

@router.post("/{game_id}/rules")
async def generate_rules(game_id: str) -> Dict[str, str]:
    """Generate the game rules."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games[game_id]
    if game.status != GameStatus.CONCEPT_GENERATED:
        raise HTTPException(status_code=400, detail="Invalid game state for rules generation")
    
    state = game
    result = rules_workflow.invoke(state, config={"configurable": {"thread_id": 1}})
    
    games[game_id].rules = result["rules"]
    games[game_id].status = GameStatus.RULES_GENERATED
    games[game_id].updated_at = datetime.now(timezone.utc)
    
    return {"status": GameStatus.RULES_GENERATED}

@router.post("/{game_id}/cards")
async def generate_cards(game_id: str) -> Dict[str, str]:
    """Generate the game cards."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games[game_id]
    if game.status != GameStatus.RULES_GENERATED:
        raise HTTPException(status_code=400, detail="Invalid game state for cards generation")
    
    state = game
    result = cards_workflow.invoke(state, config={"recursion_limit": 150, "configurable": {"thread_id": 1}})
    
    games[game_id].cards = result["cards"]
    games[game_id].status = GameStatus.CARDS_GENERATED
    games[game_id].updated_at = datetime.now(timezone.utc)
    
    return {"status": "cards_generated"} 