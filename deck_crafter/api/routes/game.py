from fastapi import APIRouter, HTTPException
from typing import Dict
import uuid
from datetime import datetime, timezone
from deck_crafter.models.user_preferences import UserPreferences
from deck_crafter.workflow.specific_workflows import (
    create_concept_workflow,
    create_rules_workflow,
    create_cards_workflow,
    create_preferences_workflow,
    create_concept_and_rules_workflow
)
from deck_crafter.services.llm_service import create_llm_service
from deck_crafter.models.state import CardGameState, GameStatus
from deck_crafter.utils.config import Config
from deck_crafter.database import init_db, save_game_state as save_game_state_to_db, get_game_state as get_game_state_from_db

router = APIRouter()

# Initialize database on startup
@router.on_event("startup")
async def startup_event():
    await init_db()

llm_service = create_llm_service(
    provider=Config.LLM_PROVIDER,
    model=Config.LLM_MODEL,
    api_key=Config.GROQ_API_KEY
)

preferences_workflow = create_preferences_workflow(llm_service)
concept_workflow = create_concept_workflow(llm_service)
rules_workflow = create_rules_workflow(llm_service)
cards_workflow = create_cards_workflow(llm_service)
concept_and_rules_workflow = create_concept_and_rules_workflow(llm_service)

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

    await save_game_state_to_db(initial_state)
    return {"game_id": game_id}

@router.get("/{game_id}")
async def get_game_state(game_id: str) -> CardGameState:
    """Get the current state of a game."""
    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")
    return state

@router.post("/{game_id}/concept")
async def generate_concept(game_id: str) -> Dict[str, str]:
    """Generate the game concept."""
    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")
    
    if state.status != GameStatus.CREATED:
        raise HTTPException(status_code=400, detail="Invalid game state for concept generation")
    
    result = concept_workflow.invoke(state, config={"configurable": {"thread_id": 1}})
    result_state = CardGameState.model_validate(result)
    
    state.concept = result_state.concept
    state.status = GameStatus.CONCEPT_GENERATED
    state.updated_at = datetime.now(timezone.utc)
    await save_game_state_to_db(state)
    
    return {"status": GameStatus.CONCEPT_GENERATED}

@router.post("/{game_id}/rules")
async def generate_rules(game_id: str) -> Dict[str, str]:
    """Generate the game rules."""
    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")
    
    if state.status != GameStatus.CONCEPT_GENERATED:
        raise HTTPException(status_code=400, detail="Invalid game state for rules generation")
    
    result = rules_workflow.invoke(state, config={"configurable": {"thread_id": 1}})
    
    state.rules = result["rules"]
    state.status = GameStatus.RULES_GENERATED
    state.updated_at = datetime.now(timezone.utc)
    await save_game_state_to_db(state)
    
    return {"status": GameStatus.RULES_GENERATED}

@router.post("/{game_id}/cards")
async def generate_cards(game_id: str) -> Dict[str, str]:
    """Generate the game cards."""
    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")
    
    if state.status != GameStatus.RULES_GENERATED:
        raise HTTPException(status_code=400, detail="Invalid game state for cards generation")
    
    result = cards_workflow.invoke(state, config={"recursion_limit": 150, "configurable": {"thread_id": 1}})
    
    state.cards = result["cards"]
    state.status = GameStatus.CARDS_GENERATED
    state.updated_at = datetime.now(timezone.utc)
    await save_game_state_to_db(state)
    
    return {"status": "cards_generated"}

@router.post("/{game_id}/concept-and-rules")
async def generate_concept_and_rules(game_id: str) -> Dict[str, str]:
    """Generate both the game concept and rules in one operation."""
    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")
    
    if state.status != GameStatus.CREATED:
        raise HTTPException(status_code=400, detail="Invalid game state for concept and rules generation")
    
    result = concept_and_rules_workflow.invoke(state, config={"configurable": {"thread_id": 1}})
    
    result_state = CardGameState.model_validate(result)
    state.concept = result_state.concept
    state.rules = result_state.rules
    state.status = GameStatus.RULES_GENERATED
    state.updated_at = datetime.now(timezone.utc)
    await save_game_state_to_db(state)
    
    return {"status": GameStatus.RULES_GENERATED} 