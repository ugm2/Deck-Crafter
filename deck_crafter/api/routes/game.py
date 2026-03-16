from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any
import logging
import uuid
from datetime import datetime, timezone
import base64
import threading

# Track which games are currently generating images
_generating_images: Dict[str, bool] = {}
# Shutdown flag to stop background tasks gracefully
_shutdown_requested = threading.Event()
from deck_crafter.models.user_preferences import UserPreferences
from deck_crafter.workflow.specific_workflows import (
    create_preferences_workflow,
    create_concept_and_rules_workflow,
    create_cards_workflow,
    create_image_generation_workflow,
    create_multi_agent_evaluation_workflow,
    create_refinement_workflow
)
from deck_crafter.workflow.evaluation_workflow import PanelEvaluationWorkflow
from deck_crafter.services.llm_service import create_llm_service, create_fallback_llm_service, GeminiService, GroqService
from deck_crafter.utils.config import Config as LLMConfig
from deck_crafter.models.state import CardGameState, GameStatus, RefinementMemory
from deck_crafter.models.chat import ChatMessage, ChatAction
from deck_crafter.utils.config import Config
from deck_crafter.database import init_db, save_game_state as save_game_state_to_db, save_game_state_sync, get_game_state as get_game_state_from_db, get_all_card_images

# Simulation imports
from deck_crafter.game_simulator.integration import run_simulation_for_game, analyze_game
from deck_crafter.game_simulator.analysis_agent import GameplayAnalysisAgent
from deck_crafter.game_simulator.rule_compiler import normalize_card_resources

router = APIRouter()

# Initialize database on startup
@router.on_event("startup")
async def startup_event():
    await init_db()
    # Clear any stale generation state from previous run
    _generating_images.clear()
    _shutdown_requested.clear()


@router.on_event("shutdown")
async def shutdown_event():
    """Signal background tasks to stop gracefully."""
    _shutdown_requested.set()
    # Wait a moment for tasks to notice
    import asyncio
    await asyncio.sleep(0.5)


# Standard LLM service (free tier)
if Config.LLM_PROVIDER == "fallback":
    standard_llm_service = create_fallback_llm_service()
elif Config.LLM_PROVIDER == "groq":
    standard_llm_service = create_llm_service(
        provider="groq",
        model=Config.GROQ_MODEL,
        api_key=Config.GROQ_API_KEY
    )
elif Config.LLM_PROVIDER == "ollama":
    standard_llm_service = create_llm_service(
        provider="ollama",
        model=Config.OLLAMA_MODEL
    )
else:
    standard_llm_service = create_llm_service(
        provider=Config.LLM_PROVIDER
    )

# Standard workflows
_panel_eval = None
if Config.GROQ_API_KEY and len(Config.EVALUATION_PANEL_MODELS) > 1:
    try:
        _panel_eval = PanelEvaluationWorkflow(
            panel_models=Config.EVALUATION_PANEL_MODELS,
            provider=Config.EVALUATION_PANEL_PROVIDER,
        )
        logging.getLogger(__name__).info(
            f"Panel evaluation enabled with {len(Config.EVALUATION_PANEL_MODELS)} models"
        )
    except Exception as e:
        logging.getLogger(__name__).warning(f"Panel evaluation init failed, using single-model: {e}")

standard_workflows = {
    'preferences': create_preferences_workflow(standard_llm_service),
    'concept_and_rules': create_concept_and_rules_workflow(standard_llm_service),
    'cards': create_cards_workflow(standard_llm_service),
    'images': create_image_generation_workflow(standard_llm_service),
    'evaluation': _panel_eval or create_multi_agent_evaluation_workflow(standard_llm_service),
    'refinement': create_refinement_workflow(standard_llm_service),
}

# Premium workflows cache (keyed by provider)
_premium_workflows_cache: Dict[str, Dict] = {}

def _create_premium_llm(provider: str, model: str = None):
    """Create premium LLM service for specified provider."""
    import logging
    logger = logging.getLogger(__name__)
    if provider == "gemini" and LLMConfig.GEMINI_API_KEY:
        gemini_model = model or LLMConfig.GEMINI_MODEL
        logger.info(f"Creating Gemini premium service with model {gemini_model}")
        return GeminiService(model=gemini_model)
    elif provider == "groq" and LLMConfig.GROQ_API_KEY:
        groq_model = model or LLMConfig.GROQ_PREMIUM_MODEL
        logger.info(f"Creating Groq premium service with model {groq_model}")
        return GroqService(model=groq_model, max_tokens=16384)
    else:
        logger.warning(f"Premium provider {provider} not available, falling back to standard")
        return standard_llm_service

def get_workflows(premium: bool = False, premium_provider: str = "gemini", model: str = None):
    if not premium:
        return standard_workflows
    # Cache key includes model to support different model configurations
    cache_key = f"{premium_provider}:{model or 'default'}"
    if cache_key not in _premium_workflows_cache:
        llm = _create_premium_llm(premium_provider, model)
        _premium_workflows_cache[cache_key] = {
            'preferences': create_preferences_workflow(llm),
            'concept_and_rules': create_concept_and_rules_workflow(llm),
            'cards': create_cards_workflow(llm),
            'images': create_image_generation_workflow(llm),
            'evaluation': _panel_eval or create_multi_agent_evaluation_workflow(llm),
            'refinement': create_refinement_workflow(llm),
        }
    return _premium_workflows_cache[cache_key]

def _get_evaluation_workflow(provider: str, model: str = None):
    """Get a cached evaluation workflow for a specific provider/model."""
    if _panel_eval:
        return _panel_eval
    cache_key = f"eval:{provider}:{model or 'default'}"
    if cache_key not in _premium_workflows_cache:
        llm = _create_premium_llm(provider, model)
        _premium_workflows_cache[cache_key] = create_multi_agent_evaluation_workflow(llm)
    return _premium_workflows_cache[cache_key]


# --- Escalation and Mandatory Simulation Helpers ---
logger = logging.getLogger(__name__)




@router.post("/start")
async def start_game(preferences: UserPreferences, premium: bool = False, premium_provider: str = "gemini", model: str = None) -> Dict[str, str]:
    """Start a new game with the given preferences."""
    game_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    workflows = get_workflows(premium, premium_provider, model)

    needs_generation = not preferences or not all([
        preferences.language,
        preferences.theme,
        preferences.game_style,
        preferences.number_of_players,
        preferences.target_audience,
        preferences.rule_complexity,
        preferences.art_style
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
        result_state = workflows['preferences'].invoke(initial_state, config={"configurable": {"thread_id": 1}})
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
    
    # If we have cards and they have images, include the image data
    if state.cards:
        images = await get_all_card_images(game_id)
        updated_cards = []
        for card in state.cards:
            if card.name in images:
                image_base64 = base64.b64encode(images[card.name]).decode('utf-8')
                updated_cards.append(card.model_copy(update={'image_data': image_base64}))
            else:
                updated_cards.append(card)
        state.cards = updated_cards
    
    return state

@router.post("/{game_id}/concept")
async def generate_concept(game_id: str, premium: bool = False, premium_provider: str = "gemini", model: str = None) -> Dict[str, str]:
    """Generate the game concept."""
    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")

    if state.status != GameStatus.CREATED:
        raise HTTPException(status_code=400, detail="Invalid game state for concept generation")

    workflows = get_workflows(premium, premium_provider, model)
    result = workflows['concept_and_rules'].invoke(state, config={"configurable": {"thread_id": 1}})
    result_state = CardGameState.model_validate(result)

    state.concept = result_state.concept
    state.status = GameStatus.CONCEPT_GENERATED
    state.updated_at = datetime.now(timezone.utc)
    await save_game_state_to_db(state)

    return {"status": GameStatus.CONCEPT_GENERATED}

@router.post("/{game_id}/rules")
async def generate_rules(game_id: str, premium: bool = False, premium_provider: str = "gemini", model: str = None) -> Dict[str, str]:
    """Generate the game rules."""
    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")

    if state.status != GameStatus.CONCEPT_GENERATED:
        raise HTTPException(status_code=400, detail="Invalid game state for rules generation")

    workflows = get_workflows(premium, premium_provider, model)
    result = workflows['concept_and_rules'].invoke(state, config={"configurable": {"thread_id": 1}})

    state.rules = result["rules"]
    state.status = GameStatus.RULES_GENERATED
    state.updated_at = datetime.now(timezone.utc)
    await save_game_state_to_db(state)

    return {"status": GameStatus.RULES_GENERATED}

@router.post("/{game_id}/cards")
async def generate_cards(game_id: str, premium: bool = False, premium_provider: str = "gemini", model: str = None) -> Dict[str, str]:
    """Generate the game cards."""
    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")

    if state.status != GameStatus.RULES_GENERATED:
        raise HTTPException(status_code=400, detail="Invalid game state for cards generation")

    workflows = get_workflows(premium, premium_provider, model)
    result = workflows['cards'].invoke(state, config={"recursion_limit": 150, "configurable": {"thread_id": 1}})

    state.cards = result["cards"]
    state.status = GameStatus.CARDS_GENERATED
    state.updated_at = datetime.now(timezone.utc)
    await save_game_state_to_db(state)

    return {"status": "cards_generated"}

@router.post("/{game_id}/concept-and-rules")
async def generate_concept_and_rules(game_id: str, premium: bool = False, premium_provider: str = "gemini", model: str = None) -> Dict[str, str]:
    """Generate both the game concept and rules in one operation."""
    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")

    if state.status != GameStatus.CREATED:
        raise HTTPException(status_code=400, detail="Invalid game state for concept and rules generation")

    workflows = get_workflows(premium, premium_provider, model)
    result = workflows['concept_and_rules'].invoke(state, config={"configurable": {"thread_id": 1}})

    result_state = CardGameState.model_validate(result)
    state.concept = result_state.concept
    state.rules = result_state.rules
    state.status = GameStatus.RULES_GENERATED
    state.updated_at = datetime.now(timezone.utc)
    await save_game_state_to_db(state)

    return {"status": GameStatus.RULES_GENERATED}

def _run_image_generation(game_id: str, state: CardGameState, use_gemini: bool = False):
    """Background task to generate images."""
    try:
        _generating_images[game_id] = True
        if use_gemini:
            from deck_crafter.agents.image_agent import GeminiImageGenerationAgent
            agent = GeminiImageGenerationAgent()
            agent.generate_images(state, shutdown_event=_shutdown_requested)
        else:
            workflows = get_workflows(False)
            workflows['images'].invoke(state, config={"configurable": {"thread_id": 1}})
    except Exception as e:
        print(f"Error generating images for {game_id}: {e}")
    finally:
        _generating_images[game_id] = False

@router.post("/{game_id}/images")
async def generate_images(
    game_id: str,
    background_tasks: BackgroundTasks,
    premium: bool = False,
    premium_provider: str = "gemini"
) -> Dict[str, Any]:
    """Generate images for all cards in the game (runs in background)."""
    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")

    if state.status not in [GameStatus.CARDS_GENERATED, GameStatus.EVALUATED, GameStatus.IMAGES_GENERATED]:
        raise HTTPException(
            status_code=400,
            detail="Cards must be generated or evaluated before generating images"
        )

    if _generating_images.get(game_id, False):
        return {"status": "generating", "message": "Image generation already in progress"}

    use_gemini = premium and premium_provider == "gemini"
    background_tasks.add_task(_run_image_generation, game_id, state, use_gemini)

    provider_name = "Gemini" if use_gemini else "FLUX (local)"
    return {"status": "generating", "message": f"Image generation started with {provider_name}"}

@router.get("/{game_id}/images/status")
async def get_image_generation_status(game_id: str) -> Dict[str, Any]:
    """Check the status of image generation."""
    is_generating = _generating_images.get(game_id, False)

    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")

    # Count images from database
    images = await get_all_card_images(game_id)
    total_cards = len(state.cards) if state.cards else 0
    completed = len(images)

    return {
        "generating": is_generating,
        "total": total_cards,
        "completed": completed,
        "remaining": total_cards - completed,
        "progress_percent": round(completed / total_cards * 100, 1) if total_cards > 0 else 0
    }

@router.post("/{game_id}/evaluate")
async def evaluate_game(game_id: str, premium: bool = False, premium_provider: str = "gemini", model: str = None) -> Dict[str, Any]:
    """Genera una evaluación experta para un juego completo usando un comité de agentes."""
    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")

    if state.status not in [GameStatus.CARDS_GENERATED, GameStatus.IMAGES_GENERATED, GameStatus.EVALUATED]:
        raise HTTPException(
            status_code=400,
            detail="Game must have at least cards generated to be evaluated."
        )

    workflows = get_workflows(premium, premium_provider, model)
    initial_eval_state = {"game_state": state}

    result_state = workflows['evaluation'].invoke(initial_eval_state, config={"configurable": {"thread_id": "eval-" + game_id}})

    final_game_state = result_state['game_state']

    # Set baseline score if this is the first evaluation
    if final_game_state.baseline_score is None:
        final_game_state.baseline_score = final_game_state.evaluation.overall_score
        final_game_state.best_score_achieved = final_game_state.evaluation.overall_score

    await save_game_state_to_db(final_game_state)

    return {
        "status": final_game_state.status.value if hasattr(final_game_state.status, 'value') else final_game_state.status,
        "evaluation_summary": final_game_state.evaluation.summary,
        "overall_score": final_game_state.evaluation.overall_score
    }


@router.post("/{game_id}/simulate")
async def simulate_game(
    game_id: str,
    num_games: int = 30,
    seed: int = 42,
    agent_type: str = "random",
    premium: bool = False,
    premium_provider: str = "gemini",
    model: str = None
) -> Dict[str, Any]:
    """
    Run gameplay simulation on a game to gather empirical data.

    Simulates multiple games and produces:
    - Statistical report (win rates, turn counts, card performance)
    - Qualitative analysis (dominant strategies, problematic cards, pacing)

    The analysis is stored in the game state and will be used by evaluation
    agents (Balance, Playability) for evidence-based scoring.

    Args:
        game_id: The game to simulate
        num_games: Number of games to simulate (default 30)
        seed: Random seed for reproducibility
        agent_type: Type of AI agent: "random" (fast), "heuristic" (smarter), "strategic" (LLM-powered)
        premium: Use premium LLM for analysis
        premium_provider: LLM provider for analysis
        model: Specific model to use
    """
    import logging
    logger = logging.getLogger(__name__)

    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")

    if not state.rules or not state.cards:
        raise HTTPException(
            status_code=400,
            detail="Game must have rules and cards to simulate"
        )

    logger.info(f"Starting simulation for game {game_id}: {num_games} games")

    # Get LLM for analysis
    llm = _create_premium_llm(premium_provider, model) if premium else standard_llm_service

    try:
        # Run simulation and compile
        game_name = state.concept.title if state.concept else "Game"
        report, warnings = run_simulation_for_game(
            rules=state.rules,
            cards=state.cards,
            game_name=game_name,
            num_games=num_games,
            seed=seed,
            llm_service=llm,
            agent_type=agent_type,
        )

        logger.info(f"Simulation complete: {report.completion_rate:.0%} completion, {report.avg_turns:.1f} avg turns")

        # Generate qualitative analysis
        language = state.concept.language if state.concept else "English"
        agent = GameplayAnalysisAgent(llm)
        analysis = agent.analyze(report, language=language)

        # Store analysis and report in state
        state.simulation_analysis = analysis
        state.simulation_report = report
        state.updated_at = datetime.now(timezone.utc)
        await save_game_state_to_db(state)

        # Build response
        return {
            "status": "simulated",
            "games_run": report.games_run,
            "games_completed": report.games_completed,
            "completion_rate": round(report.completion_rate, 2),
            "first_player_win_rate": round(report.first_player_win_rate, 2),
            "avg_turns": round(report.avg_turns, 1),
            "turn_range": {"min": report.min_turns, "max": report.max_turns},
            "compilation_warnings": warnings,
            "issues_detected": report.issues,
            "analysis": {
                "summary": analysis.summary,
                "strategic_diversity": analysis.strategic_diversity,
                "pacing_assessment": analysis.pacing_assessment,
                "comeback_potential": analysis.comeback_potential,
                "problematic_cards": [
                    {"name": c.card_name, "issue": c.issue_type, "evidence": c.evidence}
                    for c in analysis.problematic_cards
                ],
                "high_priority_fixes": analysis.high_priority_fixes,
                "fun_indicators": analysis.fun_indicators,
                "anti_fun_indicators": analysis.anti_fun_indicators,
            },
            "cards_never_played": report.cards_never_played,
            "cards_always_played": report.cards_always_played,
        }

    except Exception as e:
        logger.error(f"Simulation failed for {game_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {str(e)}"
        )


def _build_refine_step_response(result, threshold: float) -> dict:
    """Build the /refine-step response dict from a RefinementResult."""
    from deck_crafter.services.refinement_service import RefinementResult
    r: RefinementResult = result
    baseline = r.state.baseline_score or r.previous_score

    base = {
        "status": r.status,
        "iteration": r.state.evaluation_iteration,
        "previous_score": r.previous_score,
        "new_score": r.new_score,
        "threshold": threshold,
        "threshold_met": r.new_score >= threshold,
        "improved": r.improved,
        "improvement": round(r.actual_improvement, 2),
        "progress": {
            "baseline_score": round(baseline, 2),
            "current_score": round(r.new_score, 2),
            "total_improvement": round(r.new_score - baseline, 2),
            "iteration_improvement": round(r.actual_improvement, 2),
            "best_score_achieved": round(r.state.best_score_achieved or r.new_score, 2),
            "trend": (
                "regressing" if r.status == "reverted"
                else "plateau" if r.status == "stopped"
                else "improving" if r.actual_improvement > 0.05
                else "plateau" if r.actual_improvement >= -0.05
                else "regressing"
            ),
        },
    }

    if r.experiment and r.strategy:
        base["experiment"] = {
            "hypothesis": r.strategy.hypothesis,
            "intervention": r.strategy.intervention_type,
            "target_metric": r.strategy.target_metric,
            "expected": r.strategy.expected_improvement,
            "actual": round(r.actual_improvement, 2),
            "confirmed": r.experiment.hypothesis_confirmed,
            "confidence": getattr(r.strategy, 'confidence', None),
        }

    if r.status == "reverted":
        base["attempted_score"] = r.experiment.score_after if r.experiment else None
        base["reason"] = "score_degraded"
        base["failed_patterns"] = len(r.memory.failed_patterns)
        base["message"] = f"Refinement made game worse, reverted. Pattern recorded as failed."
    elif r.status == "stopped":
        base["reason"] = r.stop_reason
        base["experiments_run"] = len(r.memory.experiments)
        base["blocked_metrics"] = r.memory.get_blocked_metrics()
        base["message"] = f"Stopped: {r.stop_reason} (score: {r.new_score:.1f})"
    else:
        if r.feedback:
            base["priority_issues"] = r.feedback.priority_issues
        base["cards_regenerated"] = r.cards_changed
        base["rules_changed"] = r.rules_changed
        base["lessons_learned"] = len(r.memory.lessons_learned)
        base["failed_patterns"] = len(r.memory.failed_patterns)
        base["evaluation_summary"] = r.state.evaluation.summary if r.state.evaluation else None

    return base


@router.post("/{game_id}/refine-step")
async def refine_step(
    game_id: str,
    threshold: float = 6.0,
    premium: bool = False,
    premium_provider: str = "gemini",
    model: str = None
) -> Dict[str, Any]:
    """
    Run a single iteration of the refinement loop using reflexive scientific method.
    Returns results immediately for real-time UI updates.
    Call this in a loop from the client to show progress per iteration.
    """
    from deck_crafter.services.refinement_service import execute_refinement_step

    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")

    if not state.evaluation:
        raise HTTPException(
            status_code=400,
            detail="Game must be evaluated before refinement. Call /evaluate first."
        )

    current_score = state.evaluation.overall_score
    logger.info(f"[RefineStep] Starting refinement for game {game_id[:8]}... "
               f"(current score: {current_score:.2f}, threshold: {threshold})")

    if current_score >= threshold:
        logger.info(f"[RefineStep] Already at threshold, skipping refinement")
        return {
            "status": "threshold_met",
            "score": current_score,
            "threshold": threshold,
            "iteration": state.evaluation_iteration,
            "message": f"Score {current_score:.1f} already meets threshold {threshold}"
        }

    llm = _create_premium_llm(premium_provider, model) if premium else standard_llm_service
    workflows = get_workflows(premium, premium_provider, model)

    result = execute_refinement_step(
        state=state,
        threshold=threshold,
        llm_service=llm,
        eval_workflow=workflows['evaluation'],
        num_simulation_games=30,
        use_batch_cards=False,
    )

    await save_game_state_to_db(result.state)
    return _build_refine_step_response(result, threshold)


@router.post("/{game_id}/refine")
async def refine_game(
    game_id: str,
    max_iterations: int = 3,
    threshold: float = 6.0,
    premium: bool = False,
    premium_provider: str = "gemini",
    model: str = None
) -> Dict[str, Any]:
    """
    Run iterative refinement loop on an evaluated game.

    Regenerates rules and cards based on evaluation feedback until:
    - Score meets the threshold, OR
    - Max iterations is reached
    """
    from deck_crafter.services.refinement_service import execute_refinement_step

    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")

    if state.status != GameStatus.EVALUATED:
        raise HTTPException(
            status_code=400,
            detail="Game must be evaluated before refinement. Call /evaluate first."
        )

    if not state.evaluation:
        raise HTTPException(
            status_code=400,
            detail="No evaluation found for this game."
        )

    if state.evaluation.overall_score >= threshold:
        return {
            "status": "already_optimal",
            "message": f"Score {state.evaluation.overall_score:.1f} already meets threshold {threshold}",
            "iterations_used": 0,
            "final_score": state.evaluation.overall_score,
            "improved": False
        }

    llm = _create_premium_llm(premium_provider, model) if premium else standard_llm_service
    workflows = get_workflows(premium, premium_provider, model)

    state.max_evaluation_iterations = min(max_iterations, 5)
    state.evaluation_threshold = threshold
    state.evaluation_iteration = 0

    initial_score = state.evaluation.overall_score

    for iteration in range(1, state.max_evaluation_iterations + 1):
        ref_result = execute_refinement_step(
            state=state,
            threshold=threshold,
            llm_service=llm,
            eval_workflow=workflows['evaluation'],
        )
        state = ref_result.state
        if ref_result.stop_reason:
            break

    await save_game_state_to_db(state)

    final_score = state.best_score_achieved or state.evaluation.overall_score
    return {
        "status": "refined",
        "iterations_used": state.evaluation_iteration,
        "initial_score": initial_score,
        "final_score": final_score,
        "improved": final_score > initial_score,
        "threshold_met": final_score >= threshold,
        "evaluation_summary": state.evaluation.summary
    }


# =============================================================================
# /generate-complete - Full Pipeline with Iterative Refinement
# =============================================================================

from pydantic import BaseModel, Field
from typing import List, Optional


class GenerateCompleteRequest(BaseModel):
    """Request for the complete game generation pipeline."""
    preferences: UserPreferences
    target_threshold: float = Field(default=7.0, ge=1.0, le=10.0)
    max_refinement_iterations: int = Field(default=5, ge=1, le=30)
    num_simulation_games: int = Field(default=30, ge=10, le=100)
    premium: bool = False
    premium_provider: str = "gemini"
    model: str | None = None
    evaluation_provider: str | None = None
    evaluation_model: str | None = None


class GenerateCompleteResponse(BaseModel):
    """Response from the complete game generation pipeline."""
    game_id: str
    final_score: float
    threshold_met: bool
    iterations_used: int
    score_history: List[float]
    total_improvement: float
    stages_completed: List[str]
    game_title: str | None = None
    evaluation_summary: str | None = None
    error: str | None = None


@router.post("/generate-complete", response_model=GenerateCompleteResponse)
async def generate_complete(request: GenerateCompleteRequest):
    """
    Complete pipeline: create game → evaluate → refine iteratively until threshold.

    This endpoint orchestrates the entire game creation flow:
    1. Create game state with preferences
    2. Generate concept and rules
    3. Generate cards
    4. Run initial simulation
    5. Initial evaluation
    6. Loop: refine → re-simulate → re-evaluate until threshold or max iterations

    Returns the final game state with full history.
    """
    import logging
    logger = logging.getLogger(__name__)

    stages_completed: List[str] = []
    score_history: List[float] = []

    try:
        # Get workflows
        workflows = get_workflows(
            premium=request.premium,
            premium_provider=request.premium_provider,
            model=request.model
        )
        llm_service = _create_premium_llm(request.premium_provider, request.model) if request.premium else standard_llm_service

        # Separate evaluation workflow if evaluation_provider is specified
        if request.evaluation_provider:
            eval_workflow = _get_evaluation_workflow(request.evaluation_provider, request.evaluation_model)
            logger.info(f"[generate-complete] Using separate evaluation model: "
                       f"{request.evaluation_provider}:{request.evaluation_model or 'default'}")
        else:
            eval_workflow = workflows['evaluation']

        # =================================================================
        # STAGE 1: Create game state
        # =================================================================
        game_id = str(uuid.uuid4())
        state = CardGameState(
            game_id=game_id,
            status=GameStatus.CREATED,
            preferences=request.preferences,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            max_evaluation_iterations=request.max_refinement_iterations,
            evaluation_threshold=request.target_threshold,
        )
        stages_completed.append("game_created")
        logger.info(f"[generate-complete] Game {game_id} created")

        # =================================================================
        # STAGE 2: Generate concept and rules
        # =================================================================
        # Note: These workflows use StateGraph(CardGameState) - pass dict, get dict back
        result = workflows['concept_and_rules'].invoke(
            state.model_dump(),
            config={"configurable": {"thread_id": f"concept-{game_id}"}}
        )
        state = CardGameState(**result)
        stages_completed.append("concept_and_rules_generated")
        logger.info(f"[generate-complete] Concept and rules generated: {state.concept.title}")

        # =================================================================
        # STAGE 3: Generate cards
        # =================================================================
        result = workflows['cards'].invoke(
            state.model_dump(),
            config={"configurable": {"thread_id": f"cards-{game_id}"}}
        )
        state = CardGameState(**result)
        if state.rules and state.cards:
            normalize_card_resources(state.rules, state.cards)
        stages_completed.append("cards_generated")
        logger.info(f"[generate-complete] Generated {len(state.cards)} cards")

        # =================================================================
        # STAGE 4: Run initial simulation
        # =================================================================
        try:
            report, warnings = run_simulation_for_game(
                rules=state.rules,
                cards=state.cards,
                game_name=state.concept.title,
                num_games=request.num_simulation_games,
                llm_service=llm_service,
                use_cache=False,  # Fresh simulation
            )
            analysis_agent = GameplayAnalysisAgent(llm_service)
            state.simulation_analysis = analysis_agent.analyze(
                report, language=state.concept.language
            )
            state.simulation_report = report
            state.compilation_warnings = warnings
            stages_completed.append("simulation_completed")
            logger.info(f"[generate-complete] Simulation completed ({request.num_simulation_games} games)")
        except Exception as sim_error:
            logger.warning(f"[generate-complete] Simulation failed: {sim_error}")
            # Continue without simulation - evaluation can still work
            stages_completed.append("simulation_skipped")

        # =================================================================
        # STAGE 5: Initial evaluation
        # =================================================================
        eval_state = {"game_state": state}
        result = eval_workflow.invoke(
            eval_state,
            config={"configurable": {"thread_id": f"eval-{game_id}"}}
        )
        state = result['game_state']
        stages_completed.append("initial_evaluation")

        initial_score = state.evaluation.overall_score
        score_history.append(initial_score)
        state.baseline_score = initial_score
        state.best_score_achieved = initial_score
        logger.info(f"[generate-complete] Initial evaluation: {initial_score:.2f}")

        # Check if already at threshold
        if initial_score >= request.target_threshold:
            await save_game_state_to_db(state)
            return GenerateCompleteResponse(
                game_id=game_id,
                final_score=initial_score,
                threshold_met=True,
                iterations_used=0,
                score_history=score_history,
                total_improvement=0.0,
                stages_completed=stages_completed,
                game_title=state.concept.title,
                evaluation_summary=state.evaluation.summary,
            )

        # =================================================================
        # STAGE 6: Refinement loop (using Director agent)
        # =================================================================
        from deck_crafter.services.refinement_service import execute_refinement_step

        state.evaluation_iteration = 0
        state.previous_evaluations = []
        state.max_evaluation_iterations = request.max_refinement_iterations
        state.refinement_memory = RefinementMemory()

        for iteration in range(1, request.max_refinement_iterations + 1):
            logger.info(f"[generate-complete] Starting refinement iteration {iteration}")

            ref_result = execute_refinement_step(
                state=state,
                threshold=request.target_threshold,
                llm_service=llm_service,
                eval_workflow=eval_workflow,
                num_simulation_games=request.num_simulation_games,
                use_batch_cards=True,
            )

            state = ref_result.state
            score_history.append(ref_result.new_score)
            stages_completed.append(f"refinement_iteration_{iteration}")

            strategy_info = ref_result.strategy.intervention_type if ref_result.strategy else "N/A"
            logger.info(f"[generate-complete] Iteration {iteration}: {ref_result.new_score:.2f} "
                       f"({ref_result.status}, Director: {strategy_info})")

            # Save after every iteration to avoid losing progress
            await save_game_state_to_db(state)

            if ref_result.stop_reason:
                logger.info(f"[generate-complete] Stopping: {ref_result.stop_reason}")
                break

        # =================================================================
        # FINAL: Save and return
        # =================================================================
        await save_game_state_to_db(state)

        final_score = state.best_score_achieved or state.evaluation.overall_score
        total_improvement = final_score - initial_score

        return GenerateCompleteResponse(
            game_id=game_id,
            final_score=final_score,
            threshold_met=final_score >= request.target_threshold,
            iterations_used=state.evaluation_iteration,
            score_history=score_history,
            total_improvement=total_improvement,
            stages_completed=stages_completed,
            game_title=state.concept.title,
            evaluation_summary=state.evaluation.summary,
        )

    except Exception as e:
        logger.exception(f"[generate-complete] Pipeline failed: {e}")
        return GenerateCompleteResponse(
            game_id=game_id if 'game_id' in locals() else "unknown",
            final_score=score_history[-1] if score_history else 0.0,
            threshold_met=False,
            iterations_used=len(score_history) - 1 if score_history else 0,
            score_history=score_history,
            total_improvement=0.0,
            stages_completed=stages_completed,
            error=str(e),
        )


# =================================================================
# CHAT: Conversational game editing
# =================================================================

class ChatRequest(BaseModel):
    message: str
    run_evaluation: bool = False
    num_simulation_games: int = 30


class ChatEndpointResponse(BaseModel):
    message: str
    actions_taken: list[dict]
    state_changed: bool
    evaluation_ran: bool
    score_before: float | None = None
    score_after: float | None = None
    game_state_summary: dict | None = None


# Per-game orchestrator instances (keeps undo stack alive within server session)
_orchestrators: dict[str, "OrchestratorAgent"] = {}


def _get_orchestrator(game_id: str):
    from deck_crafter.agents.orchestrator_agent import OrchestratorAgent
    if game_id not in _orchestrators:
        llm = GroqService(model="qwen/qwen3-32b")
        _orchestrators[game_id] = OrchestratorAgent(llm)
    return _orchestrators[game_id]


@router.post("/{game_id}/chat", response_model=ChatEndpointResponse)
async def chat_with_game(game_id: str, request: ChatRequest):
    """Chat with the game editing assistant to modify, query, or evaluate a game."""
    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    orchestrator = _get_orchestrator(game_id)
    chat_history = state.chat_history or []

    score_before = state.evaluation.overall_score if state.evaluation else None

    # Get eval workflow for improve/evaluate actions
    eval_workflow = _panel_eval or standard_workflows.get("evaluation")

    # Run orchestrator
    state, response_text, actions, eval_ran = orchestrator.process_message(
        message=request.message,
        state=state,
        chat_history=chat_history,
        run_eval=request.run_evaluation,
        eval_workflow=eval_workflow,
        num_sim_games=request.num_simulation_games,
    )

    score_after = state.evaluation.overall_score if state.evaluation else None
    state_changed = any(a.success and a.intent not in ("query", "explain") for a in actions)

    # Update chat history
    user_msg = ChatMessage(role="user", content=request.message)
    assistant_msg = ChatMessage(
        role="assistant",
        content=response_text,
        actions=[ChatAction(intent=a.intent, description=a.description, target=a.target, success=a.success) for a in actions],
    )
    chat_history.append(user_msg)
    chat_history.append(assistant_msg)
    state.chat_history = chat_history

    # Save
    state.updated_at = datetime.now(timezone.utc)
    await save_game_state_to_db(state)

    # Build summary
    summary = {
        "title": state.concept.title if state.concept else None,
        "score": score_after,
        "card_count": len(state.cards) if state.cards else 0,
        "iteration": state.evaluation_iteration,
    }

    return ChatEndpointResponse(
        message=response_text,
        actions_taken=[a.model_dump() for a in actions],
        state_changed=state_changed,
        evaluation_ran=eval_ran,
        score_before=score_before,
        score_after=score_after,
        game_state_summary=summary,
    )


@router.get("/{game_id}/chat/history")
async def get_chat_history(game_id: str):
    """Get the conversation history for a game."""
    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    history = state.chat_history or []
    return [m.model_dump() for m in history]


class ChatRefinementRequest(BaseModel):
    num_iterations: int = 5
    num_simulation_games: int = 30


class ChatRefinementIterationResult(BaseModel):
    iteration: int
    score_before: float
    score_after: float
    kept: bool
    actions: list[dict]


class ChatRefinementResponse(BaseModel):
    initial_score: float
    final_score: float
    iterations_run: int
    results: list[ChatRefinementIterationResult]
    game_state_summary: dict | None = None


@router.post("/{game_id}/chat/refine", response_model=ChatRefinementResponse)
async def chat_refine_game(game_id: str, request: ChatRefinementRequest):
    """Run chat-driven refinement loop: orchestrator plans edits, evaluates, keeps or rolls back."""
    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    if not state.evaluation:
        raise HTTPException(status_code=400, detail="Game must have an evaluation before refinement")

    orchestrator = _get_orchestrator(game_id)
    eval_workflow = _panel_eval or standard_workflows.get("evaluation")

    if not eval_workflow:
        raise HTTPException(status_code=500, detail="No evaluation workflow available")

    initial_score = state.evaluation.overall_score
    if state.baseline_score is None:
        state.baseline_score = initial_score

    iteration_results: list[ChatRefinementIterationResult] = []

    def on_iteration(iteration, _state, score_before, score_after, actions, kept):
        iteration_results.append(ChatRefinementIterationResult(
            iteration=iteration,
            score_before=score_before,
            score_after=score_after,
            kept=kept,
            actions=[a.model_dump() for a in actions],
        ))

    def save_state_cb(s):
        s.updated_at = datetime.now(timezone.utc)
        save_game_state_sync(s)

    state = orchestrator.run_refinement_loop(
        state=state,
        eval_workflow=eval_workflow,
        num_iterations=request.num_iterations,
        num_sim_games=request.num_simulation_games,
        on_iteration=on_iteration,
        save_state=save_state_cb,
    )

    # Save
    state.updated_at = datetime.now(timezone.utc)
    await save_game_state_to_db(state)

    final_score = state.evaluation.overall_score if state.evaluation else initial_score
    summary = {
        "title": state.concept.title if state.concept else None,
        "score": final_score,
        "card_count": len(state.cards) if state.cards else 0,
        "iteration": state.evaluation_iteration,
    }

    return ChatRefinementResponse(
        initial_score=initial_score,
        final_score=final_score,
        iterations_run=len(iteration_results),
        results=iteration_results,
        game_state_summary=summary,
    )