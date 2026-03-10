from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any
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
from deck_crafter.services.llm_service import create_llm_service, create_fallback_llm_service, GeminiService, GroqService
from deck_crafter.utils.config import Config as LLMConfig
from deck_crafter.models.state import CardGameState, GameStatus
from deck_crafter.utils.config import Config
from deck_crafter.database import init_db, save_game_state as save_game_state_to_db, get_game_state as get_game_state_from_db, get_all_card_images

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
standard_workflows = {
    'preferences': create_preferences_workflow(standard_llm_service),
    'concept_and_rules': create_concept_and_rules_workflow(standard_llm_service),
    'cards': create_cards_workflow(standard_llm_service),
    'images': create_image_generation_workflow(standard_llm_service),
    'evaluation': create_multi_agent_evaluation_workflow(standard_llm_service),
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
        logger.info(f"Creating Groq premium service with model {LLMConfig.GROQ_PREMIUM_MODEL}")
        return GroqService(model=LLMConfig.GROQ_PREMIUM_MODEL, max_tokens=16384)
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
            'evaluation': create_multi_agent_evaluation_workflow(llm),
            'refinement': create_refinement_workflow(llm),
        }
    return _premium_workflows_cache[cache_key]

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
    import logging
    from deck_crafter.agents.director_agent import DirectorAgent
    from deck_crafter.agents.feedback_agent import FeedbackSynthesizerAgent
    from deck_crafter.agents.rules_agent import RuleGenerationAgent
    from deck_crafter.agents.card_agent import CardGenerationAgent
    from deck_crafter.models.state import RefinementMemory, RefinementExperiment

    logger = logging.getLogger(__name__)

    state = await get_game_state_from_db(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")

    if not state.evaluation:
        raise HTTPException(
            status_code=400,
            detail="Game must be evaluated before refinement. Call /evaluate first."
        )

    current_score = state.evaluation.overall_score

    # Check if already at threshold
    if current_score >= threshold:
        return {
            "status": "threshold_met",
            "score": current_score,
            "threshold": threshold,
            "iteration": state.evaluation_iteration,
            "message": f"Score {current_score:.1f} already meets threshold {threshold}"
        }

    # Get LLM service
    llm = _create_premium_llm(premium_provider, model) if premium else standard_llm_service

    # Initialize agents
    director_agent = DirectorAgent(llm)
    feedback_agent = FeedbackSynthesizerAgent(llm)
    rules_agent = RuleGenerationAgent(llm)
    card_agent = CardGenerationAgent(llm)

    # Get evaluation workflow
    workflows = get_workflows(premium, premium_provider, model)

    # Load or initialize refinement memory
    memory = state.refinement_memory or RefinementMemory()

    # Store previous evaluation
    if state.previous_evaluations is None:
        state.previous_evaluations = []
    state.previous_evaluations.append(state.evaluation)
    evaluation_before = state.evaluation

    # Save state for potential rollback
    previous_state_json = state.model_dump_json()
    previous_score = current_score

    # Step 0: Director designs experiment with memory context
    max_iterations = state.max_evaluation_iterations or 5
    cards_summary = ""
    if state.cards:
        cards_summary = "\n".join([
            f"- {c.name} (Type: {c.type}, Qty: {c.quantity}): {c.description[:80]}..."
            if len(c.description) > 80 else f"- {c.name} (Type: {c.type}, Qty: {c.quantity}): {c.description}"
            for c in state.cards
        ])

    # Log memory state for debugging
    if memory.failed_patterns:
        logger.info(f"Memory has {len(memory.failed_patterns)} failed patterns: {memory.failed_patterns[-3:]}")  # Last 3

    strategy = director_agent.design_experiment(
        evaluation=state.evaluation,
        threshold=threshold,
        iteration=state.evaluation_iteration + 1,
        max_iterations=max_iterations,
        cards_summary=cards_summary,
        memory=memory,
        previous_evaluations=state.previous_evaluations[:-1] if len(state.previous_evaluations) > 1 else None
    )
    logger.info(f"Director experiment: {strategy.intervention_type.upper()} targeting {strategy.target_metric}")
    logger.info(f"Hypothesis: {strategy.hypothesis}")
    logger.info(f"Expected improvement: +{strategy.expected_improvement} (confidence: {strategy.confidence})")
    logger.info(f"Rules: {strategy.rules_action} | Cards: {strategy.cards_action}")

    # Step 1: Synthesize feedback guided by Director's strategy
    language = state.concept.language if state.concept else "English"
    feedback = feedback_agent.synthesize(state.evaluation, state.cards, language, strategy=strategy)

    # Track what we're changing for the experiment record
    rules_changed = False
    cards_changed = []

    # Step 2: Regenerate rules based on granular control
    should_refine_rules = strategy.rules_action != "none"
    if should_refine_rules and feedback.rules_critique.lower() != "no changes needed":
        logger.info(f"Refining rules: {strategy.rules_action} on {strategy.rules_target}")
        state.critique = feedback.rules_critique
        if strategy.rules_instruction:
            state.critique += f"\n\nDIRECTOR INSTRUCTION: {strategy.rules_instruction}"

        # Choose method based on rules_action:
        # - tweak: additive only (glossary, examples, FAQ)
        # - rewrite_section: regenerate only the target section
        # - overhaul: full regeneration
        if strategy.rules_action == "tweak":
            logger.info("Using ADDITIVE rules enhancement (glossary, examples, FAQ)")
            result = rules_agent.enhance_rules(state)
        elif strategy.rules_action == "rewrite_section" and strategy.rules_target:
            logger.info(f"Using SECTION-SPECIFIC rewrite for: {strategy.rules_target}")
            result = rules_agent.rewrite_section(state, strategy.rules_target)
        else:  # overhaul or rewrite_section without target
            logger.info("Using FULL rules regeneration (overhaul)")
            result = rules_agent.generate_rules(state)

        if "rules" in result:
            state.rules = result["rules"]
            rules_changed = True
    elif strategy.rules_action == "none":
        logger.info("Skipping rules refinement (Director: rules_action=none)")

    # Step 3: Selective card regeneration based on granular control
    should_refine_cards = strategy.cards_action != "none"
    if should_refine_cards and feedback.cards_critique.lower() != "no changes needed":
        logger.info(f"Refining cards: {strategy.cards_action}")
        state.critique = feedback.cards_critique
        if strategy.cards_instruction:
            state.critique += f"\n\nDIRECTOR INSTRUCTION: {strategy.cards_instruction}"

        # Use Director's card list if available, otherwise use feedback's list
        cards_to_remove = set(strategy.cards_to_modify) if strategy.cards_to_modify else set(feedback.cards_to_regenerate or [])

        if cards_to_remove:
            original_count = len(state.cards) if state.cards else 0
            state.cards = [c for c in (state.cards or []) if c.name not in cards_to_remove]
            cards_removed = original_count - len(state.cards)
            cards_changed = list(cards_to_remove)
            logger.info(f"Keeping {len(state.cards)} cards, regenerating {cards_removed}: {cards_to_remove}")

            # Track problematic cards in memory
            for card_name in cards_to_remove:
                memory.problematic_cards[card_name] = memory.problematic_cards.get(card_name, 0) + 1
        else:
            # Only regenerate all if cards_action is regenerate_many AND no specific cards
            if strategy.cards_action == "regenerate_many":
                logger.warning("No specific cards listed with regenerate_many, regenerating all")
                cards_changed = [c.name for c in (state.cards or [])]
                state.cards = []
            else:
                logger.info("No specific cards to modify")

        # Regenerate missing cards
        total_cards = state.concept.number_of_unique_cards if state.concept else 0
        while len(state.cards) < total_cards:
            result = card_agent.generate_card(state)
            if "cards" in result:
                state.cards = result["cards"]
            else:
                break
    elif strategy.cards_action == "none":
        logger.info("Skipping cards refinement (Director: cards_action=none)")

    # Step 4: Re-evaluate
    eval_state = {"game_state": state}
    eval_result = workflows['evaluation'].invoke(eval_state, config={"configurable": {"thread_id": f"refine-eval-{game_id}"}})
    state = eval_result['game_state']
    state.evaluation_iteration += 1
    state.evaluation_threshold = threshold
    state.status = GameStatus.EVALUATED
    state.updated_at = datetime.now(timezone.utc)

    new_score = state.evaluation.overall_score
    improved = new_score > current_score
    actual_improvement = new_score - current_score

    # Update best score achieved
    if state.best_score_achieved is None or new_score > state.best_score_achieved:
        state.best_score_achieved = new_score

    # Create experiment record
    experiment = RefinementExperiment(
        iteration=state.evaluation_iteration,
        hypothesis=strategy.hypothesis,
        target_metric=strategy.target_metric,
        expected_improvement=strategy.expected_improvement,
        intervention_type=strategy.intervention_type,
        rules_changes=strategy.rules_target if rules_changed else None,
        cards_changed=cards_changed,
        score_before=previous_score,
        score_after=new_score,
        actual_improvement=actual_improvement,
    )

    # Step 5: Reflect and update memory
    if actual_improvement >= strategy.expected_improvement * 0.5:
        experiment.hypothesis_confirmed = True
    else:
        experiment.hypothesis_confirmed = False

    # Use LLM to generate reflection
    try:
        reflection = director_agent.reflect(experiment, evaluation_before, state.evaluation)
        experiment.reflection = reflection.lesson_learned

        # Update memory based on reflection
        memory.lessons_learned.append(reflection.lesson_learned)
        if reflection.should_continue_pattern:
            pattern = f"{strategy.intervention_type}: {strategy.hypothesis[:50]}"
            if pattern not in memory.successful_patterns:
                memory.successful_patterns.append(pattern)
        if reflection.pattern_to_avoid:
            memory.failed_patterns.append(reflection.pattern_to_avoid)
    except Exception as e:
        logger.warning(f"Reflection failed: {e}")
        experiment.reflection = f"Improvement: {actual_improvement:+.1f} (expected {strategy.expected_improvement:+.1f})"

    memory.experiments.append(experiment)

    # Rollback if score degraded significantly
    if new_score < previous_score - 0.1:  # Allow tiny variance
        logger.warning(f"ROLLBACK: Score degraded ({previous_score:.1f} → {new_score:.1f})")

        # Record the failed pattern with SPECIFIC details so Director won't repeat it
        failed_pattern = (
            f"FAILED [{strategy.intervention_type.upper()}]: "
            f"target={strategy.target_metric}, "
            f"rules_action={strategy.rules_action}"
            f"{f' on {strategy.rules_target}' if strategy.rules_target else ''}, "
            f"cards_action={strategy.cards_action}. "
            f"Result: {previous_score:.1f}→{new_score:.1f} (regression). "
            f"DO NOT repeat this combination."
        )
        memory.failed_patterns.append(failed_pattern)
        experiment.hypothesis_confirmed = False
        experiment.reflection = f"FAILED: Caused score regression from {previous_score:.1f} to {new_score:.1f}"

        # Rollback state but keep memory
        state = CardGameState.model_validate_json(previous_state_json)
        state.evaluation_iteration += 1  # Still count the iteration
        state.refinement_memory = memory  # Preserve memory with failed experiment
        await save_game_state_to_db(state)

        # Delta scoring for reverted state
        baseline = state.baseline_score or previous_score
        return {
            "status": "reverted",
            "iteration": state.evaluation_iteration,
            "previous_score": previous_score,
            "new_score": previous_score,  # After rollback, score is back to previous
            "attempted_score": new_score,
            "threshold": threshold,
            "threshold_met": False,
            "improved": False,
            "improvement": 0,
            "reason": "score_degraded",
            # Delta scoring
            "progress": {
                "baseline_score": round(baseline, 2),
                "current_score": round(previous_score, 2),
                "total_improvement": round(previous_score - baseline, 2),
                "iteration_improvement": 0,
                "best_score_achieved": round(state.best_score_achieved or previous_score, 2),
                "trend": "regressing",
            },
            "experiment": {
                "hypothesis": strategy.hypothesis,
                "intervention": strategy.intervention_type,
                "target_metric": strategy.target_metric,
                "expected": strategy.expected_improvement,
                "actual": round(new_score - previous_score, 2),
                "confirmed": False,
            },
            "failed_patterns": len(memory.failed_patterns),
            "message": f"Refinement made game worse ({previous_score:.1f} → {new_score:.1f}), reverted. Pattern recorded as failed."
        }

    # Stop if no improvement in last 2 iterations
    if state.previous_evaluations and len(state.previous_evaluations) >= 2:
        last_scores = [e.overall_score for e in state.previous_evaluations[-2:]]
        if all(s >= new_score for s in last_scores):
            state.refinement_memory = memory
            await save_game_state_to_db(state)

            # Delta scoring for stopped state
            baseline = state.baseline_score or current_score
            return {
                "status": "stopped",
                "iteration": state.evaluation_iteration,
                "previous_score": current_score,
                "new_score": new_score,
                "score": new_score,
                "threshold": threshold,
                "threshold_met": new_score >= threshold,
                "improved": new_score > current_score,
                "improvement": round(new_score - current_score, 2),
                "reason": "no_improvement",
                # Delta scoring
                "progress": {
                    "baseline_score": round(baseline, 2),
                    "current_score": round(new_score, 2),
                    "total_improvement": round(new_score - baseline, 2),
                    "iteration_improvement": round(new_score - current_score, 2),
                    "best_score_achieved": round(state.best_score_achieved or new_score, 2),
                    "trend": "plateau",
                },
                "experiments_run": len(memory.experiments),
                "message": f"Stopped: no improvement in last 2 iterations (score: {new_score:.1f})"
            }

    # Save state with updated memory
    state.refinement_memory = memory
    await save_game_state_to_db(state)

    # Calculate delta scoring for better progress visibility
    baseline = state.baseline_score or current_score
    total_improvement = new_score - baseline

    return {
        "status": "refined",
        "iteration": state.evaluation_iteration,
        "previous_score": current_score,
        "new_score": new_score,
        "threshold": threshold,
        "threshold_met": new_score >= threshold,
        "improved": improved,
        "improvement": round(actual_improvement, 2),
        # Delta scoring for better progress visibility
        "progress": {
            "baseline_score": round(baseline, 2),
            "current_score": round(new_score, 2),
            "total_improvement": round(total_improvement, 2),
            "iteration_improvement": round(actual_improvement, 2),
            "best_score_achieved": round(state.best_score_achieved or new_score, 2),
            "trend": "improving" if actual_improvement > 0.05 else "plateau" if actual_improvement >= -0.05 else "regressing",
        },
        "experiment": {
            "hypothesis": strategy.hypothesis,
            "intervention": strategy.intervention_type,
            "target_metric": strategy.target_metric,
            "expected": strategy.expected_improvement,
            "actual": round(actual_improvement, 2),
            "confirmed": experiment.hypothesis_confirmed,
            "confidence": strategy.confidence,
        },
        "priority_issues": feedback.priority_issues,
        "cards_regenerated": cards_changed,
        "rules_changed": rules_changed,
        "lessons_learned": len(memory.lessons_learned),
        "failed_patterns": len(memory.failed_patterns),
        "evaluation_summary": state.evaluation.summary
    }


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

    # Check if already at threshold
    if state.evaluation.overall_score >= threshold:
        return {
            "status": "already_optimal",
            "message": f"Score {state.evaluation.overall_score:.1f} already meets threshold {threshold}",
            "iterations_used": 0,
            "final_score": state.evaluation.overall_score,
            "improved": False
        }

    # Configure refinement parameters
    state.max_evaluation_iterations = min(max_iterations, 5)  # Cap at 5
    state.evaluation_threshold = threshold
    state.evaluation_iteration = 0  # Reset counter

    initial_score = state.evaluation.overall_score

    workflows = get_workflows(premium, premium_provider, model)
    refinement_state = {"game_state": state, "feedback": None, "should_stop": False}

    result = workflows['refinement'].invoke(
        refinement_state,
        config={"configurable": {"thread_id": f"refine-{game_id}"}}
    )

    final_game_state = result['game_state']
    await save_game_state_to_db(final_game_state)

    final_score = final_game_state.evaluation.overall_score
    iterations_used = final_game_state.evaluation_iteration

    return {
        "status": "refined",
        "iterations_used": iterations_used,
        "initial_score": initial_score,
        "final_score": final_score,
        "improved": final_score > initial_score,
        "threshold_met": final_score >= threshold,
        "evaluation_summary": final_game_state.evaluation.summary
    }