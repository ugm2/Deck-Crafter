from enum import Enum
from typing import Optional, List, Literal, Dict
from datetime import datetime
from pydantic import BaseModel, Field

from deck_crafter.models.card import Card
from deck_crafter.models.game_concept import GameConcept
from deck_crafter.models.rules import Rules
from deck_crafter.models.user_preferences import UserPreferences
from deck_crafter.models.evaluation import GameEvaluation


# --- Refinement Memory Models (State-of-the-Art Agentic System) ---

class RefinementExperiment(BaseModel):
    """A single refinement iteration treated as a scientific experiment."""
    iteration: int

    # Hypothesis
    hypothesis: str
    target_metric: str
    expected_improvement: float

    # Action taken
    intervention_type: Literal["surgical", "moderate", "nuclear"]
    rules_changes: Optional[str] = None
    cards_changed: List[str] = Field(default_factory=list)

    # Results
    score_before: float
    score_after: Optional[float] = None
    actual_improvement: Optional[float] = None
    hypothesis_confirmed: Optional[bool] = None

    # Post-hoc reflection
    reflection: Optional[str] = None


class RefinementMemory(BaseModel):
    """Persistent memory for the Director Agent across refinement iterations."""
    experiments: List[RefinementExperiment] = Field(default_factory=list)
    lessons_learned: List[str] = Field(default_factory=list)
    successful_patterns: List[str] = Field(default_factory=list)
    failed_patterns: List[str] = Field(default_factory=list)
    problematic_cards: Dict[str, int] = Field(default_factory=dict)


class RefinementProgress(BaseModel):
    """Tracks refinement progress with delta scoring for better visibility."""
    baseline_score: float = Field(..., description="Initial score before any refinement")
    current_score: float = Field(..., description="Current score after refinement")
    total_improvement: float = Field(..., description="current_score - baseline_score")
    iteration_improvement: float = Field(0.0, description="current_score - previous_score")
    trend: Literal["improving", "plateau", "regressing"] = Field(..., description="Score trend")
    best_score_achieved: float = Field(..., description="Highest score achieved so far")
    iterations_completed: int = Field(0, description="Number of refinement iterations completed")

    @classmethod
    def calculate(
        cls,
        baseline: float,
        current: float,
        previous: Optional[float] = None,
        best: Optional[float] = None,
        iterations: int = 0
    ) -> "RefinementProgress":
        """Calculate refinement progress metrics."""
        total_improvement = current - baseline
        iteration_improvement = (current - previous) if previous is not None else 0.0
        best_achieved = max(best or baseline, current)

        # Determine trend
        if previous is None:
            trend = "improving" if total_improvement > 0 else "plateau"
        elif iteration_improvement > 0.05:
            trend = "improving"
        elif iteration_improvement < -0.05:
            trend = "regressing"
        else:
            trend = "plateau"

        return cls(
            baseline_score=baseline,
            current_score=current,
            total_improvement=total_improvement,
            iteration_improvement=iteration_improvement,
            trend=trend,
            best_score_achieved=best_achieved,
            iterations_completed=iterations,
        )


# --- LÓGICA DE ACTUALIZACIÓN PERSONALIZADA ---
def last_write_wins(a, b):
    return b

class GameStatus(str, Enum):
    CREATED = "created"
    CONCEPT_GENERATED = "concept_generated"
    RULES_GENERATED = "rules_generated"
    CARDS_GENERATED = "cards_generated"
    IMAGES_GENERATED = "images_generated"
    EVALUATED = "evaluated"

class CardGameState(BaseModel):
    game_id: str
    status: GameStatus
    preferences: UserPreferences
    concept: Optional[GameConcept] = None
    rules: Optional[Rules] = None
    cards: Optional[List[Card]] = None
    image_paths: Optional[dict[str, str]] = None
    evaluation: Optional[GameEvaluation] = None
    created_at: datetime
    updated_at: datetime

    # Validation critique (used during generation)
    critique: Optional[str] = None
    refinement_count: int = 0

    # Post-evaluation refinement loop
    evaluation_iteration: int = 0
    max_evaluation_iterations: int = 0  # 0 = disabled (default)
    evaluation_threshold: float = 6.0
    previous_evaluations: Optional[List[GameEvaluation]] = None

    # Delta scoring for refinement progress visibility
    baseline_score: Optional[float] = None  # Initial score before any refinement
    best_score_achieved: Optional[float] = None  # Highest score achieved

    # Reflexive refinement memory (state-of-the-art agentic system)
    refinement_memory: Optional[RefinementMemory] = None

    def get_refinement_progress(self) -> Optional["RefinementProgress"]:
        """Calculate refinement progress if evaluation exists."""
        if not self.evaluation:
            return None

        current = self.evaluation.overall_score
        baseline = self.baseline_score or current
        previous = self.previous_evaluations[-1].overall_score if self.previous_evaluations else None

        return RefinementProgress.calculate(
            baseline=baseline,
            current=current,
            previous=previous,
            best=self.best_score_achieved,
            iterations=self.evaluation_iteration,
        )
