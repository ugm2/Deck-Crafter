import logging
from enum import Enum
from typing import Optional, List, Literal, Dict
from datetime import datetime
from pydantic import BaseModel, Field

from deck_crafter.models.card import Card

logger = logging.getLogger(__name__)
from deck_crafter.models.game_concept import GameConcept
from deck_crafter.models.rules import Rules
from deck_crafter.models.user_preferences import UserPreferences
from deck_crafter.models.evaluation import GameEvaluation

# Forward reference for simulation analysis (optional import to avoid circular deps)
try:
    from game_simulator.models.metrics import GameplayAnalysis, SimulationReport
except ImportError:
    GameplayAnalysis = None
    SimulationReport = None


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


class FailedPattern(BaseModel):
    """Structured record of a failed refinement approach for reliable matching."""
    iteration: int = Field(..., description="Which iteration this failed")
    target_metric: str = Field(..., description="Which metric was targeted")
    intervention_type: Literal["surgical", "moderate", "nuclear"]
    rules_action: str = Field(..., description="none/tweak/rewrite_section/overhaul")
    rules_target: Optional[str] = Field(None, description="Which section was targeted")
    cards_action: str = Field(..., description="none/stat_adjust/regenerate_few/regenerate_many")
    cards_affected: List[str] = Field(default_factory=list)

    # Results
    score_before: float
    score_after: float
    regression: float = Field(..., description="Negative number showing score drop")

    def matches(self, target_metric: str, rules_action: str, rules_target: Optional[str] = None) -> bool:
        """Check if a proposed strategy matches this failed pattern."""
        return (
            self.target_metric == target_metric and
            self.rules_action == rules_action and
            self.rules_target == rules_target
        )


class MetricCeiling(BaseModel):
    """Tracks when a metric has hit its improvement ceiling."""
    metric: str
    consecutive_failures: int = 0
    last_successful_score: float = 0.0
    ceiling_detected: bool = False
    attempts_since_last_success: int = 0


class RefinementMemory(BaseModel):
    """Persistent memory for the Director Agent across refinement iterations."""
    experiments: List[RefinementExperiment] = Field(default_factory=list)
    lessons_learned: List[str] = Field(default_factory=list)
    successful_patterns: List[str] = Field(default_factory=list)

    # CHANGED: Structured failed patterns instead of free-form strings
    failed_patterns: List[FailedPattern] = Field(default_factory=list)
    # Legacy string patterns (for backward compatibility)
    failed_pattern_strings: List[str] = Field(default_factory=list)

    problematic_cards: Dict[str, int] = Field(default_factory=dict)

    # NEW: Metric-level failure tracking
    metric_failures: Dict[str, int] = Field(
        default_factory=lambda: {
            "playability": 0, "balance": 0, "clarity": 0,
            "theme_alignment": 0, "innovation": 0
        }
    )
    metric_ceilings: Dict[str, MetricCeiling] = Field(default_factory=dict)

    # NEW: Escalation tracking
    total_failed_iterations: int = 0
    last_improvement_iteration: int = 0

    def record_failure(self, metric: str) -> None:
        """Record a metric failure and check for ceiling."""
        self.metric_failures[metric] = self.metric_failures.get(metric, 0) + 1
        self.total_failed_iterations += 1
        logger.debug(f"[RefinementMemory] Recorded failure for metric '{metric}' "
                    f"(total failures: {self.total_failed_iterations})")

        if metric not in self.metric_ceilings:
            self.metric_ceilings[metric] = MetricCeiling(metric=metric)

        ceiling = self.metric_ceilings[metric]
        ceiling.consecutive_failures += 1
        ceiling.attempts_since_last_success += 1

        # Ceiling detected after 2 consecutive failures on same metric
        if ceiling.consecutive_failures >= 2:
            ceiling.ceiling_detected = True
            logger.warning(f"[RefinementMemory] CEILING DETECTED for metric '{metric}' "
                          f"after {ceiling.consecutive_failures} consecutive failures")

    def record_success(self, metric: str, score: float, iteration: int) -> None:
        """Record a metric success, resetting failure counters."""
        logger.info(f"[RefinementMemory] Success for metric '{metric}' at iteration {iteration} "
                   f"(score: {score:.2f})")
        self.metric_failures[metric] = 0
        self.last_improvement_iteration = iteration

        if metric in self.metric_ceilings:
            ceiling = self.metric_ceilings[metric]
            if ceiling.ceiling_detected:
                logger.info(f"[RefinementMemory] Ceiling CLEARED for metric '{metric}'")
            ceiling.consecutive_failures = 0
            ceiling.last_successful_score = score
            ceiling.ceiling_detected = False
            ceiling.attempts_since_last_success = 0

    def get_blocked_metrics(self) -> List[str]:
        """Return metrics that have hit their ceiling."""
        return [m for m, c in self.metric_ceilings.items() if c.ceiling_detected]

    def should_switch_target(self, current_metric: str) -> bool:
        """True if we should stop targeting this metric."""
        return current_metric in self.get_blocked_metrics()

    def get_best_target_metric(self, current_scores: Dict[str, float]) -> str:
        """Suggest the best metric to target next, avoiding ceilings."""
        blocked = set(self.get_blocked_metrics())
        weights = {
            "playability": 2.0, "balance": 1.5, "clarity": 1.2,
            "theme_alignment": 1.0, "innovation": 0.8
        }

        # Score each metric by: (10 - current_score) * weight
        # Higher = more room for improvement * higher impact
        candidates = []
        for metric, score in current_scores.items():
            if metric not in blocked:
                room = 10 - score
                impact = room * weights.get(metric, 1.0)
                candidates.append((metric, impact))

        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0] if candidates else "playability"

    def add_failed_pattern(self, pattern: FailedPattern) -> None:
        """Add a structured failed pattern."""
        logger.warning(f"[RefinementMemory] Adding FAILED pattern: "
                      f"target={pattern.target_metric}, rules={pattern.rules_action}"
                      f"({pattern.rules_target}), regression={pattern.regression:.2f}")
        self.failed_patterns.append(pattern)
        # Also add legacy string for backward compatibility
        self.failed_pattern_strings.append(
            f"FAILED [{pattern.intervention_type.upper()}]: "
            f"target={pattern.target_metric}, rules={pattern.rules_action}"
            f"{f'({pattern.rules_target})' if pattern.rules_target else ''}, "
            f"cards={pattern.cards_action}"
        )

    def check_pattern_blocked(
        self, target_metric: str, rules_action: str, rules_target: Optional[str] = None
    ) -> bool:
        """Check if a proposed strategy matches any failed pattern."""
        for fp in self.failed_patterns:
            if fp.matches(target_metric, rules_action, rules_target):
                logger.warning(f"[RefinementMemory] Pattern BLOCKED: "
                              f"target={target_metric}, rules={rules_action}({rules_target}) "
                              f"matches failed pattern from iteration {fp.iteration}")
                return True
        return False


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

    # Simulation-based gameplay analysis (optional, from game_simulator)
    simulation_analysis: Optional["GameplayAnalysis"] = None
    # Raw simulation report with detailed statistics
    simulation_report: Optional["SimulationReport"] = None
    # Compilation warnings from rule compiler (surfaced to Clarity evaluation)
    compilation_warnings: List[str] = Field(default_factory=list)

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
