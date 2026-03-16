from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChatAction(BaseModel):
    """A single action the orchestrator executed."""
    intent: str
    description: str
    target: Optional[str] = None
    success: bool = True


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actions: Optional[list[ChatAction]] = None


class PlannedAction(BaseModel):
    intent: Literal[
        "edit_card", "edit_rule", "add_card", "remove_card",
        "regenerate_cards", "regenerate_rules",
        "improve_metric", "improve_general",
        "evaluate", "simulate",
        "query", "explain", "undo",
    ]
    reasoning: str
    params: dict = Field(default_factory=dict)


class ActionPlan(BaseModel):
    understanding: str
    actions: list[PlannedAction] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    response_language: str = "en"
