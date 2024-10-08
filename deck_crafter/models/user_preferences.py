from typing import Optional
from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    language: Optional[str] = Field("Español", description="The language of the game.")
    theme: Optional[str] = Field(
        "Fantasía tierra media ambientada en Toledo, España",
        description="The theme of the card game.",
    )
    game_style: Optional[str] = Field(
        "Un party game con mecánicas similares a Exploding Kittens, pero con giros innovadores que lo diferencian, nada de mencionar algo de un gato explosivo ni menciones nada relacionado con 'explosivo'.",
        description="The style of the game (e.g., party game, competitive).",
    )
    number_of_players: Optional[str] = Field(
        "4-12", description="The number of players the game is designed for."
    )
    target_audience: Optional[str] = Field(
        "+18, incluye contenido picante",
        description="The target audience for the game (e.g., age group).",
    )
    rule_complexity: Optional[str] = Field(
        "Reglas fáciles de aprender pero con profundidad estratégica.",
        description="The complexity of the rules (e.g., Easy, Medium, Hard).",
    )
