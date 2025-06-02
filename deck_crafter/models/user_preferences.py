from typing import Optional
from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    language: Optional[str] = None
    theme: Optional[str] = None
    game_style: Optional[str] = None
    number_of_players: Optional[str] = None
    target_audience: Optional[str] = None
    rule_complexity: Optional[str] = None
    game_description: Optional[str] = None

class RequiredUserPreferences(BaseModel):
    language: str = Field(
        ...,
        pattern=r"^[A-Z][a-z]+$",
        description="The language in which the game rules and content will be written (e.g., 'English', 'Español', 'Français')."
    )
    theme: str = Field(
        ...,
        description="The theme of the game (e.g., 'Fantasía', 'Ciencia ficción', 'Medieval')."
    )
    game_style: str = Field(
        ...,
        description="The style of the game (e.g., 'Party game', 'Competitive')."
    )
    number_of_players: str = Field(
        ..., 
        pattern=r"^[0-9]+(-[0-9]+)?$", 
        description="The number of players the game is designed for (e.g., '2-4', '5-8')."
    )
    target_audience: str = Field(
        ...,
        description="The target audience for the game (e.g., 'Niños', 'Adolescentes', 'Adultos')."
    )
    rule_complexity: str = Field(
        ...,
        description="The complexity of the rules (e.g., 'Easy', 'Medium', 'Hard')."
    )
    game_description: str = Field(
        ...,
        description="The description of the game (e.g., 'Un juego de cartas de fantasía donde los jugadores son magos que compiten por dominar diferentes escuelas de magia')."
    )
