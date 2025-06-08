import aiosqlite
import json
from datetime import datetime
from typing import Optional
from pathlib import Path
from deck_crafter.models.card import Card
from deck_crafter.models.game_concept import GameConcept
from deck_crafter.models.rules import Rules
from deck_crafter.models.state import CardGameState, GameStatus
from deck_crafter.models.user_preferences import UserPreferences

DB_PATH = Path("deck_crafter.db")

async def init_db():
    """Initialize the database and create tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                preferences TEXT NOT NULL,
                concept TEXT,
                rules TEXT,
                cards TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await db.commit()

async def save_game_state(state: CardGameState):
    """Save a game state to the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO games 
            (game_id, status, preferences, concept, rules, cards, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.game_id,
            state.status.value,
            state.preferences.model_dump_json(),
            state.concept.model_dump_json() if state.concept else None,
            state.rules.model_dump_json() if state.rules else None,
            json.dumps([card.model_dump() for card in state.cards]) if state.cards else None,
            state.created_at.isoformat(),
            state.updated_at.isoformat()
        ))
        await db.commit()

async def get_game_state(game_id: str) -> Optional[CardGameState]:
    """Retrieve a game state from the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT * FROM games WHERE game_id = ?", (game_id,))
        row = await cursor.fetchone()
    
    if not row:
        return None

    # Ensure correct deserialization for concept, rules, and cards
    concept_data = json.loads(row[3]) if row[3] else None
    rules_data = json.loads(row[4]) if row[4] else None
    cards_data = json.loads(row[5]) if row[5] else None

    return CardGameState(
        game_id=row[0],
        status=GameStatus(row[1]),
        preferences=UserPreferences.model_validate_json(row[2]),
        concept=GameConcept.model_validate(concept_data) if concept_data else None,
        rules=Rules.model_validate(rules_data) if rules_data else None,
        cards=[Card.model_validate(card) for card in cards_data] if cards_data else None,
        created_at=datetime.fromisoformat(row[6]),
        updated_at=datetime.fromisoformat(row[7])
    ) 