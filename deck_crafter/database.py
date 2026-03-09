import aiosqlite
import json
from datetime import datetime, timezone
from typing import Optional, Dict
from pathlib import Path
from deck_crafter.models.card import Card
from deck_crafter.models.game_concept import GameConcept
from deck_crafter.models.rules import Rules
from deck_crafter.models.state import CardGameState, GameStatus
from deck_crafter.models.user_preferences import UserPreferences
from deck_crafter.models.evaluation import GameEvaluation
import sqlite3

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
                evaluation TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS card_images (
                game_id TEXT NOT NULL,
                card_name TEXT NOT NULL,
                image_data BLOB NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (game_id, card_name),
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
        """)
        await db.commit()

async def save_game_state(state: CardGameState):
    """Save a game state to the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO games 
            (game_id, status, preferences, concept, rules, cards, evaluation, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.game_id,
            state.status.value if hasattr(state.status, 'value') else state.status,
            state.preferences.model_dump_json(),
            state.concept.model_dump_json() if state.concept else None,
            state.rules.model_dump_json() if state.rules else None,
            json.dumps([card.model_dump() for card in state.cards]) if state.cards else None,
            state.evaluation.model_dump_json() if state.evaluation else None,
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

    # Ensure correct deserialization for concept, rules, cards, and evaluation
    concept_data = json.loads(row[3]) if row[3] else None
    rules_data = json.loads(row[4]) if row[4] else None
    cards_data = json.loads(row[5]) if row[5] else None
    evaluation_data = json.loads(row[6]) if row[6] else None

    return CardGameState(
        game_id=row[0],
        status=GameStatus(row[1]),
        preferences=UserPreferences.model_validate_json(row[2]),
        concept=GameConcept.model_validate(concept_data) if concept_data else None,
        rules=Rules.model_validate(rules_data) if rules_data else None,
        cards=[Card.model_validate(card) for card in cards_data] if cards_data else None,
        evaluation=GameEvaluation.model_validate(evaluation_data) if evaluation_data else None,
        created_at=datetime.fromisoformat(row[7]),
        updated_at=datetime.fromisoformat(row[8])
    )

async def save_card_image(game_id: str, card_name: str, image_data: bytes):
    """Save a card image to the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO card_images 
            (game_id, card_name, image_data, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            game_id,
            card_name,
            image_data,
            datetime.now().isoformat()
        ))
        await db.commit()

async def get_card_image(game_id: str, card_name: str) -> Optional[bytes]:
    """Retrieve a card image from the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT image_data FROM card_images WHERE game_id = ? AND card_name = ?",
            (game_id, card_name)
        )
        row = await cursor.fetchone()
        return row[0] if row else None

async def get_all_card_images(game_id: str) -> Dict[str, bytes]:
    """Retrieve all card images for a game from the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT card_name, image_data FROM card_images WHERE game_id = ?",
            (game_id,)
        )
        rows = await cursor.fetchall()
        return {row[0]: row[1] for row in rows}

def save_card_image_sync(game_id: str, card_name: str, image_data: bytes) -> None:
    """Save a card image to the database synchronously."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO card_images (game_id, card_name, image_data, created_at) VALUES (?, ?, ?, ?)",
            (game_id, card_name, image_data, datetime.now(timezone.utc))
        )
        conn.commit()
    finally:
        conn.close()

def get_existing_card_images_sync(game_id: str) -> set:
    """Get set of card names that already have images."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(
            "SELECT card_name FROM card_images WHERE game_id = ?",
            (game_id,)
        )
        return {row[0] for row in cursor.fetchall()}
    finally:
        conn.close() 