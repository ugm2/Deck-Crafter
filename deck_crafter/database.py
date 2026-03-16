import aiosqlite
import json
from datetime import datetime, timezone
from typing import Optional, Dict
from pathlib import Path
from deck_crafter.models.card import Card
from deck_crafter.models.game_concept import GameConcept
from deck_crafter.models.rules import Rules
from deck_crafter.models.state import CardGameState, GameStatus, RefinementMemory
from deck_crafter.models.user_preferences import UserPreferences
from deck_crafter.models.evaluation import GameEvaluation, calculate_weighted_score
import sqlite3


def migrate_evaluation_data(data: dict) -> dict:
    """
    Migrate old 6-metric evaluation data to new 5-metric format.

    Old metrics: balance, coherence, clarity, playability, originality, fidelity
    New metrics: balance, clarity, playability, theme_alignment (merged), innovation (renamed)
    """
    if data is None:
        return None

    # Check if already in new format
    if "theme_alignment" in data and "innovation" in data:
        return data

    # Check if old format exists
    has_old_format = "coherence" in data or "fidelity" in data or "originality" in data
    if not has_old_format:
        return data

    migrated = data.copy()

    # Merge coherence + fidelity → theme_alignment
    coherence = data.get("coherence", {})
    fidelity = data.get("fidelity", {})

    if coherence or fidelity:
        # Average scores, combine analyses
        coh_score = coherence.get("score", 5) if coherence else 5
        fid_score = fidelity.get("score", 5) if fidelity else 5
        avg_score = (coh_score + fid_score) / 2

        coh_analysis = coherence.get("analysis", "") if coherence else ""
        fid_analysis = fidelity.get("analysis", "") if fidelity else ""
        combined_analysis = f"{coh_analysis}\n\n{fid_analysis}".strip()

        # Combine suggestions
        coh_suggestions = coherence.get("suggestions", []) if coherence else []
        fid_suggestions = fidelity.get("suggestions", []) if fidelity else []
        combined_suggestions = (coh_suggestions or []) + (fid_suggestions or [])

        migrated["theme_alignment"] = {
            "score": int(round(avg_score)),
            "analysis": combined_analysis or "Migrated from legacy evaluation.",
            "suggestions": combined_suggestions if combined_suggestions else None,
        }

        # Remove old fields
        migrated.pop("coherence", None)
        migrated.pop("fidelity", None)

    # Rename originality → innovation
    if "originality" in data:
        migrated["innovation"] = data["originality"]
        migrated.pop("originality", None)

    # Recalculate overall_score using weighted formula
    scores = {}
    for metric in ["balance", "clarity", "playability", "theme_alignment", "innovation"]:
        if metric in migrated and migrated[metric]:
            metric_data = migrated[metric]
            # Use final_score if available, otherwise score
            scores[metric] = metric_data.get("adjusted_score") or metric_data.get("score", 5)

    if scores:
        migrated["overall_score"] = calculate_weighted_score(scores)

    return migrated

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
                evaluation_iteration INTEGER DEFAULT 0,
                max_evaluation_iterations INTEGER DEFAULT 0,
                evaluation_threshold REAL DEFAULT 6.0,
                previous_evaluations TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Migration: add new columns if they don't exist
        try:
            await db.execute("ALTER TABLE games ADD COLUMN evaluation_iteration INTEGER DEFAULT 0")
        except aiosqlite.OperationalError:
            pass
        try:
            await db.execute("ALTER TABLE games ADD COLUMN max_evaluation_iterations INTEGER DEFAULT 0")
        except aiosqlite.OperationalError:
            pass
        try:
            await db.execute("ALTER TABLE games ADD COLUMN evaluation_threshold REAL DEFAULT 6.0")
        except aiosqlite.OperationalError:
            pass
        try:
            await db.execute("ALTER TABLE games ADD COLUMN previous_evaluations TEXT")
        except aiosqlite.OperationalError:
            pass
        try:
            await db.execute("ALTER TABLE games ADD COLUMN refinement_memory TEXT")
        except aiosqlite.OperationalError:
            pass
        try:
            await db.execute("ALTER TABLE games ADD COLUMN simulation_analysis TEXT")
        except aiosqlite.OperationalError:
            pass
        try:
            await db.execute("ALTER TABLE games ADD COLUMN simulation_report TEXT")
        except aiosqlite.OperationalError:
            pass

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
            (game_id, status, preferences, concept, rules, cards, evaluation,
             evaluation_iteration, max_evaluation_iterations, evaluation_threshold,
             previous_evaluations, refinement_memory, simulation_analysis, simulation_report,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.game_id,
            state.status.value if hasattr(state.status, 'value') else state.status,
            state.preferences.model_dump_json(),
            state.concept.model_dump_json() if state.concept else None,
            state.rules.model_dump_json() if state.rules else None,
            json.dumps([card.model_dump() for card in state.cards]) if state.cards else None,
            state.evaluation.model_dump_json() if state.evaluation else None,
            state.evaluation_iteration,
            state.max_evaluation_iterations,
            state.evaluation_threshold,
            json.dumps([e.model_dump() for e in state.previous_evaluations]) if state.previous_evaluations else None,
            state.refinement_memory.model_dump_json() if state.refinement_memory else None,
            state.simulation_analysis.model_dump_json() if state.simulation_analysis else None,
            state.simulation_report.model_dump_json() if state.simulation_report else None,
            state.created_at.isoformat(),
            state.updated_at.isoformat()
        ))
        await db.commit()

async def get_game_state(game_id: str) -> Optional[CardGameState]:
    """Retrieve a game state from the database."""
    # Import here to avoid circular dependency
    try:
        from game_simulator.models.metrics import GameplayAnalysis, SimulationReport
    except ImportError:
        GameplayAnalysis = None
        SimulationReport = None

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("""
            SELECT game_id, status, preferences, concept, rules, cards, evaluation,
                   evaluation_iteration, max_evaluation_iterations, evaluation_threshold,
                   previous_evaluations, refinement_memory, simulation_analysis, simulation_report,
                   created_at, updated_at
            FROM games WHERE game_id = ?
        """, (game_id,))
        row = await cursor.fetchone()

    if not row:
        return None

    concept_data = json.loads(row[3]) if row[3] else None
    rules_data = json.loads(row[4]) if row[4] else None
    cards_data = json.loads(row[5]) if row[5] else None
    evaluation_data = json.loads(row[6]) if row[6] else None
    previous_evals_data = json.loads(row[10]) if row[10] else None
    refinement_memory_data = json.loads(row[11]) if row[11] else None
    simulation_analysis_data = json.loads(row[12]) if row[12] else None
    simulation_report_data = json.loads(row[13]) if row[13] else None

    # Migrate old 6-metric evaluations to new 5-metric format
    evaluation_data = migrate_evaluation_data(evaluation_data)
    if previous_evals_data:
        previous_evals_data = [migrate_evaluation_data(e) for e in previous_evals_data]

    # Parse simulation data if available
    simulation_analysis = None
    if simulation_analysis_data and GameplayAnalysis:
        simulation_analysis = GameplayAnalysis.model_validate(simulation_analysis_data)

    simulation_report = None
    if simulation_report_data and SimulationReport:
        simulation_report = SimulationReport.model_validate(simulation_report_data)

    return CardGameState(
        game_id=row[0],
        status=GameStatus(row[1]),
        preferences=UserPreferences.model_validate_json(row[2]),
        concept=GameConcept.model_validate(concept_data) if concept_data else None,
        rules=Rules.model_validate(rules_data) if rules_data else None,
        cards=[Card.model_validate(card) for card in cards_data] if cards_data else None,
        evaluation=GameEvaluation.model_validate(evaluation_data) if evaluation_data else None,
        evaluation_iteration=row[7] or 0,
        max_evaluation_iterations=row[8] or 0,
        evaluation_threshold=row[9] or 6.0,
        previous_evaluations=[GameEvaluation.model_validate(e) for e in previous_evals_data] if previous_evals_data else None,
        refinement_memory=RefinementMemory.model_validate(refinement_memory_data) if refinement_memory_data else None,
        simulation_analysis=simulation_analysis,
        simulation_report=simulation_report,
        created_at=datetime.fromisoformat(row[14]),
        updated_at=datetime.fromisoformat(row[15])
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