# Plan: Conversational Game Editing

## Context

After a game is generated (or during refinement), users should be able to **talk to the system in natural language** to edit, improve, query, or restructure their game. The system must be smart enough to interpret intent, plan actions, execute them using existing agents/services, and optionally run simulation + panel evaluation after changes.

**Orchestrator model:** `qwen/qwen3-32b` via Groq

---

## Architecture

```
User message + game_id
        │
        ▼
┌─────────────────────────┐
│   OrchestratorAgent     │  (qwen/qwen3-32b)
│                         │
│  1. Classify intent     │
│  2. Plan actions        │
│  3. Execute actions     │
│  4. Summarize changes   │
│  5. (optional) Eval     │
└─────────────────────────┘
        │
        ├── CardGenerationAgent     (regenerate/modify cards)
        ├── RuleGenerationAgent     (rewrite/tweak rules)
        ├── ConceptGenerationAgent  (theme/structure changes)
        ├── DirectorAgent           (high-level "improve X")
        ├── PanelEvaluationWorkflow (evaluate game)
        ├── SimulationRunner        (simulate games)
        ├── GameplayAnalysisAgent   (interpret simulation)
        └── Direct state mutation   (simple edits like changing a number)
```

---

## Intent Classification

The orchestrator's first job is to classify the user's message into one or more intents. This determines what actions to take.

| Intent | Example | Actions |
|--------|---------|---------|
| `edit_card` | "Change Fireball damage from 3 to 5" | Mutate card fields directly |
| `edit_rule` | "Make the turn limit 15 instead of 10" | Mutate rules fields directly |
| `add_card` | "Add a healing potion card that restores 3 HP" | CardGenerationAgent (single card) |
| `remove_card` | "Remove the Skip Turn card" | Direct state mutation |
| `add_card_type` | "Add a new card type called Traps" | ConceptAgent (update types) → CardGenerationAgent |
| `regenerate_cards` | "Regenerate all common cards" | CardGenerationAgent (filtered batch) |
| `regenerate_rules` | "Rewrite the win conditions" | RuleGenerationAgent (rewrite_section) |
| `improve_metric` | "Make the game more balanced" | DirectorAgent (target: balance) |
| `improve_general` | "Make the game better" | DirectorAgent (auto-target weakest metric) |
| `evaluate` | "How good is my game?" | PanelEvaluationWorkflow + SimulationRunner |
| `simulate` | "Run a simulation" | SimulationRunner + GameplayAnalysisAgent |
| `query` | "What cards cost more than 3 mana?" | Read state, filter, respond |
| `explain` | "Why is my balance score low?" | Read evaluation, summarize |
| `undo` | "Revert the last change" | Restore previous state snapshot |
| `multi` | "Make Fireball cost 2 and add a Shield card" | Chain: edit_card + add_card |

The orchestrator can produce **multiple intents** from a single message and execute them in order.

---

## Conversation Model

### Message Schema

```python
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime

class ChatAction(BaseModel):
    """A single action the orchestrator decided to take."""
    intent: str                         # e.g. "edit_card", "improve_metric"
    description: str                    # Human-readable: "Changed Fireball damage to 5"
    target: str | None = None           # e.g. card name, rule section, metric name
    details: dict | None = None         # Action-specific payload

class ChatResponse(BaseModel):
    message: str                        # Natural language response to the user
    actions_taken: list[ChatAction]     # What was done
    state_changed: bool                 # Whether the game state was modified
    evaluation_ran: bool                # Whether eval+sim ran after changes
    evaluation_summary: str | None      # Brief eval results if ran
    score_before: float | None
    score_after: float | None
```

### Conversation History

Stored per game in a new DB column `chat_history TEXT` (JSON array of `ChatMessage`).

The orchestrator receives a **sliding window** of the last N messages (e.g. 20) as context, plus a compressed summary of older messages if the conversation is long.

This enables multi-turn:
- "Make the warrior stronger" → edits card
- "Even stronger" → knows "stronger" refers to the warrior card from context
- "Undo that" → knows what "that" is

---

## Orchestrator Agent

### Design

The `OrchestratorAgent` is a new agent in `deck_crafter/agents/orchestrator_agent.py`. It uses a **single LLM call** to classify intent + plan actions, then executes them programmatically.

```python
class OrchestratorAgent:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    def process_message(
        self,
        message: str,
        game_state: CardGameState,
        chat_history: list[ChatMessage],
        run_eval: bool = False,
    ) -> ChatResponse:
        # 1. Build context (state summary + chat history + available actions)
        # 2. LLM call → ActionPlan (list of planned actions with params)
        # 3. Execute each action against game state
        # 4. Snapshot state before changes (for undo)
        # 5. Optionally run simulation + panel evaluation
        # 6. Generate natural language summary of what was done
        ...
```

### Action Plan (LLM output)

```python
class PlannedAction(BaseModel):
    intent: str
    reasoning: str                     # Why this action
    params: dict                       # Intent-specific parameters

class ActionPlan(BaseModel):
    understanding: str                 # "The user wants to..."
    actions: list[PlannedAction]
    needs_clarification: bool          # If the request is ambiguous
    clarification_question: str | None # What to ask the user
```

When `needs_clarification=True`, the orchestrator asks instead of acting. This avoids destructive misinterpretations.

### State Summary for Context

Instead of passing the entire game state to the LLM (too large), build a concise summary:

```
Game: "Guerra de los Clones" (Star Wars, 2 players, Spanish)
Score: 6.98 (playability: 7.5, balance: 6.5, clarity: 7.0, theme: 8.5, innovation: 6.5)
Cards: 20 unique (5 Attack, 5 Defense, 5 Utility, 5 Special)
  - Fireball (Attack, cost: 3 Mana, effect: 5 damage)
  - Healing Potion (Utility, cost: 2 Mana, effect: heal 3)
  - ... (truncated, full list available)
Rules: resource=Mana (1/turn), turn_limit=20, win=reduce opponent HP to 0
Last simulation: 100% completion, 53/47 win balance, avg 14 turns
Issues: "Modulo de Salto" flagged as potentially OP
```

This keeps the orchestrator prompt small while giving it enough to make decisions.

### Action Executors

Each intent maps to an executor function:

| Intent | Executor | Implementation |
|--------|----------|----------------|
| `edit_card` | `_execute_edit_card(state, params)` | Find card by name, mutate fields directly |
| `edit_rule` | `_execute_edit_rule(state, params)` | Mutate rules fields directly |
| `add_card` | `_execute_add_card(state, params)` | LLM generates card fitting the description, append to state |
| `remove_card` | `_execute_remove_card(state, params)` | Remove card by name from state |
| `add_card_type` | `_execute_add_card_type(state, params)` | Update concept.card_types, optionally generate cards for it |
| `regenerate_cards` | `_execute_regenerate_cards(state, params)` | Filter cards, regenerate via CardGenerationAgent |
| `regenerate_rules` | `_execute_regenerate_rules(state, params)` | RuleGenerationAgent with action=rewrite_section/overhaul |
| `improve_metric` | `_execute_improve_metric(state, params)` | Full Director → refine → sim → eval cycle |
| `improve_general` | `_execute_improve_general(state, params)` | Director targets weakest weighted metric |
| `evaluate` | `_execute_evaluate(state, params)` | PanelEvaluationWorkflow + optional simulation |
| `simulate` | `_execute_simulate(state, params)` | SimulationRunner + GameplayAnalysisAgent |
| `query` | `_execute_query(state, params)` | Read state, LLM formats answer |
| `explain` | `_execute_explain(state, params)` | Read evaluation/sim data, LLM explains |
| `undo` | `_execute_undo(state, params)` | Restore previous snapshot |

---

## Undo System

Before any state-modifying action, snapshot the current state:

```python
# In OrchestratorAgent.process_message():
if any(action.modifies_state for action in plan.actions):
    snapshot = state.model_dump_json()
    # Store in a stack (new DB column or in-memory per session)
```

`undo` pops the last snapshot and restores it. Support multiple undo levels (stack of up to 10 snapshots).

New field on `CardGameState`:
```python
state_snapshots: list[str] | None = None  # JSON snapshots for undo (not persisted to DB, session-only)
```

Or a separate DB table `game_snapshots(game_id, snapshot_index, state_json, description, created_at)` for persistence.

---

## API Endpoint

### `POST /api/v1/games/{game_id}/chat`

```python
class ChatRequest(BaseModel):
    message: str                        # User's natural language message
    run_evaluation: bool = False        # Whether to run sim+eval after changes
    num_simulation_games: int = 30      # Simulation config if eval runs

class ChatEndpointResponse(BaseModel):
    response: ChatResponse              # The orchestrator's response
    game_state: CardGameState           # Updated game state (or unchanged)
    chat_history: list[ChatMessage]     # Full conversation so far
```

### `GET /api/v1/games/{game_id}/chat/history`

Returns the conversation history for a game.

---

## Streamlit UI

New page/tab: **"Edit Game"** (or integrated into existing game view).

```
┌──────────────────────────────────────────┐
│  Game: Guerra de los Clones (Score: 6.98)│
├──────────────────────────────────────────┤
│                                          │
│  [Chat messages scroll area]             │
│                                          │
│  🤖 Game created! Score: 5.56            │
│  👤 Make the game more balanced          │
│  🤖 I adjusted card costs and modified   │
│     the win condition. Score: 6.64       │
│  👤 Change Fireball damage to 7          │
│  🤖 Done! Fireball now deals 7 damage.   │
│                                          │
├──────────────────────────────────────────┤
│  [Message input]              [Send]     │
│  ☑ Run evaluation after changes          │
└──────────────────────────────────────────┘
```

### UI Components

- **Chat panel**: Scrollable message history with role-based styling
- **State panel** (sidebar): Live game state summary (cards, rules, scores)
- **Eval toggle**: Checkbox to run simulation + panel evaluation after each change
- **Undo button**: Reverts last change
- **Action log**: Expandable section showing what actions were taken per message

---

## Database Changes

`games` table — new column:
```sql
ALTER TABLE games ADD COLUMN chat_history TEXT;  -- JSON array of ChatMessage
```

Optional separate table for undo snapshots:
```sql
CREATE TABLE game_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    snapshot_index INTEGER NOT NULL,
    state_json TEXT NOT NULL,
    description TEXT,  -- "Before: edit_card Fireball"
    created_at TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);
```

---

## Files to Create/Modify

### New files
| File | Description |
|------|-------------|
| `deck_crafter/agents/orchestrator_agent.py` | OrchestratorAgent + action executors |
| `deck_crafter/models/chat.py` | ChatMessage, ChatRequest, ChatResponse, ActionPlan models |
| `streamlit_chat.py` (or new page in existing app) | Streamlit chat UI |

### Modified files
| File | Change |
|------|--------|
| `deck_crafter/api/routes/game.py` | Add `POST /{game_id}/chat` and `GET /{game_id}/chat/history` endpoints |
| `deck_crafter/models/state.py` | Add `chat_history` field to CardGameState |
| `deck_crafter/database.py` | Add `chat_history` column, migration, snapshot table |
| `streamlit_app.py` | Add navigation to chat editing page |

---

## Implementation Order

1. **Models**: `chat.py` — ChatMessage, ChatResponse, ActionPlan, PlannedAction
2. **OrchestratorAgent**: Core agent with intent classification + action planning LLM call
3. **Action executors**: One by one, starting with simple ones:
   - `edit_card`, `edit_rule`, `remove_card` (direct mutations)
   - `query`, `explain` (read-only)
   - `undo` (snapshot restore)
   - `add_card`, `regenerate_cards`, `regenerate_rules` (use existing agents)
   - `add_card_type` (concept + card generation)
   - `improve_metric`, `improve_general` (Director agent integration)
   - `evaluate`, `simulate` (run pipelines)
4. **API endpoint**: `POST /{game_id}/chat`
5. **Database**: chat_history column + snapshot table
6. **Streamlit UI**: Chat interface
7. **Multi-turn context**: Sliding window + summary compression

---

## Orchestrator Prompt Design

The orchestrator prompt should include:

1. **System role**: "You are the game editing assistant for Deck-Crafter..."
2. **Available actions**: List of intents with descriptions and required params
3. **Game state summary**: Compact representation (not full JSON)
4. **Conversation history**: Last N messages
5. **Instruction**: "Analyze the user's message. Output an ActionPlan."

The prompt must be clear about:
- When to ask for clarification vs act (ambiguous = ask)
- When multiple actions are needed (decompose complex requests)
- What params each action needs (card name, field, value, etc.)
- That `needs_clarification` should be used conservatively (prefer acting when intent is clear)

---

## Evaluation Toggle Behavior

When `run_evaluation=True` on a request that modifies state:

1. Execute all planned actions
2. Run `normalize_card_resources()` (fix resource mismatches)
3. Run simulation (30 games)
4. Run `GameplayAnalysisAgent` on simulation results
5. Run `PanelEvaluationWorkflow` (3-model panel)
6. Include score delta in response: `score_before` → `score_after`

When `run_evaluation=False`:
- Just execute actions and respond
- No simulation, no evaluation
- Fast (~2-5 seconds per message)

---

## Edge Cases

- **Card not found**: "Change Firebolt damage to 5" but card is "Fireball" → orchestrator should fuzzy match or ask
- **Conflicting actions**: "Make all cards cost 1 and improve balance" → plan sequentially, eval after both
- **Empty game**: Chat on a game with no cards/rules → limit available actions, guide user
- **Language mismatch**: User chats in English but game is in Spanish → orchestrator responds in game language (or user's language?)
- **Destructive requests**: "Delete all cards" → orchestrator should confirm before executing

---

## Verification

1. Unit tests for each action executor
2. Integration test: create game → chat "make fireball stronger" → verify card changed
3. Integration test: chat "how good is my game?" → verify evaluation runs
4. Integration test: multi-turn "change X" → "undo" → verify state restored
5. E2E: Streamlit UI manual testing with a real game
6. Load test: 20-message conversation to verify context window management
