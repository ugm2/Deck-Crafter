card_game_generator/
│
├── card_game/
│   ├── __init__.py
│   ├── agents/                  # All logic related to concept, rules, and card generation
│   │   ├── __init__.py
│   │   ├── concept_agent.py      # ConceptGenerationAgent
│   │   ├── rule_agent.py         # RuleGenerationAgent
│   │   └── card_agent.py         # CardGenerationAgent
│   │
│   ├── services/                # External service interactions (e.g., LLM, databases)
│   │   ├── __init__.py
│   │   ├── llm_service.py        # LLMService and VertexAILLM
│   │
│   ├── models/                  # All data models for the game
│   │   ├── __init__.py
│   │   ├── game_concept.py       # GameConcept model
│   │   ├── card.py               # Card model
│   │   ├── rules.py              # Rules model
│   │   └── state.py              # CardGameState and UserPreferences
│   │
│   ├── workflow/                # State graph and workflow setup
│   │   ├── __init__.py
│   │   ├── game_workflow.py      # Game workflow logic (StateGraph setup)
│   │   └── conditions.py         # Conditional logic, like should_continue function
│   │
│   ├── utils/                   # Utilities and helper functions
│   │   ├── __init__.py
│   │   ├── logger.py             # LoggerWriter and logging-related utilities
│   │   └── config.py             # Configuration settings
│   │
│   └── main.py                  # Entry point to run the application
│
├── tests/
│   ├── __init__.py
│   ├── agents/                  # Unit tests for agents
│   │   ├── test_concept_agent.py
│   │   ├── test_rule_agent.py
│   │   └── test_card_agent.py
│   ├── services/                # Unit tests for services like LLMService
│   │   └── test_llm_service.py
│   ├── models/                  # Unit tests for models (if needed)
│   │   └── test_models.py
│   └── workflow/                # Tests for state graph and workflows
│       └── test_workflow.py
│
├── requirements.txt             # External dependencies (e.g., pydantic)
└── setup.py                     # Setup script for packaging (if needed)
