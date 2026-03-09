import os

class Config:
    """
    Configuration settings for the card game generator application.
    """

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Provider configs (priority order: sambanova > groq > openrouter > deepseek)
    SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
    SAMBANOVA_MODEL = os.getenv("SAMBANOVA_MODEL", "DeepSeek-V3-0324")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")

    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    LLM_MODEL = GROQ_MODEL if LLM_PROVIDER == "groq" else OLLAMA_MODEL
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
    LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "8192"))

    LOG_FILE_PATH = "output.log"
    OUTPUT_FILE_PATH = "output.json"
