import os

class Config:
    """
    Configuration settings for the card game generator application.
    """

    # Default LLM provider
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # Options: "ollama"

    # Ollama configuration
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2") 
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Shared LLM configuration
    LLM_MODEL = os.getenv("LLM_MODEL", OLLAMA_MODEL)
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
    LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "8192"))

    # Logging configuration
    LOG_FILE_PATH = "output.log"
    OUTPUT_FILE_PATH = "output.json"
