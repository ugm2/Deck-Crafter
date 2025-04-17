import os

class Config:
    """
    Configuration settings for the card game generator application.
    """

    # Default LLM provider
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # Options: "vertexai", "ollama"

    # VertexAI configuration
    VERTEXAI_MODEL_NAME = os.getenv("VERTEXAI_MODEL_NAME", "gemini-1.5-pro-002")
    VERTEXAI_LOCATION = os.getenv("VERTEXAI_LOCATION", "us-east1")

    # Ollama configuration
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.2") 
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Shared LLM configuration
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", VERTEXAI_MODEL_NAME if LLM_PROVIDER == "vertexai" else OLLAMA_MODEL_NAME)
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
    LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "8192"))
    LLM_LOCATION = VERTEXAI_LOCATION  # Only used by VertexAI

    # Logging configuration
    LOG_FILE_PATH = "output.log"

    OUTPUT_FILE_PATH = "output.json"
