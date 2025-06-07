import os

class Config:
    """
    Configuration settings for the card game generator application.
    """

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # Options: "ollama", "groq"

    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2") 
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    LLM_MODEL = GROQ_MODEL if LLM_PROVIDER == "groq" else OLLAMA_MODEL
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
    LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "8192"))

    LOG_FILE_PATH = "output.log"
    OUTPUT_FILE_PATH = "output.json"
