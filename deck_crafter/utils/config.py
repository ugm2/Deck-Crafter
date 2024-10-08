class Config:
    """
    Configuration settings for the card game generator application.
    """

    # LLM service configuration
    LLM_MODEL_NAME = "gemini-1.5-pro-002"
    LLM_TEMPERATURE = 0.5
    LLM_MAX_OUTPUT_TOKENS = 8192
    LLM_LOCATION = "us-east1"

    # Logging configuration
    LOG_FILE_PATH = "output.log"

    OUTPUT_FILE_PATH = "output.json"
