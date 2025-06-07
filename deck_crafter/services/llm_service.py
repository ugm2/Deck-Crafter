from abc import ABC, abstractmethod
import logging
from typing import Type
from pydantic import BaseModel
from ollama import Client
from groq import Groq
import instructor
from ..utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService(ABC):
    """Base class for LLM services."""
    
    @abstractmethod
    def generate(
        self,
        output_model: Type[BaseModel],
        prompt: str,
        **context
    ) -> BaseModel:
        """Generate structured response using specified model"""
        pass

class OllamaService(LLMService):
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434", **kwargs):
        self.model_name = model
        self.client = Client(host=base_url)
        self.config = {
            'options': {
                'temperature': Config.LLM_TEMPERATURE,
                'num_predict': Config.LLM_MAX_OUTPUT_TOKENS,
                'top_p': 0.8,
                'top_k': 40,
            },
            **kwargs
        }
    
    def generate(
        self,
        output_model: Type[BaseModel],
        prompt: str,
        **context
    ) -> BaseModel:
        schema = output_model.model_json_schema()
        formatted_prompt = prompt.format(**context)
        
        response = self.client.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': formatted_prompt}],
            format=schema,
            options=self.config['options']
        )
        
        return output_model.model_validate_json(response.message.content)

class GroqService(LLMService):
    def __init__(self, model: str = "llama3-8b-8192", api_key: str = None, **kwargs):
        """
        Initialize the Groq service.
        
        Args:
            model: The model to use (default: llama3-8b-8192)
            api_key: The Groq API key. If not provided, will try to get from Config.GROQ_API_KEY
            **kwargs: Additional configuration options
        """
        self.model_name = model
        groq_client = Groq(
            api_key=api_key or Config.GROQ_API_KEY,
            max_retries=5,
            timeout=60.0,
        )
        self.client = instructor.from_groq(
            groq_client,
            mode=instructor.Mode.JSON
        )
        self.config = {
            'temperature': Config.LLM_TEMPERATURE,
            'max_tokens': Config.LLM_MAX_OUTPUT_TOKENS,
            'top_p': 0.8,
            **kwargs
        }
    
    def generate(
        self,
        output_model: Type[BaseModel],
        prompt: str,
        **context
    ) -> BaseModel:
        formatted_prompt = prompt.format(**context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                response_model=output_model,
                messages=[{'role': 'user', 'content': formatted_prompt}],
                **self.config
            )
            return response
        except Exception as e:
            logger.error(f"Error in Groq API call: {str(e)}")
            raise

class LLMProviderRegistry:
    _providers = {
        'ollama': OllamaService,
        'groq': GroqService
    }
    
    @classmethod
    def register_provider(cls, name: str, provider: Type[LLMService]):
        cls._providers[name.lower()] = provider
    
    @classmethod
    def create_provider(cls, name: str = "ollama", **kwargs) -> LLMService:
        provider = cls._providers.get(name.lower())
        if not provider:
            raise ValueError(f"Unsupported provider: {name}. Options: {list(cls._providers.keys())}")
        return provider(**kwargs)

def create_llm_service(provider: str = "ollama", **kwargs) -> LLMService:
    """
    Factory function to create an appropriate LLM service.
    Follows Dependency Inversion Principle by depending on abstractions.
    
    Args:
    - provider: The LLM provider to use (registered in LLMServiceRegistry)
    - kwargs: Configuration parameters for the LLM service
    
    Returns:
    - An instance of LLMService
    """
    return LLMProviderRegistry.create_provider(provider, **kwargs)
