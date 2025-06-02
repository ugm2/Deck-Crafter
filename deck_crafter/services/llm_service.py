from abc import ABC, abstractmethod
import logging
from typing import Type
from pydantic import BaseModel
from ollama import Client
from ..utils.config import Config

logging.basicConfig(level=logging.INFO)

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

class LLMProviderRegistry:
    _providers = {'ollama': OllamaService}
    
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
