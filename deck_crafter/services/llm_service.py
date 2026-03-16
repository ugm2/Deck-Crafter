from abc import ABC, abstractmethod
import logging
from typing import Type, List
from pydantic import BaseModel
from ollama import Client
from groq import Groq
from openai import OpenAI
import instructor
from ..utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Initializing LLM Service with provider: '{Config.LLM_PROVIDER}' and model: '{Config.LLM_MODEL}'")


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
                'temperature': Config.LLM_TEMPERATURE
            },
            **kwargs
        }

    def _remove_patterns_from_schema(self, schema: dict) -> dict:
        if isinstance(schema, dict):
            for key in list(schema.keys()):
                if key == 'pattern':
                    del schema[key]
                else:
                    self._remove_patterns_from_schema(schema[key])
        elif isinstance(schema, list):
            for item in schema:
                self._remove_patterns_from_schema(item)
        return schema

    def generate(
        self,
        output_model: Type[BaseModel],
        prompt: str,
        **context
    ) -> BaseModel:
        schema_with_patterns = output_model.model_json_schema()
        safe_schema = self._remove_patterns_from_schema(dict(schema_with_patterns))
        formatted_prompt = prompt.format(**context)

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': formatted_prompt}],
                format=safe_schema,
                options=self.config['options']
            )
            response_content = response['message']['content']
            return output_model.model_validate_json(response_content)
        except Exception as e:
            logger.error(f"Error during Ollama call: {e}")
            return None


class GroqService(LLMService):
    def __init__(self, model: str = "llama3-8b-8192", api_key: str = None, **kwargs):
        self.model_name = model
        groq_client = Groq(
            api_key=api_key or Config.GROQ_API_KEY,
            max_retries=3,
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
                max_retries=2,
                **self.config
            )
            return response
        except Exception as e:
            # Handle case where model returns array instead of object
            error_str = str(e)
            if "validation error" in error_str.lower() or "list" in error_str.lower():
                logger.warning(f"Groq validation error, likely array response: {e}")
                # Retry with stronger single-object instruction
                retry_prompt = formatted_prompt + "\n\nCRITICAL: Output a SINGLE JSON object, NOT an array. Do not wrap in []."
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    response_model=output_model,
                    messages=[{'role': 'user', 'content': retry_prompt}],
                    max_retries=1,
                    **self.config
                )
                return response
            raise


class OpenAICompatibleService(LLMService):
    """Generic OpenAI-compatible service for Sambanova, OpenRouter, DeepSeek, etc."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        extra_headers: dict = None,
        **kwargs
    ):
        self.model_name = model
        self.base_url = base_url
        openai_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60.0,
            max_retries=2,
        )
        self.client = instructor.from_openai(
            openai_client,
            mode=instructor.Mode.JSON
        )
        self.extra_headers = extra_headers or {}
        self.config = {
            'temperature': Config.LLM_TEMPERATURE,
            'max_tokens': Config.LLM_MAX_OUTPUT_TOKENS,
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
                max_retries=2,
                **self.config
            )
            return response
        except Exception as e:
            # Handle case where model returns array instead of object
            error_str = str(e)
            if "validation error" in error_str.lower() or "list" in error_str.lower():
                logger.warning(f"OpenAI-compatible validation error, likely array response: {e}")
                retry_prompt = formatted_prompt + "\n\nCRITICAL: Output a SINGLE JSON object, NOT an array. Do not wrap in []."
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    response_model=output_model,
                    messages=[{'role': 'user', 'content': retry_prompt}],
                    max_retries=1,
                    **self.config
                )
                return response
            raise


class SambanovaService(OpenAICompatibleService):
    def __init__(self, model: str = None, api_key: str = None, **kwargs):
        super().__init__(
            model=model or Config.SAMBANOVA_MODEL,
            api_key=api_key or Config.SAMBANOVA_API_KEY,
            base_url="https://api.sambanova.ai/v1",
            **kwargs
        )


class OpenRouterService(OpenAICompatibleService):
    def __init__(self, model: str = None, api_key: str = None, **kwargs):
        super().__init__(
            model=model or Config.OPENROUTER_MODEL,
            api_key=api_key or Config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            extra_headers={"HTTP-Referer": "DeckCrafter", "X-Title": "DeckCrafter"},
            **kwargs
        )


class DeepSeekService(OpenAICompatibleService):
    def __init__(self, model: str = None, api_key: str = None, **kwargs):
        super().__init__(
            model=model or Config.DEEPSEEK_MODEL,
            api_key=api_key or Config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
            **kwargs
        )


class GeminiService(OpenAICompatibleService):
    def __init__(self, model: str = None, api_key: str = None, **kwargs):
        super().__init__(
            model=model or Config.GEMINI_MODEL,
            api_key=api_key or Config.GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            **kwargs
        )


class FallbackLLMService(LLMService):
    """LLM service that tries multiple providers in order until one succeeds."""

    def __init__(self, providers: List[LLMService]):
        self.providers = providers
        self.provider_names = [type(p).__name__ for p in providers]
        logger.info(f"FallbackLLMService initialized with providers: {self.provider_names}")

    def generate(
        self,
        output_model: Type[BaseModel],
        prompt: str,
        **context
    ) -> BaseModel:
        last_error = None

        for i, provider in enumerate(self.providers):
            provider_name = self.provider_names[i]
            try:
                logger.info(f"Trying provider: {provider_name}")
                result = provider.generate(output_model, prompt, **context)
                if result is not None:
                    logger.info(f"Success with provider: {provider_name}")
                    return result
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue

        logger.error(f"All providers failed. Last error: {last_error}")
        raise last_error


class LLMProviderRegistry:
    _providers = {
        'ollama': OllamaService,
        'groq': GroqService,
        'sambanova': SambanovaService,
        'openrouter': OpenRouterService,
        'deepseek': DeepSeekService,
        'gemini': GeminiService,
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
    """Factory function to create an appropriate LLM service."""
    return LLMProviderRegistry.create_provider(provider, **kwargs)


def create_premium_llm_service() -> LLMService:
    """Create premium LLM service using Gemini or Groq.

    Falls back to standard fallback chain if not available.
    """
    if Config.GEMINI_API_KEY:
        try:
            service = GeminiService()
            logger.info(f"Premium LLM service initialized with Gemini {Config.GEMINI_MODEL}")
            return service
        except Exception as e:
            logger.warning(f"Could not initialize Gemini premium: {e}")

    if Config.GROQ_API_KEY:
        try:
            service = GroqService(model=Config.GROQ_PREMIUM_MODEL, max_tokens=16384)
            logger.info(f"Premium LLM service initialized with Groq {Config.GROQ_PREMIUM_MODEL}")
            return service
        except Exception as e:
            logger.warning(f"Could not initialize Groq premium: {e}")

    logger.warning("Premium provider not available, falling back to standard chain")
    return create_fallback_llm_service()


def create_fallback_llm_service() -> LLMService:
    """Create LLM service with automatic fallback through multiple providers.

    Priority order: Gemini > Sambanova > Groq > OpenRouter > DeepSeek > Ollama
    Only providers with configured API keys are included.
    """
    providers = []

    if Config.GEMINI_API_KEY:
        try:
            providers.append(GeminiService())
            logger.info("Added Gemini to fallback chain")
        except Exception as e:
            logger.warning(f"Could not initialize Gemini: {e}")

    if Config.SAMBANOVA_API_KEY:
        try:
            providers.append(SambanovaService())
            logger.info("Added Sambanova to fallback chain")
        except Exception as e:
            logger.warning(f"Could not initialize Sambanova: {e}")

    if Config.GROQ_API_KEY:
        try:
            providers.append(GroqService(model=Config.GROQ_MODEL))
            logger.info("Added Groq to fallback chain")
        except Exception as e:
            logger.warning(f"Could not initialize Groq: {e}")

    if Config.OPENROUTER_API_KEY:
        try:
            providers.append(OpenRouterService())
            logger.info("Added OpenRouter to fallback chain")
        except Exception as e:
            logger.warning(f"Could not initialize OpenRouter: {e}")

    if Config.DEEPSEEK_API_KEY:
        try:
            providers.append(DeepSeekService())
            logger.info("Added DeepSeek to fallback chain")
        except Exception as e:
            logger.warning(f"Could not initialize DeepSeek: {e}")

    # Always try Ollama as last resort (local, no API key needed)
    try:
        providers.append(OllamaService(model=Config.OLLAMA_MODEL))
        logger.info("Added Ollama to fallback chain")
    except Exception as e:
        logger.warning(f"Could not initialize Ollama: {e}")

    if not providers:
        raise ValueError("No LLM providers available. Configure at least one API key.")

    if len(providers) == 1:
        return providers[0]

    return FallbackLLMService(providers)
