from abc import ABC, abstractmethod
import logging
import time
import json
from typing import Dict, List, Optional, Any, Union, Type
from langchain_google_vertexai import ChatVertexAI, create_structured_runnable
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate
from vertexai.preview.generative_models import HarmBlockThreshold, HarmCategory
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
try:
    from pydantic import create_model
except ImportError:
    from pydantic.v1 import create_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logging.basicConfig(level=logging.INFO)


class LLMService(ABC):
    """
    Abstract interface for all LLM services.
    Follows the Interface Segregation Principle by keeping the interface minimal.
    """
    @abstractmethod
    def call_llm(
        self,
        structured_outputs: list,
        prompt_template: ChatPromptTemplate,
        context: dict,
    ) -> str:
        """
        Calls the Language Model with the given structured inputs and context.

        Args:
        - structured_outputs: A list of Pydantic models to guide the output.
        - prompt_template: The ChatPromptTemplate that acts as the base prompt for the LLM.
        - context: Contextual information to pass to the prompt.

        Returns:
        - The generated response from the model.
        """


class RetryStrategy(ABC):
    """
    Abstract strategy for retry behavior.
    Follows Single Responsibility Principle by separating retry logic.
    """
    @abstractmethod
    def execute_with_retry(self, operation, *args, **kwargs):
        """Execute an operation with retry logic"""
        pass


class ExponentialBackoffRetry(RetryStrategy):
    """
    Implements exponential backoff retry strategy.
    """
    def __init__(self, max_retries: int = 3, initial_backoff: int = 2, max_backoff: int = 30):
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff

    def execute_with_retry(self, operation, *args, **kwargs):
        """
        Execute operation with exponential backoff retry.
        
        Args:
        - operation: Callable to execute
        - args, kwargs: Arguments to pass to the operation
        
        Returns:
        - Result of the operation if successful
        
        Raises:
        - RuntimeError: If maximum retries exceeded
        """
        attempt = 0
        errors = []

        while attempt < self.max_retries:
            try:
                return operation(*args, errors=errors, **kwargs)
            except Exception as e:
                logging.error(f"Operation failed on attempt {attempt + 1}: {str(e)}")
                errors.append(str(e))
                self._apply_backoff(attempt)
                attempt += 1

        raise RuntimeError(
            f"Failed to execute operation after {self.max_retries} attempts."
        )

    def _apply_backoff(self, attempt: int) -> None:
        """Apply exponential backoff delay"""
        backoff_time = min(self.initial_backoff * (2**attempt), self.max_backoff)
        logging.info(f"Retrying after {backoff_time} seconds...")
        time.sleep(backoff_time)


class PromptEnhancer:
    """
    Responsible for enhancing prompts with error information.
    Follows Single Responsibility Principle.
    """
    @staticmethod
    def append_errors(
        prompt_template: ChatPromptTemplate, errors: list
    ) -> ChatPromptTemplate:
        """
        Appends error messages to the prompt.
        """
        if not errors:
            return prompt_template

        def escape_braces(s):
            return s.replace("{", "{{").replace("}", "}}")

        error_context = "\n".join(
            [f"Error {i + 1}: {escape_braces(error)}" for i, error in enumerate(errors)]
        )

        error_message = HumanMessagePromptTemplate.from_template(
            f"Previous errors encountered:\n{error_context}\n"
            f"Please correct these issues in your response."
        )

        modified_prompt = ChatPromptTemplate(
            messages=prompt_template.messages + [error_message]
        )
        logging.info(f"Enhanced prompt with error information: {errors}")
        return modified_prompt


class ModelInvoker:
    """
    Responsible for invoking the LLM with structured outputs.
    Follows Single Responsibility Principle.
    """
    @staticmethod
    def invoke(
        llm_model,
        structured_outputs: list,
        prompt_template: ChatPromptTemplate,
        context: dict,
        errors: list = None,
    ) -> str:
        """
        Invokes the LLM with structured outputs and enhanced prompt.
        """
        if not llm_model:
            raise ValueError("LLM model not initialized")
            
        # Enhance prompt with errors if any
        enhanced_prompt = PromptEnhancer.append_errors(prompt_template, errors or [])
        
        # Make sure we have valid structured outputs
        if not structured_outputs or len(structured_outputs) == 0:
            raise ValueError("Invalid or empty structured_outputs parameter")
            
        # Pass the actual model class to create_structured_runnable
        # LangChain expects the model class itself, not its schema
        try:
            return create_structured_runnable(
                function=structured_outputs[0],  # Pass the class itself
                llm=llm_model, 
                prompt=enhanced_prompt
            ).invoke(context)
        except Exception as e:
            logging.error(f"Error invoking LLM with structured output: {str(e)}")
            raise


class OllamaInvoker:
    """
    Specialized invoker for Ollama models that don't support 'functions' parameter.
    Uses output parsing instead of structured runnable.
    """
    @staticmethod
    def invoke(
        llm_model,
        structured_outputs: list,
        prompt_template: ChatPromptTemplate,
        context: dict,
        errors: list = None,
    ) -> str:
        """
        Invokes Ollama with output parsing instead of function calling.
        Focuses on clear instructions for nested structures.
        """
        if not llm_model:
            raise ValueError("Ollama model not initialized")
            
        if not structured_outputs or len(structured_outputs) == 0:
            raise ValueError("Invalid or empty structured_outputs parameter")
        
        model_cls = structured_outputs[0]
        
        # Convert the prompt template to properly formatted messages
        formatted_messages = OllamaInvoker._format_initial_messages(prompt_template, context)
        
        # Get schema info
        schema = model_cls.model_json_schema() if hasattr(model_cls, 'model_json_schema') else model_cls.schema()
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})
        
        # Generate the instruction message with specific details for CardType
        instruction_message = OllamaInvoker._create_instruction_message(required_fields, properties)
        
        # Add Pydantic errors if any
        if errors:
            error_msg = "Previous errors:\n" + "\n".join([f"- {err}" for err in errors])
            formatted_messages.append(HumanMessage(content=error_msg))
        
        formatted_messages.append(instruction_message)
        
        # Invoke the model with retry logic
        try:
            # First attempt
            response = llm_model.invoke(formatted_messages)
            content = response.content if hasattr(response, 'content') else str(response)
            result_dict = OllamaInvoker._extract_json(content)
            
            # Validate and return if successful
            if OllamaInvoker._validate_and_parse(model_cls, result_dict):
                return OllamaInvoker._parse_result(model_cls, result_dict)
                
            # If first attempt failed (likely schema issue)
            logging.warning("First attempt failed validation, retrying with clearer instructions.")
            
            # Add a more explicit retry message
            retry_msg = HumanMessage(content=(
                "Your previous response was not valid. It seems you might have provided strings for card types instead of objects, or returned the schema definition.\n\n"
                "Please ensure `card_types` is a LIST OF OBJECTS, each with `name`, `description`, `quantity`, and `unique_cards`.\n\n"
                "Return ONLY the valid JSON instance, not the schema."
            ))
            formatted_messages.append(retry_msg)
            
            # Second attempt
            response = llm_model.invoke(formatted_messages)
            content = response.content if hasattr(response, 'content') else str(response)
            result_dict = OllamaInvoker._extract_json(content)
            
            if OllamaInvoker._validate_and_parse(model_cls, result_dict):
                 return OllamaInvoker._parse_result(model_cls, result_dict)

            # If second attempt also failed
            logging.error("Second attempt failed validation.")
            raise RuntimeError("Failed to get valid output from Ollama after 2 attempts.")
                
        except Exception as e:
            logging.error(f"Error invoking Ollama: {str(e)}")
            # Fallback to simple prompt if initial attempts fail drastically
            try:
                logging.warning("Invoking with simplified prompt as fallback.")
                simple_message = OllamaInvoker._create_simplified_instruction(required_fields, properties)
                response = llm_model.invoke([simple_message])
                content = response.content if hasattr(response, 'content') else str(response)
                result_dict = OllamaInvoker._extract_json(content)
                return OllamaInvoker._parse_result(model_cls, result_dict)
            except Exception as e2:
                raise RuntimeError(f"Failed to get valid output from Ollama: {str(e)} / {str(e2)}")

    @staticmethod
    def _format_initial_messages(prompt_template, context):
        "Formats the initial prompt template messages." 
        formatted_messages = []
        for msg in prompt_template.messages:
            if hasattr(msg, 'prompt') and hasattr(msg.prompt, 'format'):
                formatted_content = msg.prompt.format(**context)
                if isinstance(msg, HumanMessagePromptTemplate):
                    formatted_messages.append(HumanMessage(content=formatted_content))
                elif isinstance(msg, AIMessagePromptTemplate):
                    formatted_messages.append(AIMessage(content=formatted_content))
                elif isinstance(msg, SystemMessagePromptTemplate):
                    formatted_messages.append(SystemMessage(content=formatted_content))
                else:
                    formatted_messages.append(HumanMessage(content=formatted_content))
            elif isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                formatted_messages.append(msg)
            else:
                try:
                    formatted_messages.append(HumanMessage(content=str(msg)))
                except Exception:
                    pass # Skip if conversion fails
        return formatted_messages

    @staticmethod
    def _create_instruction_message(required_fields, properties):
        "Creates the detailed instruction message." 
        field_descriptions = []
        example_fields = []

        for field in required_fields:
            field_info = properties.get(field, {})
            field_desc = field_info.get("description", "")
            field_type = field_info.get("type", "string")
            field_description_text = f"- {field} ({field_type}): {field_desc}"
            
            example_value = f'"Example {field}"' # Default example
            if field_type == "integer":
                example_value = 42
            elif field_type == "array" and "items" in field_info:
                item_schema = field_info["items"]
                # Specific handling for card_types list
                if field == "card_types" and ("$ref" in item_schema or item_schema.get("type") == "object"):
                    # Assuming CardType schema is available/standard
                    card_type_example = {
                        "name": "Example Card Type Name",
                        "description": "Example description of this card type.",
                        "quantity": 10,
                        "unique_cards": 3
                    }
                    # Fix: Pre-calculate the indented JSON string
                    indented_card_type_json = json.dumps(card_type_example, indent=4).replace("\n", "\n    ")
                    example_value = f'[\n    {indented_card_type_json}\n  ]'
                    field_description_text += "\n    (This must be a LIST OF OBJECTS, each with name, description, quantity, unique_cards)"
                else:
                    example_value = '["Item 1", "Item 2"]' # Generic array example
            
            field_descriptions.append(field_description_text)
            example_fields.append(f'  "{field}": {example_value}')
            
        example_json = "{\n" + ",\n".join(example_fields) + "\n}"

        return HumanMessage(content=(
            f"Create a valid JSON object with these fields:\n\n"
            f"{chr(10).join(field_descriptions)}\n\n"
            f"IMPORTANT: Create a real instance with actual values, not a schema definition.\n"
            f"DO NOT return properties or definitions, just a valid JSON object.\n\n"
            f"Example format (replace with your own values):\n{example_json}"
        ))

    @staticmethod
    def _create_simplified_instruction(required_fields, properties):
         "Creates a simplified instruction message as fallback." 
         example_fields = []
         for field in required_fields:
             field_type = properties.get(field, {}).get("type", "string")
             if field_type == "string":
                 example_fields.append(f'  "{field}": "Example value for {field}"')
             elif field_type == "integer":
                 example_fields.append(f'  "{field}": 42')
             elif field_type == "array":
                  # Simplified example for array
                  example_fields.append(f'  "{field}": [ {{"name":"CardType1", "description":"Desc1", "quantity":5, "unique_cards":2}} ]') 
             else:
                 example_fields.append(f'  "{field}": "Example value"')
         example_json = "{\n" + ",\n".join(example_fields) + "\n}"

         return HumanMessage(content=(
             f"Return ONLY a JSON object with these fields: {', '.join(required_fields)}.\n"
             f"Format: {example_json}"
         ))

    @staticmethod
    def _validate_and_parse(model_cls, result_dict):
        "Validates the dictionary against the model, returns True if valid." 
        try:
            if hasattr(model_cls, 'model_validate'):
                model_cls.model_validate(result_dict)
            elif hasattr(model_cls, 'parse_obj'):
                model_cls.parse_obj(result_dict)
            else:
                model_cls(**result_dict)
            return True
        except Exception as validation_error:
            logging.warning(f"Pydantic validation failed: {validation_error}")
            # Check if it looks like a schema was returned
            if "$defs" in result_dict or ("properties" in result_dict and len(result_dict) < 5): # Heuristic
                 logging.warning("Detected likely schema definition in response.")
            return False

    @staticmethod
    def _parse_result(model_cls, result_dict):
        "Parses the validated dictionary into the model instance." 
        if hasattr(model_cls, 'model_validate'):
            return model_cls.model_validate(result_dict)
        elif hasattr(model_cls, 'parse_obj'):
            return model_cls.parse_obj(result_dict)
        else:
            return model_cls(**result_dict)

    @staticmethod
    def _extract_json(text):
        """Extract JSON from text, handling various formats"""
        import re
        
        # Try to find JSON in code blocks first
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Look for anything that resembles a JSON object
            json_match = re.search(r'(\{[\s\S]*\})', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Just use the whole text as a last resort
                json_str = text.strip()
        
        # Try to parse the JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            logging.warning(f"Failed to parse JSON directly, attempting to fix: {json_str[:100]}...")
            
            # Replace single quotes with double quotes
            fixed_json = re.sub(r"'([^']*)'", r'"\1"', json_str)
            
            # Fix trailing commas in objects and arrays
            fixed_json = re.sub(r',\s*}', '}', fixed_json)
            fixed_json = re.sub(r',\s*\]', ']', fixed_json)
            
            # Add quotes around unquoted keys
            fixed_json = re.sub(r'(\s*?)(\w+)(\s*?):', r'\1"\2"\3:', fixed_json)
            
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                logging.error(f"Could not parse JSON even after fixes: {json_str}")
                raise ValueError(f"Invalid JSON response: {json_str[:500]}")


class BaseLLMService(LLMService):
    """
    Base implementation of LLM service with common behavior.
    """
    def __init__(self, retry_strategy: RetryStrategy = None):
        """
        Initialize with optional retry strategy.
        
        Args:
        - retry_strategy: Strategy for retrying operations
        """
        self.llm_model = None
        self.retry_strategy = retry_strategy or ExponentialBackoffRetry()
    
    def call_llm(
        self,
        structured_outputs: list,
        prompt_template: ChatPromptTemplate,
        context: dict,
    ) -> str:
        """
        Calls the LLM with structured inputs and robust error handling.
        """
        if not self.llm_model:
            raise ValueError("LLM model not initialized")
            
        # Use retry strategy to execute the model invocation
        return self.retry_strategy.execute_with_retry(
            ModelInvoker.invoke,
            self.llm_model,
            structured_outputs,
            prompt_template,
            context
        )


class VertexAILLM(BaseLLMService):
    """Implementation for Vertex AI LLM."""
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.5,
        max_output_tokens: int = 1024,
        top_p: float = 0.8,
        top_k: int = 40,
        location: str = "us-east1",
        safety_settings=None,
        retry_strategy: RetryStrategy = None,
    ):
        """
        Initialize Vertex AI LLM with model-specific parameters.
        """
        super().__init__(retry_strategy)
        
        # Initialize model
        self.llm_model = ChatVertexAI(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            location=location,
            safety_settings=safety_settings
            or {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
        )


class OllamaLLM(BaseLLMService):
    """Implementation for Ollama LLM."""
    
    def __init__(
        self,
        model_name: str = "llama3.2",
        temperature: float = 0.5,
        max_tokens: int = 2048,
        top_p: float = 0.8,
        top_k: int = 40,
        base_url: str = "http://localhost:11434",
        retry_strategy: RetryStrategy = None,
        **kwargs
    ):
        """
        Initialize Ollama LLM with model-specific parameters.
        """
        super().__init__(retry_strategy)
        
        # Initialize model
        self.llm_model = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=max_tokens,
            top_p=top_p,
            top_k=top_k,
            base_url=base_url,
            **kwargs
        )
    
    def call_llm(
        self,
        structured_outputs: list,
        prompt_template: ChatPromptTemplate,
        context: dict,
    ) -> str:
        """
        Calls the LLM with structured inputs and robust error handling.
        Uses the Ollama-specific invoker.
        """
        if not self.llm_model:
            raise ValueError("LLM model not initialized")
            
        # Use retry strategy with the OllamaInvoker
        return self.retry_strategy.execute_with_retry(
            OllamaInvoker.invoke,
            self.llm_model,
            structured_outputs,
            prompt_template,
            context
        )


class LLMServiceRegistry:
    """
    Registry for LLM service providers.
    Follows Open/Closed Principle by allowing extension without modification.
    """
    _providers = {}
    
    @classmethod
    def register(cls, provider_name: str, provider_class: Type[LLMService]):
        """Register a new LLM service provider"""
        cls._providers[provider_name.lower()] = provider_class
        
    @classmethod
    def get_provider(cls, provider_name: str) -> Type[LLMService]:
        """Get provider class by name"""
        provider = cls._providers.get(provider_name.lower())
        if not provider:
            raise ValueError(f"Unsupported LLM provider: {provider_name}. "
                            f"Available providers: {list(cls._providers.keys())}")
        return provider


# Register available providers
LLMServiceRegistry.register("vertexai", VertexAILLM)
LLMServiceRegistry.register("ollama", OllamaLLM)


def create_llm_service(provider: str = "ollama", retry_strategy: RetryStrategy = None, **kwargs) -> LLMService:
    """
    Factory function to create an appropriate LLM service.
    Follows Dependency Inversion Principle by depending on abstractions.
    
    Args:
    - provider: The LLM provider to use (registered in LLMServiceRegistry)
    - retry_strategy: Optional custom retry strategy
    - kwargs: Configuration parameters for the LLM service
    
    Returns:
    - An instance of LLMService
    """
    provider_class = LLMServiceRegistry.get_provider(provider)
    
    if retry_strategy:
        kwargs["retry_strategy"] = retry_strategy
        
    return provider_class(**kwargs)
