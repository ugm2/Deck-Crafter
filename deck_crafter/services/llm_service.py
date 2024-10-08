from abc import ABC, abstractmethod
import logging
import time
from langchain_google_vertexai import ChatVertexAI, create_structured_runnable
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from vertexai.preview.generative_models import HarmBlockThreshold, HarmCategory

logging.basicConfig(level=logging.INFO)


class LLMService(ABC):
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


class VertexAILLM(LLMService):
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 2
    MAX_BACKOFF = 30

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.5,
        max_output_tokens: int = 1024,
        top_p: float = 0.8,
        top_k: int = 40,
        location: str = "us-east1",
        safety_settings=None,
    ):
        """
        Initializes the Vertex AI model with customizable parameters.

        Args:
        - model_name: The name of the VertexAI model (e.g., 'text-bison').
        - temperature: Controls the creativity of the output. A higher value leads to more randomness.
        - max_output_tokens: The maximum number of tokens the model should generate.
        - top_p: Controls the diversity via nucleus sampling.
        - top_k: Limits the number of highest-probability vocabulary tokens considered.
        - location: Specifies the region for the Vertex AI instance.
        - safety_settings: Optional safety settings for harmful content detection.
        """
        self.llm_model = ChatVertexAI(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            location=location,
            safety_settings=safety_settings
            or {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
        )

    def call_llm(
        self,
        structured_outputs: list,
        prompt_template: ChatPromptTemplate,
        context: dict,
    ) -> str:
        """
        Calls the Vertex AI model with the structured inputs and prompt.

        Implements retry logic with exponential backoff to handle transient errors.
        In case of an error, appends all accumulated error messages to the prompt asking the LLM to fix the issue.

        Args:
        - structured_outputs: A list of Pydantic models to guide the output.
        - prompt_template: The base prompt template for the LLM to generate the response.
        - context: Additional context passed to the prompt.

        Returns:
        - The generated response from the model.
        """
        attempt = 0
        errors = []

        while attempt < self.MAX_RETRIES:
            try:
                modified_prompt = self._append_errors_to_prompt(prompt_template, errors)
                print("Prompt: \n", modified_prompt)
                print("Context: \n", context)
                response = self._invoke_llm(
                    structured_outputs, modified_prompt, context
                )

                return response

            except Exception as e:
                logging.error(f"LLM call failed on attempt {attempt + 1}: {str(e)}")
                errors.append(str(e))

                self._apply_backoff(attempt)
                attempt += 1

        raise RuntimeError(
            f"Failed to get a response from the LLM after {self.MAX_RETRIES} attempts."
        )

    def _append_errors_to_prompt(
        self, prompt_template: ChatPromptTemplate, errors: list
    ) -> ChatPromptTemplate:
        """
        Appends error messages to the prompt dynamically, escaping curly braces.
        """
        if errors:

            def escape_braces(s):
                return s.replace("{", "{{").replace("}", "}}")

            error_context = "\n".join(
                [
                    f"Error {i + 1}: {escape_braces(error)}"
                    for i, error in enumerate(errors)
                ]
            )

            # Append a human-readable message with errors to the prompt
            error_message = HumanMessagePromptTemplate.from_template(
                f"Previous errors encountered:\n{error_context}\n"
                f"Please correct these issues in your response."
            )

            # Add the error message to the existing prompt's messages
            modified_prompt = ChatPromptTemplate(
                messages=prompt_template.messages + [error_message]
            )
            logging.info(f"Retrying with modified prompt after errors: {errors}")
            return modified_prompt
        else:
            return prompt_template

    def _invoke_llm(
        self,
        structured_outputs: list,
        prompt_template: ChatPromptTemplate,
        context: dict,
    ) -> str:
        """
        Calls the LLM with the provided structured outputs and prompt.

        Args:
        - structured_outputs: The expected outputs from the LLM.
        - prompt_template: The prompt to use (modified or original).
        - context: Context data passed to the LLM.

        Returns:
        - The LLM's response.
        """
        return create_structured_runnable(
            function=structured_outputs, llm=self.llm_model, prompt=prompt_template
        ).invoke(context)

    def _apply_backoff(self, attempt: int) -> None:
        """
        Applies exponential backoff based on the current retry attempt.

        Args:
        - attempt: The current retry attempt number.
        """
        backoff_time = min(self.INITIAL_BACKOFF * (2**attempt), self.MAX_BACKOFF)
        logging.info(f"Retrying after {backoff_time} seconds...")
        time.sleep(backoff_time)
