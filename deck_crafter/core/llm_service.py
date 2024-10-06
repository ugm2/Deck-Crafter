from abc import ABC, abstractmethod
from langchain_google_vertexai import VertexAI


class LLMService(ABC):
    @abstractmethod
    def call_llm(self, prompt: str) -> str:
        """
        Calls the Language Model with the given prompt and returns the output.

        Args:
        - prompt: The input text for the model to process.

        Returns:
        - The generated response from the model.
        """


class VertexAILLM(LLMService):
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.5,
        max_output_tokens: int = 1024,
        top_p: float = 0.8,
        top_k: int = 40,
    ):
        """
        Initializes the Vertex AI model with customizable parameters.

        Args:
        - model_name: The name of the VertexAI model (e.g., 'text-bison').
        - temperature: Controls the creativity of the output. A higher value leads to more randomness.
        - max_output_tokens: The maximum number of tokens the model should generate.
        - top_p: Controls the diversity via nucleus sampling.
        - top_k: Limits the number of highest-probability vocabulary tokens considered.
        """
        self.llm_model = VertexAI(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            location="us-east1",
            safety_settings={
                "HARM_CATEGORY_UNSPECIFIED": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            },
        )

    def call_llm(self, prompt: str) -> str:
        """
        Calls the Vertex AI model with the given prompt and returns the output.

        Args:
        - prompt: The input text for the model to process.

        Returns:
        - The generated response from the model.
        """
        try:
            response = self.llm_model.invoke(prompt)
            return response
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")
