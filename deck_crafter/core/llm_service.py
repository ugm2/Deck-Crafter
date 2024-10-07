from abc import ABC, abstractmethod
from langchain_google_vertexai import VertexAI, create_structured_runnable, ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from vertexai.preview.generative_models import (
    HarmBlockThreshold,
    HarmCategory,
)


class LLMService(ABC):
    @abstractmethod
    def call_llm(self, prompt: str, records: list = None) -> str:
        """
        Calls the Language Model with the given prompt and returns the output.

        Args:
        - prompt: The input text for the model to process.
        - records: A list of records to pass to the model.

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
        self.llm_model = ChatVertexAI(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            location="us-east1",
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
        )

    def call_llm(
        self, prompt: str, records: list = None, round_trip_check: bool = True
    ) -> str:
        """
        Calls the Vertex AI model with the given prompt and returns the output.

        Args:
        - prompt: The input text for the model to process.
        - records: A list of records to pass to the model.
        - round_trip_check: Whether to perform a round-trip consistency check.

        Returns:
        - The generated response from the model.
        """
        try:
            if records:
                response = create_structured_runnable(records, self.llm_model).invoke(
                    prompt
                )
            else:
                response = self.llm_model.invoke(prompt)

            if round_trip_check and records:
                review_prompt = f"""
                Persona: You are an expert and very critic card game designer.
                Here is the generated response based on the following prompt:
                Original Prompt: {prompt}

                Response:
                {response}
                
                That was your response. Please be a critic and evaluate if that's the right response, 
                if not then generate a new reponse or modify the existing prompt.

                Also, please validate if the response matches the expected structure defined by the given schema.
                The output should have the appropriate format for the given schema.
                """
                print(
                    "\n[Round-Trip Consistency Check] Reviewing the response for consistency..."
                )
                print(f"[Review Prompt Used]:\n{review_prompt}\n")

                # Use create_structured_runnable to validate the response by mapping it to the Pydantic model
                validation_response = create_structured_runnable(
                    records, self.llm_model
                ).invoke(review_prompt)

                # If the validation fails or does not match the schema, raise an error
                if (
                    isinstance(validation_response, str)
                    and "not met" in validation_response.lower()
                ):
                    raise ValueError(
                        f"Validation failed during round-trip check: {validation_response}"
                    )
                response = validation_response
                print("[Round-Trip Consistency Check] Validation passed.")
                print(response)

            return response

        except Exception as e:
            # Retry mechanism in case the first attempt fails
            retry_prompt = f"""
            The previous LLM call failed with the following error:
            {e}
            
            Please try to generate a response again based on the original prompt:
            {prompt}
            """
            try:
                response = self.llm_model.invoke(retry_prompt)
                return response
            except Exception as retry_exception:
                raise RuntimeError(
                    f"LLM call failed after retry: {retry_exception}. Original Prompt: {prompt}\nError during retry: {e}"
                )
