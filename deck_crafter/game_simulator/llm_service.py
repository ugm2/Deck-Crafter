# llm_service.py
import os
from groq import Groq
import instructor
from pydantic import BaseModel
from typing import Type

class GroqService:
    """Servicio para interactuar con la API de Groq usando instructor."""
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("La variable de entorno GROQ_API_KEY no está configurada.")

        self.client = instructor.from_groq(
            client=Groq(
                api_key=api_key,
                max_retries=5,
                timeout=60.0,
            ),
            mode=instructor.Mode.JSON
        )
        self.model = os.getenv("GROQ_MODEL")

    def generate(self, output_model: Type[BaseModel], prompt: str) -> BaseModel:
        """
        Genera una respuesta estructurada desde Groq basada en un prompt y un modelo Pydantic.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_model=output_model,
                messages=[{"role": "user", "content": prompt}],
                max_retries=5
            )
            return response
        except Exception as e:
            print(f"Error al llamar a la API de Groq: {e}")
            raise

llm_service = GroqService()