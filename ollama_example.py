from typing import Literal
from pydantic import BaseModel, ValidationError
from ollama import chat

class Receta(BaseModel):
    nombre: str
    ingredientes: list[str]
    tiempo_preparacion: int
    dificultad: Literal['Fácil', 'Media', 'Difícil']

try:
    response = chat(
        model='llama3.2',
        messages=[{
            'role': 'user', 
            'content': 'Dame una receta de gazpacho andaluz'
        }],
        format=Receta.model_json_schema()
    )
    receta = Receta.model_validate_json(response.message.content)
    print(receta)
except ValidationError as e:
    print(f"Error de validación: {e}")
