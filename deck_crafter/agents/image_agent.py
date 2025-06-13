from io import BytesIO
from deck_crafter.models.state import CardGameState, GameStatus
from deck_crafter.services.llm_service import LLMService
from deck_crafter.database import save_card_image_sync
from mflux import Flux1, Config
from mflux.post_processing.generated_image import GeneratedImage
from datetime import datetime

class ImageGenerationAgent:
    """Agent responsible for generating images for cards."""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_images(self, state: CardGameState) -> CardGameState:
        """
        Generate images for all cards in the game state.
        
        Args:
            state: The current game state containing cards to generate images for
            
        Returns:
            Updated game state with generated images
        """
        if not state.cards:
            return state

        flux = Flux1.from_name(
            model_name="schnell",
            quantize=4
        )

        for card in state.cards:
            if not card.image_description:
                continue

            image: GeneratedImage = flux.generate_image(
                seed=hash(card.name) % 1000,  # Use card name as seed for consistency
                prompt=card.image_description,
                config=Config(
                    num_inference_steps=4,
                    height=672,
                    width=512,
                )
            )

            byte_arr = BytesIO()
            image.image.save(byte_arr, format='PNG')
            image_data = byte_arr.getvalue()
            save_card_image_sync(state.game_id, card.name, image_data)

        state.status = GameStatus.IMAGES_GENERATED
        state.updated_at = datetime.now()

        return state 