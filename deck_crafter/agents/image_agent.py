from io import BytesIO
from datetime import datetime

from deck_crafter.models.state import CardGameState, GameStatus
from deck_crafter.services.llm_service import LLMService
from deck_crafter.database import save_card_image_sync, get_existing_card_images_sync
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.common.config import ModelConfig


class ImageGenerationAgent:
    """Agent responsible for generating images for cards."""

    # Card image settings (5:7 aspect ratio for standard trading cards)
    IMAGE_WIDTH = 384
    IMAGE_HEIGHT = 538
    NUM_STEPS = 4
    QUANTIZE = 4

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_images(self, state: CardGameState) -> CardGameState:
        """
        Generate images for cards that don't have images yet.

        Args:
            state: The current game state containing cards to generate images for

        Returns:
            Updated game state with generated images
        """
        if not state.cards:
            return state

        # Get cards that already have images
        existing_images = get_existing_card_images_sync(state.game_id)
        cards_to_generate = [
            card for card in state.cards
            if card.name not in existing_images and card.image_description
        ]

        if not cards_to_generate:
            state.status = GameStatus.IMAGES_GENERATED
            state.updated_at = datetime.now()
            return state

        flux = Flux1(
            model_config=ModelConfig.from_name("schnell"),
            quantize=self.QUANTIZE,
        )

        for card in cards_to_generate:
            image = flux.generate_image(
                seed=hash(card.name) % 1000,
                prompt=card.image_description,
                width=self.IMAGE_WIDTH,
                height=self.IMAGE_HEIGHT,
                num_inference_steps=self.NUM_STEPS,
            )

            byte_arr = BytesIO()
            image.image.save(byte_arr, format='PNG')
            image_data = byte_arr.getvalue()
            save_card_image_sync(state.game_id, card.name, image_data)

        state.status = GameStatus.IMAGES_GENERATED
        state.updated_at = datetime.now()

        return state 