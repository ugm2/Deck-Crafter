from io import BytesIO
from datetime import datetime
import logging
import time
import threading

from deck_crafter.models.state import CardGameState, GameStatus
from deck_crafter.services.llm_service import LLMService
from deck_crafter.database import save_card_image_sync, get_existing_card_images_sync
from deck_crafter.utils.config import Config

logger = logging.getLogger(__name__)

# Rate limiting constants for Gemini image generation
GEMINI_REQUESTS_PER_MINUTE = 10
GEMINI_REQUEST_DELAY = 60 / GEMINI_REQUESTS_PER_MINUTE  # 6 seconds between requests
GEMINI_TIMEOUT_SECONDS = 120  # 2 minute timeout per image
GEMINI_MAX_RETRIES = 2


class ImageGenerationAgent:
    """Agent responsible for generating images for cards using local FLUX model."""

    # Card image settings (5:7 aspect ratio for standard trading cards)
    IMAGE_WIDTH = 384
    IMAGE_HEIGHT = 538
    NUM_STEPS = 4
    QUANTIZE = 4

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_images(self, state: CardGameState) -> CardGameState:
        if not state.cards:
            return state

        from mflux.models.flux.variants.txt2img.flux import Flux1
        from mflux.models.common.config import ModelConfig

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


class GeminiImageGenerationAgent:
    """Agent for generating images using Gemini's image generation API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.GEMINI_API_KEY

    def _generate_single_image(self, client, card, game_id: str, art_style: str = None) -> bool:
        """Generate a single image with retry logic. Returns True on success."""
        import base64

        style_instruction = art_style or "Digital art style with vibrant colors"

        prompt = f"""Create an illustration: {card.image_description}

ART STYLE: {style_instruction}

IMPORTANT REQUIREMENTS:
- Pure illustration only, NO text, NO borders, NO frames, NO card template
- Do NOT add any card game elements like mana costs, stats, or titles
- Portrait orientation (3:4 aspect ratio)
- Follow the art style consistently
- Clean image without any UI elements or overlays"""

        for attempt in range(GEMINI_MAX_RETRIES):
            try:
                logger.info(f"Generating image for '{card.name}' (attempt {attempt + 1}/{GEMINI_MAX_RETRIES})")

                response = client.models.generate_content(
                    model="gemini-3.1-flash-image-preview",
                    contents=[prompt],
                )

                for part in response.parts:
                    if part.inline_data is not None and part.inline_data.mime_type.startswith("image/"):
                        data = part.inline_data.data
                        if isinstance(data, str):
                            image_data = base64.b64decode(data)
                        else:
                            image_data = bytes(data) if not isinstance(data, bytes) else data
                        save_card_image_sync(game_id, card.name, image_data)
                        logger.info(f"Generated Gemini image for card: {card.name}")
                        return True

                logger.warning(f"No image data in response for {card.name}")

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {card.name}: {e}")
                if attempt < GEMINI_MAX_RETRIES - 1:
                    time.sleep(2)  # Brief delay before retry
                continue

        logger.error(f"Failed to generate image for {card.name} after {GEMINI_MAX_RETRIES} attempts")
        return False

    def generate_images(
        self,
        state: CardGameState,
        shutdown_event: threading.Event = None
    ) -> CardGameState:
        if not state.cards:
            return state

        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_SECONDS * 1000),  # milliseconds
        )

        existing_images = get_existing_card_images_sync(state.game_id)
        cards_to_generate = [
            card for card in state.cards
            if card.name not in existing_images and card.image_description
        ]

        if not cards_to_generate:
            state.status = GameStatus.IMAGES_GENERATED
            state.updated_at = datetime.now()
            return state

        # Get art style from game concept
        art_style = None
        if state.concept and hasattr(state.concept, 'art_style'):
            art_style = state.concept.art_style
            logger.info(f"Using art style: {art_style}")

        logger.info(f"Starting Gemini image generation for {len(cards_to_generate)} cards")
        successful = 0
        failed = 0

        for i, card in enumerate(cards_to_generate):
            # Check for shutdown request
            if shutdown_event and shutdown_event.is_set():
                logger.info("Shutdown requested, stopping image generation")
                break

            # Rate limiting - wait between requests
            if i > 0:
                time.sleep(GEMINI_REQUEST_DELAY)

            if self._generate_single_image(client, card, state.game_id, art_style):
                successful += 1
            else:
                failed += 1

        logger.info(f"Image generation complete: {successful} successful, {failed} failed")
        state.status = GameStatus.IMAGES_GENERATED
        state.updated_at = datetime.now()
        return state 