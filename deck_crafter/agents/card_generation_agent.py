from core.llm_service import LLMService


class CardGenerationAgent:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_card(self, state):
        theme = state["theme"]
        prompt = f"Generate a card for a {theme}-themed card game"
        card_description = self.llm_service.call_llm(prompt)

        updated_cards = state.get("cards", [])
        updated_cards.append(card_description)

        return {"cards": updated_cards}
