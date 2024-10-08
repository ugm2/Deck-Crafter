from typing import Dict, Optional
from pydantic import BaseModel, Field


class GameConcept(BaseModel):
    """
    A structured representation of the key components of a card-only game concept.
    The model defines both required and optional fields necessary for generating
    and designing the core gameplay, including theme, rules, and player dynamics.
    These examples cover a wide range of card games, emphasizing that the games
    are played exclusively with cards, without a board.
    """

    theme: str = Field(
        ...,
        description=(
            "The central theme of the game (required). "
            "Examples: 'Fantasy: Players control wizards battling for supremacy', "
            "'Sci-Fi: Explore distant planets and engage in intergalactic battles', "
            "'Pirates: Sail the seas, plunder ships, and search for treasure', "
            "'Wild West: Become an outlaw or a sheriff and duel your opponents', "
            "'Post-Apocalyptic: Survive in a world devastated by nuclear war', "
            "'Zombie Survival: Fight off hordes of zombies while scavenging for supplies', "
            "'Mythology: Command gods and heroes in epic battles', "
            "'Cyberpunk: Hack into systems, fight corporate enemies, and survive in a dystopian future'."
        ),
    )
    title: str = Field(
        ...,
        description=(
            "A catchy, thematic title for the game (required). "
            "Examples: 'Wizards Duel', 'Heroes of Avalon', 'Mystic Forces', "
            "'Evil Villains', 'Dungeon Crawl', 'Galactic Warfare', 'Pirates of the Abyss', "
            "'Shadows of the Cybernet', 'Legendary Heist', 'Cursed Kingdoms'."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "A brief but clear description of the game (required). "
            "Examples: 'A fast-paced card game where players summon creatures and cast spells to defeat their enemies', "
            "'A party game where players take turns eliminating characters from a hidden deck of assassins', "
            "'A cooperative deck-building game where players work together to fend off waves of monsters', "
            "'A strategic bluffing game where players try to gather resources while sabotaging their opponents', "
            "'A card-based survival game where players scavenge for items and fend off zombies'."
        ),
    )
    language: str = Field(
        ...,
        description=(
            "The primary language used in the game (required). "
            "Examples: 'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Russian', 'Chinese', 'Japanese'."
        ),
    )
    game_style: str = Field(
        ...,
        description=(
            "The core gameplay style, e.g., competitive or cooperative (required). "
            "Examples: 'Competitive', 'Cooperative', 'Bluffing', 'Party game', 'Deck-building'."
        ),
    )
    game_duration: str = Field(
        ...,
        description=(
            "The recommended duration of the game (required). "
            "Examples: '15-30 minutes', '30-60 minutes', '1-2 hours'."
        ),
    )
    number_of_players: str = Field(
        ...,
        description=(
            "The recommended number of players (required). "
            "Examples: '2-4 players', '3-6 players', '4-12 players (party game format)', "
            "'1-5 players (solo mode included)', '2-8 players'."
        ),
    )
    number_of_unique_cards: int = Field(
        ...,
        description=(
            "Total number of unique cards needed for the game (required). "
            "Examples: 10, 15, 20, 30, 40."
        ),
    )
    card_distribution: Dict[str, int] = Field(
        ...,
        description=(
            "A dictionary where each unique card type is a key, and the value is the quantity "
            "of that specific card in the deck. This directly reflects the number of unique cards. "
            "Examples: {'Exploding Kitten': 4, 'Defuse': 6, 'Nope': 5, 'Attack': 4, 'Skip': 4, "
            "'Favor': 4, 'Shuffle': 4, 'See the Future': 5, 'Hairy Potato Cat': 4, 'TacoCat': 4, "
            "'Cattermelon': 4}. Each card type should be reflected based on gameplay needs. "
            "Examples: {'Fireball': 3, 'Shield of Protection': 4, 'Summon Dragon': 2, 'Healing Potion': 5, "
            "'Mana Surge': 6, 'Lightning Strike': 3, 'Teleport': 2, 'Arcane Blast': 4, 'Sword of Valor': 3}, "
            "{'Space Travel': 4, 'Alien Encounter': 5, 'Resource Mining': 6, 'Black Hole': 2, 'Galactic Trade': 4, "
            "'Planet Discovery': 3, 'Space Pirates': 4, 'Starship Upgrade': 5, 'Wormhole': 3, 'Asteroid Field': 2}."
            "Each card type should be reflected based on gameplay needs."
        ),
    )
    number_of_total_cards: int = Field(
        ...,
        description=(
            "Total number of cards in the deck (required)."
            "Examples: 50, 75, 100, 150, 200."
        ),
    )
    card_actions: Dict[str, str] = Field(
        None,
        description=(
            "Description of each card's action or effect. "
            "Examples: {'Skip': 'End your turn without drawing', "
            "'Attack': 'Force the next player to take two turns'}"
        ),
    )
    target_audience: Optional[str] = Field(
        None,
        description=(
            "Age group or specific audience (optional). "
            "Examples: 'Family-friendly (Ages 8+)', 'Teenagers (Ages 13+)', 'Adults (+18)', "
            "'General audience (Ages 10+)', 'Kids (Ages 6+)'."
        ),
    )
    rule_complexity: Optional[str] = Field(
        None,
        description=(
            "Complexity level of the rules (optional). "
            "Examples: 'Easy to learn, perfect for casual play', "
            "'Medium complexity, suitable for experienced gamers', "
            "'Hard, requiring in-depth strategy and planning', "
            "'Very simple, party-style game mechanics', "
            "'Moderate complexity with multiple layers of strategy'."
        ),
    )
