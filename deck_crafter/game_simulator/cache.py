"""
Caching infrastructure for simulation results.

Uses diskcache to persist simulation reports keyed by hash of rules+cards.
This avoids re-running expensive simulations when the game hasn't changed.
"""

import hashlib
import logging
from pathlib import Path

import diskcache

from deck_crafter.models.rules import Rules
from deck_crafter.models.card import Card
from deck_crafter.game_simulator.models.metrics import SimulationReport

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".simulation_cache")
DEFAULT_TTL = 3600  # 1 hour


class SimulationCache:
    """Disk-based cache for simulation results."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache = diskcache.Cache(str(cache_dir))

    def get_game_hash(self, rules: Rules, cards: list[Card]) -> str:
        """Generate a hash key from rules and cards."""
        # Sort cards by name for consistent ordering
        sorted_cards = sorted(cards, key=lambda c: c.name)
        content = rules.model_dump_json() + "".join(
            c.model_dump_json() for c in sorted_cards
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, rules: Rules, cards: list[Card]) -> SimulationReport | None:
        """Get cached simulation report if available."""
        game_hash = self.get_game_hash(rules, cards)
        cached = self.cache.get(game_hash)
        if cached:
            logger.info(f"Cache HIT for game hash {game_hash}")
            return SimulationReport.model_validate(cached)
        logger.debug(f"Cache MISS for game hash {game_hash}")
        return None

    def set(
        self,
        rules: Rules,
        cards: list[Card],
        report: SimulationReport,
        ttl: int = DEFAULT_TTL,
    ):
        """Cache a simulation report."""
        game_hash = self.get_game_hash(rules, cards)
        self.cache.set(game_hash, report.model_dump(), expire=ttl)
        logger.info(f"Cached simulation report for game hash {game_hash}")

    def clear(self):
        """Clear all cached results."""
        self.cache.clear()
        logger.info("Cleared simulation cache")

    def close(self):
        """Close the cache."""
        self.cache.close()


# Global cache instance
_cache: SimulationCache | None = None


def get_cache() -> SimulationCache:
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        _cache = SimulationCache()
    return _cache


def get_cached_report(rules: Rules, cards: list[Card]) -> SimulationReport | None:
    """Convenience function to get a cached report."""
    return get_cache().get(rules, cards)


def cache_report(
    rules: Rules, cards: list[Card], report: SimulationReport, ttl: int = DEFAULT_TTL
):
    """Convenience function to cache a report."""
    get_cache().set(rules, cards, report, ttl)
