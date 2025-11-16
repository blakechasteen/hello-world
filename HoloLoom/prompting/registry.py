"""
Strategy Registry with Auto-Discovery

Automatically discovers and registers all strategies from the
promptly_skills/strategies/ directory. Strategies are loaded lazily
and cached for performance.
"""

from typing import Dict, List, Optional, Tuple
import importlib
import sys
from pathlib import Path
import logging

from .strategy import PromptingStrategy, StrategyContext, StrategyCategory

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Auto-discovers and registers all strategies.

    Strategies are discovered from:
    - promptly_skills/strategies/<strategy_name>/strategy.py

    Each strategy module must contain a class that inherits from
    PromptingStrategy and is not the base class itself.
    """

    def __init__(self, strategies_dir: Optional[Path] = None):
        """
        Initialize registry.

        Args:
            strategies_dir: Optional custom strategies directory.
                           Defaults to promptly_skills/strategies/
        """
        self.strategies: Dict[str, PromptingStrategy] = {}
        self._lazy_loaded: Dict[str, Path] = {}  # name -> module path

        # Determine strategies directory
        if strategies_dir is None:
            # Default: promptly_skills/strategies/ relative to this file
            repo_root = Path(__file__).parent.parent.parent
            strategies_dir = repo_root / "promptly_skills" / "strategies"

        self.strategies_dir = strategies_dir
        self._discover_strategies()

    def _discover_strategies(self):
        """
        Auto-discover strategies from strategies directory.

        Scans for directories containing strategy.py and registers
        them for lazy loading.
        """
        if not self.strategies_dir.exists():
            logger.warning(f"Strategies directory not found: {self.strategies_dir}")
            return

        logger.info(f"Discovering strategies in: {self.strategies_dir}")

        for item in self.strategies_dir.iterdir():
            if not item.is_dir():
                continue

            strategy_file = item / "strategy.py"
            if not strategy_file.exists():
                continue

            # Register for lazy loading
            strategy_name = item.name
            self._lazy_loaded[strategy_name] = strategy_file
            logger.debug(f"Discovered strategy: {strategy_name}")

        logger.info(f"Discovered {len(self._lazy_loaded)} strategies")

    def _load_strategy(self, name: str) -> Optional[PromptingStrategy]:
        """
        Lazy-load a strategy by name.

        Args:
            name: Strategy name (directory name)

        Returns:
            Loaded strategy instance or None if failed
        """
        if name not in self._lazy_loaded:
            return None

        strategy_file = self._lazy_loaded[name]

        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(
                f"promptly_skills.strategies.{name}.strategy",
                strategy_file
            )

            if spec is None or spec.loader is None:
                logger.error(f"Failed to load spec for strategy: {name}")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Find strategy class in module
            strategy_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                # Must be a class that inherits PromptingStrategy
                if (isinstance(attr, type) and
                    issubclass(attr, PromptingStrategy) and
                    attr not in (PromptingStrategy,)):

                    strategy_class = attr
                    break

            if strategy_class is None:
                logger.error(f"No PromptingStrategy class found in {strategy_file}")
                return None

            # Instantiate strategy
            strategy = strategy_class()

            # Cache it
            self.strategies[name] = strategy

            logger.info(f"Loaded strategy: {name} ({strategy.category.value})")
            return strategy

        except Exception as e:
            logger.error(f"Error loading strategy {name}: {e}", exc_info=True)
            return None

    def get(self, name: str) -> Optional[PromptingStrategy]:
        """
        Get strategy by name.

        Args:
            name: Strategy name

        Returns:
            Strategy instance or None if not found

        Example:
            >>> registry = StrategyRegistry()
            >>> verify = registry.get('verify')
            >>> print(verify.description)
            "Chain of Verification: Force model to critique its own output"
        """
        # Check cache first
        if name in self.strategies:
            return self.strategies[name]

        # Try lazy loading
        if name in self._lazy_loaded:
            return self._load_strategy(name)

        return None

    def list_all(self) -> List[Dict[str, str]]:
        """
        List all available strategies (loaded + discoverable).

        Returns:
            List of strategy info dicts with keys: name, category, description

        Example:
            >>> registry = StrategyRegistry()
            >>> strategies = registry.list_all()
            >>> for s in strategies:
            ...     print(f"{s['name']}: {s['description']}")
            verify: Chain of Verification...
            challenge: Adversarial Prompting...
        """
        strategies_info = []

        # Include already-loaded strategies
        for strategy in self.strategies.values():
            strategies_info.append({
                'name': strategy.name,
                'category': strategy.category.value,
                'description': strategy.description
            })

        # Include discoverable but not yet loaded
        for name in self._lazy_loaded.keys():
            if name not in self.strategies:
                # Load it to get metadata
                strategy = self._load_strategy(name)
                if strategy:
                    strategies_info.append({
                        'name': strategy.name,
                        'category': strategy.category.value,
                        'description': strategy.description
                    })

        return strategies_info

    def list_by_category(self, category: StrategyCategory) -> List[PromptingStrategy]:
        """
        List all strategies in a category.

        Args:
            category: Category to filter by

        Returns:
            List of strategies in that category

        Example:
            >>> registry = StrategyRegistry()
            >>> self_correction = registry.list_by_category(StrategyCategory.SELF_CORRECTION)
            >>> print([s.name for s in self_correction])
            ['verify', 'challenge']
        """
        # Ensure all strategies are loaded
        for name in list(self._lazy_loaded.keys()):
            if name not in self.strategies:
                self._load_strategy(name)

        return [s for s in self.strategies.values() if s.category == category]

    def suggest(self, context: StrategyContext, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Suggest strategies for a context (auto-detection).

        Args:
            context: Query context
            top_k: Number of suggestions to return

        Returns:
            List of (strategy_name, confidence) tuples, sorted by confidence

        Example:
            >>> registry = StrategyRegistry()
            >>> context = StrategyContext(query="analyze this security vulnerability")
            >>> suggestions = registry.suggest(context, top_k=3)
            >>> print(suggestions)
            [('challenge', 0.92), ('verify', 0.88), ('scaffold', 0.65)]
        """
        # Ensure all strategies are loaded
        for name in list(self._lazy_loaded.keys()):
            if name not in self.strategies:
                self._load_strategy(name)

        # Score all strategies
        scores = []
        for strategy in self.strategies.values():
            confidence = strategy.can_apply(context)
            scores.append((strategy.name, confidence))

        # Sort by confidence (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def register(self, strategy: PromptingStrategy):
        """
        Manually register a strategy (for testing or custom strategies).

        Args:
            strategy: Strategy instance to register

        Example:
            >>> registry = StrategyRegistry()
            >>> custom_strategy = MyCustomStrategy()
            >>> registry.register(custom_strategy)
        """
        self.strategies[strategy.name] = strategy
        logger.info(f"Manually registered strategy: {strategy.name}")

    def __len__(self) -> int:
        """Return number of discovered strategies"""
        return len(self._lazy_loaded) + len(self.strategies)

    def __contains__(self, name: str) -> bool:
        """Check if strategy exists"""
        return name in self.strategies or name in self._lazy_loaded

    def __repr__(self) -> str:
        return f"StrategyRegistry(strategies={len(self.strategies)}, discoverable={len(self._lazy_loaded)})"


# Global registry instance
_global_registry: Optional[StrategyRegistry] = None


def get_registry() -> StrategyRegistry:
    """Get or create global registry"""
    global _global_registry
    if _global_registry is None:
        _global_registry = StrategyRegistry()
    return _global_registry


def get_strategy(name: str) -> Optional[PromptingStrategy]:
    """
    Get strategy by name from global registry.

    Args:
        name: Strategy name

    Returns:
        Strategy instance or None

    Example:
        >>> verify = get_strategy('verify')
        >>> print(verify.description)
    """
    return get_registry().get(name)


def suggest_strategies(context: StrategyContext, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Suggest strategies for a context.

    Args:
        context: Query context
        top_k: Number of suggestions

    Returns:
        List of (strategy_name, confidence) tuples

    Example:
        >>> context = StrategyContext(query="analyze contract")
        >>> suggestions = suggest_strategies(context)
        >>> print(suggestions[0])
        ('verify', 0.92)
    """
    return get_registry().suggest(context, top_k=top_k)


def list_strategies(category: Optional[StrategyCategory] = None) -> List[Dict[str, str]]:
    """
    List all available strategies.

    Args:
        category: Optional category filter

    Returns:
        List of strategy info dicts

    Example:
        >>> strategies = list_strategies()
        >>> for s in strategies:
        ...     print(f"{s['name']}: {s['description']}")
    """
    registry = get_registry()

    if category is None:
        return registry.list_all()
    else:
        strategies = registry.list_by_category(category)
        return [
            {
                'name': s.name,
                'category': s.category.value,
                'description': s.description
            }
            for s in strategies
        ]
