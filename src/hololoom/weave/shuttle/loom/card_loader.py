"""
Pattern Card Loader System
===========================

Modular configuration system for HoloLoom.

Pattern Cards are YAML-based configuration modules that declaratively specify:
- Which mathematical capabilities to enable
- Which memory backends to use
- Which tools are available
- Performance/accuracy tradeoffs
- Extensions and custom features

Usage:
    # Load built-in card
    card = PatternCard.load("fast")

    # Load with overrides
    card = PatternCard.load("fast", overrides={'math': {'semantic_calculus': {'dimensions': 32}}})

    # Create shuttle from card
    shuttle = await WeavingShuttle.from_card("fast")
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class MathCapabilities:
    """
    Mathematical capabilities configuration.

    Controls which mathematical operations are exposed in the weaving process.
    """
    semantic_calculus: Dict[str, Any] = field(default_factory=dict)
    spectral_embedding: Dict[str, Any] = field(default_factory=dict)
    motif_detection: Dict[str, Any] = field(default_factory=dict)
    policy_engine: Dict[str, Any] = field(default_factory=dict)

    def to_semantic_config(self):
        """Convert to SemanticCalculusConfig."""
        from HoloLoom.semantic_calculus.config import SemanticCalculusConfig

        if not self.semantic_calculus.get('enabled', False):
            return None

        sc = self.semantic_calculus.get('config', {})
        cache_config = sc.get('cache', {})

        return SemanticCalculusConfig(
            enable_cache=cache_config.get('enabled', True),
            cache_size=cache_config.get('size', 10000),
            dimensions=sc.get('dimensions', 16),
            dt=sc.get('dt', 1.0),
            mass=sc.get('mass', 1.0),
            ethical_framework=sc.get('ethical_framework', 'compassionate'),
            compute_trajectory=sc.get('compute_trajectory', True),
            compute_ethics=sc.get('compute_ethics', True),
        )

    def is_enabled(self, capability: str) -> bool:
        """Check if specific capability is enabled."""
        cap_data = getattr(self, capability, {})
        return cap_data.get('enabled', False)


@dataclass
class MemoryConfig:
    """Memory backend configuration."""
    backend: str = "networkx"
    caching: Dict[str, Any] = field(default_factory=dict)
    retrieval: Dict[str, Any] = field(default_factory=dict)
    graph: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolsConfig:
    """Tools availability configuration."""
    enabled: List[str] = field(default_factory=list)
    disabled: List[str] = field(default_factory=list)
    configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if tool is enabled."""
        if tool_name in self.disabled:
            return False
        return tool_name in self.enabled

    def get_tool_config(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for specific tool."""
        if not self.is_tool_enabled(tool_name):
            return None
        return self.configs.get(tool_name, {})


@dataclass
class PerformanceProfile:
    """Performance optimization configuration."""
    target_latency_ms: int = 200
    max_latency_ms: int = 500
    timeout_ms: int = 2000
    optimization: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternCard:
    """
    Modular configuration card for HoloLoom.

    Pattern cards are self-contained configuration bundles that declaratively
    specify all aspects of the weaving process. They support inheritance for
    composition and can be shared as configuration recipes.

    Attributes:
        name: Unique card identifier
        display_name: Human-readable name
        description: Card description
        version: Card version (for compatibility)
        math_capabilities: Math module configuration
        memory_config: Memory backend configuration
        tools_config: Tools availability configuration
        performance_profile: Performance optimization settings
        extensions: Custom extensions and features
        extends: Parent card name for inheritance

    Example:
        >>> card = PatternCard.load("fast")
        >>> print(f"Card: {card.display_name}")
        >>> print(f"Semantic dims: {card.math_capabilities.semantic_calculus['config']['dimensions']}")
    """
    name: str
    display_name: str = ""
    description: str = ""
    version: str = "1.0"

    # Configuration modules
    math_capabilities: MathCapabilities = field(default_factory=MathCapabilities)
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    tools_config: ToolsConfig = field(default_factory=ToolsConfig)
    performance_profile: PerformanceProfile = field(default_factory=PerformanceProfile)
    extensions: Dict[str, Any] = field(default_factory=dict)

    # Inheritance
    extends: Optional[str] = None

    @classmethod
    def load(cls, card_name: str, cards_dir: Path = None, overrides: Dict[str, Any] = None) -> 'PatternCard':
        """
        Load pattern card from YAML file.

        Supports inheritance via 'extends' field. Child cards override parent values.

        Args:
            card_name: Name of card to load (without .yaml extension)
            cards_dir: Directory containing card files (default: HoloLoom/cards/)
            overrides: Runtime overrides to apply after loading

        Returns:
            Loaded PatternCard instance

        Raises:
            ValueError: If card file not found
            yaml.YAMLError: If card file is invalid YAML

        Example:
            >>> # Load built-in card
            >>> card = PatternCard.load("fast")

            >>> # Load with overrides
            >>> card = PatternCard.load("fast", overrides={
            ...     'math': {'semantic_calculus': {'config': {'dimensions': 32}}}
            ... })
        """
        # Default cards directory
        if cards_dir is None:
            cards_dir = Path(__file__).parent.parent / "cards"

        card_path = cards_dir / f"{card_name}.yaml"

        if not card_path.exists():
            raise ValueError(f"Pattern card not found: {card_path}")

        logger.info(f"Loading pattern card: {card_name} from {card_path}")

        # Load YAML
        with open(card_path) as f:
            data = yaml.safe_load(f)

        # Handle inheritance
        if 'extends' in data and data['extends']:
            parent_name = data['extends']
            logger.info(f"  Card '{card_name}' extends '{parent_name}'")
            parent_card = cls.load(parent_name, cards_dir)

            # Merge parent with child (child overrides parent)
            parent_dict = parent_card.to_dict()
            data = cls._merge_configs(parent_dict, data)

        # Apply runtime overrides
        if overrides:
            logger.info(f"  Applying runtime overrides to '{card_name}'")
            data = cls._merge_configs(data, overrides)

        # Build card from merged data
        card = cls.from_dict(data)
        logger.info(f"✓ Loaded card '{card.display_name}' (v{card.version})")

        return card

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternCard':
        """
        Construct PatternCard from dictionary.

        Args:
            data: Dictionary containing card configuration

        Returns:
            PatternCard instance
        """
        return cls(
            name=data.get('name', 'unnamed'),
            display_name=data.get('display_name', data.get('name', 'Unnamed')),
            description=data.get('description', ''),
            version=data.get('version', '1.0'),

            # Modules
            math_capabilities=MathCapabilities(**data.get('math', {})),
            memory_config=MemoryConfig(**data.get('memory', {})),
            tools_config=ToolsConfig(**data.get('tools', {})),
            performance_profile=PerformanceProfile(**data.get('performance', {})),
            extensions=data.get('extensions', {}),

            # Inheritance
            extends=data.get('extends'),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert card to dictionary representation.

        Returns:
            Dictionary containing card configuration
        """
        return {
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'version': self.version,
            'extends': self.extends,

            # Modules
            'math': {
                'semantic_calculus': self.math_capabilities.semantic_calculus,
                'spectral_embedding': self.math_capabilities.spectral_embedding,
                'motif_detection': self.math_capabilities.motif_detection,
                'policy_engine': self.math_capabilities.policy_engine,
            },
            'memory': {
                'backend': self.memory_config.backend,
                'caching': self.memory_config.caching,
                'retrieval': self.memory_config.retrieval,
                'graph': self.memory_config.graph,
            },
            'tools': {
                'enabled': self.tools_config.enabled,
                'disabled': self.tools_config.disabled,
                'configs': self.tools_config.configs,
            },
            'performance': {
                'target_latency_ms': self.performance_profile.target_latency_ms,
                'max_latency_ms': self.performance_profile.max_latency_ms,
                'timeout_ms': self.performance_profile.timeout_ms,
                'optimization': self.performance_profile.optimization,
            },
            'extensions': self.extensions,
        }

    def save(self, card_name: str = None, cards_dir: Path = None):
        """
        Save card to YAML file.

        Args:
            card_name: Name for saved card (default: use self.name)
            cards_dir: Directory to save card in
        """
        card_name = card_name or self.name
        cards_dir = cards_dir or Path(__file__).parent.parent / "cards"
        cards_dir.mkdir(parents=True, exist_ok=True)

        card_path = cards_dir / f"{card_name}.yaml"

        logger.info(f"Saving pattern card '{card_name}' to {card_path}")

        with open(card_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"✓ Saved card '{card_name}'")

    @staticmethod
    def _merge_configs(base: Dict, override: Dict) -> Dict:
        """
        Deep merge two configuration dictionaries.

        Override values take precedence over base values. Recursively merges nested dicts.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = PatternCard._merge_configs(result[key], value)
            else:
                # Override value
                result[key] = value

        return result

    def __repr__(self) -> str:
        """String representation of card."""
        return (
            f"PatternCard(name='{self.name}', "
            f"display_name='{self.display_name}', "
            f"version='{self.version}')"
        )
