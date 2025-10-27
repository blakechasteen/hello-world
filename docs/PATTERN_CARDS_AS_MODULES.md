# Pattern Cards as Configuration Modules

## The Vision

**Pattern Cards = Modular Configuration Bundles**

Instead of hardcoding "which math to expose" decisions, pattern cards become **composable configuration modules** that declaratively specify:
- Which mathematical capabilities to enable
- Which memory backends to use
- Which tools are available
- Performance/accuracy tradeoffs
- Feature flags and extensions

## Current vs. Modular Architecture

### Current: Implicit Configuration
```python
# Today: Configuration is implicit in code
pattern = "fast"
# → Hard-coded logic decides what gets created

if pattern == "fast":
    # Somewhere in orchestrator.py
    enable_semantic = True
    semantic_dims = 16
    compute_ethics = True
    # ... buried in implementation
```

### Modular: Declarative Cards
```python
# Future: Configuration is explicit, reusable module
@dataclass
class PatternCard:
    """Self-contained configuration module."""
    name: str
    description: str

    # Math exposure controls
    math_capabilities: MathCapabilities

    # Memory configuration
    memory_config: MemoryConfig

    # Tool availability
    available_tools: List[str]

    # Performance profile
    performance_profile: PerformanceProfile

    # Extensions
    extensions: Dict[str, Any]

# Cards become composable modules
fast_card = PatternCard.load("fast.yaml")
```

## Design: Pattern Cards as Modules

### 1. Card Structure (YAML-based)

```yaml
# cards/fast.yaml
name: "fast"
display_name: "Fast Mode"
description: "Balanced speed and capability for production queries"
version: "1.0"

# ============================================================
# MATH CAPABILITIES MODULE
# ============================================================
math:
  semantic_calculus:
    enabled: true
    config:
      dimensions: 16
      compute_trajectory: true
      compute_ethics: true
      ethical_framework: "compassionate"
      cache:
        enabled: true
        size: 10000

  spectral_embedding:
    enabled: true
    scales: [96, 192, 384]
    fusion_mode: "adaptive"

  motif_detection:
    enabled: true
    mode: "hybrid"  # regex + NLP
    patterns: ["question", "causality", "comparison"]

  policy_engine:
    enabled: true
    type: "neural_bandit"
    strategy: "epsilon_greedy"
    epsilon: 0.1

# ============================================================
# MEMORY MODULE
# ============================================================
memory:
  backend: "networkx"
  caching:
    vector_cache: true
    cache_size: 5000
  retrieval:
    max_shards: 10
    similarity_threshold: 0.7
  graph:
    max_depth: 2

# ============================================================
# TOOLS MODULE
# ============================================================
tools:
  enabled: ["summarize", "search", "analyze", "extract"]
  disabled: ["deep_research", "synthesis"]

  configs:
    summarize:
      max_length: 500
    search:
      max_results: 5

# ============================================================
# PERFORMANCE PROFILE
# ============================================================
performance:
  target_latency_ms: 200
  max_latency_ms: 500
  timeout_ms: 2000

  optimization:
    jit_compile: true
    parallel_threads: 4
    batch_size: 32

# ============================================================
# EXTENSIONS (Custom Features)
# ============================================================
extensions:
  narrative_depth:
    enabled: false

  semantic_monitoring:
    enabled: true
    track_velocity: true
    track_ethics: true

  custom_hooks:
    pre_query: null
    post_query: null
```

### 2. Card Composition (Inheritance)

```yaml
# cards/fused.yaml
name: "fused"
extends: "fast"  # Inherit from fast card

# Override specific modules
math:
  semantic_calculus:
    config:
      dimensions: 32  # More detail
      cache:
        size: 20000   # Larger cache

memory:
  backend: "neo4j_qdrant"  # Upgrade backend
  retrieval:
    max_shards: 20          # More context

tools:
  enabled: ["summarize", "search", "analyze", "extract", "deep_research"]

extensions:
  narrative_depth:
    enabled: true  # Enable for fused
```

### 3. Custom Cards (User-Defined)

```yaml
# cards/custom_research.yaml
name: "research"
extends: "fused"
display_name: "Deep Research Mode"

math:
  semantic_calculus:
    config:
      dimensions: 64  # Maximum detail
      compute_ethics: false  # Skip for speed

tools:
  enabled: ["deep_research", "synthesis", "cross_reference"]

performance:
  target_latency_ms: 5000  # Accept slower for thoroughness

extensions:
  # Custom extension
  citation_tracking:
    enabled: true
    format: "academic"
```

## Implementation Architecture

### 1. Card Loader System

```python
# HoloLoom/loom/card_loader.py

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import yaml
from pathlib import Path

@dataclass
class MathCapabilities:
    """Math module configuration."""
    semantic_calculus: Dict[str, Any]
    spectral_embedding: Dict[str, Any]
    motif_detection: Dict[str, Any]
    policy_engine: Dict[str, Any]

    def to_semantic_config(self) -> 'SemanticCalculusConfig':
        """Convert to SemanticCalculusConfig."""
        from HoloLoom.semantic_calculus.config import SemanticCalculusConfig

        sc = self.semantic_calculus.get('config', {})
        return SemanticCalculusConfig(
            enable_cache=sc.get('cache', {}).get('enabled', True),
            cache_size=sc.get('cache', {}).get('size', 10000),
            dimensions=sc.get('dimensions', 16),
            compute_trajectory=sc.get('compute_trajectory', True),
            compute_ethics=sc.get('compute_ethics', True),
            ethical_framework=sc.get('ethical_framework', 'compassionate'),
        )

@dataclass
class MemoryConfig:
    """Memory module configuration."""
    backend: str
    caching: Dict[str, Any]
    retrieval: Dict[str, Any]
    graph: Dict[str, Any]

@dataclass
class PerformanceProfile:
    """Performance optimization configuration."""
    target_latency_ms: int
    max_latency_ms: int
    timeout_ms: int
    optimization: Dict[str, Any]

@dataclass
class PatternCard:
    """
    Modular configuration card.

    Self-contained bundle of configuration that determines:
    - Which math gets exposed
    - Which backends are used
    - Which tools are available
    - Performance characteristics
    """
    name: str
    display_name: str
    description: str
    version: str

    # Configuration modules
    math_capabilities: MathCapabilities
    memory_config: MemoryConfig
    available_tools: List[str]
    disabled_tools: List[str]
    performance_profile: PerformanceProfile
    extensions: Dict[str, Any]

    # Inheritance
    extends: Optional[str] = None

    @classmethod
    def load(cls, card_name: str, cards_dir: Path = None) -> 'PatternCard':
        """
        Load pattern card from YAML.

        Supports inheritance via 'extends' field.
        """
        cards_dir = cards_dir or Path(__file__).parent.parent / "cards"
        card_path = cards_dir / f"{card_name}.yaml"

        if not card_path.exists():
            raise ValueError(f"Pattern card not found: {card_name}")

        with open(card_path) as f:
            data = yaml.safe_load(f)

        # Handle inheritance
        if 'extends' in data and data['extends']:
            parent_card = cls.load(data['extends'], cards_dir)
            # Merge parent with overrides
            data = cls._merge_configs(parent_card.to_dict(), data)

        # Build card from data
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternCard':
        """Construct card from dictionary."""
        return cls(
            name=data['name'],
            display_name=data.get('display_name', data['name']),
            description=data.get('description', ''),
            version=data.get('version', '1.0'),
            math_capabilities=MathCapabilities(**data.get('math', {})),
            memory_config=MemoryConfig(**data.get('memory', {})),
            available_tools=data.get('tools', {}).get('enabled', []),
            disabled_tools=data.get('tools', {}).get('disabled', []),
            performance_profile=PerformanceProfile(**data.get('performance', {})),
            extensions=data.get('extensions', {}),
            extends=data.get('extends'),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert card to dictionary."""
        # Implementation
        pass

    @staticmethod
    def _merge_configs(base: Dict, override: Dict) -> Dict:
        """Deep merge configurations."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PatternCard._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def get_tool_config(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for specific tool."""
        if tool_name in self.disabled_tools:
            return None
        if tool_name not in self.available_tools:
            return None
        # Return tool-specific config from extensions or defaults
        return self.extensions.get('tool_configs', {}).get(tool_name, {})

    def is_math_enabled(self, capability: str) -> bool:
        """Check if math capability is enabled."""
        # e.g., capability = "semantic_calculus"
        cap_data = getattr(self.math_capabilities, capability, {})
        return cap_data.get('enabled', False)
```

### 2. Orchestrator Integration

```python
# HoloLoom/weaving_shuttle.py (updated)

from HoloLoom.loom.card_loader import PatternCard

class WeavingShuttle:
    def __init__(self, card: PatternCard, ...):
        """Initialize with pattern card."""
        self.card = card
        self.cfg = self._card_to_config(card)

        # Create components based on card
        self._initialize_from_card(card)

    def _initialize_from_card(self, card: PatternCard):
        """Initialize all components from pattern card."""

        # 1. Math Capabilities
        if card.is_math_enabled('semantic_calculus'):
            sem_config = card.math_capabilities.to_semantic_config()
            self.semantic_analyzer = create_semantic_analyzer(
                self.embed_fn,
                config=sem_config
            )
        else:
            self.semantic_analyzer = None

        # 2. Memory Backend
        self.memory = create_memory_backend(card.memory_config)

        # 3. Tools
        self.tool_registry = {
            tool: self._create_tool(tool, card.get_tool_config(tool))
            for tool in card.available_tools
        }

        # 4. Performance Optimization
        self._apply_performance_profile(card.performance_profile)

        # 5. Extensions
        self._load_extensions(card.extensions)

    @classmethod
    async def from_card(cls, card_name: str, **overrides):
        """Create shuttle from card name."""
        card = PatternCard.load(card_name)

        # Apply runtime overrides
        for key, value in overrides.items():
            setattr(card, key, value)

        return cls(card=card)
```

### 3. Usage Examples

#### Basic Usage
```python
# Load built-in card
shuttle = await WeavingShuttle.from_card("fast")

# Query with card's configuration
result = await shuttle.weave(query)
```

#### Custom Card
```python
# Load custom research card
shuttle = await WeavingShuttle.from_card("custom_research")

# All math/memory/tools configured by card
result = await shuttle.weave(query)
```

#### Runtime Override
```python
# Load card but override specific settings
shuttle = await WeavingShuttle.from_card(
    "fast",
    math_capabilities={'semantic_calculus': {'dimensions': 32}}
)
```

#### Card Composition
```python
# Create new card by composing existing ones
research_card = PatternCard.load("fused")
research_card.extensions['custom_feature'] = True
research_card.save("my_research.yaml")

# Use composed card
shuttle = await WeavingShuttle.from_card("my_research")
```

## Benefits of This Design

### 1. **Declarative Configuration**
- All configuration in one place (YAML)
- Easy to understand what's enabled
- No hunting through code

### 2. **Modularity**
- Cards are self-contained
- Easy to share and reuse
- Composition via inheritance

### 3. **Version Control**
- Cards are files → can be versioned
- Track configuration changes over time
- Roll back to previous configurations

### 4. **User Customization**
- Users create custom cards for their use cases
- No code changes required
- Share cards as configuration recipes

### 5. **Testing & Validation**
- Test specific configurations easily
- A/B test cards against each other
- Validate card compatibility

### 6. **Documentation**
- Cards document themselves
- Configuration is the documentation
- Examples are working configurations

## Example Use Cases

### Use Case 1: Speed-Optimized Card
```yaml
# cards/blazing_fast.yaml
name: "blazing_fast"
display_name: "⚡ Blazing Fast Mode"

math:
  semantic_calculus:
    enabled: false  # Skip for max speed

  spectral_embedding:
    scales: [96]  # Single scale only

memory:
  backend: "simple"
  caching:
    vector_cache: false  # Skip caching overhead

performance:
  target_latency_ms: 50
  optimization:
    jit_compile: true
```

### Use Case 2: Ethics-Focused Card
```yaml
# cards/ethical_analysis.yaml
name: "ethical"
extends: "fast"

math:
  semantic_calculus:
    config:
      compute_ethics: true
      ethical_framework: "therapeutic"
      dimensions: 24  # More dimensions for nuance

tools:
  enabled: ["evaluate_ethics", "suggest_rephrasing"]

extensions:
  manipulation_detection:
    enabled: true
    sensitivity: "high"
```

### Use Case 3: Research Card
```yaml
# cards/deep_research.yaml
name: "research"
extends: "fused"

math:
  semantic_calculus:
    config:
      dimensions: 64  # Maximum detail

memory:
  backend: "neo4j_qdrant"
  retrieval:
    max_shards: 50  # Extensive context

tools:
  enabled: ["deep_research", "synthesis", "cross_reference", "citation"]

performance:
  target_latency_ms: 10000  # Accept 10s for thoroughness
```

## Migration Path

### Phase 1: Keep Current System, Add Card Loader
- Implement PatternCard system alongside current code
- Convert existing patterns (BARE/FAST/FUSED) to cards
- Maintain backward compatibility

### Phase 2: Dual Interface
- Support both old and new APIs
- `Config.fast()` → internally loads `fast.yaml`
- Gradual migration

### Phase 3: Card-Native
- Deprecate old configuration system
- All configuration via cards
- Documentation updated

## Summary

**Pattern Cards become the "which math to expose" module system.**

Instead of:
```python
if pattern == "fast":
    # Hard-coded decisions buried in orchestrator
```

We get:
```yaml
# cards/fast.yaml
math:
  semantic_calculus:
    enabled: true
    config:
      dimensions: 16
```

**Cards are:**
- ✅ Declarative (YAML, not code)
- ✅ Composable (inheritance)
- ✅ Versioned (files in git)
- ✅ Shareable (configuration recipes)
- ✅ Self-documenting (config is docs)

**Result: Configuration becomes modular, reusable, and shareable.**
