# Pattern Cards

**Modular Configuration System for HoloLoom**

Pattern Cards are YAML-based configuration modules that declaratively specify all aspects of the weaving process. They replace hard-coded configuration with shareable, version-controlled, composable configuration files.

## Quick Start

```python
from HoloLoom.loom.card_loader import PatternCard

# Load built-in card
card = PatternCard.load("fast")

# Load with overrides
card = PatternCard.load("fast", overrides={
    'math': {'semantic_calculus': {'config': {'dimensions': 32}}}
})

# Use with WeavingShuttle (future)
shuttle = await WeavingShuttle.from_card("fast")
```

## Built-in Cards

### ⚡ Bare Mode (`bare.yaml`)
**Minimal processing for maximum speed**

- **Use for**: Simple queries, speed-critical applications
- **Semantic calculus**: DISABLED
- **Embedding**: Single scale (96D)
- **Tools**: 2 (summarize, search)
- **Target latency**: 50ms
- **Memory**: NetworkX (in-memory)

### 🚀 Fast Mode (`fast.yaml`)
**Balanced speed and capability** *(recommended default)*

- **Use for**: Production queries, most use cases
- **Semantic calculus**: ENABLED (16D)
- **Embedding**: Dual scale (96D, 192D)
- **Tools**: 4 (summarize, search, analyze, extract)
- **Target latency**: 200ms
- **Memory**: NetworkX

### 🔬 Fused Mode (`fused.yaml`)
**Maximum capability for highest quality**

- **Use for**: Complex queries, research, deep analysis
- **Semantic calculus**: ENABLED (32D)
- **Embedding**: Triple scale (96D, 192D, 384D)
- **Tools**: 6 (all tools enabled)
- **Target latency**: 1000ms
- **Memory**: Neo4j + Qdrant (hybrid)

## Example Custom Cards

### 🔬 Research Mode (`research.yaml`)
Deep research with maximum detail

- Extends: `fused`
- Semantic dimensions: 64D
- Cache: 50,000 words
- Target latency: 5000ms (accepts slower)
- Custom extension: `citation_tracking`

### ⚖️ Ethical Analysis (`ethical.yaml`)
Ethics-focused with manipulation detection

- Extends: `fast`
- Ethical framework: `therapeutic`
- Semantic dimensions: 24D
- Tools: Focus on ethical evaluation
- Custom extension: `manipulation_detection`

## Card Structure

```yaml
# Card metadata
name: "fast"
display_name: "🚀 Fast Mode"
description: "Balanced speed and capability"
version: "1.0"

# Inheritance (optional)
extends: "bare"  # Inherit from another card

# Math capabilities configuration
math:
  semantic_calculus:
    enabled: true
    config:
      dimensions: 16
      compute_trajectory: true
      compute_ethics: true
      cache:
        enabled: true
        size: 10000

  spectral_embedding:
    enabled: true
    scales: [96, 192]
    fusion_mode: "adaptive"

  motif_detection:
    enabled: true
    mode: "hybrid"

  policy_engine:
    enabled: true
    type: "neural_bandit"

# Memory configuration
memory:
  backend: "networkx"
  caching:
    vector_cache: true
    cache_size: 5000
  retrieval:
    max_shards: 10

# Tools availability
tools:
  enabled: ["summarize", "search", "analyze"]
  disabled: ["deep_research"]
  configs:
    summarize:
      max_length: 500

# Performance profile
performance:
  target_latency_ms: 200
  max_latency_ms: 500
  timeout_ms: 2000

# Custom extensions
extensions:
  narrative_depth:
    enabled: false
  semantic_monitoring:
    enabled: true
```

## Features

### 1. Declarative Configuration
All configuration in one YAML file - no hunting through code.

### 2. Composability via Inheritance
```yaml
# child.yaml
extends: "parent"  # Inherit everything

# Override only what changes
math:
  semantic_calculus:
    config:
      dimensions: 32  # Override specific value
```

### 3. Runtime Overrides
```python
card = PatternCard.load("fast", overrides={
    'math': {'semantic_calculus': {'config': {'dimensions': 24}}}
})
```

### 4. Version Control Friendly
Cards are files → track with git:
```bash
git add HoloLoom/cards/my_custom.yaml
git commit -m "Add custom research configuration"
```

### 5. Shareable Configuration Recipes
Share cards like code:
```bash
# Clone repo, get cards for free
git clone repo
cd HoloLoom/cards/
# Use shared configurations
```

## Creating Custom Cards

### Method 1: Extend Existing Card (YAML)

Create `HoloLoom/cards/my_custom.yaml`:
```yaml
name: "my_custom"
extends: "fast"  # Start from fast

# Override what you need
math:
  semantic_calculus:
    config:
      dimensions: 24

performance:
  target_latency_ms: 150
```

### Method 2: Programmatic Creation

```python
from HoloLoom.loom.card_loader import PatternCard

# Load base card
card = PatternCard.load("fast")

# Customize
card.name = "my_custom"
card.display_name = "My Custom Mode"
card.math_capabilities.semantic_calculus['config']['dimensions'] = 24

# Save
card.save("my_custom")
```

## Use Cases

### Speed-Optimized Card
```yaml
name: "blazing_fast"
math:
  semantic_calculus:
    enabled: false  # Skip for max speed
performance:
  target_latency_ms: 50
```

### Ethics-Focused Card
```yaml
name: "ethical"
extends: "fast"
math:
  semantic_calculus:
    config:
      compute_ethics: true
      ethical_framework: "therapeutic"
      dimensions: 24  # More nuance
```

### Research Card
```yaml
name: "research"
extends: "fused"
math:
  semantic_calculus:
    config:
      dimensions: 64  # Maximum detail
performance:
  target_latency_ms: 10000  # Accept 10s
```

## API Reference

### PatternCard

#### Loading Cards

```python
# Load built-in card
card = PatternCard.load("fast")

# Load from custom directory
card = PatternCard.load("my_card", cards_dir=Path("./my_cards"))

# Load with overrides
card = PatternCard.load("fast", overrides={'math': {...}})
```

#### Saving Cards

```python
# Save to default directory
card.save("my_card")

# Save to custom directory
card.save("my_card", cards_dir=Path("./my_cards"))
```

#### Converting to Config

```python
# Convert to SemanticCalculusConfig
sem_config = card.math_capabilities.to_semantic_config()

# Check if math is enabled
if card.math_capabilities.is_enabled('semantic_calculus'):
    # Use semantic calculus
    pass
```

#### Tools Configuration

```python
# Check if tool is enabled
if card.tools_config.is_tool_enabled("analyze"):
    # Use tool
    pass

# Get tool config
config = card.tools_config.get_tool_config("summarize")
# Returns: {'max_length': 500}
```

## Testing

Run tests to validate cards:
```bash
cd HoloLoom
python tests/test_pattern_cards.py
```

## Demo

See complete examples:
```bash
python demos/pattern_cards_demo.py
```

## Migration Path

### Phase 1: Add Card System (Current)
- ✅ PatternCard system implemented
- ✅ Built-in cards created (BARE/FAST/FUSED)
- ✅ Tests passing
- ✅ Demo working
- ⏳ WeavingShuttle integration (optional)

### Phase 2: Dual Interface
- Support both old and new APIs
- `Config.fast()` → internally loads `fast.yaml`
- Gradual migration

### Phase 3: Card-Native
- Deprecate old configuration
- All configuration via cards
- Documentation updated

## Benefits

| Aspect | Before (Code) | After (Cards) |
|--------|--------------|---------------|
| **Configuration** | Hard-coded | Declarative YAML |
| **Reusability** | Copy-paste code | Inherit from cards |
| **Sharing** | Document settings | Share YAML file |
| **Versioning** | Code comments | Git history |
| **Customization** | Modify code | Override YAML |
| **A/B Testing** | Code branches | Different cards |

## Frequently Asked Questions

### Q: Do I need to update my existing code?
**A:** No! The card system is additive. Existing Config.fast() still works.

### Q: Can I use cards with WeavingShuttle?
**A:** Not yet, but integration is planned:
```python
# Future API
shuttle = await WeavingShuttle.from_card("fast")
```

### Q: Can cards reference other files?
**A:** Not currently, but could be added via `include` directive.

### Q: How do I share my custom cards?
**A:** Commit to git and share the repo, or copy the YAML file.

### Q: Can I override cards at runtime?
**A:** Yes! Use the `overrides` parameter:
```python
card = PatternCard.load("fast", overrides={...})
```

### Q: Are cards validated?
**A:** Basic validation happens during loading. Schema validation could be added.

### Q: Can I create card templates?
**A:** Yes! Use inheritance and placeholders in extensions.

## Next Steps

1. **Explore**: Run `python demos/pattern_cards_demo.py`
2. **Test**: Run `python tests/test_pattern_cards.py`
3. **Customize**: Create your own cards in `HoloLoom/cards/`
4. **Share**: Commit your cards to version control
5. **Iterate**: A/B test different configurations

## Contributing

To add a new built-in card:
1. Create `HoloLoom/cards/your_card.yaml`
2. Add tests in `tests/test_pattern_cards.py`
3. Document in this README
4. Submit PR

## Resources

- **Design Doc**: [docs/PATTERN_CARDS_AS_MODULES.md](../../docs/PATTERN_CARDS_AS_MODULES.md)
- **Loader Code**: [HoloLoom/loom/card_loader.py](../loom/card_loader.py)
- **Tests**: [tests/test_pattern_cards.py](../../tests/test_pattern_cards.py)
- **Demo**: [demos/pattern_cards_demo.py](../../demos/pattern_cards_demo.py)

---

**Pattern Cards: Making HoloLoom configuration modular, shareable, and composable.** 🎯
