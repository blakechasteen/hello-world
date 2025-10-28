# Building Apps on HoloLoom Framework

**Version:** 0.1.0
**Updated:** 2025-01-27

---

## Overview

HoloLoom is a **framework**, not a monolithic application. It provides semantic weaving infrastructure that you can build domain-specific analyzers on top of.

**Think of it like:**
- **Django** (framework) → **Blog, E-commerce** (apps built on it)
- **React** (framework) → **Instagram, Netflix** (apps built on it)
- **HoloLoom** (framework) → **Narrative, Code, Market analyzers** (apps built on it)

---

## Reference Implementation: hololoom-narrative

The **narrative analyzer** is the first reference app demonstrating the framework pattern.

**Stats:**
- 2400+ lines of domain logic
- 6 modules: intelligence, depth, streaming, cache, cross-domain, loop
- **Zero framework modifications** needed
- **Uses only public APIs**

Study it as a template: [hololoom_narrative/](hololoom_narrative/)

---

## Framework vs. App Boundaries

### ✅ Framework Provides

**Core Infrastructure:**
```python
from hololoom import (
    # Configuration
    Config,
    ExecutionMode,
    PatternCard,

    # Weaving Engine
    WeavingShuttle,
    WeavingOrchestrator,

    # Data Types
    Query,
    MemoryShard,
    Spacetime,
    Features,
    Context,

    # Components
    MatryoshkaEmbeddings,  # Multi-scale embeddings
    SpectralFusion,        # Spectral features
    create_policy,         # Neural decision engine
    create_motif_detector, # Pattern detection

    # Memory
    create_memory_backend, # Persistent storage
    UnifiedMemory,

    # Weaving Architecture
    LoomCommand,           # Pattern selection
    ChronoTrigger,         # Temporal control
    ResonanceShed,         # Feature fusion
    WarpSpace,             # Tensor operations
    ConvergenceEngine,     # Decision collapse
    Spacetime,             # Output fabric
    ReflectionBuffer,      # Learning loop
)
```

### ❌ Framework Does NOT Provide

**Domain-Specific Logic:**
- Character databases
- Story structure analysis
- Code patterns
- Market signals
- Scientific models
- Business rules

**These belong in APPS.**

---

## App Architecture Pattern

### 1. Directory Structure

```
hololoom_your_domain/
├── __init__.py              # Public API exports
├── core.py                  # Domain models & logic
├── analyzer.py              # Main analysis class
├── cache.py                 # Optional: domain caching
├── utils.py                 # Domain helpers
├── demos/
│   ├── quickstart.py        # Simple usage example
│   └── advanced.py          # Complex scenarios
├── tests/
│   ├── test_core.py
│   └── test_analyzer.py
├── docs/
│   ├── API.md               # Public API reference
│   └── GUIDE.md             # Usage guide
├── README.md                # Package overview
└── setup.py                 # Package config
```

### 2. Main Analyzer Class

```python
# hololoom_your_domain/analyzer.py
from typing import Optional, Dict, Any
from hololoom import WeavingShuttle, Config
from hololoom.fabric import Spacetime

class YourDomainAnalyzer:
    """
    Main entry point for your domain analysis.

    Pattern:
    1. Use framework for core processing
    2. Add domain-specific intelligence
    3. Return combined results
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize analyzer.

        Args:
            config: Framework configuration (default: Config.fused())
        """
        # Framework components
        self.config = config or Config.fused()
        self.shuttle = WeavingShuttle(cfg=self.config)

        # Domain-specific initialization
        self._init_domain_models()
        self._load_domain_knowledge()

    async def analyze(self, input_data: str) -> Dict[str, Any]:
        """
        Main analysis entry point.

        Args:
            input_data: Raw text to analyze

        Returns:
            Combined framework + domain results
        """
        # Step 1: Framework processing (weaving)
        spacetime: Spacetime = await self.shuttle.weave(input_data)

        # Step 2: Domain analysis
        domain_result = await self._domain_analysis(input_data)

        # Step 3: Combine results
        return {
            'framework': {
                'features': spacetime.features,
                'decision': spacetime.decision,
                'trace': spacetime.trace,
            },
            'domain': domain_result,
            'meta': {
                'confidence': self._calculate_confidence(domain_result),
                'model_version': self.__version__,
            }
        }

    async def _domain_analysis(self, data: str) -> Dict[str, Any]:
        """
        Your domain-specific analysis logic.

        This is where your unique intelligence lives.
        """
        # Example: Extract domain patterns
        patterns = self._detect_domain_patterns(data)

        # Example: Score against domain knowledge
        scores = self._score_patterns(patterns)

        # Example: Generate domain insights
        insights = self._generate_insights(patterns, scores)

        return {
            'patterns': patterns,
            'scores': scores,
            'insights': insights,
        }

    def _init_domain_models(self):
        """Initialize domain-specific models."""
        # Load databases, models, rules, etc.
        pass

    def _load_domain_knowledge(self):
        """Load domain knowledge base."""
        # Load patterns, templates, examples, etc.
        pass
```

### 3. Public API (__init__.py)

```python
# hololoom_your_domain/__init__.py
"""
HoloLoom Your Domain Analyzer
==============================
Description of what your analyzer does.

Features:
- Feature 1
- Feature 2
- Feature 3

Installation:
    pip install hololoom  # Framework
    pip install hololoom-your-domain  # This package

Usage:
    from hololoom_your_domain import YourDomainAnalyzer

    analyzer = YourDomainAnalyzer()
    result = await analyzer.analyze(data)
"""

__version__ = "0.1.0"

# Core analyzer
from hololoom_your_domain.analyzer import YourDomainAnalyzer

# Domain models
from hololoom_your_domain.core import (
    DomainModel,
    DomainResult,
    DomainPattern,
)

# Utilities
from hololoom_your_domain.utils import (
    load_domain_data,
    validate_input,
)

__all__ = [
    "YourDomainAnalyzer",
    "DomainModel",
    "DomainResult",
    "DomainPattern",
    "load_domain_data",
    "validate_input",
]
```

### 4. Package Configuration (setup.py)

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="hololoom-your-domain",
    version="0.1.0",
    description="Your domain analyzer built on HoloLoom framework",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/you/hololoom-your-domain",

    packages=find_packages(include=["hololoom_your_domain", "hololoom_your_domain.*"]),

    install_requires=[
        "hololoom>=0.1.0",  # Framework dependency
        # Add domain-specific deps here
    ],

    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "black>=23.0",
        ],
    },

    python_requires=">=3.9",

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
```

---

## Integration Patterns

### Pattern 1: Simple Analysis (Read-Only)

```python
from hololoom import WeavingShuttle, Config

class SimpleAnalyzer:
    """Just add domain analysis to framework output."""

    def __init__(self):
        self.shuttle = WeavingShuttle(cfg=Config.fast())

    async def analyze(self, text: str):
        # Get framework output
        spacetime = await self.shuttle.weave(text)

        # Add domain interpretation
        interpretation = self._interpret(spacetime.features)

        return {
            'spacetime': spacetime,
            'interpretation': interpretation,
        }

    def _interpret(self, features):
        # Your domain logic
        return {"insight": "..."}
```

### Pattern 2: Enhanced Memory (Write)

```python
from hololoom import WeavingShuttle, Config, create_memory_backend
from hololoom.Documentation.types import MemoryShard

class MemoryEnhancedAnalyzer:
    """Enrich memories with domain tags."""

    async def __aenter__(self):
        self.memory = await create_memory_backend(self.config)
        self.shuttle = WeavingShuttle(
            cfg=self.config,
            memory=self.memory
        )
        return self

    async def ingest(self, text: str, domain_tags: Dict):
        """Store with domain enrichment."""
        # Create shard
        shard = MemoryShard(
            text=text,
            source="user_input",
            timestamp=time.time(),
            metadata={
                **domain_tags,  # Add domain tags
                'confidence': self._calculate_confidence(text),
            }
        )

        # Store in framework memory
        await self.memory.store([shard])

    async def query(self, text: str):
        """Query with domain-aware retrieval."""
        spacetime = await self.shuttle.weave(text)

        # Domain-specific post-processing
        filtered = self._domain_filter(spacetime)

        return filtered
```

### Pattern 3: Custom Policy Integration

```python
from hololoom import WeavingShuttle, Config
from hololoom.policy import create_policy, BanditStrategy

class PolicyAwareAnalyzer:
    """Domain logic influences tool selection."""

    def __init__(self):
        config = Config.fused()

        # Custom policy with domain priors
        self.policy = create_policy(
            mem_dim=384,
            emb=None,
            scales=[96, 192, 384],
            bandit_strategy=BanditStrategy.BAYESIAN_BLEND,
            tool_priors=self._get_domain_priors(),  # Domain knowledge!
        )

        self.shuttle = WeavingShuttle(
            cfg=config,
            policy=self.policy,  # Use custom policy
        )

    def _get_domain_priors(self) -> Dict[str, float]:
        """Domain-specific tool selection priors."""
        return {
            'search': 0.3,     # Domain prefers search
            'analyze': 0.5,    # Highest prior
            'summarize': 0.2,
        }
```

### Pattern 4: Streaming Analysis

```python
from hololoom import WeavingShuttle, Config
from typing import AsyncIterator

class StreamingAnalyzer:
    """Process data streams with progressive depth."""

    async def analyze_stream(
        self,
        data_stream: AsyncIterator[str]
    ) -> AsyncIterator[Dict]:
        """Stream processing with live updates."""

        async for chunk in data_stream:
            # Process chunk
            spacetime = await self.shuttle.weave(chunk)

            # Domain-specific streaming logic
            result = self._stream_analysis(chunk, spacetime)

            # Yield intermediate result
            yield result

    def _stream_analysis(self, chunk, spacetime):
        """Domain logic for streaming."""
        return {
            'chunk': chunk,
            'spacetime': spacetime,
            'domain': self._extract_domain_features(chunk),
        }
```

---

## Best Practices

### ✅ DO

1. **Use public framework APIs only**
   ```python
   from hololoom import WeavingShuttle  # ✅ Good
   ```

2. **Declare framework dependency**
   ```python
   install_requires=["hololoom>=0.1.0"]
   ```

3. **Keep domain logic isolated**
   ```python
   # Domain code should work without framework
   domain_result = self._analyze_domain(data)  # Pure function
   ```

4. **Export clean API**
   ```python
   from hololoom_your_domain import YourAnalyzer  # ✅ Simple
   ```

5. **Document integration pattern**
   ```markdown
   ## How This Uses HoloLoom
   - Framework: Weaving + memory
   - Domain: Pattern detection
   ```

6. **Test independently**
   ```python
   # Domain logic tests (no framework)
   def test_domain_logic():
       result = analyze_pattern(data)
       assert result.confidence > 0.7

   # Integration tests (with framework)
   async def test_full_analysis():
       analyzer = YourAnalyzer()
       result = await analyzer.analyze(data)
       assert result['spacetime'] is not None
   ```

### ❌ DON'T

1. **Access framework internals**
   ```python
   from hololoom.internal.secret import SecretClass  # ❌ Bad
   from hololoom.weaving_shuttle import _private_method  # ❌ Bad
   ```

2. **Modify framework code**
   ```python
   # ❌ Don't fork framework to add features
   # ✅ Request feature addition via public API
   ```

3. **Create circular dependencies**
   ```python
   # ❌ Framework should never import your app
   # Framework → App: NEVER
   # App → Framework: OK (public API only)
   ```

4. **Mix concerns**
   ```python
   # ❌ Don't mix framework + domain in same file
   class Analyzer:
       def weave(self):  # Framework concern
           pass
       def analyze_narrative(self):  # Domain concern
           pass

   # ✅ Separate concerns
   class Analyzer:
       def __init__(self):
           self.shuttle = WeavingShuttle()  # Framework
           self.domain = DomainLogic()       # Domain
   ```

5. **Leak abstractions**
   ```python
   # ❌ Don't expose framework internals in your API
   def analyze(self) -> WeavingShuttleInternalState:
       pass

   # ✅ Expose domain results
   def analyze(self) -> DomainResult:
       pass
   ```

---

## Testing Strategy

### Framework Tests (Not Your Concern)
```python
# hololoom/tests/test_weaving.py
def test_weaving_shuttle():
    """Framework tests don't depend on apps."""
    shuttle = WeavingShuttle(cfg=Config.fast())
    result = await shuttle.weave("test")
    assert result.spacetime is not None
```

### Domain Tests (Pure Logic)
```python
# hololoom_your_domain/tests/test_domain.py
def test_pattern_detection():
    """Test domain logic without framework."""
    patterns = detect_patterns("input data")
    assert len(patterns) > 0
    assert patterns[0].confidence > 0.5
```

### Integration Tests (Framework + Domain)
```python
# hololoom_your_domain/tests/test_integration.py
import pytest
from hololoom import Config
from hololoom_your_domain import YourAnalyzer

@pytest.mark.asyncio
async def test_full_analysis():
    """Test app using framework."""
    analyzer = YourAnalyzer(config=Config.fast())
    result = await analyzer.analyze("test input")

    # Check framework output
    assert result['framework']['features'] is not None

    # Check domain output
    assert result['domain']['patterns'] is not None
    assert result['domain']['confidence'] > 0.0
```

---

## Examples by Domain

### Code Analysis App
```python
from hololoom import WeavingShuttle, Config
from hololoom_code import CodeAnalyzer

analyzer = CodeAnalyzer()
result = await analyzer.analyze(source_code)

# Domain results
print(result['patterns'])      # Design patterns detected
print(result['complexity'])    # Cyclomatic complexity
print(result['smells'])        # Code smells
print(result['suggestions'])   # Refactoring suggestions
```

### Market Intelligence App
```python
from hololoom import WeavingShuttle, Config
from hololoom_market import MarketAnalyzer

analyzer = MarketAnalyzer()
result = await analyzer.analyze(earnings_call)

# Domain results
print(result['sentiment'])     # Market sentiment
print(result['signals'])       # Trading signals
print(result['trends'])        # Emerging trends
print(result['risk_factors'])  # Risk analysis
```

### Scientific Discovery App
```python
from hololoom import WeavingShuttle, Config
from hololoom_research import ResearchAnalyzer

analyzer = ResearchAnalyzer()
result = await analyzer.analyze(paper_text)

# Domain results
print(result['methodology'])   # Research methods
print(result['findings'])      # Key findings
print(result['citations'])     # Citation analysis
print(result['novelty'])       # Novelty score
```

---

## Deployment

### Development
```bash
# Install framework
pip install hololoom

# Install your app in dev mode
cd hololoom_your_domain
pip install -e .

# Run tests
pytest tests/
```

### Production
```bash
# Install from PyPI
pip install hololoom-your-domain

# Or from source
pip install git+https://github.com/you/hololoom-your-domain
```

### Docker
```dockerfile
FROM python:3.11-slim

# Install framework
RUN pip install hololoom

# Install your app
COPY hololoom_your_domain /app/hololoom_your_domain
WORKDIR /app
RUN pip install -e hololoom_your_domain

# Run
CMD ["python", "-m", "hololoom_your_domain.server"]
```

---

## Publishing Your App

### 1. Repository Structure
```
your-repo/
├── hololoom_your_domain/    # Source code
├── tests/                   # Tests
├── docs/                    # Documentation
├── examples/                # Usage examples
├── setup.py                 # Package config
├── README.md                # Overview
├── LICENSE                  # License file
└── .github/workflows/       # CI/CD
```

### 2. README Template
```markdown
# HoloLoom Your Domain Analyzer

[Brief description]

## Built on HoloLoom Framework

This analyzer uses the [HoloLoom](https://github.com/you/hololoom) framework for semantic processing and adds domain-specific intelligence.

## Installation

\`\`\`bash
pip install hololoom  # Framework
pip install hololoom-your-domain  # This package
\`\`\`

## Quick Start

\`\`\`python
from hololoom_your_domain import YourAnalyzer

analyzer = YourAnalyzer()
result = await analyzer.analyze(data)
\`\`\`

## Features
- Feature 1
- Feature 2

## Documentation
[Link to docs]

## License
MIT
```

### 3. PyPI Publishing
```bash
# Build distribution
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*
```

---

## Getting Help

### Framework Issues
- Report at: https://github.com/you/hololoom/issues
- Framework bugs, API requests, core functionality

### App-Specific Issues
- Report in your app repo
- Domain logic, custom features

### Community
- Discussions: https://github.com/you/hololoom/discussions
- Share your app in "Show and Tell"

---

## Checklist for New Apps

- [ ] Directory structure follows pattern
- [ ] Main analyzer class implemented
- [ ] Clean public API in `__init__.py`
- [ ] `setup.py` declares `hololoom>=0.1.0` dependency
- [ ] Uses only public framework APIs (no internals)
- [ ] Domain logic isolated and testable
- [ ] Integration tests with framework
- [ ] README documents framework dependency
- [ ] Examples show usage pattern
- [ ] Tests pass independently

---

## Summary

**HoloLoom is a framework.** Your app provides domain intelligence.

**Pattern:**
1. Framework handles: weaving, memory, embeddings, decisions
2. Your app handles: domain patterns, scoring, insights
3. Combine: `{'framework': spacetime, 'domain': your_result}`

**Reference:** Study `hololoom_narrative/` as a complete example.

**Questions?** Open a discussion at https://github.com/you/hololoom/discussions
