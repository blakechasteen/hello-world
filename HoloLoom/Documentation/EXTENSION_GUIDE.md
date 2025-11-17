# HoloLoom Extension Guide

Practical guide for extending HoloLoom with new features, tools, and capabilities.

## Quick Start: Adding Your First Extension

### Example: Adding a New Feature Extractor

Let's add a **sentiment analysis** feature extractor that detects emotional tone in queries.

#### Step 1: Create Module Structure

```bash
mkdir -p HoloLoom/features/sentiment
touch HoloLoom/features/sentiment/__init__.py
touch HoloLoom/features/sentiment/analyzer.py
touch HoloLoom/features/sentiment/config.py
touch HoloLoom/features/sentiment/test_sentiment.py
```

#### Step 2: Implement Feature Extractor

**`HoloLoom/features/sentiment/config.py`**:
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""
    model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    threshold: float = 0.5
    enabled: bool = True
    cache_results: bool = True
```

**`HoloLoom/features/sentiment/analyzer.py`**:
```python
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import transformers, graceful degradation if not available
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("transformers not available, sentiment analysis disabled")
    TRANSFORMERS_AVAILABLE = False

from .config import SentimentConfig

class SentimentAnalyzer:
    """
    Analyze sentiment/emotional tone of text.

    Returns:
        - label: "POSITIVE", "NEGATIVE", "NEUTRAL"
        - score: confidence (0-1)
        - emotions: dict of emotion scores
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self._pipeline = None

        if TRANSFORMERS_AVAILABLE and self.config.enabled:
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.config.model
            )

    async def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""

        if not self.config.enabled:
            return self._neutral_result()

        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, returning neutral")
            return self._neutral_result()

        try:
            # Run sentiment analysis
            result = self._pipeline(text, truncation=True, max_length=512)[0]

            return {
                "label": result["label"],
                "score": result["score"],
                "emotions": self._detect_emotions(text),
                "confidence": result["score"]
            }

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return self._neutral_result()

    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """Detect specific emotions (joy, anger, sadness, etc.)."""
        # Simplified emotion detection
        emotions = {
            "joy": 0.0,
            "anger": 0.0,
            "sadness": 0.0,
            "fear": 0.0,
            "surprise": 0.0
        }

        # Simple keyword-based detection (replace with proper model)
        text_lower = text.lower()

        if any(word in text_lower for word in ["happy", "joy", "excited", "great"]):
            emotions["joy"] = 0.7
        if any(word in text_lower for word in ["angry", "mad", "furious", "hate"]):
            emotions["anger"] = 0.7
        if any(word in text_lower for word in ["sad", "depressed", "unhappy", "terrible"]):
            emotions["sadness"] = 0.7

        return emotions

    def _neutral_result(self) -> Dict[str, Any]:
        """Return neutral sentiment."""
        return {
            "label": "NEUTRAL",
            "score": 0.5,
            "emotions": {k: 0.0 for k in ["joy", "anger", "sadness", "fear", "surprise"]},
            "confidence": 0.5
        }
```

**`HoloLoom/features/sentiment/__init__.py`**:
```python
from .analyzer import SentimentAnalyzer
from .config import SentimentConfig

__all__ = ["SentimentAnalyzer", "SentimentConfig"]
```

#### Step 3: Add Tests

**`HoloLoom/features/sentiment/test_sentiment.py`**:
```python
import pytest
from HoloLoom.features.sentiment import SentimentAnalyzer, SentimentConfig

@pytest.mark.asyncio
async def test_positive_sentiment():
    analyzer = SentimentAnalyzer()
    result = await analyzer.analyze("I'm so happy and excited!")

    assert result["label"] in ["POSITIVE", "NEUTRAL"]  # NEUTRAL if transformers not available
    assert 0.0 <= result["score"] <= 1.0
    assert "emotions" in result

@pytest.mark.asyncio
async def test_negative_sentiment():
    analyzer = SentimentAnalyzer()
    result = await analyzer.analyze("I'm really angry and upset!")

    assert result["label"] in ["NEGATIVE", "NEUTRAL"]
    assert "anger" in result["emotions"]

@pytest.mark.asyncio
async def test_neutral_text():
    analyzer = SentimentAnalyzer()
    result = await analyzer.analyze("The weather is cloudy today.")

    assert "label" in result
    assert "score" in result

def test_disabled_analyzer():
    config = SentimentConfig(enabled=False)
    analyzer = SentimentAnalyzer(config)

    # Should work even when disabled
    import asyncio
    result = asyncio.run(analyzer.analyze("Test"))
    assert result["label"] == "NEUTRAL"
```

#### Step 4: Integrate with Orchestrator

**Update `HoloLoom/orchestrator.py`**:
```python
# Add import
from HoloLoom.features.sentiment import SentimentAnalyzer, SentimentConfig

class Orchestrator:
    def __init__(self, config: Config):
        # ... existing code ...

        # Add sentiment analyzer
        sentiment_config = SentimentConfig()
        self.sentiment = SentimentAnalyzer(sentiment_config)

    async def process(self, query: Query) -> ActionPlan:
        # ... existing feature extraction ...

        # Add sentiment analysis
        sentiment = await self.sentiment.analyze(query.text)

        # Add to metadata
        query.metadata["sentiment"] = sentiment

        # Optionally: Use sentiment in policy decision
        if sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.8:
            # User is frustrated, maybe provide more helpful response
            pass

        # Continue with existing pipeline
        # ...
```

#### Step 5: Document

**`HoloLoom/Documentation/SENTIMENT_ANALYSIS.md`**:
```markdown
# Sentiment Analysis Feature

Detects emotional tone in user queries.

## Installation

```bash
pip install transformers torch
```

## Usage

```python
from HoloLoom.features.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = await analyzer.analyze("I'm so excited about this!")

print(result)
# {
#   "label": "POSITIVE",
#   "score": 0.95,
#   "emotions": {"joy": 0.7, ...},
#   "confidence": 0.95
# }
```

## Configuration

```python
from HoloLoom.features.sentiment import SentimentConfig

config = SentimentConfig(
    model="distilbert-base-uncased-finetuned-sst-2-english",
    threshold=0.5,
    enabled=True
)
```

## Use Cases

- Detect frustrated users and offer extra help
- Adjust response tone based on user sentiment
- Track emotional patterns over time
- Filter emotionally charged queries
```

#### Step 6: Test & Commit

```bash
# Run tests
PYTHONPATH=. pytest HoloLoom/features/sentiment/test_sentiment.py -v

# If all pass, commit
git add HoloLoom/features/sentiment/
git add HoloLoom/Documentation/SENTIMENT_ANALYSIS.md
git commit -m "Add sentiment analysis feature extractor"
```

---

## Example: Adding a New Memory Backend

Let's add **Pinecone** as an alternative vector store.

#### Step 1: Create Backend Module

**`HoloLoom/memory/backends/pinecone_store.py`**:
```python
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    logger.warning("pinecone-client not installed")
    PINECONE_AVAILABLE = False

@dataclass
class PineconeConfig:
    """Pinecone configuration."""
    api_key: str
    environment: str  # e.g., "us-west1-gcp"
    index_name: str = "hololoom"
    dimension: int = 384
    metric: str = "cosine"
    pod_type: str = "p1.x1"

class PineconeStore:
    """
    Vector store using Pinecone.

    Advantages:
    - Fully managed (no infrastructure)
    - Auto-scaling
    - Low latency
    - Metadata filtering
    """

    def __init__(self, config: PineconeConfig):
        self.config = config
        self._index = None

        if not PINECONE_AVAILABLE:
            raise RuntimeError("pinecone-client not installed: pip install pinecone-client")

    async def connect(self):
        """Initialize Pinecone connection."""

        # Initialize Pinecone
        pinecone.init(
            api_key=self.config.api_key,
            environment=self.config.environment
        )

        # Create index if doesn't exist
        if self.config.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.config.index_name,
                dimension=self.config.dimension,
                metric=self.config.metric,
                pod_type=self.config.pod_type
            )
            logger.info(f"Created Pinecone index: {self.config.index_name}")

        # Connect to index
        self._index = pinecone.Index(self.config.index_name)
        logger.info(f"Connected to Pinecone index: {self.config.index_name}")

    async def upsert(
        self,
        vectors: List[tuple],  # [(id, vector, metadata), ...]
        namespace: str = ""
    ):
        """Upsert vectors to Pinecone."""

        if not self._index:
            raise RuntimeError("Not connected. Call connect() first.")

        # Pinecone expects format: (id, vector, metadata)
        self._index.upsert(vectors=vectors, namespace=namespace)

    async def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        namespace: str = ""
    ) -> List[Dict[str, Any]]:
        """Query similar vectors."""

        if not self._index:
            raise RuntimeError("Not connected. Call connect() first.")

        results = self._index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter,
            namespace=namespace
        )

        return [
            {
                "id": match["id"],
                "score": match["score"],
                "metadata": match.get("metadata", {})
            }
            for match in results["matches"]
        ]

    async def delete(self, ids: List[str], namespace: str = ""):
        """Delete vectors by IDs."""
        self._index.delete(ids=ids, namespace=namespace)

    async def close(self):
        """Close connection."""
        # Pinecone doesn't need explicit close
        pass
```

#### Step 2: Add Factory Function

**Update `HoloLoom/memory/__init__.py`**:
```python
from typing import Union
from .cache import MemoryManager
from .graph import KnowledgeGraph
from .backends.pinecone_store import PineconeStore, PineconeConfig

def create_vector_store(backend: str = "qdrant", **config):
    """
    Factory for vector stores.

    Args:
        backend: "qdrant", "pinecone", "weaviate", etc.
        **config: Backend-specific configuration

    Returns:
        Vector store instance
    """

    if backend == "qdrant":
        from .qdrant_store import QdrantStore, QdrantConfig
        return QdrantStore(QdrantConfig(**config))

    elif backend == "pinecone":
        return PineconeStore(PineconeConfig(**config))

    elif backend == "weaviate":
        # Future: add Weaviate support
        raise NotImplementedError("Weaviate backend coming soon")

    else:
        raise ValueError(f"Unknown backend: {backend}")
```

#### Step 3: Use in Orchestrator

```python
from HoloLoom.memory import create_vector_store

# Use Pinecone instead of Qdrant
vector_store = create_vector_store(
    backend="pinecone",
    api_key="your-api-key",
    environment="us-west1-gcp",
    index_name="hololoom"
)

await vector_store.connect()
```

---

## Example: Adding a New Tool

Let's add a **web search tool** using DuckDuckGo.

#### Step 1: Create Tool Module

**`HoloLoom/tools/web_search.py`**:
```python
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    logger.warning("duckduckgo_search not installed")
    DDGS_AVAILABLE = False

async def web_search(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt"  # worldwide
) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query
        max_results: Maximum number of results
        region: Region code (wt-wt for worldwide)

    Returns:
        Search results with titles, URLs, snippets
    """

    if not DDGS_AVAILABLE:
        return {
            "status": "error",
            "error": "duckduckgo_search not installed",
            "results": []
        }

    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, region=region, max_results=max_results))

        return {
            "status": "success",
            "query": query,
            "num_results": len(results),
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                }
                for r in results
            ]
        }

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "results": []
        }
```

#### Step 2: Register with MCP Server

**Update `HoloLoom/mcp/tools/__init__.py`**:
```python
from HoloLoom.mcp.server import MCPServer
from HoloLoom.tools.web_search import web_search

def register_default_tools(server: MCPServer):
    """Register default tools with MCP server."""

    # Existing tools
    # ...

    # Add web search
    @server.tool(
        name="web_search",
        description="Search the web for information"
    )
    async def search_web(
        query: str = {"description": "Search query", "required": True},
        max_results: int = {"description": "Max results", "default": 5}
    ):
        return await web_search(query, max_results)
```

#### Step 3: Add to Policy Tool List

**Update `HoloLoom/policy/unified.py`**:
```python
class NeuralCore(nn.Module):
    def __init__(self, ...):
        # Update tools list
        self.tools = [
            "calculator",
            "text_processor",
            "web_search",  # NEW
            # ... other tools
        ]

        # Update output dimension
        self.n_tools = len(self.tools)
```

---

## Extension Patterns

### Pattern 1: Plugin Architecture

Create a plugin system for third-party extensions.

**`HoloLoom/plugins/base.py`**:
```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class Plugin(ABC):
    """Base class for HoloLoom plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @abstractmethod
    async def initialize(self, orchestrator):
        """Initialize plugin with orchestrator."""
        pass

    @abstractmethod
    async def on_query(self, query: Query) -> Query:
        """Hook: Called before query processing."""
        pass

    @abstractmethod
    async def on_response(self, response: Response) -> Response:
        """Hook: Called after response generation."""
        pass
```

**Example Plugin**:
```python
from HoloLoom.plugins import Plugin

class LoggingPlugin(Plugin):
    """Log all queries and responses."""

    name = "logging"
    version = "1.0.0"

    async def initialize(self, orchestrator):
        self.logger = logging.getLogger("hololoom.plugins.logging")

    async def on_query(self, query):
        self.logger.info(f"Query: {query.text}")
        return query

    async def on_response(self, response):
        self.logger.info(f"Response: {response.text[:100]}...")
        return response

# Register plugin
orchestrator.register_plugin(LoggingPlugin())
```

### Pattern 2: Middleware System

Add middleware for cross-cutting concerns.

```python
class Middleware(ABC):
    """Middleware for request/response processing."""

    @abstractmethod
    async def process_request(self, query: Query) -> Query:
        """Process incoming query."""
        pass

    @abstractmethod
    async def process_response(self, response: Response) -> Response:
        """Process outgoing response."""
        pass

# Example: Rate limiting middleware
class RateLimitMiddleware(Middleware):
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}

    async def process_request(self, query):
        user_id = query.metadata.get("user_id", "anonymous")

        # Check rate limit
        if self._is_rate_limited(user_id):
            raise RateLimitExceeded(f"Too many requests")

        # Record request
        self._record_request(user_id)

        return query

    async def process_response(self, response):
        # Add rate limit headers
        response.headers["X-RateLimit-Remaining"] = str(self._get_remaining(user_id))
        return response
```

---

## Best Practices

### 1. **Graceful Degradation**

Always handle missing dependencies:

```python
try:
    import fancy_library
    FANCY_AVAILABLE = True
except ImportError:
    logger.warning("fancy_library not available, using fallback")
    FANCY_AVAILABLE = False

def my_function():
    if FANCY_AVAILABLE:
        return fancy_library.do_something()
    else:
        return simple_fallback()
```

### 2. **Configuration Management**

Use dataclasses for configuration:

```python
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class MyFeatureConfig:
    """Configuration for my feature."""

    # Required
    api_key: str

    # Optional with defaults
    enabled: bool = True
    timeout: int = 30

    # Complex defaults
    endpoints: List[str] = field(default_factory=lambda: ["https://api.example.com"])

    # Optional
    custom_model: Optional[str] = None
```

### 3. **Type Hints**

Always use type hints:

```python
from typing import List, Dict, Any, Optional, Union
import numpy as np

async def process_embeddings(
    embeddings: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    batch_size: int = 32
) -> List[Dict[str, Union[str, float]]]:
    """Process embeddings in batches."""
    pass
```

### 4. **Logging**

Use structured logging:

```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    logger.info("Processing query", extra={
        "query_id": query.id,
        "user_id": query.user_id,
        "duration": duration
    })
```

### 5. **Testing**

Write comprehensive tests:

```python
import pytest

@pytest.mark.asyncio
async def test_my_feature():
    """Test basic functionality."""
    result = await my_feature("input")
    assert result is not None
    assert result["status"] == "success"

@pytest.mark.asyncio
async def test_my_feature_error_handling():
    """Test error handling."""
    with pytest.raises(ValueError):
        await my_feature(invalid_input)

def test_my_feature_disabled():
    """Test with feature disabled."""
    config = MyConfig(enabled=False)
    # Should not crash
```

---

## Quick Reference

### Directory Structure for New Feature

```
HoloLoom/
└── my_feature/
    ├── __init__.py         # Public API exports
    ├── core.py             # Main implementation
    ├── config.py           # Configuration dataclass
    ├── utils.py            # Helper functions
    ├── test_my_feature.py  # Unit tests
    └── README.md           # Feature documentation
```

### Checklist for New Extension

- [ ] Create module structure
- [ ] Implement core functionality
- [ ] Add configuration
- [ ] Write unit tests (>80% coverage)
- [ ] Add integration tests
- [ ] Document API
- [ ] Add usage examples
- [ ] Handle errors gracefully
- [ ] Support graceful degradation
- [ ] Add type hints
- [ ] Add logging
- [ ] Benchmark performance
- [ ] Update orchestrator integration
- [ ] Update requirements.txt
- [ ] Commit with clear message

---

## Getting Help

- **Documentation**: Check existing modules for patterns
- **Tests**: Look at test files for usage examples
- **Community**: Ask in GitHub Discussions
- **Issues**: Report bugs or request features

---

## Summary

HoloLoom is designed for extensibility. You can:

1. **Add features**: Sentiment analysis, entity recognition, etc.
2. **Add backends**: New vector stores, databases, caches
3. **Add tools**: Web search, API clients, custom functions
4. **Add modalities**: 3D, medical imaging, sensor data
5. **Add plugins**: Custom hooks and middleware
6. **Add pipelines**: Workflows, multi-agent orchestration

Follow the patterns shown in this guide, and you'll be extending HoloLoom like a pro! 🚀
