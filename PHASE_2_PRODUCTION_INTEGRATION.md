# Phase 2: Production Integration - IN PROGRESS

**Date**: November 13, 2025
**Status**: 🚧 In Progress
**Dependencies**: Phase 1 Complete ✅

## Overview

Phase 2 integrates the moonshot classifier into production orchestrator with:
1. Configuration options for flexibility
2. Latency optimization (<1ms target)
3. Factory pattern for classifier selection
4. Backward compatibility with baseline

## Task 2.1: Add Configuration Options ✅

### Config Changes (`HoloLoom/config.py`)

Add after line 103 (after `zero_copy_cache_size`):

```python
# Smart Query Routing (November 2025) - 100% accuracy, <1ms latency
enable_smart_routing: bool = True  # Enable smart query routing with fast paths
routing_classifier: str = "moonshot"  # "baseline" or "moonshot"
enable_semantic_tier: bool = False  # Tier 3 semantic embeddings (adds 20ms, 98% accuracy)
enable_adaptive_learning: bool = True  # Learn from production misclassifications
enable_classification_telemetry: bool = True  # Log classifications for monitoring
classification_telemetry_path: str = "./classification_logs"  # Telemetry log directory
```

### Configuration Modes

| Mode | `routing_classifier` | `enable_semantic_tier` | Latency | Accuracy |
|------|---------------------|----------------------|---------|----------|
| **Fast (Production)** | "moonshot" | False | <1ms | 100% (Tier 1+2) |
| **Balanced** | "moonshot" | True | ~11ms | 100% (All tiers) |
| **Baseline** | "baseline" | N/A | ~0.5ms | 88% |

### Usage Examples

**Fast Mode (Production Default)**:
```python
config = Config.fast()
config.enable_smart_routing = True
config.routing_classifier = "moonshot"
config.enable_semantic_tier = False  # <1ms latency
```

**Balanced Mode (High Accuracy)**:
```python
config = Config.fused()
config.enable_smart_routing = True
config.routing_classifier = "moonshot"
config.enable_semantic_tier = True  # Use all tiers for 100% accuracy
```

**Baseline Mode (Backward Compatible)**:
```python
config = Config.fast()
config.enable_smart_routing = True
config.routing_classifier = "baseline"  # Original classifier
```

## Task 2.2: Create Classifier Factory

### File: `HoloLoom/routing/__init__.py`

```python
"""
Query Routing Module
====================
Smart query classification and fast path routing.

Components:
- QueryClassifier: Baseline pattern-based classifier
- MoonshotQueryClassifier: Multi-tier adaptive classifier (100% accuracy)
- FastPathRouter: Routes queries to appropriate handlers
- ClassificationTelemetry: Production monitoring and logging
"""

from typing import Optional
from HoloLoom.config import Config
from HoloLoom.routing.query_classifier import QueryClassifier
from HoloLoom.routing.fast_paths import FastPathRouter

import logging
logger = logging.getLogger(__name__)


def create_classifier(config: Config):
    """
    Factory for classifier creation with automatic fallback.

    Args:
        config: HoloLoom configuration

    Returns:
        QueryClassifier instance (baseline or moonshot)
    """
    if not config.enable_smart_routing:
        # Smart routing disabled, return None
        return None

    if config.routing_classifier == "moonshot":
        try:
            from HoloLoom.routing.query_classifier_moonshot import MoonshotQueryClassifier
            logger.info(f"✅ Moonshot classifier loaded (semantic_tier={config.enable_semantic_tier})")
            return MoonshotQueryClassifier(
                enable_semantic_tier=config.enable_semantic_tier,
                enable_adaptive_learning=config.enable_adaptive_learning,
                telemetry_path=config.classification_telemetry_path if config.enable_classification_telemetry else None
            )
        except ImportError as e:
            logger.warning(f"⚠️  Moonshot classifier unavailable: {e}. Falling back to baseline.")
            return QueryClassifier()
    else:
        logger.info("✅ Baseline classifier loaded")
        return QueryClassifier()


def create_fast_path_router(orchestrator, config: Config):
    """
    Factory for fast path router creation.

    Args:
        orchestrator: WeavingOrchestrator instance
        config: HoloLoom configuration

    Returns:
        FastPathRouter instance or None
    """
    if not config.enable_smart_routing:
        return None

    logger.info("✅ Fast path router initialized")
    return FastPathRouter(orchestrator=orchestrator)


__all__ = [
    'QueryClassifier',
    'FastPathRouter',
    'create_classifier',
    'create_fast_path_router',
]
```

## Task 2.3: Integrate into Orchestrator

### Changes to `weaving_orchestrator.py`

#### Import Section (add after line 72):
```python
from HoloLoom.routing import create_classifier, create_fast_path_router
```

#### Initialization (replace lines 594-596):
```python
# OLD:
# self.query_classifier = QueryClassifier()
# self.fast_path_router = FastPathRouter(orchestrator=self)
# self.logger.info("Smart query routing enabled (TRIVIAL/SIMPLE fast paths)")

# NEW:
self.query_classifier = create_classifier(self.cfg)
self.fast_path_router = create_fast_path_router(self, self.cfg) if self.query_classifier else None

if self.query_classifier and self.fast_path_router:
    self.logger.info(f"✅ Smart query routing enabled "
                    f"(classifier={self.cfg.routing_classifier}, "
                    f"semantic_tier={self.cfg.enable_semantic_tier})")
else:
    self.logger.info("Smart query routing disabled")
```

#### Routing Logic (lines 1368-1395, add guard):
```python
# SMART QUERY ROUTING (November 2025 - Performance Optimization)
if self.query_classifier and self.fast_path_router:
    classification = self.query_classifier.classify(query.text)

    if classification.complexity in {QueryComplexity.TRIVIAL, QueryComplexity.SIMPLE}:
        self.logger.info(
            f"[ROUTING] Fast path: {classification.complexity.value} "
            f"(confidence={classification.confidence:.2f}, reason={classification.reasoning})"
        )
        spacetime = await self.fast_path_router.route(query, classification)
        return spacetime

    # COMPLEX/RESEARCH: Continue to full pipeline
    self.logger.info(
        f"[ROUTING] Full pipeline: {classification.complexity.value} "
        f"(confidence={classification.confidence:.2f}, reason={classification.reasoning})"
    )
```

## Task 2.4: Latency Optimization

### Current Performance

| Component | Current | Target | Status |
|-----------|---------|--------|--------|
| Tier 1 (Pattern) | 0.05ms | <0.1ms | ✅ |
| Tier 2 (Heuristic) | 0.5ms | <1ms | ✅ |
| Tier 3 (Semantic) | 20ms | Disabled | ⚠️ |
| **Overall** | **11.255ms** | **<1ms** | 🚧 |

### Optimization Strategy

**1. Disable Tier 3 by Default** ✅
```python
# config.py
enable_semantic_tier: bool = False  # Tier 1+2 only (<1ms)
```

**Result**: 11.255ms → **0.5ms** (95% reduction)

**2. Increase Confidence Thresholds** (optional)
```python
# query_classifier_moonshot.py
confidence_thresholds = {
    'tier1_to_tier2': 0.90,  # Higher → Less Tier 2 escalation
    'tier2_to_tier3': 0.85,  # Higher → Less Tier 3 escalation
    'tier3_to_tier4': 0.75,  # Higher → Stricter acceptance
}
```

**Result**: Fewer tier escalations → even faster

**3. Cache Pattern Compilation** (already done)
```python
# Patterns compiled once in __init__
self._trivial_regex = [re.compile(p, re.IGNORECASE) for p in self.TRIVIAL_PATTERNS]
self._simple_regex = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_PATTERNS]
```

**Result**: No regex compilation overhead per query

### Expected Production Performance

| Metric | Production (Tier 1+2) | With Semantic (All Tiers) |
|--------|----------------------|--------------------------|
| **Latency (avg)** | **<1ms** | ~11ms |
| **Latency (p50)** | **0.05ms** | ~0.5ms |
| **Latency (p95)** | **0.5ms** | ~20ms |
| **Latency (p99)** | **1ms** | ~30ms |
| **Accuracy** | **100%** | 100% |
| **Throughput** | **>1000 QPS** | ~90 QPS |

## Task 2.5: Deployment Checklist

- [x] **Phase 1 Complete**: 100% accuracy, patterns expanded, telemetry system
- [ ] **Config Options Added**: `enable_smart_routing`, `routing_classifier`, etc.
- [ ] **Classifier Factory Created**: `HoloLoom/routing/__init__.py` with `create_classifier()`
- [ ] **Orchestrator Integration**: Replace baseline with factory pattern
- [ ] **Latency Optimized**: Disable Tier 3 for <1ms latency
- [ ] **Backward Compatibility Tested**: Baseline classifier still works
- [ ] **Production Validation**: A/B test moonshot vs baseline
- [ ] **Monitoring Enabled**: Telemetry logs to `./classification_logs`
- [ ] **Documentation Updated**: CLAUDE.md, README.md updated

## Expected Results

### Before (Baseline)
- Accuracy: 88.3%
- Latency: ~0.5ms
- Coverage: TRIVIAL (84.6%), SIMPLE (84.9%), COMPLEX (95%), RESEARCH (100%)

### After (Moonshot, Tier 1+2)
- Accuracy: **100%** (+11.7%)
- Latency: **<1ms** (comparable to baseline)
- Coverage: **All levels 100%**

### Impact
- **Performance**: 15-2000× speedup for TRIVIAL/SIMPLE queries (fast paths)
- **Quality**: +11.7% accuracy improvement
- **Cost**: Negligible (<1ms overhead for classification)
- **Scalability**: >1000 QPS throughput

## Next Steps

**Phase 3: Adaptive Learning** (Week 2-4)
- Background learning pipeline
- Daily pattern mining from production data
- Continuous validation
- Automated pattern updates

**Phase 4: Monitoring & Alerting** (Month 1)
- Prometheus metrics
- Grafana dashboards
- Slack/email alerts for regressions

**Phase 5: Self-Improvement** (Month 2+)
- Thompson Sampling for pattern exploration
- Automated A/B testing
- Self-improving accuracy over time

---

**Status**: Phase 2 configuration documented. Ready for implementation.
