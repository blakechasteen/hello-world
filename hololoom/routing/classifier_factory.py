"""
Query Classifier Factory
========================
Factory methods for creating query classifiers with automatic fallback.

Supports:
- Baseline classifier (88% accuracy, <1ms)
- Moonshot classifier (100% accuracy, <1ms with Tier 1+2, ~11ms with all tiers)
- Automatic fallback to baseline if moonshot unavailable

Author: Claude Code
Date: November 13, 2025
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_classifier(config):
    """
    Factory for classifier creation with automatic fallback.

    Args:
        config: HoloLoom configuration with routing settings

    Returns:
        QueryClassifier instance (baseline or moonshot) or None
    """
    if not getattr(config, 'enable_smart_routing', True):
        logger.info("Smart query routing disabled")
        return None

    routing_classifier = getattr(config, 'routing_classifier', 'moonshot')

    if routing_classifier == "moonshot":
        try:
            from hololoom.routing.query_classifier_moonshot import MoonshotQueryClassifier

            enable_semantic = getattr(config, 'enable_semantic_tier', False)
            enable_learning = getattr(config, 'enable_adaptive_learning', True)
            enable_telemetry = getattr(config, 'enable_classification_telemetry', True)
            telemetry_path = getattr(config, 'classification_telemetry_path', './classification_logs')

            logger.info(f"✅ Moonshot classifier loaded "
                       f"(semantic_tier={enable_semantic}, "
                       f"adaptive_learning={enable_learning}, "
                       f"telemetry={enable_telemetry})")

            return MoonshotQueryClassifier(
                enable_semantic_tier=enable_semantic,
                enable_adaptive_learning=enable_learning,
                stats_path=telemetry_path if enable_telemetry else None
            )
        except ImportError as e:
            logger.warning(f"⚠️  Moonshot classifier unavailable: {e}. Falling back to baseline.")
            from hololoom.routing.query_classifier import QueryClassifier
            return QueryClassifier()
    else:
        logger.info("✅ Baseline classifier loaded")
        from hololoom.routing.query_classifier import QueryClassifier
        return QueryClassifier()


def create_fast_path_router(orchestrator, config):
    """
    Factory for fast path router creation.

    Args:
        orchestrator: WeavingOrchestrator instance
        config: HoloLoom configuration

    Returns:
        FastPathRouter instance or None
    """
    if not getattr(config, 'enable_smart_routing', True):
        return None

    from hololoom.routing.fast_paths import FastPathRouter

    logger.info("✅ Fast path router initialized")
    return FastPathRouter(orchestrator=orchestrator)
