#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics Collection
==================

Comprehensive metrics collection for monitoring and observability.

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: lines 2016-2223, 2729-2797 (~168 lines)

This module handles:
- Reflection buffer metrics (success rates, tool recommendations)
- Recursive learning statistics (Thompson priors, policy weights, hot patterns)
- Production monitoring metrics (performance, resources, learning)

Author: Claude Code (Elegance Pass Refactoring - Phase 3)
Date: 2025-11-22
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator


logger = logging.getLogger(__name__)


def get_reflection_metrics(orchestrator: 'WeavingOrchestrator') -> Optional[Dict[str, Any]]:
    """
    Get reflection metrics if reflection is enabled.

    Args:
        orchestrator: The WeavingOrchestrator instance

    Returns:
        Dict with reflection metrics or None if reflection disabled

    Example:
        >>> metrics = get_reflection_metrics(orchestrator)
        >>> if metrics:
        ...     print(f"Success rate: {metrics['success_rate']:.1%}")
        ...     print(f"Tool recommendations: {metrics['tool_recommendations']}")
    """
    if not orchestrator.enable_reflection:
        return None

    metrics = orchestrator.reflection_buffer.get_metrics()
    return {
        'total_cycles': metrics.total_cycles,
        'success_rate': orchestrator.reflection_buffer.get_success_rate(),
        'tool_success_rates': metrics.tool_success_rates,
        'tool_recommendations': orchestrator.reflection_buffer.get_tool_recommendations(),
        'pattern_success_rates': metrics.pattern_success_rates
    }


def get_recursive_learning_stats(orchestrator: 'WeavingOrchestrator') -> Optional[Dict[str, Any]]:
    """
    Get comprehensive recursive learning statistics.

    Returns None if recursive learning is disabled.

    Args:
        orchestrator: The WeavingOrchestrator instance

    Returns:
        Dict with learning metrics, Thompson priors, policy weights, etc.

    Example:
        >>> stats = get_recursive_learning_stats(orchestrator)
        >>> if stats:
        ...     print(f"Queries processed: {stats['queries_processed']}")
        ...     print(f"Average confidence: {stats['avg_confidence']:.2f}")
        ...     for tool, priors in stats['thompson_priors'].items():
        ...         print(f"{tool}: Expected reward = {priors['expected_reward']:.2f}")
    """
    if not orchestrator.enable_recursive_learning or not orchestrator._recursive_components:
        return None

    components = orchestrator._recursive_components
    stats = {
        "enabled": True,
        "queries_processed": components['metrics'].queries_processed,
        "avg_confidence": components['metrics'].avg_confidence,
        "thompson_priors": {
            tool: {
                "expected_reward": components['thompson_priors'].get_expected_reward(tool),
                "uncertainty": components['thompson_priors'].get_uncertainty(tool),
                **priors
            }
            for tool, priors in components['thompson_priors'].tool_priors.items()
        },
        "policy_weights": {
            adapter: {
                "weight": components['policy_weights'].get_weight(adapter),
                "success_rate": components['policy_weights'].get_success_rate(adapter),
                "total_uses": components['policy_weights'].adapter_total[adapter]
            }
            for adapter in components['policy_weights'].adapter_weights.keys()
        },
    }

    # Add hot patterns if available
    if components.get('hot_tracker'):
        hot_patterns = components['hot_tracker'].get_hot_patterns(limit=10)
        stats["hot_patterns"] = [
            {
                "element_id": record.element_id,
                "heat_score": record.heat_score,
                "access_count": record.access_count,
                "success_rate": record.success_rate,
                "avg_confidence": record.avg_confidence
            }
            for record in hot_patterns
        ]

    # Add learned patterns if available
    if components.get('pattern_learner'):
        learned_patterns = components['pattern_learner'].get_hot_patterns()
        stats["learned_patterns"] = [
            {
                "motifs": pattern.motifs[:3],
                "tool": pattern.tool,
                "query_type": pattern.query_type,
                "occurrences": pattern.occurrences,
                "avg_confidence": pattern.avg_confidence
            }
            for pattern in learned_patterns[:10]
        ]

    return stats


def get_metrics(orchestrator: 'WeavingOrchestrator') -> Optional[Dict[str, Any]]:
    """
    Get production monitoring metrics.

    Returns performance, resource, and learning metrics for observability.

    Args:
        orchestrator: The WeavingOrchestrator instance

    Returns:
        Dict with metrics or None if production hardening disabled

    Example:
        >>> orch = WeavingOrchestrator(..., enable_production_hardening=True)
        >>> metrics = get_metrics(orch)
        >>> if metrics:
        ...     print(f"QPS: {metrics['performance']['qps']}")
        ...     print(f"Error rate: {metrics['performance']['error_rate']}")
        ...     print(f"P95 latency: {metrics['performance']['latency_p95']}ms")
    """
    if not orchestrator.enable_production_hardening or not orchestrator.monitor:
        logger.warning("[PRODUCTION] Monitoring not enabled")
        return None

    try:
        return {
            "performance": orchestrator.monitor.performance.get_metrics(),
            "resources": orchestrator.monitor.resources.get_metrics(),
            "learning": orchestrator.monitor.learning.get_metrics(),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"[PRODUCTION] Failed to get metrics: {e}")
        return {"error": str(e), "timestamp": time.time()}
