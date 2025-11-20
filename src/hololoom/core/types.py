#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Core Types
===================

Shared types used across core modules and patterns.

Created: 2025-01-20
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class WeaveResult:
    """
    Standard output of a single Shuttle "weave" operation.

    This is the shared type that:
    - Core modules (Shuttle) produce
    - Patterns observe and learn from
    - Runners orchestrate

    Design Philosophy:
    - Immutable (frozen after creation)
    - Self-contained (includes all context needed for learning)
    - Minimal coupling (doesn't depend on internal state)
    """

    query: str
    """The original query/question that triggered this weave."""

    context_blocks: Dict[str, Any]
    """
    Structured context retrieved/generated during weaving.

    Common keys:
    - 'fuzzy_evidence': Text chunks from warp search
    - 'structural_claims': Graph relationships from yarn
    - 'semantic_projection': Semantic calculus features
    """

    metadata: Dict[str, Any] = field(default_factory=dict)
    """
    Metadata about the weaving process.

    Common keys:
    - 'policy': Policy name used (e.g., 'mcts', 'thompson')
    - 'reward': Scalar reward signal (0.0-1.0)
    - 'confidence': Result confidence (0.0-1.0)
    - 'num_warp_chunks': Number of warp chunks retrieved
    - 'num_graph_nodes': Number of graph nodes involved
    - 'execution_time_ms': Time taken for weave
    - 'complexity': Query complexity (TRIVIAL, SIMPLE, COMPLEX, RESEARCH)
    """

    def get_reward(self, default: float = 0.0) -> float:
        """
        Extract reward from metadata, with safe default.

        Returns:
            Scalar reward (0.0-1.0), or default if not present
        """
        reward = self.metadata.get('reward', default)
        try:
            return float(reward)
        except (TypeError, ValueError):
            return default

    def get_confidence(self, default: float = 0.5) -> float:
        """
        Extract confidence from metadata, with safe default.

        Returns:
            Scalar confidence (0.0-1.0), or default if not present
        """
        confidence = self.metadata.get('confidence', default)
        try:
            return float(confidence)
        except (TypeError, ValueError):
            return default


__all__ = ['WeaveResult']
