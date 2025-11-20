#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Department - Advanced multi-hop reasoning and causal inference.

Exports:
- ReasoningDepartment: Main department class
- MultiHopReasoner: Multi-hop graph traversal
- CausalInferenceEngine: Causal relationship inference
- CounterfactualAnalyzer: "What if" scenario analysis
"""

from .reasoning import (
    ReasoningDepartment,
    MultiHopReasoner,
    CausalInferenceEngine,
    CounterfactualAnalyzer,
    ReasoningChain,
    CausalRelation,
    CounterfactualScenario
)

__all__ = [
    "ReasoningDepartment",
    "MultiHopReasoner",
    "CausalInferenceEngine",
    "CounterfactualAnalyzer",
    "ReasoningChain",
    "CausalRelation",
    "CounterfactualScenario"
]
