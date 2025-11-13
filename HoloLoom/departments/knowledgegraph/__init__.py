#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Graph Department - Graph construction, reasoning, evolution, and verification.

Exports:
- KnowledgeGraphDepartment: Main department class
- GraphConstructor: Build knowledge graphs from text
- GraphReasoner: Advanced graph reasoning
- GraphEvolver: Update graphs with new information
- GraphVerifier: Consistency checking
"""

from .knowledgegraph import (
    KnowledgeGraphDepartment,
    GraphConstructor,
    GraphReasoner,
    GraphEvolver,
    GraphVerifier,
    RelationType,
    EvolutionOperation,
    ConsistencyViolationType,
    Triple,
    GraphQuery,
    GraphPath,
    Subgraph,
    EvolutionRecord,
    ConsistencyViolation,
    VerificationReport
)

__all__ = [
    "KnowledgeGraphDepartment",
    "GraphConstructor",
    "GraphReasoner",
    "GraphEvolver",
    "GraphVerifier",
    "RelationType",
    "EvolutionOperation",
    "ConsistencyViolationType",
    "Triple",
    "GraphQuery",
    "GraphPath",
    "Subgraph",
    "EvolutionRecord",
    "ConsistencyViolation",
    "VerificationReport"
]
