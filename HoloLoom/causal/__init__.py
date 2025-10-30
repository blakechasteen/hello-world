"""
HoloLoom Causal Reasoning Engine

Layer 1 of the Cognitive Architecture - Pearl-style causal inference.

Public API:
    CausalNode: Variable in causal graph
    CausalEdge: Causal relationship with strength
    CausalDAG: Directed acyclic graph for causal models
    CausalQuery: Query for causal inference
    InterventionEngine: do() operator implementation
    CounterfactualEngine: Twin network inference
"""

from .dag import CausalNode, CausalEdge, CausalDAG, NodeType
from .query import CausalQuery, QueryType, CausalAnswer
from .intervention import InterventionEngine
from .counterfactual import CounterfactualEngine
from .neural_scm import NeuralStructuralCausalModel, NeuralMechanism
from .discovery import CausalDiscovery, ActiveCausalLearner

__all__ = [
    'CausalNode',
    'CausalEdge',
    'CausalDAG',
    'NodeType',
    'CausalQuery',
    'QueryType',
    'CausalAnswer',
    'InterventionEngine',
    'CounterfactualEngine',
    'NeuralStructuralCausalModel',
    'NeuralMechanism',
    'CausalDiscovery',
    'ActiveCausalLearner'
]
