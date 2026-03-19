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

from .counterfactual import CounterfactualEngine
from .dag import CausalDAG, CausalEdge, CausalNode, NodeType
from .discovery import ActiveCausalLearner, CausalDiscovery
from .intervention import InterventionEngine
from .neural_scm import NeuralMechanism, NeuralStructuralCausalModel
from .query import CausalAnswer, CausalQuery, QueryType
from .temporal import TemporalCausalDAG, TemporalEdge, TemporalState

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
    'ActiveCausalLearner',
    'TemporalCausalDAG',
    'TemporalEdge',
    'TemporalState'
]
