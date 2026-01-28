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
from .discovery import CausalDiscovery, ActiveCausalLearner
from .temporal import TemporalCausalDAG, TemporalEdge, TemporalState

# Conditionally import neural_scm (requires torch)
try:
    from .neural_scm import NeuralStructuralCausalModel, NeuralMechanism
    _NEURAL_AVAILABLE = True
except (ImportError, NameError):
    NeuralStructuralCausalModel = None
    NeuralMechanism = None
    _NEURAL_AVAILABLE = False

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
    'CausalDiscovery',
    'ActiveCausalLearner',
    'TemporalCausalDAG',
    'TemporalEdge',
    'TemporalState'
]

# Add neural components to __all__ only if available
if _NEURAL_AVAILABLE:
    __all__.extend(['NeuralStructuralCausalModel', 'NeuralMechanism'])
