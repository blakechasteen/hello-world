"""
NeuroHood Causal Intelligence System

Leverages HoloLoom's causal infrastructure (~2,500 lines) for:
- Neural Structural Causal Models (relationship dynamics)
- Intervention reasoning ("what if?")
- Counterfactual analysis ("what would have happened?")
- Blame attribution (PN/PS scoring)
- Temporal dynamics (relationship trajectories)

This enables residents and players to understand WHY things happen
and explore alternative timelines.
"""

from .relationship_scm import (
    RelationshipSCM,
    RelationshipCausalDAG,
    extract_relationship_data,
    simulate_relationship_outcome,
)

from .interventions import (
    NeuroHoodInterventionEngine,
    InterventionResult,
    parse_natural_language_intervention,
)

from .counterfactuals import (
    NeuroHoodCounterfactualEngine,
    CounterfactualResult,
    parse_counterfactual_query,
)

from .blame import (
    BlameAttributionSystem,
    BlameReport,
    BlameScore,
    quick_blame_attribution,
)

__all__ = [
    'RelationshipSCM',
    'RelationshipCausalDAG',
    'extract_relationship_data',
    'simulate_relationship_outcome',
    'NeuroHoodInterventionEngine',
    'InterventionResult',
    'parse_natural_language_intervention',
    'NeuroHoodCounterfactualEngine',
    'CounterfactualResult',
    'parse_counterfactual_query',
    'BlameAttributionSystem',
    'BlameReport',
    'BlameScore',
    'quick_blame_attribution',
]
