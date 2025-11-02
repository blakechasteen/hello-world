"""
HoloLoom Interpretability Framework
===================================
Explainability and transparency for all system decisions.

Modules:
- causal_explainer: Why did the system make this decision?
- feature_attribution: Which features influenced the decision?
- counterfactual: What would change the decision?
- semantic_tracing: How did the system process the query?
"""

from .causal_explainer import (
    CausalExplainer,
    Explanation,
    ExplanationType,
)
from .feature_attribution import (
    FeatureAttributor,
    Attribution,
    AttributionMethod,
)
from .counterfactual import (
    CounterfactualGenerator,
    Counterfactual,
)
from .semantic_tracing import (
    SemanticTracer,
    ProcessingTrace,
)

__all__ = [
    "CausalExplainer",
    "Explanation",
    "ExplanationType",
    "FeatureAttributor",
    "Attribution",
    "AttributionMethod",
    "CounterfactualGenerator",
    "Counterfactual",
    "SemanticTracer",
    "ProcessingTrace",
]