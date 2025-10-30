"""
Explainability - Layer 5 of Cognitive Architecture

Makes AI decisions interpretable through multiple techniques:
- Feature Attribution: Which features drove the decision?
- Attention Visualization: What did the model focus on?
- Counterfactual Explanations: What if we changed X?
- Natural Language: Human-readable explanations
- Decision Trees: Extract interpretable rules
- Provenance: Full computational lineage

Research:
- Ribeiro et al. (2016): LIME - Local Interpretable Model-Agnostic Explanations
- Lundberg & Lee (2017): SHAP - Unified approach to feature attribution
- Wachter et al. (2017): Counterfactual explanations
- Selvaraju et al. (2017): Grad-CAM - Visual explanations
- Doshi-Velez & Kim (2017): Towards rigorous science of interpretability
"""

from .feature_attribution import (
    FeatureAttributor,
    AttributionMethod,
    FeatureImportance,
    ShapleyValues,
    LimeExplainer,
)

from .attention_explainer import (
    AttentionExplainer,
    AttentionPattern,
    AttentionHeatmap,
    visualize_attention,
)

from .counterfactual_generator import (
    CounterfactualGenerator,
    Counterfactual,
    MinimalEdit,
    find_counterfactuals,
)

from .natural_language import (
    NaturalLanguageExplainer,
    ExplanationType,
    generate_explanation,
    explain_decision,
)

from .decision_tree_extractor import (
    DecisionTreeExtractor,
    extract_rules,
    RuleSet,
    visualize_tree,
)

from .provenance_tracker import (
    ProvenanceTracker,
    ComputationalTrace,
    LineageGraph,
    trace_decision,
)

from .explainer import (
    UnifiedExplainer,
    Explanation,
    explain,
)

__all__ = [
    # Feature Attribution
    'FeatureAttributor',
    'AttributionMethod',
    'FeatureImportance',
    'ShapleyValues',
    'LimeExplainer',

    # Attention
    'AttentionExplainer',
    'AttentionPattern',
    'AttentionHeatmap',
    'visualize_attention',

    # Counterfactuals
    'CounterfactualGenerator',
    'Counterfactual',
    'MinimalEdit',
    'find_counterfactuals',

    # Natural Language
    'NaturalLanguageExplainer',
    'ExplanationType',
    'generate_explanation',
    'explain_decision',

    # Decision Trees
    'DecisionTreeExtractor',
    'extract_rules',
    'RuleSet',
    'visualize_tree',

    # Provenance
    'ProvenanceTracker',
    'ComputationalTrace',
    'LineageGraph',
    'trace_decision',

    # Unified API
    'UnifiedExplainer',
    'Explanation',
    'explain',
]
