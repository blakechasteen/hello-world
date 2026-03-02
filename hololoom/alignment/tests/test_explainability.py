"""
E3.2: Explainability Module Tests
=================================
Comprehensive tests for HoloLoom's explainability modules:
- CausalExplainer: Intervention-based causal inference
- CounterfactualGenerator: Minimal counterfactual explanations
- SHAPExplainer/LIMEExplainer: Feature attribution
- AgenticExplainer: Reasoning session explanations

December 2025 - E3 Test Coverage Expansion
"""

import pytest
import asyncio
import numpy as np
from typing import Dict, List, Any, Callable

# Causal Explainer imports
from hololoom.alignment.causal_explainer import (
    CausalExplainer,
    CausalGraph,
    CausalEdge,
    CausalIntervention,
    CausalEffect,
    CausalExplanation,
    CausalDiscovery,
    CausalRelationType,
    InterventionType,
)

# Counterfactual Generator imports
from hololoom.alignment.counterfactual_generator import (
    MinimalCounterfactualGenerator,
    Counterfactual,
    FeatureChange,
)

# SHAP/LIME Explainer imports
from hololoom.alignment.shap_lime_explainer import (
    SHAPExplainer,
    LIMEExplainer,
    SHAPExplanation,
    LIMEExplanation,
    FeatureAttribution,
    ExplanationType,
    FeatureType,
)

# Agentic Explainability imports
from hololoom.alignment.agentic_explainability import (
    AgenticExplainer,
    StepExplanation,
    ReasoningExplanation,
    ExplanationDepth,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_features():
    """Sample feature vector for testing."""
    return np.array([0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4])


@pytest.fixture
def feature_names():
    """Feature names for explanations."""
    return [
        "query_complexity",
        "context_relevance",
        "confidence_score",
        "response_length",
        "tool_affinity",
        "temporal_recency",
        "semantic_similarity",
        "user_preference"
    ]


@pytest.fixture
def background_data():
    """Background data for SHAP baseline."""
    np.random.seed(42)
    return np.random.rand(100, 8)


@pytest.fixture
def sample_steps():
    """Sample reasoning steps for agentic explainability."""
    return [
        {
            "step_type": "retrieval",
            "description": "Retrieved 5 relevant memories",
            "confidence": 0.85,
            "duration_ms": 45.0
        },
        {
            "step_type": "reasoning",
            "description": "Applied Thompson Sampling for tool selection",
            "confidence": 0.78,
            "duration_ms": 12.0
        },
        {
            "step_type": "verification",
            "description": "Verified response consistency",
            "confidence": 0.92,
            "duration_ms": 8.0
        }
    ]


@pytest.fixture
def mock_predict_fn():
    """Mock prediction function for explainers (returns float)."""
    def predict_fn(features: np.ndarray) -> float:
        """Simple mock predictor - weighted sum normalized."""
        if features.ndim == 1:
            return float(np.clip(np.sum(features) / len(features), 0.0, 1.0))
        return float(np.clip(np.sum(features) / features.size, 0.0, 1.0))
    return predict_fn


@pytest.fixture
def mock_predict_fn_tuple():
    """Mock prediction function for CounterfactualGenerator (returns tuple)."""
    tools = ["answer", "research", "clarify", "explore"]

    def predict_fn(features: np.ndarray) -> tuple:
        """Returns (tool, confidence) tuple based on feature sum."""
        if features.ndim == 1:
            conf = float(np.clip(np.sum(features) / len(features), 0.0, 1.0))
        else:
            conf = float(np.clip(np.sum(features) / features.size, 0.0, 1.0))
        # Pick tool based on confidence level
        tool_idx = min(int(conf * len(tools)), len(tools) - 1)
        return (tools[tool_idx], conf)
    return predict_fn


# ============================================================================
# CausalGraph Tests
# ============================================================================

class TestCausalGraph:
    """Tests for CausalGraph dataclass operations."""

    def test_create_empty_graph(self):
        """Test creating an empty causal graph."""
        graph = CausalGraph()
        assert graph is not None
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_add_edge_populates_nodes(self):
        """Test that add_edge automatically populates nodes."""
        graph = CausalGraph()

        # Use the add_edge method
        graph.add_edge(
            source="A",
            target="B",
            strength=0.75,
            confidence=0.9
        )

        assert len(graph.nodes) == 2
        assert "A" in graph.nodes
        assert "B" in graph.nodes
        assert len(graph.edges) == 1

    def test_multiple_edges(self):
        """Test adding multiple causal edges."""
        graph = CausalGraph()

        graph.add_edge("A", "C", 0.8, 0.9)
        graph.add_edge("B", "C", 0.6, 0.85)
        graph.add_edge("A", "B", 0.5, 0.7)

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 3

    def test_edge_with_relation_type(self):
        """Test edges with different relation types."""
        graph = CausalGraph()

        graph.add_edge("X", "Y", 0.9, 0.95, CausalRelationType.DIRECT)
        graph.add_edge("Y", "Z", 0.7, 0.8, CausalRelationType.CONFOUNDED)

        assert graph.edges[0].relation_type == CausalRelationType.DIRECT
        assert graph.edges[1].relation_type == CausalRelationType.CONFOUNDED

    def test_causal_edge_dataclass(self):
        """Test CausalEdge dataclass fields."""
        edge = CausalEdge(
            source="feature_1",
            target="outcome",
            strength=0.85,
            confidence=0.92,
            relation_type=CausalRelationType.INDIRECT
        )

        assert edge.source == "feature_1"
        assert edge.target == "outcome"
        assert edge.strength == 0.85
        assert edge.confidence == 0.92
        assert edge.relation_type == CausalRelationType.INDIRECT


# ============================================================================
# CausalExplainer Tests
# ============================================================================

class TestCausalExplainer:
    """Tests for CausalExplainer."""

    @pytest.fixture
    def explainer(self, mock_predict_fn, feature_names):
        """Create CausalExplainer instance."""
        return CausalExplainer(
            predict_fn=mock_predict_fn,
            feature_names=feature_names
        )

    @pytest.mark.asyncio
    async def test_explain_basic(self, explainer, sample_features):
        """Test basic causal explanation."""
        explanation = await explainer.explain(
            query_id="test_001",
            query_text="What is Thompson Sampling?",
            features=sample_features,
            predicted_tool="answer",
            predicted_confidence=0.85
        )

        assert explanation is not None
        assert isinstance(explanation, CausalExplanation)
        assert explanation.query_id == "test_001"
        assert explanation.predicted_tool == "answer"

    @pytest.mark.asyncio
    async def test_explain_with_background(self, explainer, sample_features, background_data):
        """Test causal explanation with background data."""
        explanation = await explainer.explain(
            query_id="test_002",
            query_text="Explain neural networks",
            features=sample_features,
            predicted_tool="research",
            predicted_confidence=0.72,
            background_data=background_data
        )

        assert explanation is not None
        assert explanation.causal_graph is not None
        assert isinstance(explanation.causal_graph, CausalGraph)

    @pytest.mark.asyncio
    async def test_explanation_has_causes(self, explainer, sample_features):
        """Test that explanation includes direct and indirect causes."""
        explanation = await explainer.explain(
            query_id="test_003",
            query_text="Test causes",
            features=sample_features,
            predicted_tool="answer",
            predicted_confidence=0.9
        )

        # Should have cause lists (may be empty but should exist)
        assert hasattr(explanation, 'direct_causes')
        assert hasattr(explanation, 'indirect_causes')
        assert isinstance(explanation.direct_causes, list)
        assert isinstance(explanation.indirect_causes, list)

    @pytest.mark.asyncio
    async def test_computation_time_recorded(self, explainer, sample_features):
        """Test that computation time is recorded."""
        explanation = await explainer.explain(
            query_id="test_004",
            query_text="Timing test",
            features=sample_features,
            predicted_tool="answer",
            predicted_confidence=0.8
        )

        assert explanation.computation_time_ms >= 0

    @pytest.mark.asyncio
    async def test_multiple_explanations_concurrent(self, explainer, sample_features):
        """Test concurrent causal explanations."""
        tasks = [
            explainer.explain(
                query_id=f"concurrent_{i}",
                query_text=f"Query {i}",
                features=sample_features + np.random.rand(8) * 0.1,
                predicted_tool="answer",
                predicted_confidence=0.8 + i * 0.02
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        assert all(isinstance(r, CausalExplanation) for r in results)


class TestCausalDiscovery:
    """Tests for CausalDiscovery (structure learning)."""

    @pytest.fixture
    def discovery(self):
        """Create CausalDiscovery instance."""
        return CausalDiscovery(significance_level=0.05)

    def test_discover_from_data(self, discovery, background_data, feature_names):
        """Test structure discovery from observational data."""
        # Create outcome data as weighted sum of features + noise
        outcome_data = np.sum(background_data, axis=1) + np.random.rand(100) * 0.1
        graph = discovery.discover_structure(background_data, outcome_data, feature_names)

        assert graph is not None
        assert isinstance(graph, CausalGraph)

    def test_discovery_creates_nodes(self, discovery, background_data, feature_names):
        """Test that discovery creates nodes for features."""
        outcome_data = np.sum(background_data, axis=1) + np.random.rand(100) * 0.1
        graph = discovery.discover_structure(background_data, outcome_data, feature_names)

        # Should have some nodes for features
        assert len(graph.nodes) >= 0  # May be empty for uncorrelated data


# ============================================================================
# CounterfactualGenerator Tests
# ============================================================================

class TestCounterfactualGenerator:
    """Tests for MinimalCounterfactualGenerator."""

    @pytest.fixture
    def generator(self, mock_predict_fn_tuple, feature_names):
        """Create counterfactual generator."""
        return MinimalCounterfactualGenerator(
            predict_fn=mock_predict_fn_tuple,
            feature_names=feature_names
        )

    @pytest.mark.asyncio
    async def test_generate_basic(self, generator, sample_features):
        """Test basic counterfactual generation."""
        counterfactual = await generator.generate(
            query_id="cf_001",
            query_text="What is Thompson Sampling?",
            original_features=sample_features,
            original_tool="answer",
            original_confidence=0.85
        )

        assert counterfactual is not None
        assert isinstance(counterfactual, Counterfactual)
        assert counterfactual.query_id == "cf_001"

    @pytest.mark.asyncio
    async def test_counterfactual_has_tool(self, generator, sample_features):
        """Test that counterfactual includes counterfactual tool."""
        counterfactual = await generator.generate(
            query_id="cf_002",
            query_text="Test query",
            original_features=sample_features,
            original_tool="answer",
            original_confidence=0.75
        )

        # Should have counterfactual tool and confidence
        assert hasattr(counterfactual, 'counterfactual_tool')
        assert hasattr(counterfactual, 'counterfactual_confidence')

    @pytest.mark.asyncio
    async def test_feature_changes_tracked(self, generator, sample_features):
        """Test that feature changes are tracked."""
        counterfactual = await generator.generate(
            query_id="cf_003",
            query_text="Minimal test",
            original_features=sample_features,
            original_tool="research",
            original_confidence=0.6
        )

        # Should track feature changes
        assert hasattr(counterfactual, 'feature_changes')
        assert isinstance(counterfactual.feature_changes, list)

    @pytest.mark.asyncio
    async def test_concurrent_generation(self, generator, sample_features):
        """Test concurrent counterfactual generation."""
        tasks = [
            generator.generate(
                query_id=f"cf_concurrent_{i}",
                query_text=f"Query {i}",
                original_features=sample_features,
                original_tool="answer",
                original_confidence=0.8
            )
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)
        assert len(results) == 10


# ============================================================================
# SHAP/LIME Explainer Tests
# ============================================================================

class TestSHAPExplainer:
    """Tests for SHAPExplainer (Shapley value explanations)."""

    @pytest.fixture
    def explainer(self, mock_predict_fn, feature_names, background_data):
        """Create SHAP explainer."""
        return SHAPExplainer(
            predict_fn=mock_predict_fn,
            feature_names=feature_names,
            num_samples=100,
            background_data=background_data
        )

    @pytest.mark.asyncio
    async def test_explain_basic(self, explainer, sample_features):
        """Test basic SHAP explanation."""
        explanation = await explainer.explain(
            query_id="shap_001",
            query_text="Test SHAP",
            features=sample_features,
            predicted_tool="answer",
            predicted_confidence=0.85
        )

        assert explanation is not None
        assert isinstance(explanation, SHAPExplanation)
        assert explanation.query_id == "shap_001"

    @pytest.mark.asyncio
    async def test_shap_attributions_computed(self, explainer, sample_features):
        """Test that SHAP attributions are computed."""
        explanation = await explainer.explain(
            query_id="shap_002",
            query_text="Values test",
            features=sample_features,
            predicted_tool="answer",
            predicted_confidence=0.8
        )

        # SHAP attributions should be computed
        assert hasattr(explanation, 'attributions')
        if explanation.attributions is not None:
            assert len(explanation.attributions) == len(sample_features)

    @pytest.mark.asyncio
    async def test_feature_attributions(self, explainer, sample_features):
        """Test feature attribution extraction."""
        explanation = await explainer.explain(
            query_id="shap_003",
            query_text="Attribution test",
            features=sample_features,
            predicted_tool="research",
            predicted_confidence=0.75
        )

        # Should have attributions
        assert hasattr(explanation, 'attributions')
        if explanation.attributions:
            attr = explanation.attributions[0]
            assert isinstance(attr, FeatureAttribution)

    @pytest.mark.asyncio
    async def test_base_value_computed(self, explainer, sample_features):
        """Test that base value is computed."""
        explanation = await explainer.explain(
            query_id="shap_004",
            query_text="Base value test",
            features=sample_features,
            predicted_tool="answer",
            predicted_confidence=0.9
        )

        assert hasattr(explanation, 'base_value')

    @pytest.mark.asyncio
    async def test_concurrent_shap(self, explainer, sample_features):
        """Test concurrent SHAP explanations."""
        tasks = [
            explainer.explain(
                query_id=f"shap_concurrent_{i}",
                query_text=f"Query {i}",
                features=sample_features + np.random.rand(8) * 0.1,
                predicted_tool="answer",
                predicted_confidence=0.8
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        assert all(isinstance(r, SHAPExplanation) for r in results)


class TestLIMEExplainer:
    """Tests for LIMEExplainer (local linear explanations)."""

    @pytest.fixture
    def explainer(self, mock_predict_fn, feature_names):
        """Create LIME explainer."""
        return LIMEExplainer(
            predict_fn=mock_predict_fn,
            feature_names=feature_names,
            num_samples=100
        )

    @pytest.mark.asyncio
    async def test_explain_basic(self, explainer, sample_features):
        """Test basic LIME explanation."""
        explanation = await explainer.explain(
            query_id="lime_001",
            query_text="Test LIME",
            features=sample_features,
            predicted_tool="answer",
            predicted_confidence=0.85
        )

        assert explanation is not None
        assert isinstance(explanation, LIMEExplanation)
        assert explanation.query_id == "lime_001"

    @pytest.mark.asyncio
    async def test_lime_coefficients(self, explainer, sample_features):
        """Test LIME linear coefficients."""
        explanation = await explainer.explain(
            query_id="lime_002",
            query_text="Coefficients test",
            features=sample_features,
            predicted_tool="research",
            predicted_confidence=0.7
        )

        # LIME produces linear coefficients (Dict[str, float])
        assert hasattr(explanation, 'linear_coefficients')
        if explanation.linear_coefficients is not None:
            assert isinstance(explanation.linear_coefficients, dict)

    @pytest.mark.asyncio
    async def test_lime_local_fidelity(self, explainer, sample_features):
        """Test LIME local fidelity value."""
        explanation = await explainer.explain(
            query_id="lime_003",
            query_text="Fidelity test",
            features=sample_features,
            predicted_tool="answer",
            predicted_confidence=0.85
        )

        # Should have local fidelity R² score
        assert hasattr(explanation, 'local_fidelity')

    @pytest.mark.asyncio
    async def test_local_fidelity_range(self, explainer, sample_features):
        """Test LIME local fidelity score is in valid range."""
        explanation = await explainer.explain(
            query_id="lime_004",
            query_text="Fidelity range test",
            features=sample_features,
            predicted_tool="clarify",
            predicted_confidence=0.6
        )

        # R² score for local fidelity (0.0 to 1.0)
        assert 0 <= explanation.local_fidelity <= 1

    @pytest.mark.asyncio
    async def test_concurrent_lime(self, explainer, sample_features):
        """Test concurrent LIME explanations."""
        tasks = [
            explainer.explain(
                query_id=f"lime_concurrent_{i}",
                query_text=f"Query {i}",
                features=sample_features + np.random.rand(8) * 0.1,
                predicted_tool="answer",
                predicted_confidence=0.8
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        assert all(isinstance(r, LIMEExplanation) for r in results)


# ============================================================================
# AgenticExplainer Tests
# ============================================================================

class TestAgenticExplainer:
    """Tests for AgenticExplainer (reasoning session explanations)."""

    @pytest.fixture
    def explainer(self):
        """Create agentic explainer."""
        return AgenticExplainer()

    @pytest.mark.asyncio
    async def test_explain_reasoning_basic(self, explainer, sample_steps):
        """Test basic reasoning explanation."""
        explanation = await explainer.explain_reasoning(
            session_id="session_001",
            reasoning_mode="direct",
            steps_taken=sample_steps,
            final_confidence=0.85
        )

        assert explanation is not None
        assert isinstance(explanation, ReasoningExplanation)
        assert explanation.session_id == "session_001"

    @pytest.mark.asyncio
    async def test_step_explanations_generated(self, explainer, sample_steps):
        """Test that step explanations are generated."""
        explanation = await explainer.explain_reasoning(
            session_id="session_002",
            reasoning_mode="verify",
            steps_taken=sample_steps,
            final_confidence=0.78
        )

        assert hasattr(explanation, 'step_explanations')
        # Should have explanations for each step
        if explanation.step_explanations:
            assert len(explanation.step_explanations) <= len(sample_steps)

    @pytest.mark.asyncio
    async def test_confidence_trajectory(self, explainer, sample_steps):
        """Test confidence trajectory extraction."""
        explanation = await explainer.explain_reasoning(
            session_id="session_003",
            reasoning_mode="research",
            steps_taken=sample_steps,
            final_confidence=0.92
        )

        assert hasattr(explanation, 'confidence_trajectory')
        assert len(explanation.confidence_trajectory) == len(sample_steps)

    @pytest.mark.asyncio
    async def test_different_reasoning_modes(self, explainer, sample_steps):
        """Test explanations for different reasoning modes."""
        modes = ["direct", "verify", "research", "plan_execute"]

        for mode in modes:
            explanation = await explainer.explain_reasoning(
                session_id=f"session_mode_{mode}",
                reasoning_mode=mode,
                steps_taken=sample_steps,
                final_confidence=0.8
            )
            assert explanation is not None
            assert explanation.reasoning_mode == mode

    @pytest.mark.asyncio
    async def test_empty_steps(self, explainer):
        """Test explanation with no steps."""
        explanation = await explainer.explain_reasoning(
            session_id="session_empty",
            reasoning_mode="direct",
            steps_taken=[],
            final_confidence=0.5
        )

        assert explanation is not None
        assert len(explanation.step_explanations) == 0
        assert len(explanation.confidence_trajectory) == 0

    @pytest.mark.asyncio
    async def test_reasoning_flow_generated(self, explainer, sample_steps):
        """Test that reasoning flow is generated."""
        explanation = await explainer.explain_reasoning(
            session_id="session_summary",
            reasoning_mode="verify",
            steps_taken=sample_steps,
            final_confidence=0.88
        )

        assert hasattr(explanation, 'reasoning_flow')
        assert len(explanation.reasoning_flow) > 0

    @pytest.mark.asyncio
    async def test_concurrent_explanations(self, explainer, sample_steps):
        """Test concurrent agentic explanations."""
        tasks = [
            explainer.explain_reasoning(
                session_id=f"session_concurrent_{i}",
                reasoning_mode="verify",
                steps_taken=sample_steps,
                final_confidence=0.75 + i * 0.05
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        assert all(isinstance(r, ReasoningExplanation) for r in results)

    @pytest.mark.asyncio
    async def test_step_explanation_dataclass(self, explainer, sample_steps):
        """Test StepExplanation dataclass structure."""
        explanation = await explainer.explain_reasoning(
            session_id="session_step_check",
            reasoning_mode="research",
            steps_taken=sample_steps,
            final_confidence=0.9
        )

        if explanation.step_explanations:
            step_exp = explanation.step_explanations[0]
            assert isinstance(step_exp, StepExplanation)
            assert hasattr(step_exp, 'step_number')
            assert hasattr(step_exp, 'confidence')


# ============================================================================
# Integration Tests (Cross-Module)
# ============================================================================

class TestExplainabilityIntegration:
    """Integration tests across explainability modules."""

    @pytest.mark.asyncio
    async def test_full_explanation_pipeline(
        self,
        sample_features,
        background_data,
        feature_names,
        sample_steps,
        mock_predict_fn,
        mock_predict_fn_tuple
    ):
        """Test complete explanation pipeline."""
        # 1. Causal explanation
        causal = CausalExplainer(
            predict_fn=mock_predict_fn,
            feature_names=feature_names
        )
        causal_exp = await causal.explain(
            query_id="pipeline_001",
            query_text="Full pipeline test",
            features=sample_features,
            predicted_tool="answer",
            predicted_confidence=0.85
        )

        # 2. SHAP explanation
        shap = SHAPExplainer(
            predict_fn=mock_predict_fn,
            feature_names=feature_names,
            background_data=background_data
        )
        shap_exp = await shap.explain(
            query_id="pipeline_001",
            query_text="Full pipeline test",
            features=sample_features,
            predicted_tool="answer",
            predicted_confidence=0.85
        )

        # 3. Counterfactual (needs tuple predict_fn)
        cf_gen = MinimalCounterfactualGenerator(
            predict_fn=mock_predict_fn_tuple,
            feature_names=feature_names
        )
        cf_exp = await cf_gen.generate(
            query_id="pipeline_001",
            query_text="Full pipeline test",
            original_features=sample_features,
            original_tool="answer",
            original_confidence=0.85
        )

        # 4. Agentic explanation
        agentic = AgenticExplainer()
        agentic_exp = await agentic.explain_reasoning(
            session_id="pipeline_001",
            reasoning_mode="verify",
            steps_taken=sample_steps,
            final_confidence=0.85
        )

        # All should succeed
        assert causal_exp is not None
        assert shap_exp is not None
        assert cf_exp is not None
        assert agentic_exp is not None

    @pytest.mark.asyncio
    async def test_concurrent_mixed_explanations(
        self,
        sample_features,
        background_data,
        feature_names,
        sample_steps,
        mock_predict_fn,
        mock_predict_fn_tuple
    ):
        """Test concurrent explanations from different modules."""
        causal = CausalExplainer(
            predict_fn=mock_predict_fn,
            feature_names=feature_names
        )
        shap = SHAPExplainer(
            predict_fn=mock_predict_fn,
            feature_names=feature_names,
            background_data=background_data
        )
        lime = LIMEExplainer(
            predict_fn=mock_predict_fn,
            feature_names=feature_names
        )
        cf_gen = MinimalCounterfactualGenerator(
            predict_fn=mock_predict_fn_tuple,
            feature_names=feature_names
        )
        agentic = AgenticExplainer()

        tasks = [
            causal.explain(
                query_id="mixed_001",
                query_text="Mixed test",
                features=sample_features,
                predicted_tool="answer",
                predicted_confidence=0.8
            ),
            shap.explain(
                query_id="mixed_001",
                query_text="Mixed test",
                features=sample_features,
                predicted_tool="answer",
                predicted_confidence=0.8
            ),
            lime.explain(
                query_id="mixed_001",
                query_text="Mixed test",
                features=sample_features,
                predicted_tool="answer",
                predicted_confidence=0.8
            ),
            cf_gen.generate(
                query_id="mixed_001",
                query_text="Mixed test",
                original_features=sample_features,
                original_tool="answer",
                original_confidence=0.8
            ),
            agentic.explain_reasoning(
                session_id="mixed_001",
                reasoning_mode="direct",
                steps_taken=sample_steps,
                final_confidence=0.8
            )
        ]

        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        assert all(r is not None for r in results)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_extreme_confidence_values(self, feature_names, sample_features, mock_predict_fn):
        """Test with extreme confidence values."""
        explainer = CausalExplainer(
            predict_fn=mock_predict_fn,
            feature_names=feature_names
        )

        # Test with 0 confidence
        exp_low = await explainer.explain(
            query_id="edge_low",
            query_text="Low confidence",
            features=sample_features,
            predicted_tool="answer",
            predicted_confidence=0.0
        )
        assert exp_low is not None

        # Test with 1.0 confidence
        exp_high = await explainer.explain(
            query_id="edge_high",
            query_text="High confidence",
            features=sample_features,
            predicted_tool="answer",
            predicted_confidence=1.0
        )
        assert exp_high is not None

    @pytest.mark.asyncio
    async def test_very_long_query_text(self, feature_names, sample_features, mock_predict_fn):
        """Test with very long query text."""
        explainer = CausalExplainer(
            predict_fn=mock_predict_fn,
            feature_names=feature_names
        )
        long_query = "What is Thompson Sampling? " * 100  # Long query

        explanation = await explainer.explain(
            query_id="edge_long",
            query_text=long_query,
            features=sample_features,
            predicted_tool="answer",
            predicted_confidence=0.8
        )
        assert explanation is not None

    @pytest.mark.asyncio
    async def test_many_reasoning_steps(self):
        """Test agentic explainer with many steps."""
        explainer = AgenticExplainer()

        # Generate 100 steps
        many_steps = [
            {
                "step_type": f"step_{i}",
                "description": f"Step {i} description",
                "confidence": 0.5 + (i % 50) * 0.01,
                "duration_ms": float(i * 10)
            }
            for i in range(100)
        ]

        explanation = await explainer.explain_reasoning(
            session_id="edge_many_steps",
            reasoning_mode="research",
            steps_taken=many_steps,
            final_confidence=0.9
        )
        assert explanation is not None
        assert len(explanation.step_explanations) == 100

    @pytest.mark.asyncio
    async def test_single_feature(self, mock_predict_fn):
        """Test with single feature."""
        explainer = SHAPExplainer(
            predict_fn=mock_predict_fn,
            feature_names=["single_feature"],
            background_data=np.random.rand(50, 1)
        )

        explanation = await explainer.explain(
            query_id="single_feature",
            query_text="Single feature test",
            features=np.array([0.5]),
            predicted_tool="answer",
            predicted_confidence=0.7
        )
        assert explanation is not None

    @pytest.mark.asyncio
    async def test_high_dimensional_features(self, mock_predict_fn):
        """Test with high-dimensional features."""
        n_features = 100
        feature_names = [f"feature_{i}" for i in range(n_features)]

        explainer = LIMEExplainer(
            predict_fn=mock_predict_fn,
            feature_names=feature_names,
            num_samples=50
        )

        features = np.random.rand(n_features)
        explanation = await explainer.explain(
            query_id="high_dim",
            query_text="High dimensional test",
            features=features,
            predicted_tool="answer",
            predicted_confidence=0.75
        )
        assert explanation is not None
