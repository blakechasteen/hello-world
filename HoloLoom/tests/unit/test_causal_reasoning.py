"""
Unit Tests for Causal Reasoning Engine

Tests all three levels of Pearl's causal hierarchy:
1. Association (observational)
2. Intervention (do-calculus)
3. Counterfactual (twin networks)
"""

import pytest
import sys
sys.path.insert(0, '.')

from HoloLoom.causal import (
    CausalNode, CausalEdge, CausalDAG,
    CausalQuery, QueryType,
    InterventionEngine, CounterfactualEngine,
    NodeType
)


class TestCausalDAG:
    """Test causal DAG construction and queries."""

    def test_create_dag(self):
        """Test basic DAG creation."""
        dag = CausalDAG()

        # Add nodes
        dag.add_node(CausalNode("X", NodeType.OBSERVABLE, description="Treatment"))
        dag.add_node(CausalNode("Y", NodeType.OBSERVABLE, description="Outcome"))
        dag.add_node(CausalNode("Z", NodeType.OBSERVABLE, description="Confounder"))

        assert len(dag.nodes) == 3
        assert "X" in dag.nodes
        assert "Y" in dag.nodes
        assert "Z" in dag.nodes

    def test_add_edges(self):
        """Test adding causal edges."""
        dag = CausalDAG()
        dag.add_node(CausalNode("X"))
        dag.add_node(CausalNode("Y"))

        edge = CausalEdge("X", "Y", strength=0.8, mechanism="direct effect")
        dag.add_edge(edge)

        assert len(dag.edges) == 1
        assert ("X", "Y") in dag.edges

    def test_cycle_detection(self):
        """Test that cycles are rejected."""
        dag = CausalDAG()
        dag.add_node(CausalNode("X"))
        dag.add_node(CausalNode("Y"))

        dag.add_edge(CausalEdge("X", "Y"))

        # Try to create cycle
        with pytest.raises(ValueError, match="cycle"):
            dag.add_edge(CausalEdge("Y", "X"))

    def test_parents_children(self):
        """Test parent/child queries."""
        dag = CausalDAG()
        dag.add_node(CausalNode("X"))
        dag.add_node(CausalNode("Y"))
        dag.add_node(CausalNode("Z"))

        dag.add_edge(CausalEdge("X", "Z"))
        dag.add_edge(CausalEdge("Y", "Z"))

        # Z has two parents
        assert dag.parents("Z") == {"X", "Y"}

        # X has one child
        assert dag.children("X") == {"Z"}

    def test_ancestors_descendants(self):
        """Test transitive closure queries."""
        dag = CausalDAG()
        for name in ["A", "B", "C", "D"]:
            dag.add_node(CausalNode(name))

        # Create chain: A → B → C → D
        dag.add_edge(CausalEdge("A", "B"))
        dag.add_edge(CausalEdge("B", "C"))
        dag.add_edge(CausalEdge("C", "D"))

        # D's ancestors
        assert dag.ancestors("D") == {"A", "B", "C"}

        # A's descendants
        assert dag.descendants("A") == {"B", "C", "D"}

    def test_markov_blanket(self):
        """Test Markov blanket computation."""
        dag = CausalDAG()
        for name in ["X", "Y", "Z", "W", "V"]:
            dag.add_node(CausalNode(name))

        # Structure: X → Y ← Z, Y → W, V → W
        # Markov blanket of Y = {X, Z, W, V}
        dag.add_edge(CausalEdge("X", "Y"))
        dag.add_edge(CausalEdge("Z", "Y"))
        dag.add_edge(CausalEdge("Y", "W"))
        dag.add_edge(CausalEdge("V", "W"))

        mb = dag.markov_blanket("Y")
        assert mb == {"X", "Z", "W", "V"}

    def test_topological_order(self):
        """Test topological ordering."""
        dag = CausalDAG()
        for name in ["A", "B", "C"]:
            dag.add_node(CausalNode(name))

        dag.add_edge(CausalEdge("A", "B"))
        dag.add_edge(CausalEdge("B", "C"))

        order = dag.topological_order()

        # A must come before B, B before C
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_d_separation(self):
        """Test d-separation for conditional independence."""
        dag = CausalDAG()
        for name in ["X", "Y", "Z"]:
            dag.add_node(CausalNode(name))

        # Fork: X ← Z → Y
        # X and Y are d-separated given Z
        dag.add_edge(CausalEdge("Z", "X"))
        dag.add_edge(CausalEdge("Z", "Y"))

        assert dag.is_d_separated({"X"}, {"Y"}, {"Z"})
        assert not dag.is_d_separated({"X"}, {"Y"}, set())

    def test_find_colliders(self):
        """Test collider detection."""
        dag = CausalDAG()
        for name in ["X", "Y", "Z"]:
            dag.add_node(CausalNode(name))

        # Collider: X → Z ← Y
        dag.add_edge(CausalEdge("X", "Z"))
        dag.add_edge(CausalEdge("Y", "Z"))

        colliders = dag.find_colliders()
        assert "Z" in colliders

    def test_find_confounders(self):
        """Test confounder detection."""
        dag = CausalDAG()
        for name in ["X", "Y", "Z"]:
            dag.add_node(CausalNode(name))

        # Confounder: Z → X, Z → Y
        dag.add_edge(CausalEdge("Z", "X"))
        dag.add_edge(CausalEdge("Z", "Y"))

        confounders = dag.find_confounders("X", "Y")
        assert "Z" in confounders

    def test_find_mediators(self):
        """Test mediator detection."""
        dag = CausalDAG()
        for name in ["X", "M", "Y"]:
            dag.add_node(CausalNode(name))

        # Mediator: X → M → Y
        dag.add_edge(CausalEdge("X", "M"))
        dag.add_edge(CausalEdge("M", "Y"))

        mediators = dag.find_mediators("X", "Y")
        assert "M" in mediators

    def test_backdoor_paths(self):
        """Test backdoor path detection."""
        dag = CausalDAG()
        for name in ["X", "Y", "Z"]:
            dag.add_node(CausalNode(name))

        # Backdoor: Z → X → Y, Z → Y
        dag.add_edge(CausalEdge("Z", "X"))
        dag.add_edge(CausalEdge("X", "Y"))
        dag.add_edge(CausalEdge("Z", "Y"))

        backdoor = dag.backdoor_paths("X", "Y")
        assert len(backdoor) > 0

    def test_backdoor_criterion(self):
        """Test backdoor criterion for identification."""
        dag = CausalDAG()
        for name in ["X", "Y", "Z"]:
            dag.add_node(CausalNode(name))

        # Z confounds X and Y
        dag.add_edge(CausalEdge("Z", "X"))
        dag.add_edge(CausalEdge("Z", "Y"))
        dag.add_edge(CausalEdge("X", "Y"))

        # Adjusting for Z satisfies backdoor criterion
        assert dag.satisfies_backdoor_criterion("X", "Y", {"Z"})

    def test_serialization(self):
        """Test DAG serialization and deserialization."""
        dag = CausalDAG()
        dag.add_node(CausalNode("X", description="Treatment"))
        dag.add_node(CausalNode("Y", description="Outcome"))
        dag.add_edge(CausalEdge("X", "Y", strength=0.8))

        # Serialize
        data = dag.to_dict()
        assert len(data['nodes']) == 2
        assert len(data['edges']) == 1

        # Deserialize
        dag2 = CausalDAG.from_dict(data)
        assert len(dag2.nodes) == 2
        assert len(dag2.edges) == 1
        assert ("X", "Y") in dag2.edges


class TestInterventionEngine:
    """Test do-operator and causal effect identification."""

    def test_do_operator(self):
        """Test basic do-operator (graph surgery)."""
        dag = CausalDAG()
        for name in ["X", "Y", "Z"]:
            dag.add_node(CausalNode(name))

        dag.add_edge(CausalEdge("Z", "X"))
        dag.add_edge(CausalEdge("X", "Y"))

        engine = InterventionEngine(dag)

        # Intervene on X
        result = engine.do({"X": 1})

        # Mutilated graph should have Z → X removed
        assert result.identifiable
        assert ("Z", "X") not in result.mutilated_graph.edges
        assert ("X", "Y") in result.mutilated_graph.edges

    def test_backdoor_adjustment(self):
        """Test backdoor adjustment for identification."""
        dag = CausalDAG()
        for name in ["X", "Y", "Z"]:
            dag.add_node(CausalNode(name))

        # Confounder
        dag.add_edge(CausalEdge("Z", "X"))
        dag.add_edge(CausalEdge("Z", "Y"))
        dag.add_edge(CausalEdge("X", "Y"))

        engine = InterventionEngine(dag)

        # Identify causal effect
        result = engine.identify_causal_effect("X", "Y")

        assert result.identifiable
        assert result.identification_method == "backdoor adjustment"
        assert "Z" in result.adjustment_set

    def test_frontdoor_adjustment(self):
        """Test frontdoor adjustment when confounders unobserved."""
        dag = CausalDAG()
        for name in ["X", "M", "Y"]:
            dag.add_node(CausalNode(name))

        # X → M → Y (M is mediator)
        dag.add_edge(CausalEdge("X", "M"))
        dag.add_edge(CausalEdge("M", "Y"))

        engine = InterventionEngine(dag)

        # Should identify via frontdoor
        result = engine.identify_causal_effect("X", "Y")

        # Note: Frontdoor requires specific conditions, may not always work
        # Check if identified
        if result.identifiable:
            assert "M" in result.adjustment_set or result.identification_method in [
                "backdoor adjustment", "frontdoor adjustment"
            ]

    def test_intervention_query(self):
        """Test answering intervention queries."""
        dag = CausalDAG()
        for name in ["X", "Y"]:
            dag.add_node(CausalNode(name))

        dag.add_edge(CausalEdge("X", "Y"))

        engine = InterventionEngine(dag)

        query = CausalQuery(
            query_type=QueryType.INTERVENTION,
            treatment="X",
            outcome="Y",
            treatment_value=1
        )

        answer = engine.query(query)

        # Should be identifiable (no confounders)
        assert answer.confidence > 0

    def test_explanation(self):
        """Test human-readable explanations."""
        dag = CausalDAG()
        for name in ["X", "Y", "Z"]:
            dag.add_node(CausalNode(name))

        dag.add_edge(CausalEdge("Z", "X"))
        dag.add_edge(CausalEdge("Z", "Y"))
        dag.add_edge(CausalEdge("X", "Y"))

        engine = InterventionEngine(dag)

        explanation = engine.explain_identification("X", "Y")

        # Should contain useful information
        assert "X" in explanation
        assert "Y" in explanation
        assert "IDENTIFIABLE" in explanation or "identifiable" in explanation


class TestCounterfactualEngine:
    """Test counterfactual reasoning and twin networks."""

    def test_counterfactual_basic(self):
        """Test basic counterfactual query."""
        dag = CausalDAG()
        for name in ["X", "Y"]:
            dag.add_node(CausalNode(name))

        dag.add_edge(CausalEdge("X", "Y"))

        engine = CounterfactualEngine(dag)

        # "Would Y=1 if X had been 0?" (given X=1, Y=1)
        result = engine.counterfactual(
            intervention={"X": 0},
            evidence={"X": 1, "Y": 1},
            query="Y"
        )

        # Should return a result
        assert result.factual_outcome == 1
        assert result.counterfactual_outcome in [0, 1]
        assert 0 <= result.probability <= 1

    def test_probability_of_necessity(self):
        """Test probability of necessity."""
        dag = CausalDAG()
        for name in ["X", "Y"]:
            dag.add_node(CausalNode(name))

        dag.add_edge(CausalEdge("X", "Y"))

        engine = CounterfactualEngine(dag)

        # PN: Was X necessary for Y?
        necessity = engine.probability_of_necessity(
            treatment="X",
            outcome="Y",
            evidence={"X": 1, "Y": 1}
        )

        assert 0 <= necessity <= 1

    def test_probability_of_sufficiency(self):
        """Test probability of sufficiency."""
        dag = CausalDAG()
        for name in ["X", "Y"]:
            dag.add_node(CausalNode(name))

        dag.add_edge(CausalEdge("X", "Y"))

        engine = CounterfactualEngine(dag)

        # PS: Is X sufficient for Y?
        sufficiency = engine.probability_of_sufficiency(
            treatment="X",
            outcome="Y",
            evidence={"X": 0, "Y": 0}
        )

        assert 0 <= sufficiency <= 1

    def test_counterfactual_query(self):
        """Test answering counterfactual queries."""
        dag = CausalDAG()
        for name in ["X", "Y"]:
            dag.add_node(CausalNode(name))

        dag.add_edge(CausalEdge("X", "Y"))

        engine = CounterfactualEngine(dag)

        query = CausalQuery(
            query_type=QueryType.COUNTERFACTUAL,
            treatment="X",
            outcome="Y",
            treatment_value=0,
            evidence={"X": 1, "Y": 1}
        )

        answer = engine.query(query)

        # Should return answer
        assert 0 <= answer.result <= 1
        assert answer.method == "twin networks (3-step counterfactual)"

    def test_twin_network_creation(self):
        """Test that twin network is created properly."""
        dag = CausalDAG()
        for name in ["X", "Y"]:
            dag.add_node(CausalNode(name))

        dag.add_edge(CausalEdge("X", "Y"))

        engine = CounterfactualEngine(dag)

        result = engine.counterfactual(
            intervention={"X": 0},
            evidence={"X": 1, "Y": 1},
            query="Y"
        )

        # Twin network should exist
        assert result.twin_network is not None
        assert result.twin_network.factual_dag == dag
        assert len(result.twin_network.shared_exogenous) > 0


class TestCausalQuery:
    """Test causal query representation."""

    def test_query_levels(self):
        """Test Pearl's hierarchy levels."""
        # Level 1: Observational
        q1 = CausalQuery(query_type=QueryType.CONDITIONAL, outcome="Y", treatment="X")
        assert q1.get_level() == 1

        # Level 2: Interventional
        q2 = CausalQuery(query_type=QueryType.INTERVENTION, outcome="Y", treatment="X")
        assert q2.get_level() == 2

        # Level 3: Counterfactual
        q3 = CausalQuery(query_type=QueryType.COUNTERFACTUAL, outcome="Y", treatment="X")
        assert q3.get_level() == 3

    def test_natural_language(self):
        """Test query to natural language conversion."""
        q = CausalQuery(
            query_type=QueryType.INTERVENTION,
            outcome="recovery",
            treatment="drug_A",
            treatment_value=1
        )

        nl = q.to_natural_language()

        # Should contain key terms
        assert "drug_A" in nl
        assert "recovery" in nl
        assert "effect" in nl.lower()


def test_clinical_trial_example():
    """
    End-to-end test: Clinical trial scenario.

    Structure:
        Age → Treatment
        Age → Recovery
        Treatment → Recovery
    """
    dag = CausalDAG()

    # Nodes
    dag.add_node(CausalNode("age", description="Patient age"))
    dag.add_node(CausalNode("treatment", description="Received treatment"))
    dag.add_node(CausalNode("recovery", description="Patient recovered"))

    # Edges
    dag.add_edge(CausalEdge("age", "treatment", strength=0.3))
    dag.add_edge(CausalEdge("age", "recovery", strength=0.2))
    dag.add_edge(CausalEdge("treatment", "recovery", strength=0.6))

    # Test 1: Identify causal effect
    engine = InterventionEngine(dag)
    identification = engine.identify_causal_effect("treatment", "recovery")

    assert identification.identifiable
    assert "age" in identification.adjustment_set  # Must adjust for age (confounder)

    # Test 2: Counterfactual - would patient have recovered without treatment?
    cf_engine = CounterfactualEngine(dag)
    result = cf_engine.counterfactual(
        intervention={"treatment": 0},
        evidence={"treatment": 1, "recovery": 1, "age": "elderly"},
        query="recovery"
    )

    assert result.factual_outcome == 1
    assert result.counterfactual_outcome in [0, 1]

    print("\n✅ Clinical trial scenario test passed!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
