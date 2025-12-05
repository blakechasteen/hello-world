"""
Tests for new attack strategy generators
=========================================

Comprehensive tests for ContextOverflowGenerator and HierarchyBypassGenerator.

Author: CARTS (Continuous Adversarial Red Team System)
Date: 2025-12-05
"""

import pytest
from HoloLoom.redteam.strategy_generators.context_overflow import ContextOverflowGenerator, ContextOverflowPayload
from HoloLoom.redteam.strategy_generators.hierarchy_bypass import HierarchyBypassGenerator, HierarchyBypassPayload


class TestContextOverflowGenerator:
    """Tests for ContextOverflowGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.gen = ContextOverflowGenerator(seed=42)

    def test_initialization(self):
        """Test generator initialization."""
        assert self.gen is not None
        assert len(self.gen.filler_templates) > 0
        assert len(self.gen.filler_topics) > 0

    def test_instruction_washing_generation(self):
        """Test instruction washing payload generation."""
        target = "safety_constraint"
        payloads = self.gen.generate_instruction_washing(target)

        assert len(payloads) >= 5, "Should generate at least 5 payloads"
        assert all(isinstance(p, ContextOverflowPayload) for p in payloads)
        assert all(p.overflow_type == "instruction_washing" for p in payloads)

    def test_instruction_washing_severity(self):
        """Test that instruction washing has appropriate severity."""
        payloads = self.gen.generate_instruction_washing("test")

        severities = [p.severity_estimate for p in payloads]
        assert all(0.0 <= s <= 1.0 for s in severities), "Severity should be 0-1"
        assert min(severities) >= 0.3, "Instruction washing should have min severity"

    def test_instruction_washing_metadata(self):
        """Test instruction washing payload metadata."""
        payloads = self.gen.generate_instruction_washing("test")

        for payload in payloads:
            assert payload.payload_size_tokens > 0
            assert 0.0 <= payload.context_pollution_ratio <= 1.0
            assert payload.detection_difficulty in ["easy", "medium", "hard"]
            assert len(payload.tags) > 0

    def test_memory_poisoning_generation(self):
        """Test memory poisoning payload generation."""
        target = "safety_constraint"
        payloads = self.gen.generate_memory_poisoning(target)

        assert len(payloads) >= 5, "Should generate at least 5 payloads"
        assert all(isinstance(p, ContextOverflowPayload) for p in payloads)
        assert all(p.overflow_type == "memory_poisoning" for p in payloads)

    def test_memory_poisoning_severity(self):
        """Test that memory poisoning has appropriate severity."""
        payloads = self.gen.generate_memory_poisoning("test")

        severities = [p.severity_estimate for p in payloads]
        assert all(0.0 <= s <= 1.0 for s in severities)
        # Memory poisoning typically has higher severity
        assert max(severities) > 0.5

    def test_memory_poisoning_has_false_claims(self):
        """Test that memory poisoning contains false claims or contradictions."""
        payloads = self.gen.generate_memory_poisoning("test")

        for payload in payloads:
            # Check for keywords indicating false/misleading content
            keywords = ["false", "memory", "contradiction", "inject", "claim"]
            has_keyword = any(kw in payload.description.lower() for kw in keywords)
            assert has_keyword, f"Payload {payload.description} should indicate false content"
            assert payload.overflow_type == "memory_poisoning"

    def test_context_splitting_generation(self):
        """Test context splitting payload generation."""
        target = "safety_constraint"
        payloads = self.gen.generate_context_splitting(target)

        assert len(payloads) >= 5, "Should generate at least 5 payloads"
        assert all(isinstance(p, ContextOverflowPayload) for p in payloads)
        assert all(p.overflow_type == "context_splitting" for p in payloads)

    def test_context_splitting_multi_part(self):
        """Test that context splitting contains multi-part sequences."""
        payloads = self.gen.generate_context_splitting("test")

        # Should have payloads with multi-part sequences
        assert any("[Part" in p.payload or "[Step" in p.payload for p in payloads)

    def test_generate_all(self):
        """Test generate_all method."""
        target = "safety_constraint"
        all_payloads = self.gen.generate_all(target)

        assert len(all_payloads) >= 15, "Should generate at least 15 total payloads"
        assert all(isinstance(p, ContextOverflowPayload) for p in all_payloads)

        # Should contain all three types
        types = set(p.overflow_type for p in all_payloads)
        assert "instruction_washing" in types
        assert "memory_poisoning" in types
        assert "context_splitting" in types

    def test_payloads_sorted_by_severity(self):
        """Test that generate_all returns payloads sorted by severity."""
        all_payloads = self.gen.generate_all("test")

        severities = [p.severity_estimate for p in all_payloads]
        assert severities == sorted(severities, reverse=True), "Should be sorted by severity descending"


class TestHierarchyBypassGenerator:
    """Tests for HierarchyBypassGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.gen = HierarchyBypassGenerator(seed=42)

    def test_initialization(self):
        """Test generator initialization."""
        assert self.gen is not None
        assert len(self.gen.priority_claims) > 0
        assert len(self.gen.nesting_templates) > 0
        assert len(self.gen.meta_prefixes) > 0

    def test_priority_confusion_generation(self):
        """Test priority confusion payload generation."""
        target = "safety_constraint"
        payloads = self.gen.generate_priority_confusion(target)

        assert len(payloads) >= 5, "Should generate at least 5 payloads"
        assert all(isinstance(p, HierarchyBypassPayload) for p in payloads)
        assert all(p.bypass_type == "priority_confusion" for p in payloads)

    def test_priority_confusion_hierarchy_levels(self):
        """Test that priority confusion uses appropriate hierarchy levels."""
        payloads = self.gen.generate_priority_confusion("test")

        hierarchy_levels = [p.hierarchy_level for p in payloads]
        assert all(2 <= h <= 4 for h in hierarchy_levels), "Hierarchy levels should be 2-4"

    def test_priority_confusion_priority_claims(self):
        """Test that priority confusion contains priority claims."""
        payloads = self.gen.generate_priority_confusion("test")

        for payload in payloads:
            assert payload.priority_claim is not None
            assert payload.priority_claim != ""

    def test_nesting_attack_generation(self):
        """Test nesting attack payload generation."""
        target = "safety_constraint"
        payloads = self.gen.generate_nesting_attack(target)

        assert len(payloads) >= 5, "Should generate at least 5 payloads"
        assert all(isinstance(p, HierarchyBypassPayload) for p in payloads)
        assert all(p.bypass_type == "nesting_attack" for p in payloads)

    def test_nesting_attack_has_nesting_depth(self):
        """Test that nesting attacks specify nesting depth."""
        payloads = self.gen.generate_nesting_attack("test")

        for payload in payloads:
            assert payload.nesting_depth is not None
            assert payload.nesting_depth >= 2

    def test_nesting_attack_nesting_visible(self):
        """Test that nesting is visible in payload content."""
        payloads = self.gen.generate_nesting_attack("test")

        # Many should have nesting indicators
        nested_payloads = [p for p in payloads if ("[Level" in p.payload or "[Step" in p.payload or
                                                     "IF" in p.payload)]
        assert len(nested_payloads) > 0

    def test_meta_instructions_generation(self):
        """Test meta-instruction payload generation."""
        target = "safety_constraint"
        payloads = self.gen.generate_meta_instructions(target)

        assert len(payloads) >= 5, "Should generate at least 5 payloads"
        assert all(isinstance(p, HierarchyBypassPayload) for p in payloads)
        assert all(p.bypass_type == "meta_instructions" for p in payloads)

    def test_meta_instructions_have_meta_level(self):
        """Test that meta-instructions specify meta level."""
        payloads = self.gen.generate_meta_instructions("test")

        for payload in payloads:
            assert payload.meta_level is not None
            assert payload.meta_level >= 1

    def test_meta_instructions_hierarchy(self):
        """Test that meta-instructions have elevated hierarchy levels."""
        payloads = self.gen.generate_meta_instructions("test")

        hierarchy_levels = [p.hierarchy_level for p in payloads]
        assert all(h >= 3 for h in hierarchy_levels), "Meta-instructions should be level 3+"

    def test_generate_all(self):
        """Test generate_all method."""
        target = "safety_constraint"
        all_payloads = self.gen.generate_all(target)

        assert len(all_payloads) >= 15, "Should generate at least 15 total payloads"
        assert all(isinstance(p, HierarchyBypassPayload) for p in all_payloads)

        # Should contain all three types
        types = set(p.bypass_type for p in all_payloads)
        assert "priority_confusion" in types
        assert "nesting_attack" in types
        assert "meta_instructions" in types

    def test_payloads_sorted_by_severity(self):
        """Test that generate_all returns payloads sorted by severity."""
        all_payloads = self.gen.generate_all("test")

        severities = [p.severity_estimate for p in all_payloads]
        assert severities == sorted(severities, reverse=True), "Should be sorted by severity descending"


class TestCrossGenerator:
    """Tests comparing both generators."""

    def test_combined_payload_count(self):
        """Test combined payload generation."""
        overflow_gen = ContextOverflowGenerator()
        bypass_gen = HierarchyBypassGenerator()

        overflow_payloads = overflow_gen.generate_all("test")
        bypass_payloads = bypass_gen.generate_all("test")

        total = len(overflow_payloads) + len(bypass_payloads)
        assert total >= 30, f"Should generate at least 30 total payloads, got {total}"

    def test_payload_uniqueness(self):
        """Test that generated payloads are reasonably unique."""
        gen = ContextOverflowGenerator()
        payloads = gen.generate_all("test")

        payload_texts = [p.payload[:100] for p in payloads]
        unique_count = len(set(payload_texts))

        # At least 80% should be unique
        assert unique_count > len(payloads) * 0.8

    def test_all_payloads_have_metadata(self):
        """Test that all payloads have required metadata."""
        overflow_gen = ContextOverflowGenerator()
        bypass_gen = HierarchyBypassGenerator()

        overflow_payloads = overflow_gen.generate_all("test")
        bypass_payloads = bypass_gen.generate_all("test")

        for payload in overflow_payloads:
            assert payload.payload != ""
            assert payload.description != ""
            assert payload.expected_behavior != ""
            assert 0.0 <= payload.severity_estimate <= 1.0
            assert payload.tags

        for payload in bypass_payloads:
            assert payload.payload != ""
            assert payload.description != ""
            assert payload.expected_behavior != ""
            assert 0.0 <= payload.severity_estimate <= 1.0
            assert payload.tags


if __name__ == "__main__":
    # Run with: pytest test_new_generators.py -v
    pytest.main([__file__, "-v"])
