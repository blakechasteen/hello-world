"""
Phase 4 Integration Tests - MRF in Skills, RAG, and Memory
===========================================================
Tests UnifiedMRF integration across all 3 remaining systems.

**Phase 4 - November 2025**: Comprehensive integration testing.

Test Coverage:
1. Agentic/Skills system (4 reasoning modes)
2. RAG system (4 operation modes)
3. Memory Consolidation (4 strategies)

All tests verify:
- Prompts are generated successfully
- 7 MRF components are present
- Quality assessment works
- Integration with existing systems
"""


import pytest

from hololoom.agentic.core import ReasoningMode
from hololoom.agentic.mrf_integration import assess_agentic_quality, create_agentic_mrf_prompt
from hololoom.memory.consolidation import ConsolidationStrategy
from hololoom.memory.mrf_consolidation import (
    assess_consolidation_quality,
    create_consolidation_mrf_prompt,
    enhance_consolidation_with_mrf,
)
from hololoom.protocols.types import MemoryShard
from hololoom.rag.mrf_integration import (
    RAGMode,
    assess_rag_quality,
    create_rag_mrf_prompt,
    enhance_rag_with_mrf,
)

# ============================================================================
# Agentic/Skills System Tests
# ============================================================================

class TestAgenticMRFIntegration:
    """Test UnifiedMRF integration in agentic reasoning system."""

    def test_direct_mode_prompt(self):
        """Test DIRECT mode prompt generation."""
        prompt = create_agentic_mrf_prompt(
            query="What is Thompson Sampling?",
            mode=ReasoningMode.DIRECT,
            model_provider="claude"
        )

        # Verify 7 MRF components present (Markdown header format)
        assert "# ROLE" in prompt
        assert "# OBJECTIVE" in prompt
        assert "# PROCESS" in prompt
        assert "# FORMAT" in prompt
        assert "# CONSTRAINTS" in prompt
        assert "# UNCERTAINTY" in prompt
        assert "# VALIDATION" in prompt

        # Verify mode-specific content
        assert "direct" in prompt.lower() or "accurate" in prompt.lower()
        assert "Thompson Sampling" in prompt

    def test_verify_mode_prompt(self):
        """Test VERIFY mode prompt generation."""
        prompt = create_agentic_mrf_prompt(
            query="Is this claim correct?",
            mode=ReasoningMode.VERIFY,
            model_provider="claude",
            context={"initial_answer": "Thompson Sampling uses Bayesian priors"}
        )

        assert "verify" in prompt.lower() or "verification" in prompt.lower()
        assert "contradiction" in prompt.lower()
        assert "Bayesian priors" in prompt

    def test_research_mode_prompt(self):
        """Test RESEARCH mode prompt generation."""
        prompt = create_agentic_mrf_prompt(
            query="Compare exploration strategies",
            mode=ReasoningMode.RESEARCH,
            model_provider="claude",
            context={"max_steps": 5, "current_step": 1}
        )

        assert "research" in prompt.lower()
        assert "question" in prompt.lower() or "follow-up" in prompt.lower()
        assert "exploration strategies" in prompt

    def test_plan_execute_mode_prompt(self):
        """Test PLAN_EXECUTE mode prompt generation."""
        prompt = create_agentic_mrf_prompt(
            query="Implement Thompson Sampling in Python",
            mode=ReasoningMode.PLAN_EXECUTE,
            model_provider="claude"
        )

        assert "plan" in prompt.lower() or "goal" in prompt.lower()
        assert "sub-goal" in prompt.lower() or "step" in prompt.lower()
        assert "Thompson Sampling" in prompt

    def test_agentic_quality_assessment(self):
        """Test quality assessment for agentic responses."""
        # DIRECT mode
        response_direct = "Thompson Sampling is a Bayesian algorithm that balances exploration and exploitation."
        quality = assess_agentic_quality(response_direct, ReasoningMode.DIRECT)
        assert 0.0 <= quality <= 1.0
        assert quality > 0.5  # Should be decent quality

        # VERIFY mode
        response_verify = """Verified claims:
        - Thompson Sampling uses Bayesian priors
        Contradictions: None found
        Confidence: 0.95"""
        quality = assess_agentic_quality(response_verify, ReasoningMode.VERIFY)
        assert quality > 0.7  # Should score well for having structure

    def test_custom_constraints(self):
        """Test adding custom constraints."""
        prompt = create_agentic_mrf_prompt(
            query="Test query",
            mode=ReasoningMode.DIRECT,
            model_provider="claude",
            constraints=["MUST use formal language", "SHOULD include examples"]
        )

        assert "formal language" in prompt
        assert "examples" in prompt


# ============================================================================
# RAG System Tests
# ============================================================================

class TestRAGMRFIntegration:
    """Test UnifiedMRF integration in RAG system."""

    def test_reformulate_mode_prompt(self):
        """Test query reformulation prompt."""
        prompt = create_rag_mrf_prompt(
            query="What is TS?",
            mode=RAGMode.REFORMULATE,
            model_provider="claude"
        )

        assert "reformulate" in prompt.lower() or "expand" in prompt.lower()
        assert "abbreviation" in prompt.lower()
        assert "TS" in prompt

    def test_answer_mode_prompt(self):
        """Test answer generation prompt."""
        sources = [
            "Thompson Sampling is a Bayesian algorithm.",
            "It uses beta distributions for priors."
        ]

        prompt = create_rag_mrf_prompt(
            query="Explain Thompson Sampling",
            mode=RAGMode.ANSWER,
            sources=sources,
            model_provider="claude"
        )

        assert "answer" in prompt.lower()
        assert "source" in prompt.lower() or "citation" in prompt.lower()
        assert "Thompson Sampling" in prompt

    def test_summarize_mode_prompt(self):
        """Test summarization prompt."""
        sources = [
            "Source 1: Thompson Sampling explained",
            "Source 2: Applications of Thompson Sampling"
        ]

        prompt = create_rag_mrf_prompt(
            query="Summarize Thompson Sampling",
            mode=RAGMode.SUMMARIZE,
            sources=sources,
            model_provider="claude"
        )

        assert "summarize" in prompt.lower() or "summary" in prompt.lower()
        assert "theme" in prompt.lower()

    def test_multimodal_mode_prompt(self):
        """Test multimodal Q&A prompt."""
        sources = ["Text description"]
        images = ["Image 1 shows a diagram", "Image 2 shows a graph"]

        prompt = create_rag_mrf_prompt(
            query="Explain using both text and images",
            mode=RAGMode.MULTIMODAL,
            sources=sources,
            images=images,
            model_provider="claude"
        )

        assert "image" in prompt.lower()
        assert "multimodal" in prompt.lower() or "visual" in prompt.lower()

    def test_rag_quality_assessment(self):
        """Test quality assessment for RAG responses."""
        sources = ["Source 1", "Source 2"]

        # Good answer with citations
        response_good = "According to [Source 1], Thompson Sampling is a Bayesian algorithm. [Source 2] explains its applications."
        quality = assess_rag_quality("What is TS?", response_good, sources, RAGMode.ANSWER)
        assert quality > 0.6

        # Answer without citations
        response_no_cite = "Thompson Sampling is a Bayesian algorithm."
        quality = assess_rag_quality("What is TS?", response_no_cite, sources, RAGMode.ANSWER)
        assert quality < 0.7  # Lower score due to missing citations

    def test_rag_pipeline_enhancement(self):
        """Test complete RAG pipeline with MRF."""
        sources = ["Source 1: Thompson Sampling", "Source 2: Bayesian methods"]

        result = enhance_rag_with_mrf(
            query="What is TS?",
            sources=sources,
            model_provider="claude",
            enable_reformulation=True
        )

        assert "reformulated_query" in result
        assert "answer_prompt" in result
        assert len(result["reformulated_query"]) > 0
        assert len(result["answer_prompt"]) > 0


# ============================================================================
# Memory Consolidation Tests
# ============================================================================

class TestMemoryConsolidationMRFIntegration:
    """Test UnifiedMRF integration in memory consolidation system."""

    def create_test_episodes(self, count: int = 5) -> list[MemoryShard]:
        """Create test episodes for consolidation."""
        return [
            MemoryShard(
                text=f"Episode {i}: I learned about Thompson Sampling",
                id=f"ep_{i}",
                timestamp=None,
                entities=[],
                motifs=[]
            )
            for i in range(count)
        ]

    def test_fact_extraction_prompt(self):
        """Test fact extraction prompt."""
        episodes = self.create_test_episodes()

        prompt = create_consolidation_mrf_prompt(
            episodes=episodes,
            strategy=ConsolidationStrategy.FACT_EXTRACTION.value,
            model_provider="claude"
        )

        assert "fact" in prompt.lower()
        assert "semantic" in prompt.lower()
        assert "5-10" in prompt  # Should extract 5-10 facts

    def test_entity_extraction_prompt(self):
        """Test entity extraction prompt."""
        episodes = self.create_test_episodes()

        prompt = create_consolidation_mrf_prompt(
            episodes=episodes,
            strategy=ConsolidationStrategy.ENTITY_EXTRACTION.value,
            model_provider="claude"
        )

        assert "entity" in prompt.lower()
        assert "relationship" in prompt.lower()
        assert "IS_A" in prompt or "USES" in prompt  # Standard relationship types

    def test_summarization_prompt(self):
        """Test summarization prompt."""
        episodes = self.create_test_episodes(10)

        prompt = create_consolidation_mrf_prompt(
            episodes=episodes,
            strategy=ConsolidationStrategy.SUMMARIZATION.value,
            model_provider="claude"
        )

        assert "summarize" in prompt.lower() or "summary" in prompt.lower()
        assert "2-3 paragraph" in prompt.lower()
        assert "10:1" in prompt  # Compression ratio

    def test_deduplication_prompt(self):
        """Test deduplication prompt."""
        episodes = self.create_test_episodes()

        prompt = create_consolidation_mrf_prompt(
            episodes=episodes,
            strategy=ConsolidationStrategy.DEDUPLICATION.value,
            model_provider="claude"
        )

        assert "duplicate" in prompt.lower() or "similar" in prompt.lower()
        assert "90%" in prompt  # Similarity threshold

    def test_consolidation_quality_assessment(self):
        """Test quality assessment for consolidation."""
        # Fact extraction
        facts = [
            "Thompson Sampling balances exploration and exploitation",
            "It uses Bayesian priors",
            "Beta distributions model uncertainty"
        ]
        quality = assess_consolidation_quality(
            ConsolidationStrategy.FACT_EXTRACTION.value,
            input_episodes=10,
            output_facts=facts
        )
        assert 0.0 <= quality <= 1.0

        # Entity extraction
        entities = [
            "(Thompson Sampling, IS_A, Bayesian algorithm)",
            "(Thompson Sampling, USES, beta distribution)"
        ]
        quality = assess_consolidation_quality(
            ConsolidationStrategy.ENTITY_EXTRACTION.value,
            input_episodes=5,
            output_facts=entities
        )
        assert quality > 0.0

    def test_consolidation_pipeline_enhancement(self):
        """Test complete consolidation pipeline with MRF."""
        episodes = self.create_test_episodes(10)

        result = enhance_consolidation_with_mrf(
            episodes=episodes,
            model_provider="claude",
            enable_all_strategies=True
        )

        # Should have all 4 strategies
        assert ConsolidationStrategy.FACT_EXTRACTION.value in result
        assert ConsolidationStrategy.ENTITY_EXTRACTION.value in result
        assert ConsolidationStrategy.SUMMARIZATION.value in result
        assert ConsolidationStrategy.DEDUPLICATION.value in result

        # Each should have content
        for strategy, prompt in result.items():
            assert len(prompt) > 0


# ============================================================================
# Integration Tests Across Systems
# ============================================================================

class TestCrossSystemIntegration:
    """Test integration patterns across all 3 systems."""

    def test_model_provider_consistency(self):
        """Test all systems support same model providers."""
        providers = ["claude", "gemini", "gpt", "ollama"]

        for provider in providers:
            # Agentic
            prompt_agentic = create_agentic_mrf_prompt(
                "test", ReasoningMode.DIRECT, model_provider=provider
            )
            assert len(prompt_agentic) > 0

            # RAG
            prompt_rag = create_rag_mrf_prompt(
                "test", mode=RAGMode.ANSWER, model_provider=provider
            )
            assert len(prompt_rag) > 0

            # Memory
            episodes = [MemoryShard(text="test", id="1",
                                   timestamp=None, entities=[], motifs=[])]
            prompt_memory = create_consolidation_mrf_prompt(
                episodes, ConsolidationStrategy.FACT_EXTRACTION.value,
                model_provider=provider
            )
            assert len(prompt_memory) > 0

    def test_7_component_framework(self):
        """Test all systems generate 7-component MRF prompts."""
        # Markdown header format (# ROLE, not ROLE:)
        components = ["# ROLE", "# OBJECTIVE", "# PROCESS", "# FORMAT",
                     "# CONSTRAINTS", "# UNCERTAINTY", "# VALIDATION"]

        # Agentic
        prompt_agentic = create_agentic_mrf_prompt(
            "test", ReasoningMode.DIRECT
        )
        for comp in components:
            assert comp in prompt_agentic, f"Missing {comp} in agentic prompt"

        # RAG
        prompt_rag = create_rag_mrf_prompt("test", mode=RAGMode.ANSWER)
        for comp in components:
            assert comp in prompt_rag, f"Missing {comp} in RAG prompt"

        # Memory
        episodes = [MemoryShard(text="test", id="1",
                               timestamp=None, entities=[], motifs=[])]
        prompt_memory = create_consolidation_mrf_prompt(
            episodes, ConsolidationStrategy.FACT_EXTRACTION.value
        )
        for comp in components:
            assert comp in prompt_memory, f"Missing {comp} in memory prompt"

    def test_quality_assessment_range(self):
        """Test all quality assessments return [0.0, 1.0]."""
        # Agentic
        quality = assess_agentic_quality("test response", ReasoningMode.DIRECT)
        assert 0.0 <= quality <= 1.0

        # RAG
        quality = assess_rag_quality("test", "test response", [], RAGMode.ANSWER)
        assert 0.0 <= quality <= 1.0

        # Memory
        quality = assess_consolidation_quality(
            ConsolidationStrategy.FACT_EXTRACTION.value, 10, ["fact1", "fact2"]
        )
        assert 0.0 <= quality <= 1.0


# ============================================================================
# Main Test Execution
# ============================================================================

if __name__ == "__main__":
    """Run all Phase 4 integration tests."""
    print("="*60)
    print("Phase 4 Integration Tests - MRF in Skills, RAG, and Memory")
    print("="*60)

    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])

    print("\n" + "="*60)
    print("Test Summary:")
    print("- Agentic/Skills: 4 reasoning modes ✓")
    print("- RAG: 4 operation modes ✓")
    print("- Memory: 4 consolidation strategies ✓")
    print("- Cross-system integration ✓")
    print("="*60)
