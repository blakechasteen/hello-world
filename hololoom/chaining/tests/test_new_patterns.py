"""
Tests for New Chain Templates and Evaluation System
=====================================================
Comprehensive test suite for the 8 new chain templates
and the evaluation framework.

Author: HoloLoom Architecture Team
Date: December 2025
"""

import pytest
import asyncio
from typing import Dict, Any

# Chain imports
from hololoom.chaining import (
    Chain,
    ChainStep,
    StepType,
    ChainPatterns,
    Conditions,
    CommonConditions,
)

# New condition imports
from hololoom.chaining import (
    FactCheckConditions,
    CodeReviewConditions,
    SafetyConditions,
    HallucinationConditions,
    RAGConditions,
    MemoryConditions,
    AgentConditions,
)

# Evaluation imports
from hololoom.chaining import (
    LLMJudge,
    ChainEvaluator,
    JudgeCriteria,
    JudgeConfig,
    JudgeScore,
    JudgeResult,
    TestCase,
    ChainEvalResult,
    EvalPresets,
    create_evaluator,
    create_judge,
)


# =============================================================================
# Test New Chain Templates
# =============================================================================

class TestNewChainPatterns:
    """Test the 8 new chain pattern templates."""

    def test_fact_check_pattern_exists(self):
        """fact_check pattern should be available."""
        chain = ChainPatterns.fact_check()
        assert isinstance(chain, Chain)
        assert chain.name == "fact_check"
        assert chain.entry_point == "extract_claims"

    def test_fact_check_has_required_steps(self):
        """fact_check should have all required steps."""
        chain = ChainPatterns.fact_check()
        expected_steps = ["extract_claims", "verify_sources", "check_all_verified",
                         "generate_verdict", "flag_unverified", "generate_verdict_with_warnings"]
        for step_name in expected_steps:
            assert step_name in chain.steps, f"Missing step: {step_name}"

    def test_code_review_pattern_exists(self):
        """code_review pattern should be available."""
        chain = ChainPatterns.code_review()
        assert isinstance(chain, Chain)
        assert chain.name == "code_review"
        assert chain.entry_point == "security_scan"

    def test_code_review_has_required_steps(self):
        """code_review should have all required steps."""
        chain = ChainPatterns.code_review()
        expected_steps = ["security_scan", "check_security", "style_check", "suggest_fixes",
                         "flag_security_issues"]
        for step_name in expected_steps:
            assert step_name in chain.steps, f"Missing step: {step_name}"

    def test_summarize_pattern_exists(self):
        """summarize pattern should be available."""
        chain = ChainPatterns.summarize()
        assert isinstance(chain, Chain)
        assert chain.name == "summarize"
        assert chain.entry_point == "extract_key_points"

    def test_summarize_has_required_steps(self):
        """summarize should have all required steps."""
        chain = ChainPatterns.summarize()
        expected_steps = ["extract_key_points", "draft_summary", "verify_accuracy",
                         "check_accuracy", "finalize_summary", "refine_summary"]
        for step_name in expected_steps:
            assert step_name in chain.steps, f"Missing step: {step_name}"

    def test_safety_gated_pattern_exists(self):
        """safety_gated pattern should be available."""
        chain = ChainPatterns.safety_gated()
        assert isinstance(chain, Chain)
        assert chain.name == "safety_gated"
        assert chain.entry_point == "safety_check"

    def test_safety_gated_has_required_steps(self):
        """safety_gated should have all required steps."""
        chain = ChainPatterns.safety_gated()
        expected_steps = ["safety_check", "check_safety", "execute", "audit",
                         "block_unsafe", "output_safe"]
        for step_name in expected_steps:
            assert step_name in chain.steps, f"Missing step: {step_name}"

    def test_memory_augmented_pattern_exists(self):
        """memory_augmented pattern should be available."""
        chain = ChainPatterns.memory_augmented()
        assert isinstance(chain, Chain)
        assert chain.name == "memory_augmented"
        assert chain.entry_point == "recall"

    def test_memory_augmented_has_required_steps(self):
        """memory_augmented should have all required steps."""
        chain = ChainPatterns.memory_augmented()
        expected_steps = ["recall", "execute", "check_new_knowledge", "store", "output"]
        for step_name in expected_steps:
            assert step_name in chain.steps, f"Missing step: {step_name}"

    def test_hallucination_guard_pattern_exists(self):
        """hallucination_guard pattern should be available."""
        chain = ChainPatterns.hallucination_guard()
        assert isinstance(chain, Chain)
        assert chain.name == "hallucination_guard"
        assert chain.entry_point == "generate"

    def test_hallucination_guard_has_required_steps(self):
        """hallucination_guard should have all required steps."""
        chain = ChainPatterns.hallucination_guard()
        expected_steps = ["generate", "verify_claims", "check_all_sourced",
                         "output_verified", "filter_unsourced", "regenerate_with_sources"]
        for step_name in expected_steps:
            assert step_name in chain.steps, f"Missing step: {step_name}"

    def test_rag_optimized_pattern_exists(self):
        """rag_optimized pattern should be available."""
        chain = ChainPatterns.rag_optimized()
        assert isinstance(chain, Chain)
        assert chain.name == "rag_optimized"
        assert chain.entry_point == "retrieve"

    def test_rag_optimized_has_required_steps(self):
        """rag_optimized should have all required steps."""
        chain = ChainPatterns.rag_optimized()
        expected_steps = ["retrieve", "rerank", "generate", "cite", "verify_citations",
                         "check_citations", "output_with_citations", "fix_citations"]
        for step_name in expected_steps:
            assert step_name in chain.steps, f"Missing step: {step_name}"

    def test_agent_planning_pattern_exists(self):
        """agent_planning pattern should be available."""
        chain = ChainPatterns.agent_planning()
        assert isinstance(chain, Chain)
        assert chain.name == "agent_planning"
        assert chain.entry_point == "decompose"

    def test_agent_planning_has_required_steps(self):
        """agent_planning should have all required steps."""
        chain = ChainPatterns.agent_planning()
        expected_steps = ["decompose", "plan", "execute_step", "check_more_steps",
                         "reflect", "check_retry", "output_result"]
        for step_name in expected_steps:
            assert step_name in chain.steps, f"Missing step: {step_name}"


# =============================================================================
# Test Domain-Specific Conditions
# =============================================================================

class TestFactCheckConditions:
    """Test conditions for fact-checking chains."""

    def test_all_claims_verified_true(self):
        """all_claims_verified should return True when all claims are verified."""
        condition = FactCheckConditions.all_claims_verified()
        ctx = {
            "claims": [
                {"text": "Claim 1", "verified": True},
                {"text": "Claim 2", "status": "verified"},
            ]
        }
        assert condition(ctx) is True

    def test_all_claims_verified_false(self):
        """all_claims_verified should return False with unverified claims."""
        condition = FactCheckConditions.all_claims_verified()
        ctx = {
            "claims": [
                {"text": "Claim 1", "verified": True},
                {"text": "Claim 2", "verified": False},
            ]
        }
        assert condition(ctx) is False

    def test_all_claims_verified_empty(self):
        """all_claims_verified should return False with no claims."""
        condition = FactCheckConditions.all_claims_verified()
        ctx = {"claims": []}
        assert condition(ctx) is False

    def test_citation_coverage_above(self):
        """citation_coverage_above should check citation ratio."""
        condition = FactCheckConditions.citation_coverage_above(0.5)
        ctx = {
            "claims": [
                {"text": "Claim 1", "citations": ["source1"]},
                {"text": "Claim 2", "sources": ["source2"]},
                {"text": "Claim 3"},  # No citations
            ]
        }
        assert condition(ctx) is True  # 2/3 = 0.67 >= 0.5

    def test_has_unverified_claims(self):
        """has_unverified_claims should detect unverified claims."""
        condition = FactCheckConditions.has_unverified_claims()
        ctx = {
            "claims": [
                {"text": "Claim 1", "verified": True},
                {"text": "Claim 2", "verified": False},
            ]
        }
        assert condition(ctx) is True


class TestCodeReviewConditions:
    """Test conditions for code review chains."""

    def test_no_security_issues_true(self):
        """no_security_issues should return True when no issues."""
        condition = CodeReviewConditions.no_security_issues()
        ctx = {"security_issues": [], "security_score": 1.0}
        assert condition(ctx) is True

    def test_no_security_issues_false(self):
        """no_security_issues should return False with issues."""
        condition = CodeReviewConditions.no_security_issues()
        ctx = {"security_issues": ["SQL injection"], "security_score": 0.5}
        assert condition(ctx) is False

    def test_style_score_above(self):
        """style_score_above should check style score threshold."""
        condition = CodeReviewConditions.style_score_above(0.8)
        assert condition({"style_score": 0.9}) is True
        assert condition({"style_score": 0.7}) is False

    def test_has_critical_issues(self):
        """has_critical_issues should detect critical severity."""
        condition = CodeReviewConditions.has_critical_issues()
        ctx = {
            "issues": [
                {"type": "warning", "severity": "low"},
                {"type": "error", "severity": "critical"},
            ]
        }
        assert condition(ctx) is True


class TestSafetyConditions:
    """Test conditions for safety-gated chains."""

    def test_safety_score_above(self):
        """safety_score_above should check safety threshold."""
        condition = SafetyConditions.safety_score_above(0.8)
        assert condition({"safety_score": 0.9}) is True
        assert condition({"safety_score": 0.7}) is False

    def test_audit_trail_complete(self):
        """audit_trail_complete should check for required entries."""
        condition = SafetyConditions.audit_trail_complete()
        ctx = {
            "audit_trail": [
                {"type": "input", "timestamp": "2025-12-06T10:00:00"},
                {"type": "decision", "timestamp": "2025-12-06T10:00:01"},
                {"type": "output", "timestamp": "2025-12-06T10:00:02"},
            ]
        }
        assert condition(ctx) is True

    def test_audit_trail_incomplete(self):
        """audit_trail_complete should return False when missing entries."""
        condition = SafetyConditions.audit_trail_complete()
        ctx = {
            "audit_trail": [
                {"type": "input", "timestamp": "2025-12-06T10:00:00"},
            ]
        }
        assert condition(ctx) is False

    def test_risk_level_acceptable(self):
        """risk_level_acceptable should check for low/medium risk."""
        condition = SafetyConditions.risk_level_acceptable()
        assert condition({"risk_level": "low"}) is True
        assert condition({"risk_level": "medium"}) is True
        assert condition({"risk_level": "high"}) is False


class TestHallucinationConditions:
    """Test conditions for hallucination guard chains."""

    def test_all_claims_sourced_true(self):
        """all_claims_sourced should return True when all claims have sources."""
        condition = HallucinationConditions.all_claims_sourced()
        ctx = {
            "claims": [
                {"text": "Claim 1", "sourced": True},
                {"text": "Claim 2", "sources": ["source1"]},
            ]
        }
        assert condition(ctx) is True

    def test_all_claims_sourced_false(self):
        """all_claims_sourced should return False with unsourced claims."""
        condition = HallucinationConditions.all_claims_sourced()
        ctx = {
            "claims": [
                {"text": "Claim 1", "sourced": True},
                {"text": "Claim 2", "sources": []},
            ]
        }
        assert condition(ctx) is False

    def test_hallucination_score_below(self):
        """hallucination_score_below should check score threshold."""
        condition = HallucinationConditions.hallucination_score_below(0.3)
        assert condition({"hallucination_score": 0.1}) is True
        assert condition({"hallucination_score": 0.5}) is False


class TestRAGConditions:
    """Test conditions for RAG chains."""

    def test_retrieval_quality_above(self):
        """retrieval_quality_above should check quality threshold."""
        condition = RAGConditions.retrieval_quality_above(0.7)
        assert condition({"retrieval_quality": 0.8}) is True
        assert condition({"retrieval_quality": 0.6}) is False

    def test_has_relevant_documents(self):
        """has_relevant_documents should check for relevant docs."""
        condition = RAGConditions.has_relevant_documents(2)
        ctx = {
            "retrieved_documents": [
                {"id": "1", "relevance": 0.8},
                {"id": "2", "relevance": 0.7},
                {"id": "3", "relevance": 0.3},
            ]
        }
        assert condition(ctx) is True

    def test_rerank_improved(self):
        """rerank_improved should detect improvement."""
        condition = RAGConditions.rerank_improved()
        ctx = {"original_ranking_score": 0.6, "reranked_score": 0.8}
        assert condition(ctx) is True


class TestMemoryConditions:
    """Test conditions for memory-augmented chains."""

    def test_memory_recall_successful(self):
        """memory_recall_successful should check for recalled memories."""
        condition = MemoryConditions.memory_recall_successful()
        assert condition({"recalled_memories": [{"id": "1"}]}) is True
        assert condition({"recalled_memories": []}) is False

    def test_memory_confidence_above(self):
        """memory_confidence_above should check average confidence."""
        condition = MemoryConditions.memory_confidence_above(0.7)
        ctx = {
            "recalled_memories": [
                {"id": "1", "confidence": 0.8},
                {"id": "2", "confidence": 0.9},
            ]
        }
        assert condition(ctx) is True  # avg = 0.85 >= 0.7

    def test_new_knowledge_detected(self):
        """new_knowledge_detected should check flag."""
        condition = MemoryConditions.new_knowledge_detected()
        assert condition({"new_knowledge_detected": True}) is True
        assert condition({"new_knowledge_detected": False}) is False


class TestAgentConditions:
    """Test conditions for agent planning chains."""

    def test_plan_valid(self):
        """plan_valid should check plan validity."""
        condition = AgentConditions.plan_valid()
        ctx = {"plan": {"steps": [{"id": 1}], "valid": True}}
        assert condition(ctx) is True
        ctx = {"plan": {"steps": [], "valid": True}}
        assert condition(ctx) is False

    def test_all_steps_complete(self):
        """all_steps_complete should check all steps are done."""
        condition = AgentConditions.all_steps_complete()
        ctx = {
            "plan": {
                "steps": [
                    {"id": 1, "status": "complete"},
                    {"id": 2, "status": "complete"},
                ]
            }
        }
        assert condition(ctx) is True

    def test_goal_achieved(self):
        """goal_achieved should check goal flag."""
        condition = AgentConditions.goal_achieved()
        assert condition({"goal_achieved": True}) is True
        assert condition({"goal_achieved": False}) is False

    def test_subtasks_remaining(self):
        """subtasks_remaining should detect incomplete subtasks."""
        condition = AgentConditions.subtasks_remaining()
        ctx = {
            "subtasks": [
                {"id": 1, "status": "complete"},
                {"id": 2, "status": "pending"},
            ]
        }
        assert condition(ctx) is True


# =============================================================================
# Test Evaluation System
# =============================================================================

class TestJudgeCriteria:
    """Test evaluation criteria enum."""

    def test_all_criteria_exist(self):
        """All expected criteria should exist."""
        expected = [
            "quality", "relevance", "coherence", "accuracy",
            "helpfulness", "safety", "creativity", "conciseness",
            "completeness", "citation_accuracy", "source_coverage",
            "step_coherence"
        ]
        actual = [c.value for c in JudgeCriteria]
        for criterion in expected:
            assert criterion in actual, f"Missing criterion: {criterion}"


class TestJudgeConfig:
    """Test judge configuration."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = JudgeConfig(criteria=[JudgeCriteria.QUALITY])
        assert config.provider == "ollama"
        assert config.model == "llama3.2:3b"
        assert config.temperature == 0.3
        assert config.use_rubric is True

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = JudgeConfig(
            criteria=[JudgeCriteria.QUALITY, JudgeCriteria.ACCURACY],
            provider="anthropic",
            model="claude-3-sonnet",
            temperature=0.5
        )
        assert config.provider == "anthropic"
        assert config.model == "claude-3-sonnet"
        assert len(config.criteria) == 2


class TestEvalPresets:
    """Test evaluation presets."""

    def test_quality_eval(self):
        """quality_eval should have standard quality criteria."""
        config = EvalPresets.quality_eval()
        criteria_values = [c.value for c in config.criteria]
        assert "quality" in criteria_values
        assert "relevance" in criteria_values
        assert "coherence" in criteria_values
        assert config.provider == "ollama"

    def test_safety_eval(self):
        """safety_eval should focus on safety."""
        config = EvalPresets.safety_eval()
        criteria_values = [c.value for c in config.criteria]
        assert "safety" in criteria_values
        assert "helpfulness" in criteria_values
        assert "accuracy" in criteria_values

    def test_rag_eval(self):
        """rag_eval should include citation criteria."""
        config = EvalPresets.rag_eval()
        criteria_values = [c.value for c in config.criteria]
        assert "citation_accuracy" in criteria_values
        assert "source_coverage" in criteria_values

    def test_chain_eval(self):
        """chain_eval should include step coherence."""
        config = EvalPresets.chain_eval()
        criteria_values = [c.value for c in config.criteria]
        assert "step_coherence" in criteria_values

    def test_comprehensive_eval(self):
        """comprehensive_eval should include many criteria."""
        config = EvalPresets.comprehensive_eval()
        assert len(config.criteria) >= 5


class TestLLMJudge:
    """Test LLM judge functionality."""

    def test_judge_creation(self):
        """LLMJudge should be creatable."""
        judge = LLMJudge()
        assert judge.provider == "ollama"
        assert judge.model == "llama3.2:3b"

    def test_judge_with_custom_provider(self):
        """LLMJudge should accept custom provider."""
        judge = LLMJudge(provider="anthropic", model="claude-3-sonnet")
        assert judge.provider == "anthropic"
        assert judge.model == "claude-3-sonnet"

    @pytest.mark.asyncio
    async def test_evaluate_mock(self):
        """evaluate should return JudgeResult (mock mode)."""
        judge = LLMJudge()
        config = EvalPresets.quality_eval()

        result = await judge.evaluate(
            task="Explain Thompson Sampling",
            output="Thompson Sampling is a Bayesian algorithm...",
            config=config
        )

        assert isinstance(result, JudgeResult)
        assert len(result.scores) == len(config.criteria)
        assert 0.0 <= result.overall_score <= 1.0
        assert result.summary != ""


class TestChainEvaluator:
    """Test chain evaluator."""

    def test_evaluator_creation(self):
        """ChainEvaluator should be creatable."""
        evaluator = create_evaluator()
        assert evaluator.judge is not None

    def test_evaluator_with_judge(self):
        """ChainEvaluator should accept custom judge."""
        judge = create_judge(provider="ollama", model="llama3.2:3b")
        evaluator = ChainEvaluator(judge=judge)
        assert evaluator.judge is judge


class TestTestCase:
    """Test TestCase dataclass."""

    def test_test_case_creation(self):
        """TestCase should be creatable with required fields."""
        tc = TestCase(
            input_data={"query": "What is AI?"},
            expected_output="AI stands for Artificial Intelligence..."
        )
        assert tc.input_data["query"] == "What is AI?"
        assert tc.expected_output is not None

    def test_test_case_to_dict(self):
        """TestCase should serialize to dict."""
        tc = TestCase(
            input_data={"query": "test"},
            expected_output="result",
            metadata={"source": "unit_test"}
        )
        d = tc.to_dict()
        assert d["input_data"]["query"] == "test"
        assert d["metadata"]["source"] == "unit_test"


class TestJudgeResult:
    """Test JudgeResult dataclass."""

    def test_judge_result_creation(self):
        """JudgeResult should be creatable."""
        scores = [
            JudgeScore(criterion="quality", score=0.8, reasoning="Good"),
            JudgeScore(criterion="relevance", score=0.9, reasoning="Very relevant"),
        ]
        result = JudgeResult(
            output="Test output",
            scores=scores,
            overall_score=0.85,
            summary="quality: 0.80, relevance: 0.90"
        )
        assert result.overall_score == 0.85
        assert len(result.scores) == 2

    def test_get_score_by_criterion(self):
        """get_score_by_criterion should find correct score."""
        scores = [
            JudgeScore(criterion="quality", score=0.8, reasoning="Good"),
            JudgeScore(criterion="relevance", score=0.9, reasoning="Very relevant"),
        ]
        result = JudgeResult(
            output="Test",
            scores=scores,
            overall_score=0.85,
            summary=""
        )
        quality_score = result.get_score_by_criterion("quality")
        assert quality_score is not None
        assert quality_score.score == 0.8


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for chains with evaluation."""

    def test_pattern_with_conditions(self):
        """Chain patterns should work with domain conditions."""
        chain = ChainPatterns.fact_check()

        # Simulate context with verified claims
        ctx = {
            "claims": [
                {"text": "Claim 1", "verified": True, "citations": ["source1"]},
                {"text": "Claim 2", "verified": True, "citations": ["source2"]},
            ],
            "confidence": 0.9
        }

        # Test conditions
        assert FactCheckConditions.all_claims_verified()(ctx) is True
        assert FactCheckConditions.citation_coverage_above(0.9)(ctx) is True
        assert Conditions.confidence_above(0.8)(ctx) is True

    def test_safety_chain_with_conditions(self):
        """Safety chain should work with safety conditions."""
        chain = ChainPatterns.safety_gated()

        ctx = {
            "safety_score": 0.95,
            "risk_level": "low",
            "audit_trail": [
                {"type": "input", "timestamp": "2025-12-06T10:00:00"},
                {"type": "decision", "timestamp": "2025-12-06T10:00:01"},
                {"type": "output", "timestamp": "2025-12-06T10:00:02"},
            ]
        }

        assert SafetyConditions.safety_score_above(0.9)(ctx) is True
        assert SafetyConditions.risk_level_acceptable()(ctx) is True
        assert SafetyConditions.audit_trail_complete()(ctx) is True

    def test_all_patterns_are_chains(self):
        """All pattern methods should return Chain instances."""
        patterns = [
            ChainPatterns.fact_check,
            ChainPatterns.code_review,
            ChainPatterns.summarize,
            ChainPatterns.safety_gated,
            ChainPatterns.memory_augmented,
            ChainPatterns.hallucination_guard,
            ChainPatterns.rag_optimized,
            ChainPatterns.agent_planning,
        ]

        for pattern_fn in patterns:
            chain = pattern_fn()
            assert isinstance(chain, Chain), f"{pattern_fn.__name__} should return Chain"
            assert chain.name != "", f"{pattern_fn.__name__} should have a name"
            assert chain.entry_point != "", f"{pattern_fn.__name__} should have entry_point"
            assert len(chain.steps) > 0, f"{pattern_fn.__name__} should have steps"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
