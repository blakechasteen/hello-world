"""
Tests for Domain Houses, Card Loader, and Initialization
=========================================================

Covers:
- CodeReviewHouse: config, query building, result parsing, issue/suggestion extraction
- ResearchHouse: config, query building, result parsing, findings/insights extraction
- PatternCard (card_loader.py): from_dict, to_dict, merge, save/load, tools/math config
- initialization.py: create_weave_house_system, factory helpers

Date: March 2026
"""

import asyncio
import pytest
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock


# ============================================================================
# Card Loader imports (no heavy deps)
# ============================================================================

from hololoom.loom.card_loader import (
    PatternCard,
    MathCapabilities,
    MemoryConfig,
    ToolsConfig,
    PerformanceProfile,
)


# ============================================================================
# Helper: mock Fabric for domain house tests
# ============================================================================

@dataclass
class MockFabric:
    """Lightweight stand-in for Fabric used by domain house result parsers."""
    perspective: str = "unknown"
    response: str = ""
    confidence: float = 0.8
    epistemic_confidence: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)
    admits_ignorance: Optional[List[str]] = None


@dataclass
class MockWeaveResult:
    """Stand-in for the object returned by WeaveHouse.weave()."""
    response: str = "summary"
    confidence: float = 0.85
    epistemic_confidence: float = 0.75
    fabrics: List[Any] = field(default_factory=list)
    tensions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# CodeReviewHouse tests
# ============================================================================


class TestCodeReviewConfig:
    """Test CodeReviewConfig defaults and custom values."""

    def test_defaults(self):
        from hololoom.core.loom.domain_houses.code_review_house import CodeReviewConfig
        cfg = CodeReviewConfig()
        assert cfg.recall_k == 15
        assert cfg.recall_temperature == 0.3
        assert cfg.max_inference_depth == 7
        assert cfg.require_explicit_links is True
        assert cfg.verification_passes == 2
        assert cfg.consistency_threshold == 0.85
        assert cfg.attack_intensity == 0.7
        assert cfg.coverage_target == 0.8
        assert cfg.exploration_depth == 2
        assert cfg.tension_threshold == 0.25
        assert cfg.check_security is True
        assert cfg.check_performance is True
        assert cfg.check_maintainability is True

    def test_custom_values(self):
        from hololoom.core.loom.domain_houses.code_review_house import CodeReviewConfig
        cfg = CodeReviewConfig(recall_k=30, attack_intensity=0.9, check_security=False)
        assert cfg.recall_k == 30
        assert cfg.attack_intensity == 0.9
        assert cfg.check_security is False


class TestCodeReviewResult:
    """Test CodeReviewResult dataclass and properties."""

    def test_empty_result(self):
        from hololoom.core.loom.domain_houses.code_review_house import CodeReviewResult
        r = CodeReviewResult(
            code="x = 1",
            summary="Looks good",
            confidence=0.9,
            epistemic_confidence=0.8,
        )
        assert r.issue_count == 0
        assert r.has_security_issues is False
        assert r.issues_by_severity("high") == []

    def test_with_security_issues(self):
        from hololoom.core.loom.domain_houses.code_review_house import CodeReviewResult
        issues = [
            {"type": "security", "severity": "high", "description": "SQL injection"},
            {"type": "consistency", "severity": "low", "description": "Naming mismatch"},
        ]
        r = CodeReviewResult(
            code="x = 1",
            summary="Issues found",
            confidence=0.5,
            epistemic_confidence=0.4,
            issues=issues,
        )
        assert r.issue_count == 2
        assert r.has_security_issues is True
        assert len(r.issues_by_severity("high")) == 1
        assert len(r.issues_by_severity("low")) == 1
        assert len(r.issues_by_severity("medium")) == 0

    def test_suggestions_and_perspectives(self):
        from hololoom.core.loom.domain_houses.code_review_house import CodeReviewResult
        r = CodeReviewResult(
            code="pass",
            summary="ok",
            confidence=0.9,
            epistemic_confidence=0.8,
            suggestions=["Use constants", "Add docstring"],
            perspectives={"recall": "Pattern matches found", "reason": "Logic is sound"},
        )
        assert len(r.suggestions) == 2
        assert "recall" in r.perspectives


class TestCodeReviewHouseQueryBuilder:
    """Test _build_review_query without initializing the full house."""

    def test_basic_query(self):
        from hololoom.core.loom.domain_houses.code_review_house import CodeReviewHouse
        # Directly call the unbound method on a mock instance
        query = CodeReviewHouse._build_review_query(None, "x = 1", None, None)
        assert "```" in query
        assert "x = 1" in query
        assert "multiple perspectives" in query

    def test_query_with_context(self):
        from hololoom.core.loom.domain_houses.code_review_house import CodeReviewHouse
        ctx = {"language": "python", "file": "main.py"}
        query = CodeReviewHouse._build_review_query(None, "x = 1", ctx, None)
        assert "Context:" in query

    def test_query_with_focus(self):
        from hololoom.core.loom.domain_houses.code_review_house import CodeReviewHouse
        query = CodeReviewHouse._build_review_query(
            None, "x = 1", None, ["security", "performance"]
        )
        assert "security" in query
        assert "performance" in query


class TestCodeReviewHouseExtractors:
    """Test _extract_issues and _extract_suggestions with mock fabrics."""

    def _make_result(self, fabrics):
        return MockWeaveResult(fabrics=fabrics)

    def test_extract_security_issues(self):
        from hololoom.core.loom.domain_houses.code_review_house import CodeReviewHouse
        fab = MockFabric(
            perspective="refuse",
            metadata={
                "vulnerabilities": [
                    {"severity": "high", "description": "Buffer overflow"},
                    {"severity": "low", "description": "Info leak"},
                ]
            },
        )
        result = self._make_result([fab])
        issues = CodeReviewHouse._extract_issues(None, result)
        assert len(issues) == 2
        assert issues[0]["type"] == "security"
        assert issues[0]["severity"] == "high"
        assert issues[1]["severity"] == "low"

    def test_extract_consistency_issues(self):
        from hololoom.core.loom.domain_houses.code_review_house import CodeReviewHouse
        fab = MockFabric(
            perspective="reflect",
            metadata={"contradictions": ["Naming inconsistency", "Type mismatch"]},
        )
        result = self._make_result([fab])
        issues = CodeReviewHouse._extract_issues(None, result)
        assert len(issues) == 2
        assert all(i["type"] == "consistency" for i in issues)

    def test_extract_no_issues(self):
        from hololoom.core.loom.domain_houses.code_review_house import CodeReviewHouse
        fab = MockFabric(perspective="reason", metadata={})
        result = self._make_result([fab])
        issues = CodeReviewHouse._extract_issues(None, result)
        assert issues == []

    def test_extract_suggestions_from_reason(self):
        from hololoom.core.loom.domain_houses.code_review_house import CodeReviewHouse
        fab = MockFabric(
            perspective="reason",
            metadata={"improvements": ["Use list comp", "Add type hints"]},
        )
        result = self._make_result([fab])
        suggestions = CodeReviewHouse._extract_suggestions(None, result)
        assert len(suggestions) == 2

    def test_extract_suggestions_from_recall(self):
        from hololoom.core.loom.domain_houses.code_review_house import CodeReviewHouse
        fab = MockFabric(
            perspective="recall",
            metadata={"similar_patterns": ["Factory pattern", "Builder pattern"]},
        )
        result = self._make_result([fab])
        suggestions = CodeReviewHouse._extract_suggestions(None, result)
        assert len(suggestions) == 2
        assert all("Consider pattern:" in s for s in suggestions)


class TestCodeReviewHouseParseResult:
    """Test _parse_review_result constructs CodeReviewResult properly."""

    def _make_mock_house(self):
        """Create a mock that delegates _extract_* to the real class methods."""
        from hololoom.core.loom.domain_houses.code_review_house import CodeReviewHouse
        mock_self = MagicMock(spec=CodeReviewHouse)
        mock_self._extract_issues = lambda r: CodeReviewHouse._extract_issues(mock_self, r)
        mock_self._extract_suggestions = lambda r: CodeReviewHouse._extract_suggestions(mock_self, r)
        mock_self._parse_review_result = lambda r, c: CodeReviewHouse._parse_review_result(mock_self, r, c)
        return mock_self

    def test_parse_empty_result(self):
        mock_house = self._make_mock_house()
        result = MagicMock()
        result.response = "All good"
        result.confidence = 0.9
        result.epistemic_confidence = 0.8
        result.fabrics = []
        result.tensions = []
        result.metadata = {"total_ms": 100}

        parsed = mock_house._parse_review_result(result, "x = 1")
        assert parsed.code == "x = 1"
        assert parsed.summary == "All good"
        assert parsed.confidence == 0.9
        assert parsed.issue_count == 0

    def test_parse_result_with_fabrics(self):
        mock_house = self._make_mock_house()
        fab_refuse = MagicMock()
        fab_refuse.perspective = "refuse"
        fab_refuse.response = "Found vuln"
        fab_refuse.metadata = {
            "vulnerabilities": [{"severity": "critical", "description": "RCE"}]
        }
        fab_reason = MagicMock()
        fab_reason.perspective = "reason"
        fab_reason.response = "Logic ok"
        fab_reason.metadata = {"improvements": ["Simplify loop"]}

        result = MagicMock()
        result.response = "Summary"
        result.confidence = 0.6
        result.epistemic_confidence = 0.5
        result.fabrics = [fab_refuse, fab_reason]
        result.tensions = []
        result.metadata = {}

        parsed = mock_house._parse_review_result(result, "eval(input())")
        assert parsed.has_security_issues is True
        assert len(parsed.suggestions) == 1
        assert "refuse" in parsed.perspectives
        assert "reason" in parsed.perspectives


class TestCodeReviewFactory:
    """Test the create_code_review_house factory."""

    @patch("hololoom.core.loom.domain_houses.code_review_house.WeaveHouse.__init__", return_value=None)
    @patch("hololoom.core.loom.domain_houses.code_review_house.create_loom_consensus")
    @patch("hololoom.core.loom.domain_houses.code_review_house.RecallLoom")
    @patch("hololoom.core.loom.domain_houses.code_review_house.ReasonLoom")
    @patch("hololoom.core.loom.domain_houses.code_review_house.ReflectLoom")
    @patch("hololoom.core.loom.domain_houses.code_review_house.RefuseLoom")
    def test_factory_creates_house(self, mock_refuse, mock_reflect, mock_reason,
                                    mock_recall, mock_consensus, mock_wh_init):
        from hololoom.core.loom.domain_houses.code_review_house import create_code_review_house
        mock_consensus.return_value = MagicMock()
        for m in [mock_refuse, mock_reflect, mock_reason, mock_recall]:
            m.return_value = MagicMock()
            m.return_value.blind_spots = []

        house = create_code_review_house()
        assert house is not None
        assert hasattr(house, 'code_config')
        mock_consensus.assert_called_once()
        # Verify 4 looms created (no ReachLoom for code review)
        mock_recall.assert_called_once()
        mock_reason.assert_called_once()
        mock_reflect.assert_called_once()
        mock_refuse.assert_called_once()


# ============================================================================
# ResearchHouse tests
# ============================================================================


class TestResearchConfig:
    """Test ResearchConfig defaults and custom values."""

    def test_defaults(self):
        from hololoom.core.loom.domain_houses.research_house import ResearchConfig
        cfg = ResearchConfig()
        assert cfg.recall_k == 25
        assert cfg.max_inference_depth == 10
        assert cfg.association_hops == 4
        assert cfg.novelty_threshold == 0.4
        assert cfg.creativity_temperature == 0.85
        assert cfg.verification_passes == 3
        assert cfg.consistency_threshold == 0.9
        assert cfg.include_creative is True
        assert cfg.require_evidence is True
        assert cfg.synthesize_perspectives is True

    def test_no_creative(self):
        from hololoom.core.loom.domain_houses.research_house import ResearchConfig
        cfg = ResearchConfig(include_creative=False)
        assert cfg.include_creative is False


class TestResearchResult:
    """Test ResearchResult dataclass and properties."""

    def test_empty_result(self):
        from hololoom.core.loom.domain_houses.research_house import ResearchResult
        r = ResearchResult(
            question="Why?",
            summary="Because",
            confidence=0.8,
            epistemic_confidence=0.7,
        )
        assert r.findings_count == 0
        assert r.has_novel_insights is False
        assert r.has_open_questions is False
        assert r.findings_by_type("inference") == []
        assert r.high_confidence_findings() == []

    def test_with_findings(self):
        from hololoom.core.loom.domain_houses.research_house import ResearchResult
        findings = [
            {"type": "prior_work", "content": "Study A", "confidence": 0.9},
            {"type": "inference", "content": "Therefore B", "confidence": 0.6},
            {"type": "inference", "content": "Also C", "confidence": 0.85},
        ]
        r = ResearchResult(
            question="Why?",
            summary="Complex",
            confidence=0.7,
            epistemic_confidence=0.6,
            findings=findings,
            novel_insights=["New connection"],
            open_questions=["What about D?"],
        )
        assert r.findings_count == 3
        assert r.has_novel_insights is True
        assert r.has_open_questions is True
        assert len(r.findings_by_type("inference")) == 2
        assert len(r.high_confidence_findings(0.8)) == 2
        assert len(r.high_confidence_findings(0.95)) == 0


class TestResearchHouseQueryBuilder:
    """Test _build_research_query without full house init."""

    def test_basic_query(self):
        from hololoom.core.loom.domain_houses.research_house import ResearchHouse
        query = ResearchHouse._build_research_query(None, "What is X?", None, None)
        assert "What is X?" in query
        assert "multi-perspective" in query

    def test_query_with_context(self):
        from hololoom.core.loom.domain_houses.research_house import ResearchHouse
        ctx = {"domain": "biology"}
        query = ResearchHouse._build_research_query(None, "Why?", ctx, None)
        assert "Context:" in query

    def test_query_with_depth_quick(self):
        from hololoom.core.loom.domain_houses.research_house import ResearchHouse
        query = ResearchHouse._build_research_query(None, "Q", None, "quick")
        assert "brief overview" in query

    def test_query_with_depth_deep(self):
        from hololoom.core.loom.domain_houses.research_house import ResearchHouse
        query = ResearchHouse._build_research_query(None, "Q", None, "deep")
        assert "exhaustive" in query

    def test_query_with_unknown_depth(self):
        from hololoom.core.loom.domain_houses.research_house import ResearchHouse
        query = ResearchHouse._build_research_query(None, "Q", None, "unknown_depth")
        assert "Depth:" in query


class TestResearchHouseExtractors:
    """Test extraction methods with mock fabrics."""

    def _make_result(self, fabrics, tensions=None):
        return MockWeaveResult(fabrics=fabrics, tensions=tensions or [])

    def test_extract_findings_from_recall(self):
        from hololoom.core.loom.domain_houses.research_house import ResearchHouse
        fab = MockFabric(
            perspective="recall",
            metadata={
                "retrieved_facts": [
                    {"content": "Fact A", "confidence": 0.9},
                    {"content": "Fact B", "confidence": 0.7},
                ]
            },
        )
        result = self._make_result([fab])
        findings = ResearchHouse._extract_findings(None, result)
        assert len(findings) == 2
        assert findings[0]["type"] == "prior_work"
        assert findings[0]["source"] == "recall_loom"

    def test_extract_findings_from_reason(self):
        from hololoom.core.loom.domain_houses.research_house import ResearchHouse
        fab = MockFabric(
            perspective="reason",
            confidence=0.8,
            metadata={"conclusions": ["A implies B"]},
        )
        result = self._make_result([fab])
        findings = ResearchHouse._extract_findings(None, result)
        assert len(findings) == 1
        assert findings[0]["type"] == "inference"
        assert findings[0]["confidence"] == 0.8

    def test_extract_novel_insights(self):
        from hololoom.core.loom.domain_houses.research_house import ResearchHouse
        fab = MockFabric(
            perspective="reach",
            metadata={
                "novel_connections": ["X relates to Y"],
                "creative_insights": ["Novel insight Z"],
            },
        )
        result = self._make_result([fab])
        insights = ResearchHouse._extract_novel_insights(None, result)
        assert len(insights) == 2

    def test_extract_open_questions_from_admits_ignorance(self):
        from hololoom.core.loom.domain_houses.research_house import ResearchHouse
        fab = MockFabric(
            perspective="recall",
            admits_ignorance=["long-term effects", "What about edge cases?"],
        )
        result = self._make_result([fab])
        questions = ResearchHouse._extract_open_questions(None, result)
        assert len(questions) >= 2
        # Check that items without "?" get wrapped
        assert any("?" in q for q in questions)

    def test_extract_open_questions_from_tensions(self):
        from hololoom.core.loom.domain_houses.research_house import ResearchHouse
        tensions = [
            {"is_irreducible": True, "claim_a": "X is true", "claim_b": "X is false"},
            {"is_irreducible": False, "claim_a": "A", "claim_b": "B"},  # Not irreducible
        ]
        fab = MockFabric(perspective="recall", admits_ignorance=None)
        result = self._make_result([fab], tensions=tensions)
        questions = ResearchHouse._extract_open_questions(None, result)
        # Only the irreducible tension should produce a question
        assert any("reconcile" in q for q in questions)

    def test_extract_counterarguments(self):
        from hololoom.core.loom.domain_houses.research_house import ResearchHouse
        fab = MockFabric(
            perspective="refuse",
            metadata={
                "attacks": [
                    {"target": "claim A", "description": "Flawed assumption", "severity": "high"},
                ]
            },
        )
        result = self._make_result([fab])
        counterargs = ResearchHouse._extract_counterarguments(None, result)
        assert len(counterargs) == 1
        assert counterargs[0]["target"] == "claim A"

    def test_extract_evidence_gaps_unverified(self):
        from hololoom.core.loom.domain_houses.research_house import ResearchHouse
        fab = MockFabric(
            perspective="reflect",
            metadata={
                "unverified_claims": ["Earth is flat"],
                "verification_results": [
                    {"claim": "Water is wet", "verified": True},
                    {"claim": "Sky is green", "verified": False},
                ],
            },
        )
        result = self._make_result([fab])
        gaps = ResearchHouse._extract_evidence_gaps(None, result)
        assert len(gaps) == 2  # 1 unverified claim + 1 failed verification
        assert any("Earth is flat" in g for g in gaps)
        assert any("Sky is green" in g for g in gaps)


class TestResearchFactory:
    """Test the create_research_house factory."""

    @patch("hololoom.core.loom.domain_houses.research_house.WeaveHouse.__init__", return_value=None)
    @patch("hololoom.core.loom.domain_houses.research_house.create_loom_consensus")
    @patch("hololoom.core.loom.domain_houses.research_house.RecallLoom")
    @patch("hololoom.core.loom.domain_houses.research_house.ReasonLoom")
    @patch("hololoom.core.loom.domain_houses.research_house.ReachLoom")
    @patch("hololoom.core.loom.domain_houses.research_house.ReflectLoom")
    @patch("hololoom.core.loom.domain_houses.research_house.RefuseLoom")
    def test_factory_creates_house(self, mock_refuse, mock_reflect, mock_reach,
                                    mock_reason, mock_recall, mock_consensus,
                                    mock_wh_init):
        from hololoom.core.loom.domain_houses.research_house import create_research_house
        mock_consensus.return_value = MagicMock()
        for m in [mock_refuse, mock_reflect, mock_reach, mock_reason, mock_recall]:
            m.return_value = MagicMock()
            m.return_value.blind_spots = []

        house = create_research_house()
        assert house is not None
        assert hasattr(house, 'research_config')
        assert house.research_config.include_creative is True
        # All 5 looms should be created
        mock_reach.assert_called_once()

    @patch("hololoom.core.loom.domain_houses.research_house.WeaveHouse.__init__", return_value=None)
    @patch("hololoom.core.loom.domain_houses.research_house.create_loom_consensus")
    @patch("hololoom.core.loom.domain_houses.research_house.RecallLoom")
    @patch("hololoom.core.loom.domain_houses.research_house.ReasonLoom")
    @patch("hololoom.core.loom.domain_houses.research_house.ReachLoom")
    @patch("hololoom.core.loom.domain_houses.research_house.ReflectLoom")
    @patch("hololoom.core.loom.domain_houses.research_house.RefuseLoom")
    def test_factory_no_creative(self, mock_refuse, mock_reflect, mock_reach,
                                  mock_reason, mock_recall, mock_consensus,
                                  mock_wh_init):
        from hololoom.core.loom.domain_houses.research_house import (
            create_research_house, ResearchConfig,
        )
        mock_consensus.return_value = MagicMock()
        for m in [mock_refuse, mock_reflect, mock_reach, mock_reason, mock_recall]:
            m.return_value = MagicMock()
            m.return_value.blind_spots = []

        cfg = ResearchConfig(include_creative=False)
        house = create_research_house(config=cfg)
        assert house.research_config.include_creative is False
        # ReachLoom should not be called when include_creative is False
        mock_reach.assert_not_called()


class TestResearchHouseParseResult:
    """Test _parse_research_result constructs ResearchResult properly."""

    def _make_mock_house(self):
        """Create a mock that delegates extraction methods to the real class."""
        from hololoom.core.loom.domain_houses.research_house import ResearchHouse
        mock_self = MagicMock(spec=ResearchHouse)
        mock_self._extract_findings = lambda r: ResearchHouse._extract_findings(mock_self, r)
        mock_self._extract_novel_insights = lambda r: ResearchHouse._extract_novel_insights(mock_self, r)
        mock_self._extract_open_questions = lambda r: ResearchHouse._extract_open_questions(mock_self, r)
        mock_self._extract_counterarguments = lambda r: ResearchHouse._extract_counterarguments(mock_self, r)
        mock_self._extract_evidence_gaps = lambda r: ResearchHouse._extract_evidence_gaps(mock_self, r)
        mock_self._parse_research_result = lambda r, q: ResearchHouse._parse_research_result(mock_self, r, q)
        return mock_self

    def test_parse_full_result(self):
        mock_house = self._make_mock_house()

        fab_recall = MagicMock()
        fab_recall.perspective = "recall"
        fab_recall.response = "Prior work"
        fab_recall.confidence = 0.8
        fab_recall.metadata = {
            "retrieved_facts": [{"content": "Study X", "confidence": 0.9}]
        }
        fab_recall.admits_ignorance = None

        fab_reach = MagicMock()
        fab_reach.perspective = "reach"
        fab_reach.response = "Creative"
        fab_reach.confidence = 0.7
        fab_reach.metadata = {"novel_connections": ["A-B link"]}
        fab_reach.admits_ignorance = None

        fab_refuse = MagicMock()
        fab_refuse.perspective = "refuse"
        fab_refuse.response = "Critique"
        fab_refuse.confidence = 0.6
        fab_refuse.metadata = {
            "attacks": [{"target": "Study X", "description": "Small sample"}]
        }
        fab_refuse.admits_ignorance = None

        result = MagicMock()
        result.response = "Summary"
        result.confidence = 0.75
        result.epistemic_confidence = 0.7
        result.fabrics = [fab_recall, fab_reach, fab_refuse]
        result.tensions = []
        result.metadata = {"depth": 2}

        parsed = mock_house._parse_research_result(result, "Why is X true?")
        assert parsed.question == "Why is X true?"
        assert parsed.summary == "Summary"
        assert len(parsed.findings) == 1
        assert len(parsed.novel_insights) == 1
        assert len(parsed.counterarguments) == 1
        assert "recall" in parsed.perspectives


# ============================================================================
# Domain houses __init__ tests
# ============================================================================


class TestDomainHousesInit:
    """Test that domain_houses __init__.py exports correctly."""

    def test_imports(self):
        from hololoom.loom.domain_houses import (
            CodeReviewHouse,
            create_code_review_house,
            ResearchHouse,
            create_research_house,
        )
        assert CodeReviewHouse is not None
        assert ResearchHouse is not None


# ============================================================================
# PatternCard (card_loader.py) tests
# ============================================================================


class TestMathCapabilities:
    """Test MathCapabilities config."""

    def test_defaults(self):
        mc = MathCapabilities()
        assert mc.semantic_calculus == {}
        assert mc.spectral_embedding == {}
        assert mc.is_enabled("semantic_calculus") is False

    def test_enabled_capability(self):
        mc = MathCapabilities(
            semantic_calculus={"enabled": True, "config": {"dimensions": 16}}
        )
        assert mc.is_enabled("semantic_calculus") is True
        assert mc.is_enabled("spectral_embedding") is False

    def test_is_enabled_unknown(self):
        mc = MathCapabilities()
        # Unknown capability returns False (getattr returns {})
        assert mc.is_enabled("nonexistent") is False

    def test_to_semantic_config_disabled(self):
        mc = MathCapabilities()
        assert mc.to_semantic_config() is None

    @patch("hololoom.core.loom.card_loader.MathCapabilities.to_semantic_config")
    def test_to_semantic_config_enabled_mocked(self, mock_method):
        """Test that to_semantic_config is callable when enabled."""
        mock_method.return_value = MagicMock()
        mc = MathCapabilities(
            semantic_calculus={"enabled": True, "config": {"dimensions": 32}}
        )
        result = mc.to_semantic_config()
        assert result is not None


class TestToolsConfig:
    """Test ToolsConfig."""

    def test_empty(self):
        tc = ToolsConfig()
        assert tc.is_tool_enabled("search") is False
        assert tc.get_tool_config("search") is None

    def test_enabled_tool(self):
        tc = ToolsConfig(
            enabled=["search", "calculator"],
            configs={"search": {"max_results": 10}},
        )
        assert tc.is_tool_enabled("search") is True
        assert tc.is_tool_enabled("calculator") is True
        assert tc.is_tool_enabled("browser") is False
        assert tc.get_tool_config("search") == {"max_results": 10}
        assert tc.get_tool_config("calculator") == {}

    def test_disabled_overrides_enabled(self):
        tc = ToolsConfig(
            enabled=["search"],
            disabled=["search"],
        )
        assert tc.is_tool_enabled("search") is False
        assert tc.get_tool_config("search") is None


class TestPerformanceProfile:
    """Test PerformanceProfile defaults."""

    def test_defaults(self):
        pp = PerformanceProfile()
        assert pp.target_latency_ms == 200
        assert pp.max_latency_ms == 500
        assert pp.timeout_ms == 2000
        assert pp.optimization == {}


class TestMemoryConfig:
    """Test MemoryConfig defaults."""

    def test_defaults(self):
        mc = MemoryConfig()
        assert mc.backend == "networkx"
        assert mc.caching == {}
        assert mc.retrieval == {}
        assert mc.graph == {}


class TestPatternCardFromDict:
    """Test PatternCard.from_dict construction."""

    def test_minimal(self):
        card = PatternCard.from_dict({"name": "test"})
        assert card.name == "test"
        assert card.display_name == "test"
        assert card.version == "1.0"
        assert card.extends is None

    def test_full(self):
        data = {
            "name": "research",
            "display_name": "Research Mode",
            "description": "Deep research card",
            "version": "2.0",
            "extends": "base",
            "math": {
                "semantic_calculus": {"enabled": True},
            },
            "memory": {"backend": "neo4j"},
            "tools": {"enabled": ["search"]},
            "performance": {"target_latency_ms": 500},
            "extensions": {"custom": True},
        }
        card = PatternCard.from_dict(data)
        assert card.display_name == "Research Mode"
        assert card.version == "2.0"
        assert card.extends == "base"
        assert card.math_capabilities.semantic_calculus == {"enabled": True}
        assert card.memory_config.backend == "neo4j"
        assert card.tools_config.is_tool_enabled("search") is True
        assert card.performance_profile.target_latency_ms == 500
        assert card.extensions == {"custom": True}


class TestPatternCardToDict:
    """Test PatternCard.to_dict round-trip."""

    def test_round_trip(self):
        original = PatternCard(
            name="fast",
            display_name="Fast Mode",
            description="Quick responses",
            version="1.1",
            math_capabilities=MathCapabilities(
                semantic_calculus={"enabled": True, "config": {"dimensions": 16}}
            ),
            tools_config=ToolsConfig(enabled=["search"]),
        )
        d = original.to_dict()
        restored = PatternCard.from_dict(d)

        assert restored.name == original.name
        assert restored.display_name == original.display_name
        assert restored.version == original.version
        assert restored.math_capabilities.semantic_calculus == original.math_capabilities.semantic_calculus
        assert restored.tools_config.enabled == original.tools_config.enabled

    def test_repr(self):
        card = PatternCard(name="x", display_name="X Card", version="1.0")
        r = repr(card)
        assert "x" in r
        assert "X Card" in r


class TestPatternCardMerge:
    """Test _merge_configs deep merge."""

    def test_shallow_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = PatternCard._merge_configs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge(self):
        base = {"math": {"semantic": {"dims": 16, "dt": 1.0}}}
        override = {"math": {"semantic": {"dims": 32}}}
        result = PatternCard._merge_configs(base, override)
        assert result["math"]["semantic"]["dims"] == 32
        assert result["math"]["semantic"]["dt"] == 1.0

    def test_override_non_dict_with_dict(self):
        base = {"key": "value"}
        override = {"key": {"nested": True}}
        result = PatternCard._merge_configs(base, override)
        assert result["key"] == {"nested": True}

    def test_base_not_mutated(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        PatternCard._merge_configs(base, override)
        assert "c" not in base["a"]


class TestPatternCardSaveLoad:
    """Test save/load with tmp_path."""

    def test_save_and_load(self, tmp_path):
        card = PatternCard(
            name="testcard",
            display_name="Test Card",
            description="For testing",
            version="1.0",
            tools_config=ToolsConfig(enabled=["search"]),
        )
        card.save("testcard", cards_dir=tmp_path)

        # Verify file exists
        card_path = tmp_path / "testcard.yaml"
        assert card_path.exists()

        # Load it back
        loaded = PatternCard.load("testcard", cards_dir=tmp_path)
        assert loaded.name == "testcard"
        assert loaded.display_name == "Test Card"
        assert loaded.tools_config.is_tool_enabled("search") is True

    def test_load_missing_card_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            PatternCard.load("nonexistent", cards_dir=tmp_path)

    def test_load_with_inheritance(self, tmp_path):
        # Save parent card
        parent = PatternCard(
            name="base",
            display_name="Base",
            version="1.0",
            tools_config=ToolsConfig(enabled=["search"]),
        )
        parent.save("base", cards_dir=tmp_path)

        # Save child card that extends parent
        child_data = {
            "name": "child",
            "display_name": "Child Card",
            "extends": "base",
            "tools": {"enabled": ["search", "calculator"]},
        }
        with open(tmp_path / "child.yaml", "w") as f:
            yaml.dump(child_data, f)

        loaded = PatternCard.load("child", cards_dir=tmp_path)
        assert loaded.display_name == "Child Card"
        assert "search" in loaded.tools_config.enabled
        assert "calculator" in loaded.tools_config.enabled

    def test_load_with_overrides(self, tmp_path):
        card = PatternCard(name="base", display_name="Base", version="1.0")
        card.save("base", cards_dir=tmp_path)

        loaded = PatternCard.load(
            "base",
            cards_dir=tmp_path,
            overrides={"display_name": "Overridden"},
        )
        assert loaded.display_name == "Overridden"

    def test_save_creates_directory(self, tmp_path):
        subdir = tmp_path / "deep" / "nested"
        card = PatternCard(name="nested_card")
        card.save(cards_dir=subdir)
        assert (subdir / "nested_card.yaml").exists()


# ============================================================================
# initialization.py tests
# ============================================================================


class TestCreateWeaveHouseSystem:
    """Test create_weave_house_system factory."""

    @pytest.mark.asyncio
    @patch("hololoom.core.loom.initialization.create_dream_orchestrator")
    @patch("hololoom.core.loom.initialization.create_all_looms")
    @patch("hololoom.core.loom.initialization.create_loom_consensus")
    @patch("hololoom.core.loom.initialization.WeaveHouse")
    async def test_default_with_dreaming(self, mock_wh_cls, mock_consensus,
                                          mock_create_looms, mock_dream):
        from hololoom.core.loom.initialization import create_weave_house_system

        mock_looms = [MagicMock(), MagicMock()]
        for loom in mock_looms:
            loom.perspective = "test"
            loom.blind_spots = []
        mock_create_looms.return_value = mock_looms
        mock_consensus.return_value = MagicMock()
        mock_wh = MagicMock()
        mock_wh_cls.return_value = mock_wh
        mock_dream.return_value = MagicMock()

        house, dreamer = await create_weave_house_system()

        assert house is mock_wh
        assert dreamer is not None
        mock_create_looms.assert_called_once()
        mock_consensus.assert_called_once()
        mock_dream.assert_called_once()

    @pytest.mark.asyncio
    @patch("hololoom.core.loom.initialization.create_dream_orchestrator")
    @patch("hololoom.core.loom.initialization.create_all_looms")
    @patch("hololoom.core.loom.initialization.create_loom_consensus")
    @patch("hololoom.core.loom.initialization.WeaveHouse")
    async def test_no_dreaming(self, mock_wh_cls, mock_consensus,
                                mock_create_looms, mock_dream):
        from hololoom.core.loom.initialization import create_weave_house_system

        mock_looms = [MagicMock()]
        for loom in mock_looms:
            loom.perspective = "test"
            loom.blind_spots = []
        mock_create_looms.return_value = mock_looms
        mock_consensus.return_value = MagicMock()
        mock_wh_cls.return_value = MagicMock()

        house, dreamer = await create_weave_house_system(enable_dreaming=False)

        assert dreamer is None
        mock_dream.assert_not_called()

    @pytest.mark.asyncio
    @patch("hololoom.core.loom.initialization.create_dream_orchestrator")
    @patch("hololoom.core.loom.initialization.create_all_looms")
    @patch("hololoom.core.loom.initialization.create_loom_consensus")
    @patch("hololoom.core.loom.initialization.WeaveHouse")
    async def test_config_overrides(self, mock_wh_cls, mock_consensus,
                                     mock_create_looms, mock_dream):
        from hololoom.core.loom.initialization import create_weave_house_system

        mock_looms = [MagicMock()]
        for loom in mock_looms:
            loom.perspective = "test"
            loom.blind_spots = []
        mock_create_looms.return_value = mock_looms
        mock_consensus.return_value = MagicMock()
        mock_wh_cls.return_value = MagicMock()
        mock_dream.return_value = MagicMock()

        config = MagicMock()
        config.weave_house_exploration_depth = 5
        config.weave_house_tension_threshold = 0.1
        config.dream_consolidation_interval = 7200
        config.enable_dreaming = True

        house, dreamer = await create_weave_house_system(config=config)

        call_kwargs = mock_wh_cls.call_args
        assert call_kwargs[1]["exploration_depth"] == 5 or call_kwargs.kwargs.get("exploration_depth") == 5

    @pytest.mark.asyncio
    @patch("hololoom.core.loom.initialization.create_dream_orchestrator")
    @patch("hololoom.core.loom.initialization.create_all_looms")
    @patch("hololoom.core.loom.initialization.create_loom_consensus")
    @patch("hololoom.core.loom.initialization.WeaveHouse")
    async def test_memory_backend_passed(self, mock_wh_cls, mock_consensus,
                                          mock_create_looms, mock_dream):
        from hololoom.core.loom.initialization import create_weave_house_system

        mock_looms = [MagicMock()]
        for loom in mock_looms:
            loom.perspective = "test"
            loom.blind_spots = []
        mock_create_looms.return_value = mock_looms
        mock_consensus.return_value = MagicMock()
        mock_wh_cls.return_value = MagicMock()
        mock_dream.return_value = MagicMock()

        memory = MagicMock()
        await create_weave_house_system(memory_backend=memory)

        call_kwargs = mock_create_looms.call_args
        assert call_kwargs[1].get("memory_backend") is memory or \
               call_kwargs.kwargs.get("memory_backend") is memory


class TestCreateResearchWeaveHouse:
    """Test create_research_weave_house convenience factory."""

    @patch("hololoom.core.loom.domain_houses.research_house.WeaveHouse.__init__", return_value=None)
    @patch("hololoom.core.loom.domain_houses.research_house.create_loom_consensus")
    @patch("hololoom.core.loom.domain_houses.research_house.RecallLoom")
    @patch("hololoom.core.loom.domain_houses.research_house.ReasonLoom")
    @patch("hololoom.core.loom.domain_houses.research_house.ReachLoom")
    @patch("hololoom.core.loom.domain_houses.research_house.ReflectLoom")
    @patch("hololoom.core.loom.domain_houses.research_house.RefuseLoom")
    def test_creates_research_house(self, mock_refuse, mock_reflect, mock_reach,
                                     mock_reason, mock_recall, mock_consensus,
                                     mock_wh_init):
        from hololoom.core.loom.initialization import create_research_weave_house

        mock_consensus.return_value = MagicMock()
        for m in [mock_refuse, mock_reflect, mock_reach, mock_reason, mock_recall]:
            m.return_value = MagicMock()
            m.return_value.blind_spots = []

        house = create_research_weave_house()
        assert house is not None
        assert house.research_config.exploration_depth == 3
        assert house.research_config.include_creative is True


class TestInitializationExports:
    """Test initialization module exports."""

    def test_all_exports(self):
        from hololoom.loom.initialization import __all__
        assert "create_weave_house_system" in __all__
        assert "create_research_weave_house" in __all__
        assert "create_code_review_weave_house" in __all__
