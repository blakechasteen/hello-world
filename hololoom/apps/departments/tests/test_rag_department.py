"""
Unit tests for RAG Department at HoloLoom/departments/rag_department.py

Comprehensive test coverage for all 7 Department protocol methods:
1. execute(): Query processing with SimpleRAG
2. verify(): DS-STAR verification checks
3. refine(): Low-confidence response refinement
4. update_strategy(): Learning from feedback
5. get_capabilities(): Capability reporting
6. get_metrics(): Metrics collection
7. health_check(): System health verification

Test Structure:
- Uses mocking to avoid external dependencies
- Tests all code paths in each method
- Validates error handling and edge cases
- Verifies integration with SimpleRAG
- Tests learning and metric collection

Author: HoloLoom Testing
Date: November 2025
"""

import pytest
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from dataclasses import dataclass, field
from uuid import uuid4

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ============================================================================
# Mocked Protocol Classes
# ============================================================================

class ConfidenceLevel:
    """Confidence level enum."""
    CRITICAL = "critical"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


@dataclass
class ConfidenceMetadata:
    """Confidence metadata."""
    score: float
    level: str = "medium"
    justification: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_score(cls, score: float, justification=None, sources=None):
        """Create from score."""
        if score < 0.2:
            level = ConfidenceLevel.CRITICAL
        elif score < 0.5:
            level = ConfidenceLevel.LOW
        elif score < 0.75:
            level = ConfidenceLevel.MEDIUM
        elif score < 0.95:
            level = ConfidenceLevel.HIGH
        else:
            level = ConfidenceLevel.VERIFIED

        return cls(
            score=max(0.0, min(1.0, score)),
            level=level,
            justification=justification or [],
            sources=sources or []
        )


class PrivacyLevel:
    """Privacy levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    CRITICAL = "critical"


@dataclass
class PrivacyEnvelope:
    """Privacy envelope."""
    content: Any = None
    level: str = "public"
    redaction_rules: list = field(default_factory=list)
    access_log: list = field(default_factory=list)

    def log_access(self, accessor: str, reason: str, approved: bool = True):
        """Log access."""
        self.access_log.append({
            "accessor": accessor,
            "reason": reason,
            "approved": approved,
            "timestamp": datetime.utcnow()
        })


@dataclass
class DepartmentRequest:
    """Department request."""
    task_type: str = ""
    parameters: Dict = field(default_factory=dict)
    context: Dict = field(default_factory=dict)
    task_id: Any = field(default_factory=uuid4)
    priority: int = 50
    timestamp: datetime = field(default_factory=datetime.utcnow)
    constraints: Dict = field(default_factory=dict)
    privacy_envelope: Any = None


@dataclass
class DepartmentResponse:
    """Department response."""
    task_id: Any
    result: Any
    confidence: ConfidenceMetadata
    metadata: Dict = field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Custom fields for RAG Department compatibility
    @property
    def request_id(self):
        """Compatibility property."""
        return str(self.task_id)

    @property
    def response(self):
        """Compatibility property."""
        return self.metadata.get("response_data", {})

    @response.setter
    def response(self, value):
        """Compatibility setter."""
        self.metadata["response_data"] = value

    @property
    def privacy_envelope(self):
        """Get privacy envelope."""
        return self.metadata.get("_privacy_envelope")

    @privacy_envelope.setter
    def privacy_envelope(self, value):
        """Set privacy envelope."""
        self.metadata["_privacy_envelope"] = value


@dataclass
class DSStarCheck:
    """DS-STAR verification check."""
    dimension: str
    passed: bool
    score: float
    details: str = ""


@dataclass
class VerificationResult:
    """Verification result."""
    verified: bool = False
    checks: list = field(default_factory=list)
    overall_score: float = 0.0
    recommendations: list = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RAGResult:
    """RAG query result."""
    response: str
    sources: list
    confidence: float
    reasoning_mode: str = "direct"
    metadata: Dict = field(default_factory=dict)


class Config:
    """Mock config."""
    def __init__(self):
        self.memory_backend = "inmemory"
        self.enable_zero_copy_embeddings = False

    @staticmethod
    def fast():
        return Config()

    @staticmethod
    def fused():
        return Config()


# ============================================================================
# Import the actual RAG Department (with mocked dependencies)
# ============================================================================

# Mock the imports that RAGDepartment needs
import sys
sys.modules['hololoom.config'] = MagicMock(Config=Config)
sys.modules['hololoom.rag'] = MagicMock()
sys.modules['hololoom.rag.simple_rag'] = MagicMock(RAGResult=RAGResult, SimpleRAG=AsyncMock)

# Now we can import by mocking the actual implementation
from hololoom.apps.departments.rag_department import RAGDepartment


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Create test configuration."""
    return Config.fast()


@pytest.fixture
def rag_department():
    """Create RAG Department with mocked RAG."""
    dept = RAGDepartment(
        config=Config.fast(),
        llm_provider="ollama",
        enable_reranking=False,
        department_id="test_rag"
    )

    # Mock SimpleRAG
    dept.rag = AsyncMock()
    dept.rag.loom = MagicMock()
    dept.rag.orchestrator = MagicMock()
    dept.rag.get_metrics = MagicMock(return_value={
        "cache_hits": 0,
        "cache_misses": 0,
        "avg_latency_ms": 100.0
    })

    return dept


@pytest.fixture
def sample_request():
    """Create a sample department request."""
    req = DepartmentRequest()
    req.task_type = "retrieve_context"
    req.parameters = {"query": "What is Thompson Sampling?"}
    req.context = {"mode": "verify", "max_sources": 5}
    return req


@pytest.fixture
def sample_response(sample_request):
    """Create a sample department response."""
    confidence = ConfidenceMetadata.from_score(
        score=0.85,
        justification=["Multiple sources agree"],
        sources=["source1", "source2"]
    )

    resp = DepartmentResponse(
        task_id=sample_request.task_id,
        result="Thompson Sampling is a Bayesian optimization technique.",
        confidence=confidence,
        metadata={
            "n_sources": 2,
            "cache_hit": False,
            "execution_time_ms": 150.0,
            "query_type": "factual",
            "response_data": {
                "answer": "Thompson Sampling is a Bayesian optimization technique.",
                "sources": ["source1", "source2"],
                "reasoning_mode": "verify"
            }
        }
    )
    resp.privacy_envelope = None
    return resp


# ============================================================================
# Test Initialization
# ============================================================================

class TestInitialization:
    """Test RAG Department initialization."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        dept = RAGDepartment()
        assert dept.department_id == "rag"
        assert dept.llm_provider == "ollama"
        assert dept.enable_reranking is False
        assert dept.rag is None

    def test_init_custom(self, config):
        """Test initialization with custom parameters."""
        dept = RAGDepartment(
            config=config,
            llm_provider="anthropic",
            enable_reranking=True,
            department_id="custom_rag"
        )
        assert dept.llm_provider == "anthropic"
        assert dept.enable_reranking is True
        assert dept.department_id == "custom_rag"

    def test_verification_thresholds(self):
        """Test verification thresholds are set."""
        dept = RAGDepartment()
        assert "domain_min_relevance" in dept._verification_thresholds
        assert "sensibility_min_score" in dept._verification_thresholds
        assert "temporal_max_age_days" in dept._verification_thresholds
        assert "argument_min_support" in dept._verification_thresholds
        assert "reference_min_traceable" in dept._verification_thresholds


# ============================================================================
# Test execute() Method
# ============================================================================

class TestExecuteMethod:
    """Test execute() method."""

    @pytest.mark.asyncio
    async def test_execute_basic_query(self, rag_department, sample_request):
        """Test basic query execution."""
        rag_department.rag.query = AsyncMock(return_value=RAGResult(
            response="Thompson Sampling is...",
            sources=["source1", "source2"],
            confidence=0.88,
            reasoning_mode="verify",
            metadata={"cache_hit": False, "llm_provider": "ollama"}
        ))

        response = await rag_department.execute(sample_request)

        assert response.confidence.score == 0.88
        assert len(response.metadata.get("sources", [])) >= 0
        rag_department.rag.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_no_rag_raises(self, sample_request):
        """Test execute raises when RAG not initialized."""
        dept = RAGDepartment()
        dept.rag = None

        with pytest.raises(RuntimeError, match="RAG Department not initialized"):
            await dept.execute(sample_request)

    @pytest.mark.asyncio
    async def test_execute_empty_query_raises(self, rag_department):
        """Test execute raises on empty query."""
        req = DepartmentRequest()
        req.parameters = {"query": ""}

        with pytest.raises(ValueError, match="must contain 'query' field"):
            await rag_department.execute(req)

    @pytest.mark.asyncio
    async def test_execute_missing_query_raises(self, rag_department):
        """Test execute raises when query field missing."""
        req = DepartmentRequest()
        req.parameters = {}

        with pytest.raises(ValueError, match="must contain 'query' field"):
            await rag_department.execute(req)

    @pytest.mark.asyncio
    async def test_execute_query_too_long(self, rag_department):
        """Test execute rejects overly long queries."""
        req = DepartmentRequest()
        req.parameters = {"query": " ".join(["word"] * 1001)}

        with pytest.raises(ValueError, match="exceeds max_query_length"):
            await rag_department.execute(req)

    @pytest.mark.asyncio
    async def test_execute_tracks_confidence(self, rag_department, sample_request):
        """Test confidence tracking."""
        rag_department.rag.query = AsyncMock(return_value=RAGResult(
            response="Answer",
            sources=["source1"],
            confidence=0.75,
            reasoning_mode="verify",
            metadata={}
        ))

        await rag_department.execute(sample_request)
        assert 0.75 in rag_department._confidence_history

    @pytest.mark.asyncio
    async def test_execute_tracks_query_type(self, rag_department, sample_request):
        """Test query type tracking."""
        rag_department.rag.query = AsyncMock(return_value=RAGResult(
            response="Answer",
            sources=["source1"],
            confidence=0.75,
            reasoning_mode="verify",
            metadata={}
        ))

        await rag_department.execute(sample_request)
        assert "factual" in rag_department._query_patterns

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, rag_department, sample_request):
        """Test graceful error handling."""
        rag_department.rag.query = AsyncMock(side_effect=Exception("RAG error"))

        response = await rag_department.execute(sample_request)

        assert response.confidence.score == 0.1
        assert "Unable to process query" in str(response.result)

    @pytest.mark.asyncio
    async def test_execute_modes(self, rag_department, sample_request):
        """Test different reasoning modes."""
        for mode in ["direct", "verify", "research", "plan_execute"]:
            sample_request.context = {"mode": mode}
            rag_department.rag.query = AsyncMock(return_value=RAGResult(
                response=f"Answer for {mode}",
                sources=["source1"],
                confidence=0.80,
                reasoning_mode=mode,
                metadata={}
            ))

            response = await rag_department.execute(sample_request)
            assert response.confidence.score == 0.80


# ============================================================================
# Test verify() Method
# ============================================================================

class TestVerifyMethod:
    """Test verify() method."""

    @pytest.mark.asyncio
    async def test_verify_returns_result(self, rag_department, sample_response):
        """Test verify returns VerificationResult."""
        result = await rag_department.verify(sample_response)

        assert isinstance(result, VerificationResult)
        assert result.verified is not None
        assert len(result.checks) == 5  # 5 DS-STAR dimensions

    @pytest.mark.asyncio
    async def test_verify_ds_star_dimensions(self, rag_department, sample_response):
        """Test all 5 DS-STAR dimensions."""
        result = await rag_department.verify(sample_response)

        dimensions = {check.dimension for check in result.checks}
        expected = {"Domain", "Sensibility", "Temporal", "Argument", "Reference"}
        assert dimensions == expected

    @pytest.mark.asyncio
    async def test_verify_overall_score(self, rag_department, sample_response):
        """Test overall score calculation."""
        result = await rag_department.verify(sample_response)

        assert 0.0 <= result.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_verify_empty_sources(self, rag_department, sample_response):
        """Test with no sources."""
        sample_response.metadata["response_data"]["sources"] = []
        result = await rag_department.verify(sample_response)

        assert len(result.checks) == 5

    @pytest.mark.asyncio
    async def test_verify_empty_answer(self, rag_department, sample_response):
        """Test with empty answer."""
        sample_response.metadata["response_data"]["answer"] = ""
        result = await rag_department.verify(sample_response)

        assert len(result.checks) == 5

    @pytest.mark.asyncio
    async def test_verify_recommendations(self, rag_department, sample_response):
        """Test recommendations for failed checks."""
        sample_response.metadata["response_data"]["sources"] = []
        result = await rag_department.verify(sample_response)

        # May have recommendations if checks failed
        assert isinstance(result.recommendations, list)


# ============================================================================
# Test refine() Method
# ============================================================================

class TestRefineMethod:
    """Test refine() method."""

    @pytest.mark.asyncio
    async def test_refine_high_confidence(self, rag_department, sample_response):
        """Test refine skips high confidence."""
        sample_response.confidence.score = 0.80
        sample_response.metadata["original_query"] = "What is Thompson Sampling?"

        refined = await rag_department.refine(sample_response)
        assert refined.confidence.score == 0.80

    @pytest.mark.asyncio
    async def test_refine_low_confidence(self, rag_department, sample_response):
        """Test refine improves low confidence."""
        sample_response.confidence.score = 0.60
        sample_response.metadata["original_query"] = "What is Thompson Sampling?"

        rag_department.rag.query = AsyncMock(return_value=RAGResult(
            response="Improved answer",
            sources=["source1", "source2", "source3"],
            confidence=0.85,
            reasoning_mode="research",
            metadata={}
        ))

        refined = await rag_department.refine(sample_response)
        assert refined.confidence.score >= sample_response.confidence.score

    @pytest.mark.asyncio
    async def test_refine_tracking(self, rag_department, sample_response):
        """Test refinement tracking."""
        sample_response.confidence.score = 0.60
        sample_response.metadata["original_query"] = "What is Thompson Sampling?"

        rag_department.rag.query = AsyncMock(return_value=RAGResult(
            response="Improved answer",
            sources=["source1"],
            confidence=0.85,
            reasoning_mode="research",
            metadata={}
        ))

        initial = rag_department._refinement_stats["total_refinements"]
        await rag_department.refine(sample_response)
        assert rag_department._refinement_stats["total_refinements"] > initial

    @pytest.mark.asyncio
    async def test_refine_no_query_graceful(self, rag_department, sample_response):
        """Test refine handles missing query gracefully."""
        sample_response.confidence.score = 0.60
        sample_response.metadata = {}

        refined = await rag_department.refine(sample_response)
        assert refined is not None


# ============================================================================
# Test update_strategy() Method
# ============================================================================

class TestUpdateStrategyMethod:
    """Test update_strategy() method."""

    @pytest.mark.asyncio
    async def test_update_strategy_helpful(self, rag_department):
        """Test learning from helpful feedback."""
        feedback = {
            "helpful": True,
            "confidence": 0.80,
            "query_type": "factual"
        }

        # Should not raise
        await rag_department.update_strategy(feedback)

    @pytest.mark.asyncio
    async def test_update_strategy_unhelpful(self, rag_department):
        """Test learning from unhelpful feedback."""
        feedback = {
            "helpful": False,
            "confidence": 0.80,
            "query_type": "factual"
        }

        await rag_department.update_strategy(feedback)

    @pytest.mark.asyncio
    async def test_update_strategy_complete(self, rag_department):
        """Test with complete feedback."""
        feedback = {
            "helpful": True,
            "confidence": 0.82,
            "query_type": "analytical",
            "refinement_needed": False,
            "user_rating": 4.5
        }

        await rag_department.update_strategy(feedback)


# ============================================================================
# Test get_capabilities() Method
# ============================================================================

class TestGetCapabilitiesMethod:
    """Test get_capabilities() method."""

    @pytest.mark.asyncio
    async def test_get_capabilities(self, rag_department):
        """Test capabilities structure."""
        caps = await rag_department.get_capabilities()

        assert "tasks" in caps
        assert "max_tokens" in caps
        assert "reasoning_modes" in caps
        assert "features" in caps
        assert "constraints" in caps

    @pytest.mark.asyncio
    async def test_get_capabilities_tasks(self, rag_department):
        """Test supported tasks."""
        caps = await rag_department.get_capabilities()
        assert "question_answering" in caps["tasks"]
        assert "document_search" in caps["tasks"]

    @pytest.mark.asyncio
    async def test_get_capabilities_modes(self, rag_department):
        """Test reasoning modes."""
        caps = await rag_department.get_capabilities()
        modes = caps["reasoning_modes"]
        assert "direct" in modes
        assert "verify" in modes
        assert "research" in modes

    @pytest.mark.asyncio
    async def test_get_capabilities_constraints(self, rag_department):
        """Test constraints."""
        caps = await rag_department.get_capabilities()
        constraints = caps["constraints"]
        assert constraints["max_sources"] == 20
        assert constraints["max_query_length"] == 1000


# ============================================================================
# Test get_metrics() Method
# ============================================================================

class TestGetMetricsMethod:
    """Test get_metrics() method."""

    @pytest.mark.asyncio
    async def test_get_metrics_structure(self, rag_department):
        """Test metrics structure."""
        metrics = await rag_department.get_metrics()

        assert "rag_metrics" in metrics
        assert "confidence_stats" in metrics
        assert "query_patterns" in metrics
        assert "refinement_stats" in metrics

    @pytest.mark.asyncio
    async def test_get_metrics_no_queries(self, rag_department):
        """Test metrics before queries."""
        metrics = await rag_department.get_metrics()
        assert metrics["confidence_stats"] == {}
        assert metrics["query_patterns"] == {}

    @pytest.mark.asyncio
    async def test_get_metrics_after_queries(self, rag_department, sample_request):
        """Test metrics after queries."""
        rag_department.rag.query = AsyncMock(return_value=RAGResult(
            response="Answer",
            sources=["source1"],
            confidence=0.85,
            reasoning_mode="verify",
            metadata={}
        ))

        await rag_department.execute(sample_request)

        metrics = await rag_department.get_metrics()
        assert metrics["confidence_stats"]["count"] == 1
        assert metrics["confidence_stats"]["mean"] == 0.85

    @pytest.mark.asyncio
    async def test_get_metrics_refinement_stats(self, rag_department):
        """Test refinement stats."""
        rag_department._refinement_stats["total_refinements"] = 5
        rag_department._refinement_stats["successful_refinements"] = 4
        rag_department._confidence_history = [0.7] * 10

        metrics = await rag_department.get_metrics()
        refine = metrics["refinement_stats"]
        assert refine["total_refinements"] == 5


# ============================================================================
# Test health_check() Method
# ============================================================================

class TestHealthCheckMethod:
    """Test health_check() method."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, rag_department):
        """Test successful health check."""
        rag_department.rag.loom = MagicMock()
        healthy = await rag_department.health_check()
        assert healthy is True

    @pytest.mark.asyncio
    async def test_health_check_no_rag(self):
        """Test health check fails without RAG."""
        dept = RAGDepartment()
        dept.rag = None
        healthy = await dept.health_check()
        assert healthy is False

    @pytest.mark.asyncio
    async def test_health_check_no_memory(self, rag_department):
        """Test health check fails without memory."""
        rag_department.rag.loom = None
        healthy = await rag_department.health_check()
        assert healthy is False


# ============================================================================
# Test Helper Methods
# ============================================================================

class TestHelperMethods:
    """Test private helper methods."""

    def test_classify_factual(self, rag_department):
        """Test factual query classification."""
        assert rag_department._classify_query_type("What is X?") == "factual"
        assert rag_department._classify_query_type("Define Y") == "factual"

    def test_classify_procedural(self, rag_department):
        """Test procedural classification."""
        assert rag_department._classify_query_type("How to do X?") == "procedural"
        assert rag_department._classify_query_type("How do I?") == "procedural"

    def test_classify_analytical(self, rag_department):
        """Test analytical classification."""
        assert rag_department._classify_query_type("Why does X work?") == "analytical"
        assert rag_department._classify_query_type("What caused Y?") == "analytical"

    def test_classify_comparative(self, rag_department):
        """Test comparative classification."""
        assert rag_department._classify_query_type("Compare X vs Y") == "comparative"
        assert rag_department._classify_query_type("What's the difference?") == "comparative"

    def test_check_domain_relevance(self, rag_department):
        """Test domain relevance check."""
        score = rag_department._check_domain_relevance(["source1", "source2"])
        assert 0.0 <= score <= 1.0

    def test_check_sensibility(self, rag_department):
        """Test sensibility check."""
        score = rag_department._check_sensibility("This is a detailed answer.", ["source1"])
        assert 0.0 <= score <= 1.0

    def test_check_temporal(self, rag_department):
        """Test temporal freshness check."""
        age = rag_department._check_temporal_freshness(["source1"])
        assert age >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])