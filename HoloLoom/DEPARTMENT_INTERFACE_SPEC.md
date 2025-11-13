# HoloLoom Department Interface Specification

**Version**: 1.0.0
**Date**: November 9, 2025
**Status**: DRAFT - Core Architecture

---

## Executive Summary

This document defines the **Department Protocol** - the generic interface that all HoloLoom departments must implement. By standardizing on this protocol, we enable:

1. **Modularity**: Departments can be swapped, upgraded, or replaced independently
2. **B2B Flexibility**: Different industries use different department sets
3. **Marketplace Ready**: Third-party departments can be built and distributed
4. **Confidence-First**: All departments speak the same confidence language

---

## Core Principles

### 1. Confidence is the Universal Currency
Every department input and output includes confidence scores. This enables:
- Adaptive learning rates (high confidence = slow updates, low confidence = fast updates)
- Intelligent verification (low confidence triggers deep checks)
- Context compaction (high confidence = minimal detail, low confidence = full audit trail)

### 2. Departments are Nested Optimization Problems
Each department:
- Learns at its own rate
- Maintains its own memory system (short/medium/long-term)
- Updates its strategy based on outcomes
- Never catastrophically forgets (separate optimization spaces)

### 3. Privacy is Non-Negotiable
All departments:
- Process sensitive data in TEE (Trusted Execution Environment)
- Only share privacy-preserved insights
- Support verifiable output (external parties can validate)

---

## Department Protocol (Python)

```python
from typing import Protocol, Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

class ConfidenceLevel(Enum):
    """Standard confidence tiers with associated behaviors"""
    CRITICAL = (0.95, 1.00)   # Established knowledge, weekly updates
    HIGH = (0.85, 0.94)        # Validated approach, daily updates
    MEDIUM = (0.65, 0.84)      # Active learning, hourly updates
    LOW = (0.40, 0.64)         # Exploration, per-task updates
    UNCERTAIN = (0.00, 0.39)   # Rapid iteration, immediate updates

    def __init__(self, min_conf: float, max_conf: float):
        self.min_conf = min_conf
        self.max_conf = max_conf

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Map confidence score to tier"""
        if score >= 0.95: return cls.CRITICAL
        elif score >= 0.85: return cls.HIGH
        elif score >= 0.65: return cls.MEDIUM
        elif score >= 0.40: return cls.LOW
        else: return cls.UNCERTAIN

@dataclass
class ConfidenceMetadata:
    """Rich confidence information for nested learning"""
    score: float  # 0.0-1.0
    level: ConfidenceLevel
    justification: List[str]  # Human-readable reasons
    uncertainty_sources: List[str]  # What causes uncertainty?
    calibration_history: Optional[Dict[str, float]] = None  # Prior accuracy
    learning_rate: str = ""  # "weekly" | "daily" | "hourly" | "per-task"

    def __post_init__(self):
        self.level = ConfidenceLevel.from_score(self.score)
        # Map confidence level to learning rate
        rate_map = {
            ConfidenceLevel.CRITICAL: "weekly",
            ConfidenceLevel.HIGH: "daily",
            ConfidenceLevel.MEDIUM: "hourly",
            ConfidenceLevel.LOW: "per-task",
            ConfidenceLevel.UNCERTAIN: "immediate"
        }
        self.learning_rate = rate_map[self.level]

@dataclass
class DepartmentRequest:
    """Standardized request format for all departments"""
    task_id: str
    task_type: str  # "extract_entities" | "query_data" | "verify_decision" | etc.
    parameters: Dict[str, Any]

    # Confidence negotiation
    confidence_expected: float  # Requestor's expectation (0.0-1.0)
    context_preference: str  # "minimal" | "moderate" | "detailed" | "exhaustive"

    # Privacy envelope
    privacy_level: str  # "public" | "aggregate" | "confidential" | "tee_only"

    # Session management
    session_id: Optional[str] = None  # Resume previous session
    parent_task_id: Optional[str] = None  # Part of larger workflow

@dataclass
class DepartmentResponse:
    """Standardized response format for all departments"""
    task_id: str
    result: Any  # Department-specific result

    # Confidence reporting
    confidence: ConfidenceMetadata

    # Context detail (aligned with confidence)
    detail_level: str  # "minimal" | "moderate" | "detailed" | "exhaustive"
    reasoning: Optional[Dict[str, Any]] = None  # Decision chain (if detail_level != minimal)
    alternatives_considered: Optional[List[Dict[str, Any]]] = None  # Other approaches tried

    # Learning signals
    learning_signals: Dict[str, Any] = None  # Patterns discovered, confidence drift, etc.

    # Session state
    session_state: Optional[Dict[str, Any]] = None  # For resumption
    institutional_memory: Optional[Dict[str, Any]] = None  # Long-term patterns

    # Privacy verification
    privacy_metadata: Optional[Dict[str, str]] = None  # TEE attestation, DP parameters

@dataclass
class VerificationResult:
    """Result from DS-STAR verification loop"""
    sufficient: bool  # Did the department's work meet expectations?
    confidence_valid: bool  # Was reported confidence accurate?
    reasoning_sound: bool  # Was the decision chain logical?
    alternative_paths: List[str]  # Other approaches to consider
    refinement_suggestions: Optional[Dict[str, Any]] = None  # How to improve
    escalation_needed: bool = False  # Human-in-the-loop required?

class Department(Protocol):
    """
    Core protocol that all HoloLoom departments must implement.

    This enables:
    - Pluggable departments (swap implementations)
    - B2B customization (different departments per industry)
    - Marketplace ecosystem (third-party departments)
    - Confidence-driven nested learning
    """

    # Department Identity
    name: str  # "MasterWeaver" | "Infrastructure" | "Context" | etc.
    domain: str  # "beekeeping" | "healthcare" | "finance" | "generic"
    version: str  # Semantic versioning

    # Capabilities
    supported_tasks: List[str]  # Task types this department handles
    confidence_range: tuple[float, float]  # Expected confidence range (e.g., (0.40, 0.85))

    # Memory Systems
    short_term_memory: Dict[str, Any]  # Milliseconds-seconds (current task state)
    medium_term_memory: Dict[str, Any]  # Minutes-hours (session state)
    long_term_memory: Dict[str, Any]  # Days-weeks (institutional memory)

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Main execution method. All departments must implement this.

        Args:
            request: Standardized request with confidence expectations

        Returns:
            DepartmentResponse with confidence metadata and learning signals

        Behavior:
        - Process request according to department's domain expertise
        - Track confidence throughout execution
        - Log intermediate results to appropriate memory tier
        - Return response with confidence justification
        """
        ...

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """
        Self-verification step (DS-STAR pattern).

        Args:
            response: The department's own response

        Returns:
            VerificationResult indicating if work is sufficient

        Behavior:
        - Check if confidence matches actual quality
        - Validate reasoning chain
        - Identify alternative approaches
        - Suggest refinements if insufficient
        """
        ...

    async def refine(
        self,
        request: DepartmentRequest,
        prior_response: DepartmentResponse,
        verification: VerificationResult
    ) -> DepartmentResponse:
        """
        Refinement step (DS-STAR Router pattern).

        Args:
            request: Original request
            prior_response: Previous attempt
            verification: Why it was insufficient

        Returns:
            Improved DepartmentResponse

        Behavior:
        - Apply verification feedback
        - Try alternative approach if suggested
        - Increase detail level if low confidence
        - Log pattern for institutional learning
        """
        ...

    async def update_strategy(self, learning_signals: List[Dict[str, Any]]) -> None:
        """
        Institutional learning (Magic Cycle).

        Args:
            learning_signals: Accumulated patterns from recent tasks

        Behavior:
        - Analyze patterns across multiple tasks
        - Update department's core strategy
        - Recalibrate confidence model
        - Adjust learning rate based on accuracy

        Frequency:
        - CRITICAL confidence: Weekly updates
        - HIGH confidence: Daily updates
        - MEDIUM confidence: Hourly updates
        - LOW/UNCERTAIN: Per-task updates
        """
        ...

    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Resume previous session.

        Args:
            session_id: Identifier for prior session

        Returns:
            Session state with confidence-indexed detail level

        Behavior:
        - High-confidence sessions: Return summary only
        - Low-confidence sessions: Return full audit trail
        """
        ...

    async def get_institutional_memory(self, pattern_type: str) -> Dict[str, Any]:
        """
        Query long-term patterns.

        Args:
            pattern_type: Type of pattern to retrieve

        Returns:
            Learned patterns with confidence scores

        Examples:
        - "successful_strategies": What works well?
        - "failure_modes": Common problems?
        - "confidence_drift": Where are we over/underconfident?
        """
        ...

    async def health_check(self) -> Dict[str, Any]:
        """
        Department health and readiness.

        Returns:
            Status, confidence distribution, learning rate, memory usage
        """
        ...

# Helper Types

@dataclass
class DepartmentManifest:
    """Metadata for department marketplace"""
    name: str
    version: str
    domain: str  # "generic" | "beekeeping" | "healthcare" | etc.
    author: str
    license: str
    description: str
    supported_tasks: List[str]
    confidence_range: tuple[float, float]
    dependencies: List[str]
    privacy_guarantees: List[str]  # ["TEE", "differential_privacy", "verifiable_output"]
    marketplace_url: Optional[str] = None
    documentation_url: Optional[str] = None

@dataclass
class DepartmentConfig:
    """Runtime configuration for department"""
    learning_rate_override: Optional[str] = None  # Override default learning rate
    confidence_threshold: float = 0.75  # Minimum acceptable confidence
    verification_enabled: bool = True  # Run DS-STAR verification?
    privacy_mode: str = "tee_only"  # "public" | "aggregate" | "confidential" | "tee_only"
    session_persistence: bool = True  # Store session state?
    institutional_learning: bool = True  # Enable magic cycle?
```

---

## Example Implementation: Generic Context Department

```python
from typing import List, Dict, Any, Optional
import logging

class ContextDepartment(Department):
    """
    Generic context enrichment department.
    Maps to current HoloLoom implementation.
    """

    def __init__(self, config: DepartmentConfig):
        self.name = "Context"
        self.domain = "generic"  # Works for all domains
        self.version = "1.0.0"
        self.supported_tasks = [
            "enrich_context",
            "retrieve_memories",
            "expand_knowledge_graph",
            "multi_scale_embedding"
        ]
        self.confidence_range = (0.55, 0.88)  # From architecture doc

        # Memory systems
        self.short_term_memory = {}  # Current context request
        self.medium_term_memory = {}  # Enriched context artifacts
        self.long_term_memory = {}  # Learned patterns (which context helps which dept?)

        # Existing HoloLoom components
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        from HoloLoom.config import Config

        self.orchestrator = WeavingOrchestrator(cfg=Config.fused())
        self.config = config
        self.logger = logging.getLogger(f"Department.{self.name}")

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Use existing HoloLoom orchestrator for context enrichment"""
        from HoloLoom.documentation.types import Query

        # Extract query from request
        query_text = request.parameters.get("query", "")
        query = Query(text=query_text)

        # Log to short-term memory
        self.short_term_memory[request.task_id] = {
            "query": query_text,
            "timestamp": datetime.now(),
            "confidence_expected": request.confidence_expected
        }

        # Execute with HoloLoom
        async with self.orchestrator as orch:
            spacetime = await orch.weave(query)

        # Map spacetime to confidence metadata
        confidence = ConfidenceMetadata(
            score=spacetime.confidence,
            justification=[
                f"Matryoshka scale: {spacetime.metadata.get('matryoshka_scale')}D",
                f"Knowledge graph nodes: {spacetime.metadata.get('kg_nodes', 0)}",
                f"Thompson samples: {spacetime.metadata.get('thompson_samples', 0)}"
            ],
            uncertainty_sources=[
                "Query ambiguity" if spacetime.confidence < 0.7 else "",
                "Limited context" if spacetime.metadata.get('kg_nodes', 0) < 5 else ""
            ],
            calibration_history=self._get_calibration_history()
        )

        # Determine detail level based on confidence
        detail_level = self._determine_detail_level(confidence.score, request.context_preference)

        # Build response
        response = DepartmentResponse(
            task_id=request.task_id,
            result=spacetime.output,
            confidence=confidence,
            detail_level=detail_level,
            reasoning=self._extract_reasoning(spacetime, detail_level),
            alternatives_considered=None,  # TODO: Extract from spacetime trace
            learning_signals={
                "cache_hit": spacetime.metadata.get("cache_hit", False),
                "latency_ms": spacetime.metadata.get("latency_ms", 0)
            },
            session_state=self._build_session_state(request, spacetime),
            institutional_memory=None  # Returned separately via get_institutional_memory()
        )

        # Log to medium-term memory
        self.medium_term_memory[request.task_id] = {
            "response": response,
            "timestamp": datetime.now()
        }

        return response

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """Check if context enrichment was sufficient"""
        confidence = response.confidence.score

        # Confidence validation
        confidence_valid = True
        reasoning_sound = True
        alternative_paths = []

        # Check if confidence matches quality indicators
        result_quality = self._assess_result_quality(response.result)
        confidence_mismatch = abs(confidence - result_quality) > 0.15

        if confidence_mismatch:
            confidence_valid = False
            alternative_paths.append("Recalibrate confidence model")

        # Check if detail level matches confidence
        expected_detail = self._determine_detail_level(confidence, "auto")
        if response.detail_level != expected_detail:
            reasoning_sound = False
            alternative_paths.append(f"Adjust detail level to {expected_detail}")

        # Determine if sufficient
        sufficient = confidence_valid and reasoning_sound and confidence >= 0.65

        return VerificationResult(
            sufficient=sufficient,
            confidence_valid=confidence_valid,
            reasoning_sound=reasoning_sound,
            alternative_paths=alternative_paths,
            refinement_suggestions={
                "increase_context": not sufficient and confidence < 0.7,
                "expand_kg": not sufficient and "kg_nodes" in response.learning_signals
            } if not sufficient else None,
            escalation_needed=confidence < 0.40  # Very low confidence
        )

    async def refine(
        self,
        request: DepartmentRequest,
        prior_response: DepartmentResponse,
        verification: VerificationResult
    ) -> DepartmentResponse:
        """Apply verification feedback to improve response"""

        # Apply refinement suggestions
        if verification.refinement_suggestions:
            if verification.refinement_suggestions.get("increase_context"):
                # Re-run with FUSED mode instead of FAST
                request.parameters["mode"] = "FUSED"

            if verification.refinement_suggestions.get("expand_kg"):
                # Enable knowledge graph expansion
                request.parameters["expand_kg"] = True

        # Re-execute with refinements
        refined_response = await self.execute(request)

        # Log pattern for institutional learning
        self.long_term_memory.setdefault("refinement_patterns", []).append({
            "original_confidence": prior_response.confidence.score,
            "refined_confidence": refined_response.confidence.score,
            "improvement": refined_response.confidence.score - prior_response.confidence.score,
            "refinement_type": list(verification.refinement_suggestions.keys())[0],
            "timestamp": datetime.now()
        })

        return refined_response

    async def update_strategy(self, learning_signals: List[Dict[str, Any]]) -> None:
        """Update context enrichment strategy based on accumulated patterns"""

        # Analyze refinement patterns
        refinement_patterns = self.long_term_memory.get("refinement_patterns", [])

        if len(refinement_patterns) >= 10:
            # Calculate average improvement per refinement type
            improvements = {}
            for pattern in refinement_patterns[-50:]:  # Last 50 refinements
                rtype = pattern["refinement_type"]
                improvements.setdefault(rtype, []).append(pattern["improvement"])

            # Update strategy: prioritize refinements with highest average improvement
            avg_improvements = {
                rtype: sum(imps) / len(imps)
                for rtype, imps in improvements.items()
            }

            self.long_term_memory["preferred_refinements"] = sorted(
                avg_improvements.items(),
                key=lambda x: x[1],
                reverse=True
            )

            self.logger.info(f"Updated strategy: preferred_refinements = {self.long_term_memory['preferred_refinements']}")

    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session with confidence-indexed detail"""
        session = self.medium_term_memory.get(session_id)
        if not session:
            return None

        response = session["response"]
        confidence = response.confidence.score

        # High confidence: summary only
        if confidence >= 0.85:
            return {
                "status": "COMPLETE",
                "confidence": confidence,
                "result_summary": str(response.result)[:200],
                "artifacts": "compact"
            }

        # Low confidence: full audit trail
        else:
            return {
                "status": "PENDING_REVIEW" if confidence < 0.65 else "COMPLETE",
                "confidence": confidence,
                "full_response": response,
                "artifacts": "exhaustive"
            }

    async def get_institutional_memory(self, pattern_type: str) -> Dict[str, Any]:
        """Query learned patterns"""
        if pattern_type == "successful_strategies":
            return {
                "preferred_refinements": self.long_term_memory.get("preferred_refinements", []),
                "high_confidence_configs": []  # TODO: Track configs that yield high confidence
            }

        elif pattern_type == "confidence_drift":
            return {
                "calibration_history": self._get_calibration_history()
            }

        return {}

    async def health_check(self) -> Dict[str, Any]:
        """Department health"""
        return {
            "status": "healthy",
            "name": self.name,
            "version": self.version,
            "confidence_range": self.confidence_range,
            "learning_rate": "hourly",  # Medium confidence
            "memory_usage": {
                "short_term_entries": len(self.short_term_memory),
                "medium_term_entries": len(self.medium_term_memory),
                "long_term_patterns": len(self.long_term_memory)
            }
        }

    # Private helpers

    def _determine_detail_level(self, confidence: float, preference: str) -> str:
        """Map confidence to detail level"""
        if preference != "auto":
            return preference

        if confidence >= 0.90:
            return "minimal"
        elif confidence >= 0.75:
            return "moderate"
        elif confidence >= 0.50:
            return "detailed"
        else:
            return "exhaustive"

    def _extract_reasoning(self, spacetime, detail_level: str) -> Optional[Dict[str, Any]]:
        """Extract reasoning chain based on detail level"""
        if detail_level == "minimal":
            return None

        return {
            "trace": spacetime.trace,
            "metadata": spacetime.metadata
        }

    def _assess_result_quality(self, result: Any) -> float:
        """Heuristic quality assessment"""
        # TODO: Implement domain-specific quality metrics
        return 0.75

    def _get_calibration_history(self) -> Dict[str, float]:
        """Confidence calibration metrics"""
        # TODO: Track predicted vs actual confidence over time
        return {}

    def _build_session_state(self, request: DepartmentRequest, spacetime) -> Dict[str, Any]:
        """Build session state for resumption"""
        return {
            "session_id": request.session_id or request.task_id,
            "timestamp": datetime.now().isoformat(),
            "confidence": spacetime.confidence,
            "mode": request.parameters.get("mode", "FAST")
        }
```

---

## Department Lifecycle

```
INSTANTIATION
  ├─ Load configuration
  ├─ Initialize memory systems
  └─ Register with orchestration
       ↓
EXECUTION LOOP
  ├─ Receive request (DepartmentRequest)
  ├─ Execute task
  ├─ Self-verify (DS-STAR)
  ├─ Refine if insufficient
  └─ Return response (DepartmentResponse)
       ↓
LEARNING LOOP (periodic)
  ├─ Accumulate learning signals
  ├─ Analyze patterns
  ├─ Update strategy
  └─ Recalibrate confidence
       ↓
HEALTH MONITORING
  ├─ Track memory usage
  ├─ Monitor confidence distribution
  └─ Adjust learning rate
```

---

## Confidence Contract (Department ↔ Orchestration)

### Request Confidence Expectations

```python
request = DepartmentRequest(
    confidence_expected=0.85,
    context_preference="minimal"
)
# Translation: "I expect high-quality (0.85+) with minimal context detail"
```

### Response Confidence Reporting

```python
response = DepartmentResponse(
    confidence=ConfidenceMetadata(
        score=0.87,
        justification=["Deterministic query", "50+ similar patterns"],
        uncertainty_sources=[],
        learning_rate="daily"
    )
)
# Translation: "Delivered 0.87 (above expectation), high confidence → daily updates"
```

### Confidence Mismatch Handling

```python
if response.confidence.score < request.confidence_expected - 0.15:
    # Significant confidence drop (>15%)
    verification_result = await department.verify(response)

    if not verification_result.sufficient:
        # Trigger refinement
        refined_response = await department.refine(request, response, verification_result)

        if refined_response.confidence.score >= request.confidence_expected:
            # Refinement successful
            return refined_response
        else:
            # Escalate to human
            if verification_result.escalation_needed:
                await orchestration.escalate_to_human(request, refined_response)
```

---

## Privacy Envelope Specification

### TEE Processing

```python
@dataclass
class PrivacyEnvelope:
    """Privacy guarantees for department execution"""
    tee_enabled: bool  # Running in Trusted Execution Environment?
    differential_privacy: bool  # DP applied to outputs?
    verifiable_output: bool  # External parties can verify?
    data_residency: str  # "tee_only" | "encrypted_storage" | "aggregate_only"
    attestation: Optional[str] = None  # TEE attestation signature
    dp_epsilon: Optional[float] = None  # DP privacy budget (if applicable)
    dp_delta: Optional[float] = None  # DP failure probability

# Example usage
response = DepartmentResponse(
    result=aggregate_statistics,
    privacy_metadata={
        "tee_attestation": "SGX_SIGNATURE_HERE",
        "dp_epsilon": 0.1,
        "dp_delta": 1e-5,
        "verifiable_code_hash": "sha256:abc123..."
    }
)
```

### Data Flow Boundaries

```
RAW INPUT (Sensitive)
  ↓
TEE BOUNDARY ─────────────────────────────────┐
  ├─ Department processes raw data            │
  ├─ Extracts structured insights             │ ← No raw data exits
  ├─ Applies differential privacy             │
  └─ Generates verifiable attestation         │
       ↓                                       │
PRIVACY-PRESERVED OUTPUT ──────────────────────┘
  (Aggregated, anonymized, verifiable)
```

---

## Marketplace Specification

### Department Registry

```python
class DepartmentMarketplace:
    """Central registry for pluggable departments"""

    def register_department(self, manifest: DepartmentManifest, implementation: Department):
        """Register department for discovery"""
        ...

    def discover_departments(
        self,
        domain: Optional[str] = None,
        task_type: Optional[str] = None,
        min_confidence: Optional[float] = None
    ) -> List[DepartmentManifest]:
        """Find departments matching criteria"""
        ...

    def install_department(self, manifest: DepartmentManifest) -> Department:
        """Download and instantiate department"""
        ...

    def verify_department(self, manifest: DepartmentManifest) -> bool:
        """Verify signature, license, privacy guarantees"""
        ...
```

### Example: Installing Beekeeping MasterWeaver

```python
marketplace = DepartmentMarketplace()

# Discover beekeeping departments
beekeeping_depts = marketplace.discover_departments(domain="beekeeping")

# Find MasterWeaver
masterweaver_manifest = next(
    d for d in beekeeping_depts
    if d.name == "MasterWeaver"
)

# Verify before install
if marketplace.verify_department(masterweaver_manifest):
    masterweaver = marketplace.install_department(masterweaver_manifest)

    # Configure
    config = DepartmentConfig(
        confidence_threshold=0.70,
        privacy_mode="tee_only"
    )

    # Ready to use
    response = await masterweaver.execute(request)
```

---

## Testing Requirements

All departments MUST pass:

### 1. Confidence Contract Tests
```python
def test_confidence_reporting():
    """Department reports confidence accurately"""
    response = await dept.execute(request)
    assert 0.0 <= response.confidence.score <= 1.0
    assert response.confidence.justification  # Non-empty
    assert response.confidence.level == ConfidenceLevel.from_score(response.confidence.score)

def test_confidence_mismatch_handling():
    """Department handles confidence mismatches"""
    request = DepartmentRequest(confidence_expected=0.90, ...)
    response = await dept.execute(request)

    if response.confidence.score < 0.75:  # Below expectation
        verification = await dept.verify(response)
        assert verification.refinement_suggestions is not None
```

### 2. DS-STAR Loop Tests
```python
def test_verification_loop():
    """Department can verify and refine its own work"""
    response = await dept.execute(request)
    verification = await dept.verify(response)

    if not verification.sufficient:
        refined = await dept.refine(request, response, verification)
        assert refined.confidence.score > response.confidence.score

def test_learning_loop():
    """Department learns from accumulated signals"""
    signals = [...]  # 10+ learning signals
    await dept.update_strategy(signals)

    institutional_memory = await dept.get_institutional_memory("successful_strategies")
    assert len(institutional_memory) > 0
```

### 3. Privacy Tests
```python
def test_tee_execution():
    """Sensitive data stays in TEE"""
    response = await dept.execute(request_with_sensitive_data)
    assert response.privacy_metadata["tee_attestation"]
    assert "raw_data" not in str(response.result)

def test_differential_privacy():
    """Aggregates have DP applied"""
    response = await dept.execute(aggregate_request)
    assert response.privacy_metadata["dp_epsilon"] <= 1.0
    assert response.privacy_metadata["dp_delta"] <= 1e-5
```

### 4. Memory System Tests
```python
def test_multi_timescale_memory():
    """Department maintains memory at multiple timescales"""
    await dept.execute(request)

    assert len(dept.short_term_memory) > 0  # Immediate
    assert len(dept.medium_term_memory) > 0  # Session
    # Long-term populated after update_strategy()

def test_session_resumption():
    """Department can resume previous session"""
    session_id = "test_session_123"
    response1 = await dept.execute(DepartmentRequest(session_id=session_id, ...))

    session_state = await dept.get_session_state(session_id)
    assert session_state["confidence"] == response1.confidence.score
```

---

## Integration Patterns

### Pattern 1: Single Department Call

```python
# Simple request → response
request = DepartmentRequest(
    task_id="extract_entities_001",
    task_type="extract_entities",
    parameters={"transcript": audio_transcript},
    confidence_expected=0.75,
    context_preference="moderate"
)

response = await masterweaver.execute(request)

if response.confidence.score >= 0.75:
    # High confidence → accept
    entities = response.result
else:
    # Low confidence → verify and refine
    verification = await masterweaver.verify(response)
    if not verification.sufficient:
        response = await masterweaver.refine(request, response, verification)
```

### Pattern 2: Multi-Department Workflow

```python
# Step 1: MasterWeaver extracts entities
entities_response = await masterweaver.execute(extract_request)

# Step 2: Context enriches with knowledge graph
context_request = DepartmentRequest(
    task_type="expand_knowledge_graph",
    parameters={"entities": entities_response.result},
    confidence_expected=entities_response.confidence.score  # Inherit confidence
)
context_response = await context.execute(context_request)

# Step 3: Verification validates entire workflow
verification_request = DepartmentRequest(
    task_type="validate_workflow",
    parameters={
        "entities": entities_response,
        "context": context_response
    },
    confidence_expected=0.85
)
verification_response = await verification.execute(verification_request)
```

### Pattern 3: Learning Loop

```python
# Accumulate learning signals from multiple tasks
learning_signals = []

for task in completed_tasks:
    response = await dept.execute(task.request)
    learning_signals.append(response.learning_signals)

# Periodic update (hourly, daily, weekly based on confidence)
if len(learning_signals) >= dept.learning_threshold:
    await dept.update_strategy(learning_signals)
    learning_signals.clear()
```

---

## Next Steps

1. **Implement Base Classes**: Create abstract base class for `Department` protocol
2. **Port Context Department**: Refactor current HoloLoom to implement `ContextDepartment`
3. **Build MCP Adapter**: Wrap departments in MCP servers for cross-process communication
4. **Create Department Registry**: Local marketplace for development
5. **Add Privacy Layer**: TEE integration and differential privacy aggregation
6. **Build First Domain Set**: Beekeeping departments (MasterWeaver, Infrastructure, Context)

---

**Status**: DRAFT - Ready for Implementation
**Next Document**: [HOLOLOOM_CONTEXT_DEPARTMENT_MAPPING.md](HOLOLOOM_CONTEXT_DEPARTMENT_MAPPING.md)