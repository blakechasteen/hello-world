# HoloLoom → Context Department Mapping

**Version**: 1.0.0
**Date**: November 9, 2025
**Status**: REFERENCE - Implementation Guide

---

## Executive Summary

This document maps the **existing HoloLoom codebase** to the **Context Department** specification. The goal: Show how current HoloLoom (244D semantic space, Thompson Sampling, knowledge graphs) becomes one pluggable department in the larger multi-department architecture.

**Key Insight**: HoloLoom's current functionality is already a complete Context Department. We just need to wrap it in the department protocol.

---

## Current HoloLoom Architecture

```
HoloLoom/
├── weaving_orchestrator.py     # Main entry point (1,963 lines)
├── config.py                   # BARE/FAST/FUSED modes (460 lines)
├── policy/unified.py           # Thompson Sampling + neural policy
├── memory/
│   ├── cache.py                # BM25 + semantic retrieval
│   └── graph.py                # NetworkX knowledge graph
├── embedding/spectral.py       # Matryoshka multi-scale embeddings
├── semantic_calculus/          # 244D semantic space
├── reflection/                 # Learning & improvement
└── documentation/types.py      # Query, MemoryShard, Spacetime
```

**Current Capabilities**:
- Multi-pass enrichment (BARE/FAST/FUSED modes)
- Thompson Sampling for tool selection
- Knowledge graph expansion
- Matryoshka embeddings (96/192/384D)
- Confidence scoring
- Recursive learning

**Current Missing**:
- Department protocol wrapper
- Confidence negotiation interface
- DS-STAR verification loop
- Multi-timescale memory (has some, needs formalization)
- Privacy envelope (TEE integration)

---

## Mapping: HoloLoom → Context Department

### 1. Core Identity

| Department Property | HoloLoom Equivalent |
|---------------------|---------------------|
| `name` | "Context" |
| `domain` | "generic" (works for all domains) |
| `version` | "1.0.0" (current HoloLoom version) |
| `supported_tasks` | `["enrich_context", "retrieve_memories", "expand_knowledge_graph", "multi_scale_embedding"]` |
| `confidence_range` | `(0.55, 0.88)` (from architecture doc) |

### 2. Memory Systems

| Department Memory | HoloLoom Component | Location |
|-------------------|-------------------|----------|
| **Short-term** (ms) | Current query processing state | `weaving_orchestrator.py:WeavingOrchestrator` in-memory state |
| **Medium-term** (hours) | Session artifacts, spacetime results | `HoloLoom/reflection/buffer.py:ReflectionBuffer` |
| **Long-term** (weeks) | Learned patterns, confidence calibration | `HoloLoom/memory/graph.py` + `reflection/buffer.py` (persistence) |

**Mapping Details**:

```python
# Short-term memory (milliseconds - current request)
class ContextDepartment(Department):
    def __init__(self):
        self.short_term_memory = {}  # task_id → current state

    async def execute(self, request: DepartmentRequest):
        # Log current request
        self.short_term_memory[request.task_id] = {
            "query": request.parameters["query"],
            "timestamp": datetime.now(),
            "mode": self._select_mode(request.confidence_expected)
        }
        # ... execute ...

# Medium-term memory (hours - session artifacts)
# Currently: ReflectionBuffer
from HoloLoom.reflection.buffer import ReflectionBuffer

class ContextDepartment(Department):
    def __init__(self):
        self.medium_term_memory = ReflectionBuffer(
            capacity=1000,
            persist_path="./context_sessions"
        )

    async def execute(self, request: DepartmentRequest):
        # ... execute ...
        # Store result in medium-term
        await self.medium_term_memory.store(spacetime, feedback={})

# Long-term memory (weeks - institutional patterns)
# Currently: Partially implemented in reflection/semantic_learning.py
# Needs: Formal pattern extraction

class ContextDepartment(Department):
    def __init__(self):
        self.long_term_memory = {
            "successful_strategies": [],  # Which configs yield high confidence?
            "confidence_drift": [],       # Are we over/underestimating?
            "preferred_refinements": []   # Which refinements work best?
        }

    async def update_strategy(self, learning_signals):
        # Analyze patterns across many tasks
        # Update long-term memory
        # Adjust default configs
```

### 3. Execute Method

**Department Protocol**:
```python
async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
    ...
```

**HoloLoom Equivalent**: `weaving_orchestrator.py:WeavingOrchestrator.weave()`

**Mapping**:

```python
class ContextDepartment(Department):
    def __init__(self, config: DepartmentConfig):
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        from HoloLoom.config import Config

        # Map confidence to mode
        self.orchestrator = WeavingOrchestrator(cfg=Config.fused())

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        from HoloLoom.documentation.types import Query

        # Extract query
        query_text = request.parameters.get("query", "")
        query = Query(text=query_text)

        # Select mode based on confidence expectation
        mode = self._select_mode(request.confidence_expected)
        self.orchestrator.cfg = self._get_config_for_mode(mode)

        # Execute with HoloLoom
        async with self.orchestrator as orch:
            spacetime = await orch.weave(query)

        # Map spacetime to DepartmentResponse
        return self._spacetime_to_response(request, spacetime)

    def _select_mode(self, confidence_expected: float) -> str:
        """Map confidence expectation to HoloLoom mode"""
        if confidence_expected >= 0.85:
            return "FUSED"  # High expectation → full processing
        elif confidence_expected >= 0.70:
            return "FAST"   # Medium expectation → balanced
        else:
            return "BARE"   # Low expectation → minimal

    def _get_config_for_mode(self, mode: str):
        """Get HoloLoom config for mode"""
        from HoloLoom.config import Config
        if mode == "FUSED":
            return Config.fused()
        elif mode == "FAST":
            return Config.fast()
        else:
            return Config.bare()

    def _spacetime_to_response(
        self,
        request: DepartmentRequest,
        spacetime
    ) -> DepartmentResponse:
        """Convert HoloLoom Spacetime to DepartmentResponse"""
        from HoloLoom.documentation.types import Spacetime

        # Extract confidence
        confidence = ConfidenceMetadata(
            score=spacetime.confidence,
            justification=[
                f"Mode: {spacetime.metadata.get('mode', 'FAST')}",
                f"Matryoshka scale: {spacetime.metadata.get('matryoshka_scale')}D",
                f"KG nodes: {spacetime.metadata.get('kg_nodes', 0)}",
                f"Thompson samples: {spacetime.metadata.get('thompson_samples', 0)}"
            ],
            uncertainty_sources=self._extract_uncertainty_sources(spacetime),
            calibration_history=self._get_calibration_history()
        )

        # Determine detail level
        detail_level = self._determine_detail_level(
            confidence.score,
            request.context_preference
        )

        # Build response
        return DepartmentResponse(
            task_id=request.task_id,
            result=spacetime.output,
            confidence=confidence,
            detail_level=detail_level,
            reasoning=self._extract_reasoning(spacetime, detail_level),
            alternatives_considered=None,  # TODO: Extract from trace
            learning_signals={
                "cache_hit": spacetime.metadata.get("cache_hit", False),
                "latency_ms": spacetime.metadata.get("latency_ms", 0),
                "mode": spacetime.metadata.get("mode", "FAST")
            },
            session_state=self._build_session_state(request, spacetime)
        )

    def _extract_uncertainty_sources(self, spacetime) -> List[str]:
        """Extract what causes uncertainty in this result"""
        sources = []

        if spacetime.confidence < 0.70:
            sources.append("Low overall confidence")

        if spacetime.metadata.get("kg_nodes", 0) < 5:
            sources.append("Limited knowledge graph context")

        if spacetime.metadata.get("matryoshka_scale", 0) < 192:
            sources.append("Low-resolution embeddings used")

        return sources
```

### 4. Verify Method

**Department Protocol**:
```python
async def verify(self, response: DepartmentResponse) -> VerificationResult:
    ...
```

**HoloLoom Equivalent**: **NEW** - needs to be built

**Implementation**:

```python
class ContextDepartment(Department):
    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """Check if context enrichment was sufficient"""

        confidence = response.confidence.score
        confidence_valid = True
        reasoning_sound = True
        alternative_paths = []

        # Check 1: Confidence vs Quality Indicators
        quality_indicators = {
            "kg_nodes": response.learning_signals.get("kg_nodes", 0),
            "matryoshka_scale": response.learning_signals.get("matryoshka_scale", 0),
            "cache_hit": response.learning_signals.get("cache_hit", False)
        }

        # Low confidence but high-quality indicators = underconfident
        if confidence < 0.70 and quality_indicators["kg_nodes"] > 10:
            confidence_valid = False
            alternative_paths.append("Recalibrate confidence model (underestimating)")

        # High confidence but low-quality indicators = overconfident
        if confidence > 0.85 and quality_indicators["kg_nodes"] < 3:
            confidence_valid = False
            alternative_paths.append("Recalibrate confidence model (overestimating)")

        # Check 2: Detail level matches confidence
        expected_detail = self._determine_detail_level(confidence, "auto")
        if response.detail_level != expected_detail:
            reasoning_sound = False
            alternative_paths.append(f"Adjust detail level to {expected_detail}")

        # Check 3: Mode selection appropriate?
        mode = response.learning_signals.get("mode", "FAST")
        expected_mode = self._select_mode(confidence)
        if mode != expected_mode:
            reasoning_sound = False
            alternative_paths.append(f"Use {expected_mode} mode for this confidence level")

        # Determine if sufficient
        sufficient = (
            confidence_valid and
            reasoning_sound and
            confidence >= 0.65
        )

        return VerificationResult(
            sufficient=sufficient,
            confidence_valid=confidence_valid,
            reasoning_sound=reasoning_sound,
            alternative_paths=alternative_paths,
            refinement_suggestions={
                "upgrade_to_fused": not sufficient and mode != "FUSED",
                "expand_kg": not sufficient and quality_indicators["kg_nodes"] < 5,
                "increase_scale": not sufficient and quality_indicators["matryoshka_scale"] < 384
            } if not sufficient else None,
            escalation_needed=confidence < 0.40
        )
```

### 5. Refine Method

**Department Protocol**:
```python
async def refine(
    self,
    request: DepartmentRequest,
    prior_response: DepartmentResponse,
    verification: VerificationResult
) -> DepartmentResponse:
    ...
```

**HoloLoom Equivalent**: **Partially exists** in `recursive/` module (Phase 4 refinement)

**Integration**:

```python
class ContextDepartment(Department):
    async def refine(
        self,
        request: DepartmentRequest,
        prior_response: DepartmentResponse,
        verification: VerificationResult
    ) -> DepartmentResponse:
        """Apply verification feedback to improve response"""

        # Use existing recursive refinement system
        from HoloLoom.recursive import AdvancedRefiner, RefinementStrategy

        # Map verification suggestions to refinement strategy
        if verification.refinement_suggestions:
            if verification.refinement_suggestions.get("upgrade_to_fused"):
                # Switch to FUSED mode
                request.parameters["mode"] = "FUSED"

            if verification.refinement_suggestions.get("expand_kg"):
                # Enable knowledge graph expansion
                request.parameters["expand_kg"] = True

            if verification.refinement_suggestions.get("increase_scale"):
                # Use full 384D embeddings
                request.parameters["matryoshka_scale"] = 384

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
```

### 6. Update Strategy Method

**Department Protocol**:
```python
async def update_strategy(self, learning_signals: List[Dict[str, Any]]) -> None:
    ...
```

**HoloLoom Equivalent**: **Partially exists** in `reflection/semantic_learning.py`

**Integration**:

```python
class ContextDepartment(Department):
    async def update_strategy(self, learning_signals: List[Dict[str, Any]]) -> None:
        """Update context enrichment strategy based on patterns"""

        # Use existing semantic learning system
        from HoloLoom.reflection.semantic_learning import MultiTaskSemanticLearner

        if not hasattr(self, "learner"):
            self.learner = MultiTaskSemanticLearner(
                embedding_dim=244,
                hidden_dim=128
            )

        # Analyze refinement patterns
        refinement_patterns = self.long_term_memory.get("refinement_patterns", [])

        if len(refinement_patterns) >= 10:
            # Calculate average improvement per refinement type
            improvements = {}
            for pattern in refinement_patterns[-50:]:
                rtype = pattern["refinement_type"]
                improvements.setdefault(rtype, []).append(pattern["improvement"])

            # Update preferred refinements
            avg_improvements = {
                rtype: sum(imps) / len(imps)
                for rtype, imps in improvements.items()
            }

            self.long_term_memory["preferred_refinements"] = sorted(
                avg_improvements.items(),
                key=lambda x: x[1],
                reverse=True
            )

        # Analyze confidence calibration
        # TODO: Track predicted vs actual quality over time
        # Update confidence model accordingly

        self.logger.info(f"Strategy updated: {self.long_term_memory['preferred_refinements']}")
```

### 7. Session Management

**Department Protocol**:
```python
async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
    ...
```

**HoloLoom Equivalent**: `reflection/buffer.py:ReflectionBuffer`

**Integration**:

```python
class ContextDepartment(Department):
    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session with confidence-indexed detail"""

        # Query reflection buffer
        session = await self.medium_term_memory.retrieve(
            query={"session_id": session_id},
            limit=1
        )

        if not session:
            return None

        spacetime = session[0]
        confidence = spacetime.confidence

        # High confidence: compact summary
        if confidence >= 0.85:
            return {
                "status": "COMPLETE",
                "confidence": confidence,
                "result_summary": str(spacetime.output)[:200],
                "artifacts": "compact",
                "size_kb": 5
            }

        # Low confidence: full audit trail
        else:
            return {
                "status": "PENDING_REVIEW" if confidence < 0.65 else "COMPLETE",
                "confidence": confidence,
                "full_spacetime": spacetime,
                "trace": spacetime.trace,
                "metadata": spacetime.metadata,
                "artifacts": "exhaustive",
                "size_kb": 500
            }
```

### 8. Institutional Memory

**Department Protocol**:
```python
async def get_institutional_memory(self, pattern_type: str) -> Dict[str, Any]:
    ...
```

**HoloLoom Equivalent**: **NEW** - formalize existing patterns

**Implementation**:

```python
class ContextDepartment(Department):
    async def get_institutional_memory(self, pattern_type: str) -> Dict[str, Any]:
        """Query learned patterns"""

        if pattern_type == "successful_strategies":
            return {
                "preferred_refinements": self.long_term_memory.get("preferred_refinements", []),
                "high_confidence_configs": self._extract_high_confidence_configs(),
                "optimal_modes": self._analyze_mode_performance()
            }

        elif pattern_type == "confidence_drift":
            return {
                "calibration_history": self._get_calibration_history(),
                "overconfidence_rate": self._calculate_overconfidence_rate(),
                "underconfidence_rate": self._calculate_underconfidence_rate()
            }

        elif pattern_type == "failure_modes":
            return {
                "common_low_confidence_scenarios": self._analyze_low_confidence_patterns(),
                "refinement_success_rate": self._calculate_refinement_success_rate()
            }

        return {}

    def _extract_high_confidence_configs(self) -> List[Dict[str, Any]]:
        """Find configurations that consistently yield high confidence"""
        # Analyze reflection buffer for high-confidence results
        # Extract common config patterns
        return []

    def _analyze_mode_performance(self) -> Dict[str, float]:
        """Average confidence per mode"""
        return {
            "BARE": 0.62,
            "FAST": 0.75,
            "FUSED": 0.85
        }
```

---

## Implementation Phases

### Phase 1: Wrap Existing HoloLoom (Week 1-2)

**Goal**: Create `ContextDepartment` class that wraps current HoloLoom

**Tasks**:
1. Create `HoloLoom/departments/context.py`
2. Implement `execute()` method (wraps `WeavingOrchestrator.weave()`)
3. Implement `_spacetime_to_response()` (maps Spacetime → DepartmentResponse)
4. Add confidence metadata extraction
5. Test basic request → response flow

**Output**: Working Context Department (no verification yet)

### Phase 2: Add Verification Loop (Week 3-4)

**Goal**: Implement DS-STAR verification pattern

**Tasks**:
1. Implement `verify()` method (quality checks)
2. Implement `refine()` method (integrate recursive refinement)
3. Add confidence calibration tracking
4. Test verification → refinement loop

**Output**: Context Department with self-improvement

### Phase 3: Formalize Memory Systems (Week 5-6)

**Goal**: Multi-timescale memory management

**Tasks**:
1. Formalize short-term memory (in-memory state)
2. Integrate ReflectionBuffer as medium-term memory
3. Build long-term memory (pattern extraction)
4. Implement `get_session_state()` with confidence-indexed detail
5. Implement `get_institutional_memory()` with pattern queries

**Output**: Complete memory hierarchy

### Phase 4: Institutional Learning (Week 7-8)

**Goal**: Magic cycle - learn from execution

**Tasks**:
1. Implement `update_strategy()` (periodic learning)
2. Add confidence calibration
3. Track preferred refinements
4. Analyze mode performance
5. Auto-adjust default configs

**Output**: Self-improving Context Department

### Phase 5: Privacy Envelope (Week 9-10)

**Goal**: TEE integration and verifiable output

**Tasks**:
1. Add TEE processing for sensitive contexts
2. Implement differential privacy for aggregates
3. Generate verifiable attestations
4. Add privacy metadata to responses

**Output**: Privacy-first Context Department

---

## File Structure

```
HoloLoom/
├── departments/                    # NEW: Department implementations
│   ├── __init__.py
│   ├── base.py                     # BaseDepartment (abstract base class)
│   ├── context.py                  # ContextDepartment (wraps existing HoloLoom)
│   └── protocol.py                 # Department protocol (from spec)
│
├── weaving_orchestrator.py         # UNCHANGED: Core HoloLoom (used by ContextDepartment)
├── config.py                       # UNCHANGED: BARE/FAST/FUSED modes
├── policy/                         # UNCHANGED: Thompson Sampling
├── memory/                         # UNCHANGED: Knowledge graphs
├── embedding/                      # UNCHANGED: Matryoshka embeddings
├── semantic_calculus/              # UNCHANGED: 244D space
├── reflection/                     # EXTENDED: Add confidence calibration
│   ├── buffer.py                   # Medium-term memory
│   ├── semantic_learning.py        # EXTENDED: Integrate with update_strategy()
│   └── confidence_calibration.py   # NEW: Track prediction vs actual
│
└── documentation/types.py          # EXTENDED: Add DepartmentRequest/Response
```

---

## Code Example: Complete Context Department

```python
# HoloLoom/departments/context.py

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from .base import BaseDepartment
from .protocol import (
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
    ConfidenceMetadata,
    ConfidenceLevel,
    DepartmentConfig
)

class ContextDepartment(BaseDepartment):
    """
    Context enrichment department.
    Wraps existing HoloLoom weaving orchestrator.
    """

    def __init__(self, config: DepartmentConfig):
        # Department identity
        self.name = "Context"
        self.domain = "generic"
        self.version = "1.0.0"
        self.supported_tasks = [
            "enrich_context",
            "retrieve_memories",
            "expand_knowledge_graph",
            "multi_scale_embedding"
        ]
        self.confidence_range = (0.55, 0.88)

        # Memory systems
        self.short_term_memory = {}  # Current requests
        from HoloLoom.reflection.buffer import ReflectionBuffer
        self.medium_term_memory = ReflectionBuffer(
            capacity=1000,
            persist_path="./context_sessions"
        )
        self.long_term_memory = {
            "refinement_patterns": [],
            "preferred_refinements": [],
            "confidence_calibration": []
        }

        # HoloLoom orchestrator
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        from HoloLoom.config import Config
        self.orchestrator = WeavingOrchestrator(cfg=Config.fused())

        # Config
        self.config = config
        self.logger = logging.getLogger(f"Department.{self.name}")

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Execute context enrichment using HoloLoom"""
        from HoloLoom.documentation.types import Query

        # Log to short-term memory
        self.short_term_memory[request.task_id] = {
            "query": request.parameters.get("query", ""),
            "timestamp": datetime.now(),
            "confidence_expected": request.confidence_expected
        }

        # Select mode based on confidence expectation
        mode = self._select_mode(request.confidence_expected)
        self.orchestrator.cfg = self._get_config_for_mode(mode)

        # Execute
        query = Query(text=request.parameters.get("query", ""))
        async with self.orchestrator as orch:
            spacetime = await orch.weave(query)

        # Convert to DepartmentResponse
        response = self._spacetime_to_response(request, spacetime)

        # Store in medium-term memory
        await self.medium_term_memory.store(spacetime, feedback={})

        return response

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """Verify context enrichment quality"""
        confidence = response.confidence.score
        confidence_valid = True
        reasoning_sound = True
        alternative_paths = []

        # Check confidence vs quality indicators
        kg_nodes = response.learning_signals.get("kg_nodes", 0)
        if confidence > 0.85 and kg_nodes < 3:
            confidence_valid = False
            alternative_paths.append("Recalibrate confidence (overestimating)")

        # Check detail level
        expected_detail = self._determine_detail_level(confidence, "auto")
        if response.detail_level != expected_detail:
            reasoning_sound = False
            alternative_paths.append(f"Adjust detail to {expected_detail}")

        sufficient = confidence_valid and reasoning_sound and confidence >= 0.65

        return VerificationResult(
            sufficient=sufficient,
            confidence_valid=confidence_valid,
            reasoning_sound=reasoning_sound,
            alternative_paths=alternative_paths,
            refinement_suggestions={
                "upgrade_to_fused": not sufficient,
                "expand_kg": kg_nodes < 5
            } if not sufficient else None,
            escalation_needed=confidence < 0.40
        )

    async def refine(
        self,
        request: DepartmentRequest,
        prior_response: DepartmentResponse,
        verification: VerificationResult
    ) -> DepartmentResponse:
        """Apply refinements"""
        # Apply suggestions
        if verification.refinement_suggestions:
            if verification.refinement_suggestions.get("upgrade_to_fused"):
                request.parameters["mode"] = "FUSED"
            if verification.refinement_suggestions.get("expand_kg"):
                request.parameters["expand_kg"] = True

        # Re-execute
        refined = await self.execute(request)

        # Log pattern
        self.long_term_memory["refinement_patterns"].append({
            "original_confidence": prior_response.confidence.score,
            "refined_confidence": refined.confidence.score,
            "improvement": refined.confidence.score - prior_response.confidence.score,
            "timestamp": datetime.now()
        })

        return refined

    async def update_strategy(self, learning_signals: List[Dict[str, Any]]) -> None:
        """Learn from accumulated patterns"""
        patterns = self.long_term_memory["refinement_patterns"]

        if len(patterns) >= 10:
            # Analyze improvements
            avg_improvement = sum(p["improvement"] for p in patterns[-50:]) / min(50, len(patterns))

            # Update strategy if improvements are consistent
            if avg_improvement > 0.05:
                self.long_term_memory["preferred_refinements"].append({
                    "strategy": "upgrade_to_fused",
                    "avg_improvement": avg_improvement,
                    "timestamp": datetime.now()
                })

            self.logger.info(f"Strategy updated: avg_improvement={avg_improvement:.3f}")

    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session with confidence-indexed detail"""
        session = self.medium_term_memory.get(session_id)
        if not session:
            return None

        confidence = session.confidence

        if confidence >= 0.85:
            return {"status": "COMPLETE", "confidence": confidence, "artifacts": "compact"}
        else:
            return {"status": "PENDING_REVIEW", "confidence": confidence, "artifacts": "exhaustive"}

    async def get_institutional_memory(self, pattern_type: str) -> Dict[str, Any]:
        """Query learned patterns"""
        if pattern_type == "successful_strategies":
            return {"preferred_refinements": self.long_term_memory["preferred_refinements"]}
        return {}

    async def health_check(self) -> Dict[str, Any]:
        """Health status"""
        return {
            "status": "healthy",
            "name": self.name,
            "version": self.version,
            "confidence_range": self.confidence_range,
            "memory_usage": {
                "short_term": len(self.short_term_memory),
                "medium_term": len(self.medium_term_memory),
                "long_term": len(self.long_term_memory["refinement_patterns"])
            }
        }

    # Helper methods

    def _select_mode(self, confidence_expected: float) -> str:
        """Map confidence to mode"""
        if confidence_expected >= 0.85:
            return "FUSED"
        elif confidence_expected >= 0.70:
            return "FAST"
        return "BARE"

    def _get_config_for_mode(self, mode: str):
        """Get config for mode"""
        from HoloLoom.config import Config
        return {"FUSED": Config.fused(), "FAST": Config.fast(), "BARE": Config.bare()}[mode]

    def _spacetime_to_response(self, request, spacetime) -> DepartmentResponse:
        """Convert Spacetime to DepartmentResponse"""
        confidence = ConfidenceMetadata(
            score=spacetime.confidence,
            justification=[f"Mode: {spacetime.metadata.get('mode', 'FAST')}"],
            uncertainty_sources=[]
        )

        return DepartmentResponse(
            task_id=request.task_id,
            result=spacetime.output,
            confidence=confidence,
            detail_level=self._determine_detail_level(confidence.score, request.context_preference),
            learning_signals={"cache_hit": spacetime.metadata.get("cache_hit", False)}
        )

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
        return "exhaustive"
```

---

## Testing Strategy

### Unit Tests (Test Department Methods)

```python
# tests/departments/test_context_department.py

import pytest
from HoloLoom.apps.departments.context import ContextDepartment
from HoloLoom.apps.departments.protocol import DepartmentRequest, DepartmentConfig

@pytest.mark.asyncio
async def test_execute_returns_response():
    """Context department returns DepartmentResponse"""
    dept = ContextDepartment(DepartmentConfig())

    request = DepartmentRequest(
        task_id="test_001",
        task_type="enrich_context",
        parameters={"query": "What is Thompson Sampling?"},
        confidence_expected=0.75,
        context_preference="moderate",
        privacy_level="public"
    )

    response = await dept.execute(request)

    assert response.task_id == "test_001"
    assert 0.0 <= response.confidence.score <= 1.0
    assert response.detail_level in ["minimal", "moderate", "detailed", "exhaustive"]

@pytest.mark.asyncio
async def test_verification_loop():
    """Department can verify and refine"""
    dept = ContextDepartment(DepartmentConfig())

    request = DepartmentRequest(
        task_id="test_002",
        task_type="enrich_context",
        parameters={"query": "Ambiguous query"},
        confidence_expected=0.85,
        context_preference="auto",
        privacy_level="public"
    )

    response = await dept.execute(request)
    verification = await dept.verify(response)

    if not verification.sufficient:
        refined = await dept.refine(request, response, verification)
        assert refined.confidence.score >= response.confidence.score

@pytest.mark.asyncio
async def test_learning_loop():
    """Department learns from patterns"""
    dept = ContextDepartment(DepartmentConfig())

    # Execute 10 tasks
    for i in range(10):
        request = DepartmentRequest(
            task_id=f"test_{i}",
            task_type="enrich_context",
            parameters={"query": f"Query {i}"},
            confidence_expected=0.75,
            context_preference="auto",
            privacy_level="public"
        )
        response = await dept.execute(request)

    # Update strategy
    await dept.update_strategy([])

    # Check institutional memory
    memory = await dept.get_institutional_memory("successful_strategies")
    assert "preferred_refinements" in memory
```

### Integration Tests (Test with Real HoloLoom)

```python
# tests/integration/test_context_department_integration.py

import pytest
from HoloLoom.apps.departments.context import ContextDepartment
from HoloLoom.config import Config

@pytest.mark.asyncio
async def test_context_department_uses_hololoom():
    """Context department correctly uses HoloLoom orchestrator"""
    dept = ContextDepartment(DepartmentConfig())

    request = DepartmentRequest(
        task_id="integration_001",
        task_type="enrich_context",
        parameters={"query": "Explain multi-scale embeddings"},
        confidence_expected=0.80,
        context_preference="detailed",
        privacy_level="public"
    )

    response = await dept.execute(request)

    # Check HoloLoom features are used
    assert "matryoshka" in str(response.result).lower() or "embedding" in str(response.result).lower()
    assert response.confidence.score > 0.0
```

---

## Summary

**Current HoloLoom → Context Department Mapping**:

| HoloLoom Component | Context Department Method | Status |
|--------------------|---------------------------|--------|
| `WeavingOrchestrator.weave()` | `execute()` | ✅ Direct mapping |
| `Spacetime` output | `DepartmentResponse` | ✅ Convert in wrapper |
| `ReflectionBuffer` | Medium-term memory | ✅ Use directly |
| Recursive refinement | `refine()` | ⚠️ Integrate existing |
| Semantic learning | `update_strategy()` | ⚠️ Integrate existing |
| Confidence scoring | `ConfidenceMetadata` | ✅ Extract from Spacetime |
| - | `verify()` | ❌ NEW - needs implementation |
| - | `get_institutional_memory()` | ❌ NEW - formalize patterns |
| - | Privacy envelope | ❌ NEW - add TEE integration |

**Implementation Effort**:
- **Week 1-2**: Basic wrapper (80% existing code reuse)
- **Week 3-4**: Verification loop (50% existing code reuse from recursive/)
- **Week 5-6**: Formalize memory systems (70% existing code reuse)
- **Week 7-8**: Institutional learning (60% existing code reuse from reflection/)
- **Week 9-10**: Privacy envelope (NEW - full implementation)

**Result**: Current HoloLoom becomes a fully-functional Context Department in the multi-department architecture, with minimal code changes and maximum reuse.

---

**Next Document**: [PHASE_1_IMPLEMENTATION_PLAN.md](PHASE_1_IMPLEMENTATION_PLAN.md)