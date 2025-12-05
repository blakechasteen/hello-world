"""
HoloLoom Bridge - Phase 2 Production Integration
================================================

High-level bridge from NeuroHood dreams to HoloLoom's production systems.

Extends Phase 1 adapter with:
- DreamSpinner for MemoryShard conversion
- DreamDepartment for DS-STAR verification
- Production-grade error handling
- Complete audit trail

Usage:
    from NeuroHood.dreams.hololoom_bridge import HoloLoomBridge

    async with HoloLoomBridge() as bridge:
        # Store with DS-STAR verification
        result = await bridge.store_dream(narrative, "resident_001")
        print(f"Stored: {result.stored}, Verification: {result.verification_passed}")

        # Query semantically
        dreams = await bridge.query_dreams("water symbolism")

        # Get analysis
        analysis = await bridge.analyze_dream(narrative)

Author: Phase 2 Production Integration
Date: November 2025
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

# Phase 1 adapter
from .hololoom_adapter import (
    store_dream_narrative,
    recall_dreams,
    reflect_on_dream,
    is_hololoom_available,
    DreamMemoryBridge,
    HOLOLOOM_AVAILABLE
)

# Try to import Phase 2 components
try:
    from HoloLoom.departments.dream import (
        DreamDepartment,
        DreamDSStarVerifier,
        DreamAction,
        VerificationResult,
        create_dream_department
    )
    DREAM_DEPARTMENT_AVAILABLE = True
except ImportError:
    DREAM_DEPARTMENT_AVAILABLE = False
    DreamDepartment = None

try:
    from HoloLoom.spinningWheel.dream_spinner import (
        DreamSpinner,
        DreamSpinnerConfig
    )
    DREAM_SPINNER_AVAILABLE = True
except ImportError:
    DREAM_SPINNER_AVAILABLE = False
    DreamSpinner = None

try:
    from HoloLoom.protocols.types import MemoryShard
    TYPES_AVAILABLE = True
except ImportError:
    TYPES_AVAILABLE = False
    MemoryShard = None


logger = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class StoreResult:
    """Result of dream storage operation."""
    stored: bool
    memory_id: Optional[str] = None
    verification_passed: Optional[bool] = None
    verification_confidence: Optional[float] = None
    verification_messages: List[str] = field(default_factory=list)
    shards_created: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stored": self.stored,
            "memory_id": self.memory_id,
            "verification_passed": self.verification_passed,
            "verification_confidence": self.verification_confidence,
            "verification_messages": self.verification_messages,
            "shards_created": self.shards_created,
            "error": self.error
        }


@dataclass
class QueryResult:
    """Result of dream query operation."""
    count: int
    dreams: List[Dict[str, Any]] = field(default_factory=list)
    query: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "dreams": self.dreams,
            "query": self.query,
            "error": self.error
        }


@dataclass
class AnalysisResult:
    """Result of dream analysis operation."""
    title: Optional[str] = None
    archetype: Optional[str] = None
    mood: Optional[str] = None
    interpretation: Optional[str] = None
    symbols: List[str] = field(default_factory=list)
    literary_inspirations: List[str] = field(default_factory=list)
    emotional_profile: Dict[str, float] = field(default_factory=dict)
    protagonist_state: Dict[str, float] = field(default_factory=dict)
    verification: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "archetype": self.archetype,
            "mood": self.mood,
            "interpretation": self.interpretation,
            "symbols": self.symbols,
            "literary_inspirations": self.literary_inspirations,
            "emotional_profile": self.emotional_profile,
            "protagonist_state": self.protagonist_state,
            "verification": self.verification,
            "error": self.error
        }


@dataclass
class SpinResult:
    """Result of dream spinning operation."""
    shard_count: int
    shards: List[Any] = field(default_factory=list)  # List[MemoryShard]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_count": self.shard_count,
            "shards": [
                {
                    "id": s.id,
                    "text": s.text[:100] + "..." if len(s.text) > 100 else s.text,
                    "entities": s.entities[:5] if hasattr(s, 'entities') else [],
                    "motifs": s.motifs[:5] if hasattr(s, 'motifs') else []
                }
                for s in self.shards[:10]
            ],
            "error": self.error
        }


# =============================================================================
# HoloLoom Bridge
# =============================================================================

class HoloLoomBridge:
    """
    Production bridge from NeuroHood dreams to HoloLoom.

    Provides:
    - DS-STAR verified dream storage
    - Semantic dream querying
    - Dream analysis and interpretation
    - MemoryShard conversion via DreamSpinner
    - Complete audit trail

    Graceful degradation:
    - Without HoloLoom: Returns empty results, no crashes
    - Without DreamDepartment: Falls back to Phase 1 adapter
    - Without DreamSpinner: Skips MemoryShard conversion

    Usage:
        async with HoloLoomBridge() as bridge:
            result = await bridge.store_dream(narrative, "resident_001")
            dreams = await bridge.query_dreams("ocean journey")
            analysis = await bridge.analyze_dream(narrative)
    """

    def __init__(
        self,
        enable_verification: bool = True,
        strict_verification: bool = False,
        enable_spinner: bool = True,
        auto_spin_on_store: bool = True
    ):
        """
        Initialize HoloLoom Bridge.

        Args:
            enable_verification: Run DS-STAR verification on store
            strict_verification: Require all DS-STAR checks to pass
            enable_spinner: Enable MemoryShard conversion
            auto_spin_on_store: Auto-create MemoryShards when storing
        """
        self.enable_verification = enable_verification
        self.strict_verification = strict_verification
        self.enable_spinner = enable_spinner and DREAM_SPINNER_AVAILABLE
        self.auto_spin_on_store = auto_spin_on_store

        # Components (initialized on __aenter__)
        self._department: Optional[Any] = None
        self._spinner: Optional[Any] = None
        self._memory_bridge: Optional[DreamMemoryBridge] = None
        self._verifier: Optional[Any] = None

        self._started = False

        # Statistics
        self.stats = {
            "dreams_stored": 0,
            "dreams_queried": 0,
            "dreams_analyzed": 0,
            "shards_created": 0,
            "verifications_passed": 0,
            "verifications_failed": 0,
            "errors": 0,
            "start_time": None
        }

    async def __aenter__(self) -> 'HoloLoomBridge':
        """Enter async context and initialize components."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and cleanup."""
        await self.stop()

    async def start(self):
        """Start bridge and initialize components."""
        if self._started:
            return

        logger.info("Starting HoloLoom Bridge (Phase 2)")

        # Initialize DreamDepartment if available
        if DREAM_DEPARTMENT_AVAILABLE:
            self._department = create_dream_department(
                verify_on_store=self.enable_verification,
                strict_verification=self.strict_verification,
                enable_spinner=self.enable_spinner
            )
            await self._department.start()
            logger.info("DreamDepartment initialized")
        else:
            logger.warning("DreamDepartment not available, using Phase 1 adapter")

        # Initialize DreamSpinner if available
        if self.enable_spinner and DREAM_SPINNER_AVAILABLE:
            self._spinner = DreamSpinner(DreamSpinnerConfig())
            logger.info("DreamSpinner initialized")

        # Initialize standalone verifier for quick verification
        if DREAM_DEPARTMENT_AVAILABLE:
            self._verifier = DreamDSStarVerifier(strict=self.strict_verification)

        # Initialize Phase 1 memory bridge
        self._memory_bridge = DreamMemoryBridge()
        await self._memory_bridge.__aenter__()
        logger.info(f"DreamMemoryBridge initialized (available: {self._memory_bridge.available})")

        self._started = True
        self.stats["start_time"] = datetime.now()
        logger.info("HoloLoom Bridge started successfully")

    async def stop(self):
        """Stop bridge and cleanup."""
        if not self._started:
            return

        logger.info("Stopping HoloLoom Bridge")

        # Cleanup department
        if self._department:
            await self._department.stop()
            self._department = None

        # Cleanup memory bridge
        if self._memory_bridge:
            await self._memory_bridge.__aexit__(None, None, None)
            self._memory_bridge = None

        self._started = False
        logger.info("HoloLoom Bridge stopped")

    # =========================================================================
    # Core API
    # =========================================================================

    async def store_dream(
        self,
        narrative: Any,
        resident_id: str,
        skip_verification: bool = False
    ) -> StoreResult:
        """
        Store dream narrative with DS-STAR verification.

        Args:
            narrative: DreamNarrative object
            resident_id: Resident identifier
            skip_verification: Skip DS-STAR verification

        Returns:
            StoreResult with storage outcome and verification details
        """
        if not self._started:
            return StoreResult(stored=False, error="Bridge not started")

        try:
            # Use department if available (full Phase 2)
            if self._department:
                result = await self._department.process({
                    "action": "store",
                    "narrative": narrative,
                    "resident_id": resident_id
                })

                verification = result.get("verification", {})

                # Also create MemoryShards if auto_spin enabled
                shards_created = 0
                if self.auto_spin_on_store and self._spinner and result.get("success"):
                    spin_result = await self._spinner.spin(narrative, resident_id=resident_id)
                    shards_created = len(spin_result.shards) if spin_result.shards else 0
                    self.stats["shards_created"] += shards_created

                if result.get("success"):
                    self.stats["dreams_stored"] += 1
                    if verification.get("passed"):
                        self.stats["verifications_passed"] += 1
                    else:
                        self.stats["verifications_failed"] += 1

                return StoreResult(
                    stored=result.get("success", False),
                    memory_id=result.get("result", {}).get("memory_id"),
                    verification_passed=verification.get("passed"),
                    verification_confidence=verification.get("confidence"),
                    verification_messages=verification.get("messages", []),
                    shards_created=shards_created,
                    error=result.get("error")
                )

            else:
                # Fall back to Phase 1 adapter
                if self._memory_bridge and self._memory_bridge.available:
                    # Run standalone verification if requested
                    verification_result = None
                    if self.enable_verification and not skip_verification and self._verifier:
                        verification_result = self._verifier.verify(narrative)
                        if not verification_result.passed:
                            self.stats["verifications_failed"] += 1
                            return StoreResult(
                                stored=False,
                                verification_passed=False,
                                verification_confidence=verification_result.confidence,
                                verification_messages=verification_result.messages,
                                error="DS-STAR verification failed"
                            )
                        self.stats["verifications_passed"] += 1

                    memory = await self._memory_bridge.store(narrative, resident_id)

                    # Spin if enabled
                    shards_created = 0
                    if self.auto_spin_on_store and self._spinner:
                        spin_result = await self._spinner.spin(narrative, resident_id=resident_id)
                        shards_created = len(spin_result.shards) if spin_result.shards else 0
                        self.stats["shards_created"] += shards_created

                    self.stats["dreams_stored"] += 1

                    return StoreResult(
                        stored=memory is not None,
                        memory_id=memory.id if memory else None,
                        verification_passed=verification_result.passed if verification_result else None,
                        verification_confidence=verification_result.confidence if verification_result else None,
                        verification_messages=verification_result.messages if verification_result else [],
                        shards_created=shards_created
                    )
                else:
                    return StoreResult(
                        stored=False,
                        error="HoloLoom not available"
                    )

        except Exception as e:
            logger.error(f"Error storing dream: {e}", exc_info=True)
            self.stats["errors"] += 1
            return StoreResult(stored=False, error=str(e))

    async def query_dreams(
        self,
        query: str,
        resident_id: Optional[str] = None,
        limit: int = 10
    ) -> QueryResult:
        """
        Query dreams semantically.

        Args:
            query: Search query (e.g., "water symbolism", "transformation dreams")
            resident_id: Optional filter by resident
            limit: Maximum results

        Returns:
            QueryResult with matching dreams
        """
        if not self._started:
            return QueryResult(count=0, error="Bridge not started")

        try:
            # Use department if available
            if self._department:
                result = await self._department.process({
                    "action": "query",
                    "query": query,
                    "resident_id": resident_id,
                    "limit": limit
                })

                if result.get("success"):
                    self.stats["dreams_queried"] += 1
                    res = result.get("result", {})
                    return QueryResult(
                        count=res.get("count", 0),
                        dreams=res.get("dreams", []),
                        query=query
                    )
                else:
                    return QueryResult(
                        count=0,
                        query=query,
                        error=result.get("error")
                    )

            else:
                # Fall back to Phase 1 adapter
                if self._memory_bridge and self._memory_bridge.available:
                    memories = await self._memory_bridge.recall(
                        query,
                        resident_id=resident_id,
                        limit=limit
                    )

                    self.stats["dreams_queried"] += 1

                    return QueryResult(
                        count=len(memories),
                        dreams=[
                            {
                                "id": m.id if hasattr(m, 'id') else None,
                                "text": m.text[:200] + "..." if len(m.text) > 200 else m.text,
                                "archetype": m.context.get("archetype") if hasattr(m, 'context') else None
                            }
                            for m in memories
                        ],
                        query=query
                    )
                else:
                    return QueryResult(count=0, query=query, error="HoloLoom not available")

        except Exception as e:
            logger.error(f"Error querying dreams: {e}", exc_info=True)
            self.stats["errors"] += 1
            return QueryResult(count=0, query=query, error=str(e))

    async def analyze_dream(self, narrative: Any) -> AnalysisResult:
        """
        Analyze dream for insights.

        Args:
            narrative: DreamNarrative object

        Returns:
            AnalysisResult with dream analysis and interpretation
        """
        if not self._started:
            return AnalysisResult(error="Bridge not started")

        try:
            # Use department if available
            if self._department:
                result = await self._department.process({
                    "action": "analyze",
                    "narrative": narrative
                })

                if result.get("success"):
                    self.stats["dreams_analyzed"] += 1
                    res = result.get("result", {})
                    verification = result.get("verification", {})

                    return AnalysisResult(
                        title=res.get("title"),
                        archetype=res.get("archetype"),
                        mood=res.get("mood"),
                        interpretation=res.get("jungian_interpretation"),
                        symbols=res.get("symbols", []),
                        literary_inspirations=res.get("literary_inspirations", []),
                        emotional_profile=res.get("emotional_profile", {}),
                        protagonist_state=res.get("protagonist_state", {}),
                        verification=verification
                    )
                else:
                    return AnalysisResult(error=result.get("error"))

            else:
                # Manual analysis without department
                self.stats["dreams_analyzed"] += 1

                analysis = AnalysisResult(
                    title=getattr(narrative, 'title', None),
                    archetype=getattr(narrative.archetype, 'value', str(narrative.archetype)) if hasattr(narrative, 'archetype') else None,
                    mood=getattr(narrative.mood, 'value', str(narrative.mood)) if hasattr(narrative, 'mood') else None,
                    interpretation=getattr(narrative, 'jungian_interpretation', None),
                    symbols=getattr(narrative, 'symbols_used', []),
                    literary_inspirations=getattr(narrative, 'literary_inspirations', [])
                )

                # Calculate emotional profile
                if hasattr(narrative, 'acts') and narrative.acts:
                    intensities = []
                    for act in narrative.acts:
                        if hasattr(act, 'emotional_intensity'):
                            intensities.append(act.emotional_intensity)

                    if intensities:
                        analysis.emotional_profile = {
                            "min": min(intensities),
                            "max": max(intensities),
                            "avg": sum(intensities) / len(intensities)
                        }

                # Add protagonist state
                if hasattr(narrative, 'protagonist_state'):
                    analysis.protagonist_state = narrative.protagonist_state

                # Add verification if verifier available
                if self._verifier:
                    v = self._verifier.verify(narrative)
                    analysis.verification = v.to_dict()

                return analysis

        except Exception as e:
            logger.error(f"Error analyzing dream: {e}", exc_info=True)
            self.stats["errors"] += 1
            return AnalysisResult(error=str(e))

    async def spin_dream(
        self,
        narrative: Any,
        resident_id: Optional[str] = None
    ) -> SpinResult:
        """
        Convert dream to MemoryShards.

        Args:
            narrative: DreamNarrative object
            resident_id: Optional resident identifier

        Returns:
            SpinResult with MemoryShards
        """
        if not self._started:
            return SpinResult(shard_count=0, error="Bridge not started")

        if not self._spinner:
            return SpinResult(shard_count=0, error="DreamSpinner not available")

        try:
            result = await self._spinner.spin(narrative, resident_id=resident_id)

            if result.shards:
                self.stats["shards_created"] += len(result.shards)

            return SpinResult(
                shard_count=len(result.shards) if result.shards else 0,
                shards=result.shards or []
            )

        except Exception as e:
            logger.error(f"Error spinning dream: {e}", exc_info=True)
            self.stats["errors"] += 1
            return SpinResult(shard_count=0, error=str(e))

    async def verify_dream(
        self,
        narrative: Any,
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Run DS-STAR verification on dream.

        Args:
            narrative: DreamNarrative object
            strict: Require all checks to pass

        Returns:
            Verification result dict
        """
        if not self._started:
            return {"passed": False, "error": "Bridge not started"}

        try:
            # Use department if available
            if self._department:
                result = await self._department.process({
                    "action": "verify",
                    "narrative": narrative,
                    "verify_strict": strict
                })

                verification = result.get("verification", {})
                if verification.get("passed"):
                    self.stats["verifications_passed"] += 1
                else:
                    self.stats["verifications_failed"] += 1

                return verification

            elif self._verifier:
                # Use standalone verifier
                verifier = DreamDSStarVerifier(strict=strict)
                v = verifier.verify(narrative)

                if v.passed:
                    self.stats["verifications_passed"] += 1
                else:
                    self.stats["verifications_failed"] += 1

                return v.to_dict()

            else:
                return {"passed": True, "message": "No verifier available"}

        except Exception as e:
            logger.error(f"Error verifying dream: {e}", exc_info=True)
            self.stats["errors"] += 1
            return {"passed": False, "error": str(e)}

    async def reflect(
        self,
        dream_memories: List[Any],
        feedback: Dict[str, Any]
    ) -> bool:
        """
        Provide feedback on dreams for learning.

        Args:
            dream_memories: Dreams to reflect on
            feedback: Feedback dict (e.g., {"meaningful": True, "emotional_impact": 0.8})

        Returns:
            True if reflection was recorded
        """
        if not self._started:
            return False

        try:
            # Use department if available
            if self._department:
                result = await self._department.process({
                    "action": "reflect",
                    "dream_memories": dream_memories,
                    "feedback": feedback
                })
                return result.get("success", False)

            elif self._memory_bridge and self._memory_bridge.available:
                await self._memory_bridge.reflect(dream_memories, feedback)
                return True

            return False

        except Exception as e:
            logger.error(f"Error reflecting: {e}", exc_info=True)
            self.stats["errors"] += 1
            return False

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @property
    def available(self) -> bool:
        """Check if HoloLoom is available."""
        return HOLOLOOM_AVAILABLE

    @property
    def department_available(self) -> bool:
        """Check if DreamDepartment is available."""
        return DREAM_DEPARTMENT_AVAILABLE and self._department is not None

    @property
    def spinner_available(self) -> bool:
        """Check if DreamSpinner is available."""
        return DREAM_SPINNER_AVAILABLE and self._spinner is not None

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        uptime = None
        if self.stats["start_time"]:
            uptime = (datetime.now() - self.stats["start_time"]).total_seconds()

        return {
            **self.stats,
            "uptime_seconds": uptime,
            "hololoom_available": self.available,
            "department_available": self.department_available,
            "spinner_available": self.spinner_available,
            "memory_bridge_available": self._memory_bridge.available if self._memory_bridge else False
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from underlying systems."""
        metrics = {
            "bridge": self.get_stats()
        }

        if self._department:
            metrics["department"] = self._department.get_stats()

        if self._memory_bridge:
            metrics["memory_bridge"] = self._memory_bridge.get_metrics()

        return metrics


# =============================================================================
# Convenience Functions
# =============================================================================

def is_phase2_available() -> bool:
    """Check if Phase 2 components are available."""
    return DREAM_DEPARTMENT_AVAILABLE and DREAM_SPINNER_AVAILABLE


async def quick_store(
    narrative: Any,
    resident_id: str,
    verify: bool = True
) -> StoreResult:
    """
    Quick helper to store a dream.

    Args:
        narrative: DreamNarrative object
        resident_id: Resident identifier
        verify: Run DS-STAR verification

    Returns:
        StoreResult
    """
    async with HoloLoomBridge(enable_verification=verify) as bridge:
        return await bridge.store_dream(narrative, resident_id)


async def quick_query(
    query: str,
    limit: int = 10
) -> QueryResult:
    """
    Quick helper to query dreams.

    Args:
        query: Search query
        limit: Maximum results

    Returns:
        QueryResult
    """
    async with HoloLoomBridge() as bridge:
        return await bridge.query_dreams(query, limit=limit)


async def quick_analyze(narrative: Any) -> AnalysisResult:
    """
    Quick helper to analyze a dream.

    Args:
        narrative: DreamNarrative object

    Returns:
        AnalysisResult
    """
    async with HoloLoomBridge() as bridge:
        return await bridge.analyze_dream(narrative)
