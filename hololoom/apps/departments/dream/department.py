"""
Dream Department - Phase 2 NeuroHood-HoloLoom Integration
=========================================================

Production-grade department for dream processing with DS-STAR verification.

Provides:
- Dream narrative storage with full metadata
- Semantic dream querying (symbol, emotion, archetype)
- Dream analysis and interpretation
- DS-STAR verification for dream content quality

Usage:
    dept = DreamDepartment()
    await dept.start()

    result = await dept.process({
        "action": "store",
        "narrative": narrative,
        "resident_id": "r001"
    })

    await dept.stop()

Author: Phase 2 Integration
Date: November 2025
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..protocol import DepartmentProtocol

# Graceful degradation - works without NeuroHood
try:
    from NeuroHood.dreams.narrative_generator import DreamNarrative, NarrativeAct
    from NeuroHood.dreams.hololoom_adapter import (
        store_dream_narrative,
        recall_dreams,
        reflect_on_dream,
        is_hololoom_available,
        DreamMemoryBridge
    )
    NEUROHOOD_AVAILABLE = True
except ImportError:
    NEUROHOOD_AVAILABLE = False
    DreamNarrative = None
    NarrativeAct = None

# Try to import DreamSpinner
try:
    from hololoom.spinningWheel.dream_spinner import DreamSpinner
    DREAM_SPINNER_AVAILABLE = True
except ImportError:
    DREAM_SPINNER_AVAILABLE = False
    DreamSpinner = None


logger = logging.getLogger(__name__)


# =============================================================================
# DS-STAR Verification Types
# =============================================================================

class DSStarCheck(Enum):
    """DS-STAR verification dimensions."""
    DOMAIN = "domain"           # Dream-domain validity
    SENSIBILITY = "sensibility" # Semantic coherence
    TEMPORAL = "temporal"       # Temporal consistency
    ARGUMENT = "argument"       # Logical structure
    REFERENCE = "reference"     # Symbol/archetype references


@dataclass
class VerificationResult:
    """Result of DS-STAR verification."""
    passed: bool
    checks: Dict[DSStarCheck, bool] = field(default_factory=dict)
    scores: Dict[DSStarCheck, float] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def failed_checks(self) -> List[DSStarCheck]:
        return [check for check, passed in self.checks.items() if not passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": {k.value: v for k, v in self.checks.items()},
            "scores": {k.value: v for k, v in self.scores.items()},
            "messages": self.messages,
            "confidence": self.confidence,
            "failed_checks": [c.value for c in self.failed_checks]
        }


class DreamAction(Enum):
    """Available dream department actions."""
    STORE = "store"           # Store dream narrative
    QUERY = "query"           # Query dreams semantically
    ANALYZE = "analyze"       # Analyze dream for insights
    REFLECT = "reflect"       # Reflect on dream quality
    VERIFY = "verify"         # DS-STAR verification only
    BATCH_STORE = "batch_store"  # Store multiple dreams
    SPIN = "spin"             # Convert to MemoryShards


@dataclass
class DreamRequest:
    """Request to Dream Department."""
    action: DreamAction
    narrative: Optional[Any] = None  # DreamNarrative
    narratives: Optional[List[Any]] = None  # For batch_store
    resident_id: Optional[str] = None
    query: Optional[str] = None
    limit: int = 10
    feedback: Optional[Dict[str, Any]] = None
    dream_memories: Optional[List[Any]] = None
    verify_strict: bool = False


@dataclass
class DreamResponse:
    """Response from Dream Department."""
    success: bool
    action: DreamAction
    result: Optional[Any] = None
    verification: Optional[VerificationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# =============================================================================
# DS-STAR Verifier
# =============================================================================

class DreamDSStarVerifier:
    """
    DS-STAR verification for dream content.

    Checks:
    - Domain: Is this valid dream content?
    - Sensibility: Does the narrative make semantic sense?
    - Temporal: Is the temporal structure consistent?
    - Argument: Is there logical narrative progression?
    - Reference: Are symbols and archetypes properly referenced?
    """

    # Known archetypes
    VALID_ARCHETYPES = {
        "journey", "confrontation", "transformation", "hero",
        "shadow", "trickster", "mentor", "threshold", "rebirth"
    }

    # Known moods
    VALID_MOODS = {
        "nightmare", "wonder", "mysterious", "transformative",
        "peaceful", "anxious", "ecstatic", "melancholic", "lucid"
    }

    # Jungian symbols
    JUNGIAN_SYMBOLS = {
        "shadow", "anima", "animus", "self", "persona", "water",
        "fire", "earth", "air", "tree", "snake", "bird", "mother",
        "father", "child", "ocean", "mountain", "cave", "mirror",
        "death", "rebirth", "transformation", "threshold", "crossing"
    }

    def __init__(self, strict: bool = False):
        """
        Initialize verifier.

        Args:
            strict: If True, require all checks to pass
        """
        self.strict = strict

    def verify(self, narrative: Any) -> VerificationResult:
        """
        Run DS-STAR verification on dream narrative.

        Args:
            narrative: DreamNarrative object

        Returns:
            VerificationResult with all check outcomes
        """
        checks = {}
        scores = {}
        messages = []

        # Domain check
        domain_score, domain_msgs = self._check_domain(narrative)
        checks[DSStarCheck.DOMAIN] = domain_score >= 0.5
        scores[DSStarCheck.DOMAIN] = domain_score
        messages.extend(domain_msgs)

        # Sensibility check
        sense_score, sense_msgs = self._check_sensibility(narrative)
        checks[DSStarCheck.SENSIBILITY] = sense_score >= 0.5
        scores[DSStarCheck.SENSIBILITY] = sense_score
        messages.extend(sense_msgs)

        # Temporal check
        temp_score, temp_msgs = self._check_temporal(narrative)
        checks[DSStarCheck.TEMPORAL] = temp_score >= 0.5
        scores[DSStarCheck.TEMPORAL] = temp_score
        messages.extend(temp_msgs)

        # Argument check
        arg_score, arg_msgs = self._check_argument(narrative)
        checks[DSStarCheck.ARGUMENT] = arg_score >= 0.5
        scores[DSStarCheck.ARGUMENT] = arg_score
        messages.extend(arg_msgs)

        # Reference check
        ref_score, ref_msgs = self._check_reference(narrative)
        checks[DSStarCheck.REFERENCE] = ref_score >= 0.5
        scores[DSStarCheck.REFERENCE] = ref_score
        messages.extend(ref_msgs)

        # Calculate overall pass/fail
        if self.strict:
            passed = all(checks.values())
        else:
            # Pass if at least 3/5 checks pass
            passed = sum(checks.values()) >= 3

        # Weighted confidence score
        weights = {
            DSStarCheck.DOMAIN: 0.25,
            DSStarCheck.SENSIBILITY: 0.2,
            DSStarCheck.TEMPORAL: 0.15,
            DSStarCheck.ARGUMENT: 0.2,
            DSStarCheck.REFERENCE: 0.2
        }
        confidence = sum(scores[k] * weights[k] for k in scores)

        return VerificationResult(
            passed=passed,
            checks=checks,
            scores=scores,
            messages=messages,
            confidence=confidence
        )

    def _check_domain(self, narrative: Any) -> tuple[float, List[str]]:
        """Check domain validity - is this dream content?"""
        messages = []
        score = 0.0

        # Check has title
        if hasattr(narrative, 'title') and narrative.title:
            score += 0.2
        else:
            messages.append("Domain: Missing dream title")

        # Check has archetype
        if hasattr(narrative, 'archetype'):
            archetype = narrative.archetype
            if hasattr(archetype, 'value'):
                archetype = archetype.value
            if str(archetype).lower() in self.VALID_ARCHETYPES:
                score += 0.25
            else:
                messages.append(f"Domain: Unknown archetype '{archetype}'")
                score += 0.1  # Partial credit
        else:
            messages.append("Domain: Missing archetype")

        # Check has mood
        if hasattr(narrative, 'mood'):
            mood = narrative.mood
            if hasattr(mood, 'value'):
                mood = mood.value
            if str(mood).lower() in self.VALID_MOODS:
                score += 0.25
            else:
                messages.append(f"Domain: Unknown mood '{mood}'")
                score += 0.1
        else:
            messages.append("Domain: Missing mood")

        # Check has acts (dream structure)
        if hasattr(narrative, 'acts') and narrative.acts:
            if len(narrative.acts) >= 1:
                score += 0.15
            if len(narrative.acts) >= 3:
                score += 0.15
        else:
            messages.append("Domain: No narrative acts")

        return min(score, 1.0), messages

    def _check_sensibility(self, narrative: Any) -> tuple[float, List[str]]:
        """Check semantic sensibility - does it make sense?"""
        messages = []
        score = 0.0

        # Check acts have descriptions
        if hasattr(narrative, 'acts') and narrative.acts:
            valid_descriptions = 0
            for act in narrative.acts:
                if hasattr(act, 'description') and act.description:
                    if len(act.description) >= 20:  # Minimum length
                        valid_descriptions += 1

            if valid_descriptions > 0:
                score += 0.4 * (valid_descriptions / len(narrative.acts))

        # Check emotional intensity is reasonable (0-1 range)
        if hasattr(narrative, 'acts') and narrative.acts:
            valid_intensities = 0
            for act in narrative.acts:
                if hasattr(act, 'emotional_intensity'):
                    intensity = act.emotional_intensity
                    if 0 <= intensity <= 1:
                        valid_intensities += 1

            if valid_intensities > 0:
                score += 0.3 * (valid_intensities / len(narrative.acts))

        # Check protagonist state values are reasonable
        if hasattr(narrative, 'protagonist_state') and narrative.protagonist_state:
            valid_states = sum(
                1 for v in narrative.protagonist_state.values()
                if isinstance(v, (int, float)) and 0 <= v <= 1
            )
            if valid_states > 0:
                score += 0.3 * min(valid_states / 3, 1.0)  # Expect at least 3 states

        if score < 0.5:
            messages.append("Sensibility: Dream content lacks semantic coherence")

        return min(score, 1.0), messages

    def _check_temporal(self, narrative: Any) -> tuple[float, List[str]]:
        """Check temporal consistency."""
        messages = []
        score = 0.0

        # Check acts are numbered sequentially
        if hasattr(narrative, 'acts') and narrative.acts:
            act_numbers = []
            for act in narrative.acts:
                if hasattr(act, 'act_number'):
                    act_numbers.append(act.act_number)

            if act_numbers:
                # Check sequential
                expected = list(range(1, len(act_numbers) + 1))
                if sorted(act_numbers) == expected:
                    score += 0.5
                else:
                    messages.append("Temporal: Act numbers not sequential")
                    score += 0.2

        # Check has timestamp
        if hasattr(narrative, 'generation_timestamp'):
            if narrative.generation_timestamp:
                score += 0.3

        # Check act types form reasonable progression
        if hasattr(narrative, 'acts') and narrative.acts:
            act_types = []
            for act in narrative.acts:
                if hasattr(act, 'act_type'):
                    act_types.append(act.act_type)

            # Classic three-act structure: setup → climax → resolution
            if len(act_types) >= 3:
                has_setup = any('setup' in t.lower() for t in act_types[:2] if t)
                has_climax = any('climax' in t.lower() for t in act_types if t)
                has_resolution = any('resolution' in t.lower() for t in act_types[-2:] if t)

                structure_score = (has_setup + has_climax + has_resolution) / 3
                score += 0.2 * structure_score

        if score < 0.5:
            messages.append("Temporal: Weak temporal structure")

        return min(score, 1.0), messages

    def _check_argument(self, narrative: Any) -> tuple[float, List[str]]:
        """Check logical argument structure."""
        messages = []
        score = 0.0

        # Check has interpretation (logical conclusion)
        if hasattr(narrative, 'jungian_interpretation') and narrative.jungian_interpretation:
            if len(narrative.jungian_interpretation) >= 30:
                score += 0.4
            else:
                score += 0.2
        else:
            messages.append("Argument: Missing Jungian interpretation")

        # Check emotional arc (intensity should vary)
        if hasattr(narrative, 'acts') and len(narrative.acts) >= 2:
            intensities = []
            for act in narrative.acts:
                if hasattr(act, 'emotional_intensity'):
                    intensities.append(act.emotional_intensity)

            if intensities:
                # Expect some variation in intensity
                intensity_range = max(intensities) - min(intensities)
                if intensity_range >= 0.2:
                    score += 0.3
                else:
                    messages.append("Argument: Flat emotional arc")
                    score += 0.1

        # Check symbols support interpretation
        if hasattr(narrative, 'symbols_used') and narrative.symbols_used:
            if len(narrative.symbols_used) >= 2:
                score += 0.3
        else:
            messages.append("Argument: No supporting symbols")

        return min(score, 1.0), messages

    def _check_reference(self, narrative: Any) -> tuple[float, List[str]]:
        """Check symbol and archetype references."""
        messages = []
        score = 0.0

        # Check symbols have Jungian validity
        if hasattr(narrative, 'symbols_used') and narrative.symbols_used:
            jungian_count = sum(
                1 for s in narrative.symbols_used
                if s.lower() in self.JUNGIAN_SYMBOLS
            )
            ratio = jungian_count / len(narrative.symbols_used)
            score += 0.4 * ratio

            if ratio < 0.5:
                messages.append(f"Reference: Only {jungian_count}/{len(narrative.symbols_used)} recognized Jungian symbols")

        # Check literary inspirations
        if hasattr(narrative, 'literary_inspirations') and narrative.literary_inspirations:
            score += 0.3 * min(len(narrative.literary_inspirations) / 2, 1.0)
        else:
            messages.append("Reference: No literary inspirations")

        # Check symbols in acts
        if hasattr(narrative, 'acts') and narrative.acts:
            acts_with_symbols = 0
            for act in narrative.acts:
                if hasattr(act, 'symbols') and act.symbols:
                    acts_with_symbols += 1

            if narrative.acts:
                score += 0.3 * (acts_with_symbols / len(narrative.acts))

        return min(score, 1.0), messages


# =============================================================================
# Dream Department
# =============================================================================

class DreamDepartment(DepartmentProtocol):
    """
    Dream Department - NeuroHood-HoloLoom Bridge

    Provides production-grade dream processing with DS-STAR verification.

    Features:
    - Store dream narratives with full metadata
    - Query dreams semantically
    - Analyze dreams for Jungian insights
    - DS-STAR verification for quality assurance
    - Integration with DreamSpinner for MemoryShard conversion

    Usage:
        dept = DreamDepartment()
        await dept.start()

        result = await dept.process({
            "action": "store",
            "narrative": narrative,
            "resident_id": "r001"
        })

        await dept.stop()
    """

    def __init__(
        self,
        verify_on_store: bool = True,
        strict_verification: bool = False,
        enable_spinner: bool = True
    ):
        """
        Initialize Dream Department.

        Args:
            verify_on_store: Run DS-STAR verification before storing
            strict_verification: Require all DS-STAR checks to pass
            enable_spinner: Enable DreamSpinner for MemoryShard conversion
        """
        self.verify_on_store = verify_on_store
        self.strict_verification = strict_verification
        self.enable_spinner = enable_spinner and DREAM_SPINNER_AVAILABLE

        self.verifier = DreamDSStarVerifier(strict=strict_verification)
        self.spinner: Optional[Any] = None
        self.bridge: Optional[Any] = None

        self.started = False

        # Statistics
        self.stats = {
            "dreams_stored": 0,
            "dreams_queried": 0,
            "verifications_passed": 0,
            "verifications_failed": 0,
            "errors": 0,
            "start_time": None
        }

    async def start(self):
        """Start Dream Department."""
        if self.started:
            logger.warning("Dream Department already started")
            return

        logger.info("Starting Dream Department")

        # Initialize DreamSpinner if enabled
        if self.enable_spinner and DREAM_SPINNER_AVAILABLE:
            from hololoom.spinningWheel.dream_spinner import DreamSpinner, DreamSpinnerConfig
            self.spinner = DreamSpinner(DreamSpinnerConfig())
            logger.info("DreamSpinner initialized")

        # Initialize DreamMemoryBridge if available
        if NEUROHOOD_AVAILABLE:
            self.bridge = DreamMemoryBridge()
            await self.bridge.__aenter__()
            logger.info(f"DreamMemoryBridge initialized (HoloLoom available: {self.bridge.available})")

        self.started = True
        self.stats["start_time"] = datetime.now()
        logger.info("Dream Department started successfully")

    async def stop(self):
        """Stop Dream Department."""
        if not self.started:
            return

        logger.info("Stopping Dream Department")

        # Clean up bridge
        if self.bridge:
            await self.bridge.__aexit__(None, None, None)
            self.bridge = None

        self.started = False
        logger.info("Dream Department stopped")

    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Dream Department request.

        Args:
            request: Request dictionary with:
                - action: DreamAction (store, query, analyze, reflect, verify)
                - narrative: DreamNarrative object (for store, analyze, verify)
                - query: Search query (for query action)
                - resident_id: Resident identifier
                - feedback: Feedback dict (for reflect action)
                - limit: Result limit (for query)

        Returns:
            Response dictionary with:
                - success: bool
                - action: Action performed
                - result: Action-specific result
                - verification: DS-STAR verification result (if applicable)
                - metadata: Additional metadata
                - error: Error message (if failed)
        """
        if not self.started:
            return {
                "success": False,
                "error": "Department not started. Call start() first."
            }

        # Parse request
        try:
            dream_request = self._parse_request(request)
        except ValueError as e:
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": f"Invalid request: {str(e)}"
            }

        # Process action
        try:
            start_time = datetime.now()
            response = await self._process_action(dream_request)
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Add metadata
            if response.metadata is None:
                response.metadata = {}
            response.metadata["department"] = "dream"
            response.metadata["duration_ms"] = duration_ms
            response.metadata["neurohood_available"] = NEUROHOOD_AVAILABLE
            response.metadata["spinner_available"] = DREAM_SPINNER_AVAILABLE

            return self._response_to_dict(response)

        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": f"Processing error: {str(e)}"
            }

    async def health_check(self) -> bool:
        """Check department health."""
        if not self.started:
            return False

        # Check bridge health
        if self.bridge and hasattr(self.bridge, 'available'):
            if not self.bridge.available:
                logger.warning("DreamMemoryBridge not available")
                return True  # Graceful degradation - still healthy

        return True

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _parse_request(self, request: Dict[str, Any]) -> DreamRequest:
        """Parse raw request into DreamRequest."""
        action_str = request.get("action")
        if not action_str:
            raise ValueError("Missing 'action' field")

        try:
            action = DreamAction(action_str)
        except ValueError:
            raise ValueError(f"Invalid action: {action_str}. Valid: {[a.value for a in DreamAction]}")

        return DreamRequest(
            action=action,
            narrative=request.get("narrative"),
            narratives=request.get("narratives"),
            resident_id=request.get("resident_id"),
            query=request.get("query"),
            limit=request.get("limit", 10),
            feedback=request.get("feedback"),
            dream_memories=request.get("dream_memories"),
            verify_strict=request.get("verify_strict", self.strict_verification)
        )

    async def _process_action(self, request: DreamRequest) -> DreamResponse:
        """Process dream action."""
        action_handlers = {
            DreamAction.STORE: self._handle_store,
            DreamAction.QUERY: self._handle_query,
            DreamAction.ANALYZE: self._handle_analyze,
            DreamAction.REFLECT: self._handle_reflect,
            DreamAction.VERIFY: self._handle_verify,
            DreamAction.BATCH_STORE: self._handle_batch_store,
            DreamAction.SPIN: self._handle_spin
        }

        handler = action_handlers.get(request.action)
        if not handler:
            return DreamResponse(
                success=False,
                action=request.action,
                error=f"Unknown action: {request.action}"
            )

        return await handler(request)

    async def _handle_store(self, request: DreamRequest) -> DreamResponse:
        """Handle store action."""
        if not request.narrative:
            return DreamResponse(
                success=False,
                action=DreamAction.STORE,
                error="Missing 'narrative' for store action"
            )

        if not request.resident_id:
            return DreamResponse(
                success=False,
                action=DreamAction.STORE,
                error="Missing 'resident_id' for store action"
            )

        # Run DS-STAR verification if enabled
        verification = None
        if self.verify_on_store:
            verification = self.verifier.verify(request.narrative)

            if not verification.passed:
                self.stats["verifications_failed"] += 1
                return DreamResponse(
                    success=False,
                    action=DreamAction.STORE,
                    verification=verification,
                    error=f"DS-STAR verification failed: {verification.failed_checks}"
                )
            else:
                self.stats["verifications_passed"] += 1

        # Store via bridge
        if self.bridge and self.bridge.available:
            memory = await self.bridge.store(request.narrative, request.resident_id)
            self.stats["dreams_stored"] += 1

            return DreamResponse(
                success=True,
                action=DreamAction.STORE,
                result={
                    "memory_id": memory.id if memory else None,
                    "stored": memory is not None
                },
                verification=verification,
                metadata={
                    "resident_id": request.resident_id,
                    "archetype": getattr(request.narrative.archetype, 'value', str(request.narrative.archetype)) if hasattr(request.narrative, 'archetype') else None
                }
            )
        else:
            # Graceful degradation - return success but note HoloLoom unavailable
            return DreamResponse(
                success=True,
                action=DreamAction.STORE,
                result={
                    "memory_id": None,
                    "stored": False,
                    "reason": "HoloLoom not available"
                },
                verification=verification,
                metadata={
                    "resident_id": request.resident_id,
                    "hololoom_available": False
                }
            )

    async def _handle_query(self, request: DreamRequest) -> DreamResponse:
        """Handle query action."""
        if not request.query:
            return DreamResponse(
                success=False,
                action=DreamAction.QUERY,
                error="Missing 'query' for query action"
            )

        if self.bridge and self.bridge.available:
            memories = await self.bridge.recall(
                request.query,
                resident_id=request.resident_id,
                limit=request.limit
            )
            self.stats["dreams_queried"] += 1

            return DreamResponse(
                success=True,
                action=DreamAction.QUERY,
                result={
                    "count": len(memories),
                    "dreams": [
                        {
                            "id": m.id if hasattr(m, 'id') else None,
                            "text": m.text[:200] + "..." if len(m.text) > 200 else m.text,
                            "archetype": m.context.get("archetype") if hasattr(m, 'context') else None,
                            "mood": m.context.get("mood") if hasattr(m, 'context') else None
                        }
                        for m in memories
                    ]
                },
                metadata={
                    "query": request.query,
                    "resident_id": request.resident_id,
                    "limit": request.limit
                }
            )
        else:
            return DreamResponse(
                success=True,
                action=DreamAction.QUERY,
                result={"count": 0, "dreams": [], "reason": "HoloLoom not available"},
                metadata={"hololoom_available": False}
            )

    async def _handle_analyze(self, request: DreamRequest) -> DreamResponse:
        """Handle analyze action - provide dream insights."""
        if not request.narrative:
            return DreamResponse(
                success=False,
                action=DreamAction.ANALYZE,
                error="Missing 'narrative' for analyze action"
            )

        narrative = request.narrative

        # Extract analysis
        analysis = {
            "title": getattr(narrative, 'title', 'Untitled'),
            "archetype": getattr(narrative.archetype, 'value', str(narrative.archetype)) if hasattr(narrative, 'archetype') else None,
            "mood": getattr(narrative.mood, 'value', str(narrative.mood)) if hasattr(narrative, 'mood') else None,
            "jungian_interpretation": getattr(narrative, 'jungian_interpretation', None),
            "symbols": getattr(narrative, 'symbols_used', []),
            "literary_inspirations": getattr(narrative, 'literary_inspirations', []),
            "emotional_profile": {}
        }

        # Calculate emotional profile
        if hasattr(narrative, 'acts') and narrative.acts:
            intensities = []
            for act in narrative.acts:
                if hasattr(act, 'emotional_intensity'):
                    intensities.append(act.emotional_intensity)

            if intensities:
                analysis["emotional_profile"] = {
                    "min_intensity": min(intensities),
                    "max_intensity": max(intensities),
                    "avg_intensity": sum(intensities) / len(intensities),
                    "intensity_range": max(intensities) - min(intensities),
                    "act_count": len(narrative.acts)
                }

        # Add protagonist state
        if hasattr(narrative, 'protagonist_state') and narrative.protagonist_state:
            analysis["protagonist_state"] = narrative.protagonist_state

        # Run DS-STAR verification for quality assessment
        verification = self.verifier.verify(narrative)

        return DreamResponse(
            success=True,
            action=DreamAction.ANALYZE,
            result=analysis,
            verification=verification,
            metadata={
                "ds_star_confidence": verification.confidence
            }
        )

    async def _handle_reflect(self, request: DreamRequest) -> DreamResponse:
        """Handle reflect action - provide feedback to learning system."""
        if not request.feedback:
            return DreamResponse(
                success=False,
                action=DreamAction.REFLECT,
                error="Missing 'feedback' for reflect action"
            )

        if self.bridge and self.bridge.available:
            dreams = request.dream_memories or []
            await self.bridge.reflect(dreams, request.feedback)

            return DreamResponse(
                success=True,
                action=DreamAction.REFLECT,
                result={
                    "reflected": True,
                    "dreams_count": len(dreams)
                },
                metadata={"feedback": request.feedback}
            )
        else:
            return DreamResponse(
                success=True,
                action=DreamAction.REFLECT,
                result={"reflected": False, "reason": "HoloLoom not available"},
                metadata={"hololoom_available": False}
            )

    async def _handle_verify(self, request: DreamRequest) -> DreamResponse:
        """Handle verify action - DS-STAR verification only."""
        if not request.narrative:
            return DreamResponse(
                success=False,
                action=DreamAction.VERIFY,
                error="Missing 'narrative' for verify action"
            )

        # Create verifier with request's strict setting
        verifier = DreamDSStarVerifier(strict=request.verify_strict)
        verification = verifier.verify(request.narrative)

        if verification.passed:
            self.stats["verifications_passed"] += 1
        else:
            self.stats["verifications_failed"] += 1

        return DreamResponse(
            success=True,
            action=DreamAction.VERIFY,
            result={"passed": verification.passed},
            verification=verification,
            metadata={
                "strict_mode": request.verify_strict
            }
        )

    async def _handle_batch_store(self, request: DreamRequest) -> DreamResponse:
        """Handle batch store action."""
        if not request.narratives:
            return DreamResponse(
                success=False,
                action=DreamAction.BATCH_STORE,
                error="Missing 'narratives' for batch_store action"
            )

        if not request.resident_id:
            return DreamResponse(
                success=False,
                action=DreamAction.BATCH_STORE,
                error="Missing 'resident_id' for batch_store action"
            )

        results = []
        stored_count = 0
        failed_count = 0

        for narrative in request.narratives:
            # Verify if enabled
            verification = None
            if self.verify_on_store:
                verification = self.verifier.verify(narrative)
                if not verification.passed:
                    self.stats["verifications_failed"] += 1
                    results.append({
                        "stored": False,
                        "reason": "DS-STAR verification failed",
                        "verification": verification.to_dict()
                    })
                    failed_count += 1
                    continue
                else:
                    self.stats["verifications_passed"] += 1

            # Store
            if self.bridge and self.bridge.available:
                memory = await self.bridge.store(narrative, request.resident_id)
                results.append({
                    "stored": memory is not None,
                    "memory_id": memory.id if memory else None,
                    "verification": verification.to_dict() if verification else None
                })
                if memory:
                    stored_count += 1
                    self.stats["dreams_stored"] += 1
            else:
                results.append({
                    "stored": False,
                    "reason": "HoloLoom not available"
                })

        return DreamResponse(
            success=True,
            action=DreamAction.BATCH_STORE,
            result={
                "total": len(request.narratives),
                "stored": stored_count,
                "failed": failed_count,
                "details": results
            },
            metadata={
                "resident_id": request.resident_id
            }
        )

    async def _handle_spin(self, request: DreamRequest) -> DreamResponse:
        """Handle spin action - convert to MemoryShards."""
        if not self.spinner:
            return DreamResponse(
                success=False,
                action=DreamAction.SPIN,
                error="DreamSpinner not available"
            )

        if not request.narrative and not request.narratives:
            return DreamResponse(
                success=False,
                action=DreamAction.SPIN,
                error="Missing 'narrative' or 'narratives' for spin action"
            )

        # Spin single or batch
        narratives = request.narratives or [request.narrative]

        shards = []
        for narrative in narratives:
            result = await self.spinner.spin(narrative, resident_id=request.resident_id)
            if result.shards:
                shards.extend(result.shards)

        return DreamResponse(
            success=True,
            action=DreamAction.SPIN,
            result={
                "shard_count": len(shards),
                "shards": [
                    {
                        "id": s.id,
                        "text": s.text[:100] + "..." if len(s.text) > 100 else s.text,
                        "entities": s.entities[:5],
                        "motifs": s.motifs[:5],
                        "episode": s.episode
                    }
                    for s in shards[:10]  # Limit output
                ]
            },
            metadata={
                "narratives_processed": len(narratives)
            }
        )

    def _response_to_dict(self, response: DreamResponse) -> Dict[str, Any]:
        """Convert response to dictionary."""
        result = {
            "success": response.success,
            "action": response.action.value
        }

        if response.result is not None:
            result["result"] = response.result

        if response.verification:
            result["verification"] = response.verification.to_dict()

        if response.metadata:
            result["metadata"] = response.metadata

        if response.error:
            result["error"] = response.error

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get department statistics."""
        uptime = None
        if self.stats["start_time"]:
            uptime = (datetime.now() - self.stats["start_time"]).total_seconds()

        return {
            **self.stats,
            "uptime_seconds": uptime,
            "bridge_available": self.bridge.available if self.bridge else False,
            "spinner_available": self.spinner is not None,
            "neurohood_available": NEUROHOOD_AVAILABLE
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_dream_department(
    verify_on_store: bool = True,
    strict_verification: bool = False,
    enable_spinner: bool = True
) -> DreamDepartment:
    """
    Factory function to create Dream Department.

    Args:
        verify_on_store: Run DS-STAR verification before storing
        strict_verification: Require all DS-STAR checks to pass
        enable_spinner: Enable DreamSpinner for MemoryShard conversion

    Returns:
        Configured DreamDepartment instance
    """
    return DreamDepartment(
        verify_on_store=verify_on_store,
        strict_verification=strict_verification,
        enable_spinner=enable_spinner
    )


async def create_and_start_dream_department(**kwargs) -> DreamDepartment:
    """Create and start Dream Department."""
    dept = create_dream_department(**kwargs)
    await dept.start()
    return dept
