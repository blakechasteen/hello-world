"""
HoloLoom Agentic Intelligence Core
===================================
Self-directed reasoning with verification loops.

Extends HoloLoom's recursive learning system with autonomous query generation,
multi-step reasoning, and verification protocols.

Philosophy:
"Don't just answer - understand, verify, refine."

Integration Points:
- Builds on recursive.FullLearningEngine (Phase 5)
- Uses alignment.audit_trail for decision logging
- Leverages ReflectionBuffer for learning
- Extends action_items for goal tracking
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime

from HoloLoom.protocols.types import Query, Context, MemoryShard
from HoloLoom.fabric.spacetime import Spacetime
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.recursive import FullLearningEngine, ActionItemTracker, ActionStatus
from HoloLoom.alignment.audit_trail import AuditTrail, DecisionType, OutcomeType


logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

class ReasoningMode(Enum):
    """Agentic reasoning modes."""
    DIRECT = "direct"              # Single-pass answer
    VERIFY = "verify"              # Answer + verification loop
    RESEARCH = "research"          # Multi-query exploration
    PLAN_EXECUTE = "plan_execute"  # Goal decomposition + execution


@dataclass
class AgenticIntent:
    """
    User intent with goal tracking.

    Extends action items with research/verification goals.
    """
    intent_id: str
    original_query: str
    goal: str
    sub_goals: List[str] = field(default_factory=list)
    evidence_gathered: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.85
    max_verification_loops: int = 3

    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    status: ActionStatus = ActionStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of verification loop."""
    verified: bool
    confidence: float
    contradictions: List[str]
    supporting_evidence: List[str]
    suggested_refinements: List[str]

    # Provenance
    verification_queries: List[str]
    sources_checked: List[str]


@dataclass
class AgenticResult:
    """Complete agentic reasoning result."""
    spacetime: Spacetime
    intent: AgenticIntent
    reasoning_mode: ReasoningMode
    verification: Optional[VerificationResult] = None

    # Multi-step tracking
    steps_taken: List[Dict[str, Any]] = field(default_factory=list)
    total_queries: int = 0
    total_duration_ms: float = 0.0

    # Consciousness Integration - Phase 1 (Nov 2025)
    # Epistemic confidence aggregated across all reasoning steps
    # Tracks system's self-awareness of knowledge gaps throughout multi-query reasoning
    aggregated_epistemic_confidence: Optional[float] = None


# ============================================================================
# Agentic Orchestrator
# ============================================================================

class AgenticOrchestrator:
    """
    Autonomous reasoning orchestrator.

    Wraps WeavingOrchestrator with self-directed reasoning capabilities:
    - Generates verification queries
    - Decomposes complex goals
    - Tracks multi-step reasoning
    - Learns from verification outcomes

    Usage:
        async with AgenticOrchestrator(config, shards) as agent:
            result = await agent.reason(
                query="Is Thompson Sampling optimal for this task?",
                mode=ReasoningMode.VERIFY
            )
    """

    def __init__(
        self,
        learning_engine: FullLearningEngine,
        audit_trail: Optional[AuditTrail] = None,
        enable_verification: bool = True,
        enable_goal_tracking: bool = True,
        llm: Optional[Any] = None,  # LLM for intelligent query generation
        awareness_layer: Optional[Any] = None,  # Consciousness Integration - Phase 1 (Nov 2025)
        epistemic_threshold: float = 0.3  # Early stopping threshold for epistemic confidence
    ):
        self.learning_engine = learning_engine
        self.audit_trail = audit_trail or AuditTrail()
        self.enable_verification = enable_verification
        self.enable_goal_tracking = enable_goal_tracking
        self.llm = llm  # LLM for agentic search
        self.awareness_layer = awareness_layer  # Consciousness Integration - Phase 1
        self.epistemic_threshold = epistemic_threshold  # Early stopping threshold

        # Goal tracker (extends action items)
        self.goal_tracker = ActionItemTracker() if enable_goal_tracking else None

        self.logger = logging.getLogger(__name__)

        # Initialize LLM if not provided but available in orchestrator
        if self.llm is None and hasattr(learning_engine, 'orchestrator'):
            orchestrator = learning_engine.orchestrator
            if hasattr(orchestrator, 'tool_executor') and hasattr(orchestrator.tool_executor, 'llm'):
                self.llm = orchestrator.tool_executor.llm
                if self.llm:
                    self.logger.info("LLM-activated agentic search enabled")

        # Consciousness Integration - Phase 1 (Nov 2025)
        # Extract awareness_layer from learning_engine's orchestrator if not provided
        if self.awareness_layer is None and hasattr(learning_engine, 'orchestrator'):
            orchestrator = learning_engine.orchestrator
            if hasattr(orchestrator, 'awareness_layer'):
                self.awareness_layer = orchestrator.awareness_layer
                if self.awareness_layer:
                    self.logger.info("Awareness layer extracted from orchestrator for epistemic tracking")

    async def reason(
        self,
        query: Query,
        mode: ReasoningMode = ReasoningMode.DIRECT,
        confidence_threshold: float = 0.85,
        max_steps: int = 5
    ) -> AgenticResult:
        """
        Execute agentic reasoning with selected mode.

        Args:
            query: User query
            mode: Reasoning mode (DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE)
            confidence_threshold: Minimum confidence for accepting result
            max_steps: Maximum reasoning steps

        Returns:
            AgenticResult with complete reasoning trace
        """
        start_time = datetime.now()

        # Create intent
        intent = AgenticIntent(
            intent_id=f"intent_{int(start_time.timestamp())}",
            original_query=query.text,
            goal=self._extract_goal(query),
            confidence_threshold=confidence_threshold
        )

        # Log decision
        self.audit_trail.log_decision(
            decision_type=DecisionType.TOOL_SELECTION,
            outcome=OutcomeType.APPROVED,
            reason=f"Agentic reasoning mode: {mode.value}",
            query_text=query.text,
            metadata={"mode": mode.value, "intent_id": intent.intent_id}
        )

        # Route to appropriate handler
        if mode == ReasoningMode.DIRECT:
            result = await self._direct_answer(query, intent)
        elif mode == ReasoningMode.VERIFY:
            result = await self._verify_answer(query, intent, max_steps)
        elif mode == ReasoningMode.RESEARCH:
            result = await self._research_query(query, intent, max_steps)
        elif mode == ReasoningMode.PLAN_EXECUTE:
            result = await self._plan_and_execute(query, intent, max_steps)
        else:
            raise ValueError(f"Unknown reasoning mode: {mode}")

        # Calculate duration
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        result.total_duration_ms = duration_ms

        return result

    async def _direct_answer(
        self,
        query: Query,
        intent: AgenticIntent
    ) -> AgenticResult:
        """Direct answer without verification."""
        spacetime = await self.learning_engine.weave(query)

        # Consciousness Integration - Phase 1 (Nov 2025)
        # Extract epistemic confidence from spacetime
        epistemic_conf = self._extract_epistemic_confidence(spacetime)

        # Generate LLM answer if available
        if self.llm:
            try:
                # Extract context from retrieved memories
                context_text = ""
                if hasattr(spacetime, 'context_used') and spacetime.context_used:
                    context_text = "\n\n".join([str(ctx) for ctx in spacetime.context_used[:3]])
                elif hasattr(spacetime, 'memories') and spacetime.memories:
                    context_text = "\n\n".join([m.text for m in spacetime.memories[:3]])

                # Build prompt
                system_prompt = "You are a helpful AI assistant. Answer the question based on the provided context."

                if context_text:
                    user_prompt = f"""Context:
{context_text}

Question: {query.text}

Please provide a clear and concise answer based on the context above."""
                else:
                    user_prompt = f"Question: {query.text}\n\nPlease provide a helpful answer."

                # Generate answer
                self.logger.info(f"[AGENTIC] Generating LLM answer for: {query.text}")
                llm_response = await self.llm.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=500,
                    temperature=0.7
                )

                # Set response in spacetime
                spacetime.response = llm_response.content
                self.logger.info(f"[AGENTIC] LLM answer generated ({len(llm_response.content)} chars)")

            except Exception as e:
                self.logger.warning(f"[AGENTIC] LLM answer generation failed: {e}")
                spacetime.response = f"Retrieved relevant information but could not generate answer: {e}"
        else:
            self.logger.warning("[AGENTIC] No LLM available for answer generation")
            spacetime.response = "LLM unavailable - retrieved context but cannot generate natural language answer"

        return AgenticResult(
            spacetime=spacetime,
            intent=intent,
            reasoning_mode=ReasoningMode.DIRECT,
            total_queries=1,
            steps_taken=[{
                "type": "direct_answer",
                "query": query.text,
                "confidence": spacetime.confidence,
                "epistemic_confidence": epistemic_conf,  # Consciousness Integration - Phase 1
                "llm_generated": bool(self.llm)
            }],
            aggregated_epistemic_confidence=epistemic_conf  # Single step, so no aggregation needed
        )

    async def _verify_answer(
        self,
        query: Query,
        intent: AgenticIntent,
        max_loops: int
    ) -> AgenticResult:
        """Answer with verification loop."""
        steps = []
        epistemic_confidences = []  # Consciousness Integration - Phase 1 (Nov 2025)

        # Step 1: Initial answer
        self.logger.info(f"[AGENTIC] Initial answer: {query.text}")
        spacetime = await self.learning_engine.weave(query)

        # Consciousness Integration - Phase 1
        epistemic_conf = self._extract_epistemic_confidence(spacetime)
        if epistemic_conf is not None:
            epistemic_confidences.append(epistemic_conf)

        steps.append({
            "type": "initial_answer",
            "query": query.text,
            "confidence": spacetime.confidence,
            "epistemic_confidence": epistemic_conf,  # Consciousness Integration - Phase 1
            "tool_used": spacetime.metadata.get("tool_used")
        })

        # Step 2: Generate verification queries
        verification_queries = self._generate_verification_queries(query, spacetime)

        # Step 3: Execute verification loops
        verification = await self._run_verification_loops(
            original_query=query,
            initial_result=spacetime,
            verification_queries=verification_queries,
            max_loops=max_loops,
            steps=steps,
            epistemic_confidences=epistemic_confidences  # Consciousness Integration - Phase 1
        )

        # Step 4: Refine if needed
        if not verification.verified and verification.suggested_refinements:
            self.logger.info("[AGENTIC] Verification failed, refining...")
            refined_query = Query(text=verification.suggested_refinements[0])
            spacetime = await self.learning_engine.weave(
                refined_query,
                enable_refinement=True
            )

            # Consciousness Integration - Phase 1
            epistemic_conf = self._extract_epistemic_confidence(spacetime)
            if epistemic_conf is not None:
                epistemic_confidences.append(epistemic_conf)

            steps.append({
                "type": "refinement",
                "query": refined_query.text,
                "confidence": spacetime.confidence,
                "epistemic_confidence": epistemic_conf  # Consciousness Integration - Phase 1
            })

        # Consciousness Integration - Phase 1
        # Aggregate epistemic confidence across all steps
        aggregated_epistemic = self._aggregate_epistemic_confidence(epistemic_confidences)

        return AgenticResult(
            spacetime=spacetime,
            intent=intent,
            reasoning_mode=ReasoningMode.VERIFY,
            verification=verification,
            total_queries=len(steps),
            steps_taken=steps,
            aggregated_epistemic_confidence=aggregated_epistemic  # Consciousness Integration - Phase 1
        )

    async def _research_query(
        self,
        query: Query,
        intent: AgenticIntent,
        max_steps: int
    ) -> AgenticResult:
        """Multi-query exploration with LLM-activated intelligent search and epistemic tracking."""
        steps = []
        evidence = []
        initial_findings = None
        epistemic_confidences = []  # Consciousness Integration - Phase 1 (Nov 2025)

        # Step 1: Generate research questions (LLM-activated)
        research_queries = await self._generate_research_queries(
            query,
            max_queries=max_steps,
            initial_findings=initial_findings
        )

        # Step 2: Execute research queries with epistemic tracking
        for i, rq in enumerate(research_queries):
            # Consciousness Integration - Phase 1
            # Early stopping if epistemic confidence drops too low
            if self._should_stop_early(i, epistemic_confidences, max_steps):
                self.logger.warning(
                    f"[EPISTEMIC] Stopping research early at query {i+1} "
                    f"due to low epistemic confidence"
                )
                break

            self.logger.info(f"[AGENTIC] Research query {i+1}/{len(research_queries)}: {rq}")
            result = await self.learning_engine.weave(Query(text=rq))

            # Consciousness Integration - Phase 1
            epistemic_conf = self._extract_epistemic_confidence(result)
            if epistemic_conf is not None:
                epistemic_confidences.append(epistemic_conf)

            finding = result.response if hasattr(result, 'response') else str(result)
            evidence.append(finding)
            steps.append({
                "type": "research_query",
                "query": rq,
                "confidence": result.confidence,
                "epistemic_confidence": epistemic_conf,  # Consciousness Integration - Phase 1
                "findings": finding[:200]
            })

            # Update initial_findings for next iteration (adaptive exploration)
            if i == 0:
                initial_findings = finding[:500]  # Use first finding to guide subsequent queries

        # Step 3: Synthesize findings
        synthesis_query = self._create_synthesis_query(query, evidence)
        final_result = await self.learning_engine.weave(Query(text=synthesis_query))

        # Consciousness Integration - Phase 1
        epistemic_conf = self._extract_epistemic_confidence(final_result)
        if epistemic_conf is not None:
            epistemic_confidences.append(epistemic_conf)

        steps.append({
            "type": "synthesis",
            "query": synthesis_query,
            "confidence": final_result.confidence,
            "epistemic_confidence": epistemic_conf,  # Consciousness Integration - Phase 1
            "sources": len(evidence)
        })

        intent.evidence_gathered = evidence

        # Consciousness Integration - Phase 1
        # Aggregate epistemic confidence across all research steps
        aggregated_epistemic = self._aggregate_epistemic_confidence(epistemic_confidences)

        return AgenticResult(
            spacetime=final_result,
            intent=intent,
            reasoning_mode=ReasoningMode.RESEARCH,
            total_queries=len(steps),
            steps_taken=steps,
            aggregated_epistemic_confidence=aggregated_epistemic  # Consciousness Integration - Phase 1
        )

    async def _plan_and_execute(
        self,
        query: Query,
        intent: AgenticIntent,
        max_steps: int
    ) -> AgenticResult:
        """Goal decomposition and execution with epistemic tracking."""
        steps = []
        epistemic_confidences = []  # Consciousness Integration - Phase 1 (Nov 2025)

        # Step 1: Decompose goal into sub-goals
        sub_goals = self._decompose_goal(query)
        intent.sub_goals = sub_goals

        self.logger.info(f"[AGENTIC] Decomposed into {len(sub_goals)} sub-goals")

        # Step 2: Execute sub-goals with epistemic tracking
        results = []
        for i, sub_goal in enumerate(sub_goals[:max_steps]):
            # Consciousness Integration - Phase 1
            # Early stopping if epistemic confidence drops too low
            if self._should_stop_early(i, epistemic_confidences, max_steps):
                self.logger.warning(
                    f"[EPISTEMIC] Stopping sub-goal execution early at step {i+1} "
                    f"due to low epistemic confidence"
                )
                break

            self.logger.info(f"[AGENTIC] Executing sub-goal {i+1}: {sub_goal}")
            result = await self.learning_engine.weave(Query(text=sub_goal))
            results.append(result)

            # Consciousness Integration - Phase 1
            epistemic_conf = self._extract_epistemic_confidence(result)
            if epistemic_conf is not None:
                epistemic_confidences.append(epistemic_conf)

            steps.append({
                "type": "sub_goal",
                "goal": sub_goal,
                "confidence": result.confidence,
                "epistemic_confidence": epistemic_conf,  # Consciousness Integration - Phase 1
                "completed": result.confidence >= intent.confidence_threshold
            })

        # Step 3: Synthesize
        synthesis_query = f"Based on the following findings, {query.text}:\n"
        for i, r in enumerate(results):
            synthesis_query += f"\n{i+1}. {r.metadata.get('response', '')[:200]}"

        final_result = await self.learning_engine.weave(Query(text=synthesis_query))

        # Consciousness Integration - Phase 1
        epistemic_conf = self._extract_epistemic_confidence(final_result)
        if epistemic_conf is not None:
            epistemic_confidences.append(epistemic_conf)

        steps.append({
            "type": "synthesis",
            "confidence": final_result.confidence,
            "epistemic_confidence": epistemic_conf,  # Consciousness Integration - Phase 1
            "sub_goals_completed": sum(1 for s in steps if s.get("completed", False))
        })

        # Consciousness Integration - Phase 1
        # Aggregate epistemic confidence across all steps
        aggregated_epistemic = self._aggregate_epistemic_confidence(epistemic_confidences)

        return AgenticResult(
            spacetime=final_result,
            intent=intent,
            reasoning_mode=ReasoningMode.PLAN_EXECUTE,
            total_queries=len(steps),
            steps_taken=steps,
            aggregated_epistemic_confidence=aggregated_epistemic  # Consciousness Integration - Phase 1
        )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _extract_epistemic_confidence(self, spacetime: Spacetime) -> Optional[float]:
        """
        Extract epistemic confidence from spacetime metadata.

        Consciousness Integration - Phase 1 (Nov 2025)
        Epistemic confidence represents the system's self-awareness of knowledge gaps.

        Args:
            spacetime: Spacetime result from weaving

        Returns:
            Epistemic confidence (0.0-1.0) or None if unavailable
        """
        if not spacetime or not hasattr(spacetime, 'metadata'):
            return None

        awareness_metadata = spacetime.metadata.get('awareness', {})
        if not awareness_metadata:
            return None

        # Calculate epistemic confidence from awareness metrics
        # Similar to RAG integration: 70% coherence + 30% activation
        coherence = awareness_metadata.get('coherence', 0.5)
        activation = awareness_metadata.get('activation_level', 0.5)

        epistemic_confidence = (0.7 * coherence) + (0.3 * activation)

        # Penalize if semantic shift detected (indicates unstable understanding)
        if awareness_metadata.get('shift_detected', False):
            epistemic_confidence *= 0.8

        return epistemic_confidence

    def _should_stop_early(
        self,
        current_step: int,
        epistemic_confidences: List[float],
        max_steps: int
    ) -> bool:
        """
        Determine if reasoning should stop early due to low epistemic confidence.

        Consciousness Integration - Phase 1 (Nov 2025)
        Early stopping prevents wasted computation when system is very uncertain.

        Args:
            current_step: Current step number (0-indexed)
            epistemic_confidences: List of epistemic confidences so far
            max_steps: Maximum allowed steps

        Returns:
            True if should stop early, False otherwise
        """
        if not epistemic_confidences or current_step >= max_steps:
            return False

        # Check if last 2 steps have very low epistemic confidence
        recent_confidences = epistemic_confidences[-2:]
        if len(recent_confidences) >= 2:
            avg_recent = sum(recent_confidences) / len(recent_confidences)
            if avg_recent < self.epistemic_threshold:
                self.logger.warning(
                    f"[EPISTEMIC] Early stopping at step {current_step}: "
                    f"avg epistemic confidence {avg_recent:.3f} < threshold {self.epistemic_threshold}"
                )
                return True

        return False

    def _aggregate_epistemic_confidence(
        self,
        epistemic_confidences: List[float]
    ) -> Optional[float]:
        """
        Aggregate epistemic confidence across all reasoning steps.

        Consciousness Integration - Phase 1 (Nov 2025)

        Uses weighted average: recent steps weighted higher than early steps.

        Args:
            epistemic_confidences: List of epistemic confidences from each step

        Returns:
            Aggregated epistemic confidence or None
        """
        if not epistemic_confidences:
            return None

        # Weighted average: more recent steps have higher weight
        # Weight formula: step_weight = (step_idx + 1) / total_steps
        total_weight = 0.0
        weighted_sum = 0.0

        for idx, conf in enumerate(epistemic_confidences):
            weight = (idx + 1) / len(epistemic_confidences)  # Linear ramp: 1/n, 2/n, ..., n/n
            weighted_sum += conf * weight
            total_weight += weight

        aggregated = weighted_sum / total_weight if total_weight > 0 else None

        if aggregated is not None:
            self.logger.info(
                f"[EPISTEMIC] Aggregated epistemic confidence: {aggregated:.3f} "
                f"(from {len(epistemic_confidences)} steps)"
            )

        return aggregated

    def _extract_goal(self, query: Query) -> str:
        """Extract high-level goal from query."""
        # Simple heuristic - improve with NLP in production
        text = query.text.lower()

        if any(word in text for word in ["how", "what", "why"]):
            return f"Understand: {query.text}"
        elif any(word in text for word in ["should", "better", "optimal"]):
            return f"Decide: {query.text}"
        elif any(word in text for word in ["plan", "design", "implement"]):
            return f"Plan: {query.text}"
        else:
            return f"Answer: {query.text}"

    def _generate_verification_queries(
        self,
        query: Query,
        spacetime: Spacetime
    ) -> List[str]:
        """Generate queries to verify initial answer."""
        # Extract key claims from response
        response = spacetime.metadata.get("response", "")

        # Generate verification queries
        queries = [
            f"What are potential weaknesses in this answer: {response[:100]}?",
            f"Are there alternative perspectives on {query.text}?",
            f"What evidence contradicts the claim that {response[:100]}?"
        ]

        return queries

    async def _run_verification_loops(
        self,
        original_query: Query,
        initial_result: Spacetime,
        verification_queries: List[str],
        max_loops: int,
        steps: List[Dict],
        epistemic_confidences: List[float]  # Consciousness Integration - Phase 1 (Nov 2025)
    ) -> VerificationResult:
        """Execute verification loops with epistemic tracking."""
        contradictions = []
        supporting = []
        sources = []

        for i, vq in enumerate(verification_queries[:max_loops]):
            # Consciousness Integration - Phase 1
            # Early stopping if epistemic confidence is very low
            if self._should_stop_early(i + 1, epistemic_confidences, max_loops):
                self.logger.warning(
                    f"[EPISTEMIC] Stopping verification early at step {i+1} "
                    f"due to low epistemic confidence"
                )
                break

            self.logger.info(f"[AGENTIC] Verification: {vq}")
            result = await self.learning_engine.weave(Query(text=vq))

            # Consciousness Integration - Phase 1
            epistemic_conf = self._extract_epistemic_confidence(result)
            if epistemic_conf is not None:
                epistemic_confidences.append(epistemic_conf)

            response = result.metadata.get("response", "")
            sources.append(vq)

            # Simple heuristic - improve with semantic analysis
            if "however" in response.lower() or "but" in response.lower():
                contradictions.append(response[:200])
            else:
                supporting.append(response[:200])

            steps.append({
                "type": "verification",
                "query": vq,
                "confidence": result.confidence,
                "epistemic_confidence": epistemic_conf,  # Consciousness Integration - Phase 1
                "finding": "contradiction" if contradictions else "supporting"
            })

        # Determine if verified
        verified = len(contradictions) == 0 and initial_result.confidence >= 0.8

        # Generate refinement suggestions if not verified
        refinements = []
        if not verified:
            refinements.append(
                f"{original_query.text} (considering: {contradictions[0] if contradictions else 'low confidence'})"
            )

        return VerificationResult(
            verified=verified,
            confidence=initial_result.confidence,
            contradictions=contradictions,
            supporting_evidence=supporting,
            suggested_refinements=refinements,
            verification_queries=verification_queries,
            sources_checked=sources
        )

    async def _generate_research_queries(
        self,
        query: Query,
        max_queries: int,
        initial_findings: Optional[str] = None
    ) -> List[str]:
        """
        Generate research queries using LLM for intelligent exploration.

        Uses LLM to analyze gaps and generate targeted follow-up questions.
        Falls back to templates if LLM unavailable.
        """
        # Try LLM-activated intelligent query generation
        if self.llm and hasattr(self.llm, 'is_available') and self.llm.is_available():
            try:
                # Build prompt for LLM to generate research questions
                system_prompt = (
                    "You are a research assistant. Generate specific follow-up questions "
                    "to explore a topic thoroughly. Focus on gaps, tradeoffs, and practical implications."
                )

                if initial_findings:
                    user_prompt = f"""Original query: {query.text}

Initial findings: {initial_findings}

Based on these findings, what follow-up questions would help complete understanding?
Generate {max_queries} specific research questions, one per line.
Focus on:
- Gaps in the initial findings
- Practical applications and tradeoffs
- Edge cases or limitations
- Related concepts that provide context

Questions:"""
                else:
                    user_prompt = f"""Query: {query.text}

Generate {max_queries} research questions to explore this topic thoroughly, one per line.
Focus on:
- Key concepts and definitions
- Practical applications and use cases
- Tradeoffs and limitations
- Common misconceptions
- Recent developments

Questions:"""

                # Call LLM
                response = await self.llm.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=300,
                    temperature=0.7
                )

                # Parse questions from LLM response
                questions = self._parse_research_queries(response.content, max_queries)

                if questions:
                    self.logger.info(f"[AGENTIC] LLM generated {len(questions)} research queries")
                    return questions

            except Exception as e:
                self.logger.warning(f"LLM query generation failed: {e}, using fallback")

        # Fallback to template-based queries if LLM unavailable
        self.logger.info("[AGENTIC] Using template-based research queries (LLM unavailable)")
        base = query.text

        queries = [
            f"What are the key concepts in {base}?",
            f"What are the tradeoffs of {base}?",
            f"What are practical applications of {base}?",
            f"What are common misconceptions about {base}?",
            f"What are recent developments in {base}?"
        ]

        return queries[:max_queries]

    def _parse_research_queries(self, llm_response: str, max_queries: int) -> List[str]:
        """Parse research queries from LLM response."""
        lines = llm_response.strip().split('\n')
        queries = []

        for line in lines:
            # Clean up line (remove numbering, bullets, etc.)
            cleaned = line.strip()
            # Remove common prefixes
            for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•', 'Q:', 'Question:']:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()

            # Skip empty lines or very short lines
            if len(cleaned) > 10 and '?' in cleaned:
                queries.append(cleaned)

            if len(queries) >= max_queries:
                break

        return queries

    def _decompose_goal(self, query: Query) -> List[str]:
        """Decompose complex goal into sub-goals."""
        # Simple heuristic decomposition
        text = query.text

        sub_goals = [
            f"What are the prerequisites for {text}?",
            f"What are the main steps in {text}?",
            f"What are potential challenges with {text}?",
            f"What are success criteria for {text}?"
        ]

        return sub_goals

    def _create_synthesis_query(self, original: Query, evidence: List[str]) -> str:
        """Create synthesis query from gathered evidence."""
        return f"""
Based on the following research findings, provide a comprehensive answer to: {original.text}

Findings:
{chr(10).join(f'{i+1}. {e[:200]}...' for i, e in enumerate(evidence))}

Synthesize these findings into a coherent answer.
"""

    async def close(self):
        """Cleanup resources."""
        await self.learning_engine.close()
        self.logger.info("AgenticOrchestrator closed")


# ============================================================================
# Factory Functions
# ============================================================================

async def create_agentic_orchestrator(
    config,
    shards: List[MemoryShard],
    enable_verification: bool = True,
    enable_goal_tracking: bool = True,
    audit_trail: Optional[AuditTrail] = None
) -> AgenticOrchestrator:
    """
    Create agentic orchestrator with full learning system.

    Args:
        config: HoloLoom config
        shards: Memory shards
        enable_verification: Enable verification loops
        enable_goal_tracking: Enable goal/intent tracking
        audit_trail: Optional audit trail (creates new if None)

    Returns:
        AgenticOrchestrator ready to use
    """
    # Create full learning engine
    learning_engine = FullLearningEngine(
        cfg=config,
        shards=shards,
        enable_background_learning=True
    )

    # Initialize the learning engine's async context
    await learning_engine.__aenter__()

    return AgenticOrchestrator(
        learning_engine=learning_engine,
        audit_trail=audit_trail,
        enable_verification=enable_verification,
        enable_goal_tracking=enable_goal_tracking
    )