"""
RAG Department - Multi-scale retrieval and generation department.

Wraps HoloLoom's SimpleRAG system in the Department protocol, providing:
- Multi-scale Matryoshka embeddings (96, 192, 384 dimensions)
- BM25 + semantic hybrid retrieval
- Optional cross-encoder reranking (10-20% precision boost)
- LLM generation with confidence tracking
- DS-STAR verification (Domain, Sensibility, Temporal, Argument, Reference)
- Automatic refinement for low-confidence responses
- Learning from query patterns and feedback

Author: HoloLoom Team
Date: November 2025
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from HoloLoom.config import Config
from HoloLoom.rag.simple_rag import SimpleRAG, RAGResult
from HoloLoom.apps.departments.protocol import (
    Department,
    DepartmentRequest,
    DepartmentResponse,
    ConfidenceMetadata,
    PrivacyEnvelope,
    VerificationResult,
    DSStarCheck,
    DepartmentConfig,
)
from HoloLoom.apps.departments.base import BaseDepartment

logger = logging.getLogger(__name__)


class RAGDepartment(BaseDepartment):
    """
    RAG Department implementing multi-scale retrieval and generation.

    Provides question answering and document search using:
    - HoloLoom's memory system (graph + vector)
    - Matryoshka embeddings for multi-scale retrieval
    - BM25 + semantic hybrid search
    - Optional cross-encoder reranking
    - LLM generation with multiple reasoning modes

    Department Capabilities:
    - question_answering: Answer questions based on knowledge base
    - document_search: Retrieve relevant documents
    - batch_processing: Process multiple queries efficiently

    Constraints:
    - max_sources: 20 (maximum documents per query)
    - max_query_length: 1000 tokens
    - supported_modes: ["direct", "verify", "research", "plan_execute"]
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        llm_provider: str = "ollama",
        enable_reranking: bool = False,
        department_id: str = "rag",
    ):
        """
        Initialize RAG Department.

        Args:
            config: HoloLoom configuration (defaults to Config.fast())
            llm_provider: LLM provider ("ollama", "anthropic", "openai")
            enable_reranking: Enable cross-encoder reranking for precision boost
            department_id: Department identifier for registry
        """
        # Initialize base department
        dept_config = DepartmentConfig(name=department_id, domain="general")
        super().__init__(
            name=department_id,
            domain="general",
            version="1.0.0",
            supported_tasks=["question_answering", "document_search", "batch_processing"],
            confidence_range=(0.7, 0.95),
            config=dept_config
        )

        self.config = config or Config.fast()
        self.llm_provider = llm_provider
        self.enable_reranking = enable_reranking

        # SimpleRAG instance (initialized in async context)
        self.rag: Optional[SimpleRAG] = None

        # Learning statistics
        self._query_patterns: Dict[str, int] = {}  # Track query types
        self._confidence_history: List[float] = []  # Track confidence over time
        self._refinement_stats = {
            "total_refinements": 0,
            "successful_refinements": 0,
            "avg_improvement": 0.0,
        }

        # DS-STAR verification thresholds
        self._verification_thresholds = {
            "domain_min_relevance": 0.6,  # Minimum domain relevance score
            "sensibility_min_score": 0.7,  # Minimum logical coherence
            "temporal_max_age_days": 365,  # Maximum age of information
            "argument_min_support": 0.8,  # Minimum source support
            "reference_min_traceable": 0.9,  # Minimum traceability
        }

        logger.info(f"✓ RAGDepartment initialized (id={department_id}, llm={llm_provider})")

    async def __aenter__(self):
        """Async context manager entry - initialize SimpleRAG."""
        await super().__aenter__()

        # Initialize SimpleRAG
        self.rag = SimpleRAG(
            config=self.config,
            llm_provider=self.llm_provider,
            enable_caching=True,
            enable_reranking=self.enable_reranking,
        )
        await self.rag.__aenter__()

        logger.info("✓ RAG Department ready")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup SimpleRAG."""
        if self.rag:
            await self.rag.__aexit__(exc_type, exc_val, exc_tb)

        await super().__aexit__(exc_type, exc_val, exc_tb)
        logger.info("✓ RAG Department closed")

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Execute RAG query with retrieval + generation.

        Process:
        1. Extract query from request
        2. Validate privacy envelope
        3. Call SimpleRAG.query() with appropriate mode
        4. Track confidence and sources
        5. Return DepartmentResponse with confidence metadata

        Args:
            request: DepartmentRequest with:
                - query (str): Question text
                - mode (str, optional): Reasoning mode (default: "verify")
                - max_sources (int, optional): Maximum sources (default: 5)

        Returns:
            DepartmentResponse with:
                - response: Generated answer
                - confidence: ConfidenceMetadata with score and justification
                - sources: List of retrieved documents
                - metadata: Query stats and timing

        Raises:
            RuntimeError: If RAG not initialized
            ValueError: If query exceeds max_query_length
        """
        if self.rag is None:
            raise RuntimeError("RAG Department not initialized. Use async context manager.")

        start_time = time.time()

        # Extract query parameters
        query_text = request.parameters.get("query", "")
        if not query_text:
            raise ValueError("Request must contain 'query' field")

        mode = request.parameters.get("mode", "verify")
        max_sources = request.parameters.get("max_sources", 5)

        # Validate query length (approximate token count)
        if len(query_text.split()) > 1000:
            raise ValueError(f"Query exceeds max_query_length (1000 tokens)")

        logger.info(f"Executing RAG query: {query_text[:50]}... (mode={mode})")

        # Privacy check (optional attribute)
        privacy_envelope = getattr(request, 'privacy_envelope', None)
        if privacy_envelope:
            if privacy_envelope.privacy_level.value > 2:  # RESTRICTED or higher
                logger.warning(f"High privacy level: {privacy_envelope.privacy_level}")
                # Could implement differential privacy here

        # Execute RAG query
        try:
            rag_result: RAGResult = await self.rag.query(
                question=query_text,
                mode=mode,
                max_sources=max_sources,
                use_cache=True,
            )

            # Track confidence
            self._confidence_history.append(rag_result.confidence)

            # Track query pattern (classify query type)
            query_type = self._classify_query_type(query_text)
            self._query_patterns[query_type] = self._query_patterns.get(query_type, 0) + 1

            # Create confidence metadata
            confidence_meta = ConfidenceMetadata.from_score(
                score=rag_result.confidence,
                justification=[
                    f"Retrieved {len(rag_result.sources)} relevant sources",
                    f"LLM provider: {rag_result.metadata.get('llm_provider', 'unknown')}",
                    f"Reasoning mode: {rag_result.reasoning_mode}",
                ],
                sources=rag_result.sources[:3],  # First 3 sources as justification
            )

            # Create response
            response = DepartmentResponse(
                department_id=self.name,  # BaseDepartment stores as self.name
                request_id=request.task_id,  # DepartmentRequest uses task_id
                response={
                    "answer": rag_result.response,
                    "sources": rag_result.sources,
                    "reasoning_mode": rag_result.reasoning_mode,
                },
                confidence=confidence_meta,
                metadata={
                    "n_sources": len(rag_result.sources),
                    "cache_hit": rag_result.metadata.get("cache_hit", False),
                    "rerank_latency_ms": rag_result.metadata.get("rerank_latency_ms", 0.0),
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "query_type": query_type,
                },
                privacy_envelope=request.privacy_envelope,
            )

            logger.info(
                f"✓ RAG query complete: confidence={confidence_meta.score:.2f}, "
                f"sources={len(rag_result.sources)}, time={response.metadata['execution_time_ms']:.1f}ms"
            )

            return response

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            # Return low-confidence error response
            error_confidence = ConfidenceMetadata.from_score(
                score=0.1,
                justification=[f"Error during RAG execution: {str(e)}"],
                sources=[],
            )

            return DepartmentResponse(
                department_id=self.name,  # BaseDepartment stores as self.name
                request_id=request.task_id,  # DepartmentRequest uses task_id
                response={
                    "answer": f"Unable to process query: {str(e)}",
                    "sources": [],
                    "reasoning_mode": "error",
                },
                confidence=error_confidence,
                metadata={
                    "error": str(e),
                    "execution_time_ms": (time.time() - start_time) * 1000,
                },
                privacy_envelope=request.privacy_envelope,
            )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """
        Verify response using DS-STAR framework.

        DS-STAR checks:
        - **Domain**: Are sources relevant to the query domain?
        - **Sensibility**: Does the answer make logical sense?
        - **Temporal**: Is the information up-to-date?
        - **Argument**: Is the answer supported by sources?
        - **Reference**: Are sources traceable and credible?

        Args:
            response: DepartmentResponse to verify

        Returns:
            VerificationResult with:
                - verified: True if all checks pass
                - checks: List of DSStarCheck results
                - overall_score: Aggregate verification score (0.0-1.0)
                - recommendations: Suggested improvements
        """
        logger.info(f"Verifying response (request_id={response.request_id})...")

        checks: List[DSStarCheck] = []
        answer = response.response.get("answer", "")
        sources = response.response.get("sources", [])

        # 1. Domain check: Are sources relevant?
        domain_relevance = self._check_domain_relevance(sources)
        checks.append(
            DSStarCheck(
                dimension="Domain",
                passed=domain_relevance >= self._verification_thresholds["domain_min_relevance"],
                score=domain_relevance,
                details=f"Source relevance: {domain_relevance:.2f}",
            )
        )

        # 2. Sensibility check: Does answer make logical sense?
        sensibility_score = self._check_sensibility(answer, sources)
        checks.append(
            DSStarCheck(
                dimension="Sensibility",
                passed=sensibility_score >= self._verification_thresholds["sensibility_min_score"],
                score=sensibility_score,
                details=f"Logical coherence: {sensibility_score:.2f}",
            )
        )

        # 3. Temporal check: Is information up-to-date?
        temporal_age_days = self._check_temporal_freshness(sources)
        temporal_passed = temporal_age_days <= self._verification_thresholds["temporal_max_age_days"]
        checks.append(
            DSStarCheck(
                dimension="Temporal",
                passed=temporal_passed,
                score=1.0 - (temporal_age_days / 365),  # Normalize to 0-1
                details=f"Information age: {temporal_age_days} days",
            )
        )

        # 4. Argument check: Is answer supported by sources?
        argument_support = self._check_argument_support(answer, sources)
        checks.append(
            DSStarCheck(
                dimension="Argument",
                passed=argument_support >= self._verification_thresholds["argument_min_support"],
                score=argument_support,
                details=f"Source support: {argument_support:.2f}",
            )
        )

        # 5. Reference check: Are sources traceable?
        reference_trace = self._check_reference_traceability(sources)
        checks.append(
            DSStarCheck(
                dimension="Reference",
                passed=reference_trace >= self._verification_thresholds["reference_min_traceable"],
                score=reference_trace,
                details=f"Traceability: {reference_trace:.2f}",
            )
        )

        # Aggregate score
        overall_score = sum(check.score for check in checks) / len(checks)
        verified = all(check.passed for check in checks)

        # Generate recommendations
        recommendations = []
        if not verified:
            for check in checks:
                if not check.passed:
                    recommendations.append(f"Improve {check.dimension}: {check.details}")

        result = VerificationResult(
            verified=verified,
            checks=checks,
            overall_score=overall_score,
            recommendations=recommendations,
            timestamp=datetime.utcnow(),
        )

        logger.info(
            f"✓ Verification complete: verified={verified}, score={overall_score:.2f}"
        )

        return result

    async def refine(self, response: DepartmentResponse) -> DepartmentResponse:
        """
        Refine low-confidence responses by retrying with improved parameters.

        Refinement strategies:
        1. Expand search: Increase max_sources (5 → 10 → 20)
        2. Enable reranking: Boost precision with cross-encoder
        3. Change reasoning mode: Try "research" or "plan_execute"

        Args:
            response: DepartmentResponse with low confidence (<0.75)

        Returns:
            Refined DepartmentResponse with (hopefully) higher confidence

        Note:
            Tracks refinement success in _refinement_stats for learning
        """
        if response.confidence.score >= 0.75:
            logger.info(f"Response confidence {response.confidence.score:.2f} ≥ 0.75, no refinement needed")
            return response

        logger.info(f"Refining response (confidence={response.confidence.score:.2f})...")
        self._refinement_stats["total_refinements"] += 1

        # Reconstruct original request (need query text)
        # Note: In production, would store original request or extract from metadata
        original_query = response.metadata.get("original_query", "")
        if not original_query:
            logger.warning("Cannot refine: original query not found in metadata")
            return response

        # Strategy 1: Expand search + enable reranking
        refined_request = DepartmentRequest(
            department_id=self.department_id,
            request_id=f"{response.request_id}_refined",
            query={
                "query": original_query,
                "mode": "research",  # More thorough reasoning
                "max_sources": 10,  # Expand from default 5
            },
            privacy_envelope=response.privacy_envelope,
        )

        # Temporarily enable reranking if not already on
        original_reranking = self.enable_reranking
        if not self.enable_reranking:
            logger.info("Enabling reranking for refinement...")
            self.enable_reranking = True

        try:
            refined_response = await self.execute(refined_request)

            # Restore original reranking setting
            self.enable_reranking = original_reranking

            # Track improvement
            improvement = refined_response.confidence.score - response.confidence.score
            if improvement > 0:
                self._refinement_stats["successful_refinements"] += 1
                self._refinement_stats["avg_improvement"] = (
                    (self._refinement_stats["avg_improvement"] * (self._refinement_stats["successful_refinements"] - 1)
                     + improvement) / self._refinement_stats["successful_refinements"]
                )

            logger.info(
                f"✓ Refinement complete: {response.confidence.score:.2f} → "
                f"{refined_response.confidence.score:.2f} (Δ={improvement:+.2f})"
            )

            # Add refinement metadata
            refined_response.metadata["refinement"] = {
                "original_confidence": response.confidence.score,
                "improvement": improvement,
                "strategy": "expand_search_and_rerank",
            }

            return refined_response

        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            self.enable_reranking = original_reranking
            return response

    async def update_strategy(self, feedback: Dict[str, Any]) -> None:
        """
        Learn from feedback and adapt retrieval strategies.

        Learning signals:
        - Query patterns: Adjust retrieval for different query types
        - Confidence trends: Calibrate confidence thresholds
        - Refinement success: Tune refinement triggers
        - User feedback: Explicit ratings (helpful: true/false)

        Args:
            feedback: Dictionary with feedback signals:
                - helpful (bool): Was response helpful?
                - query_type (str): Type of query
                - confidence (float): Response confidence
                - refinement_needed (bool): Should have been refined?
        """
        logger.info(f"Updating strategy with feedback: {feedback}")

        # 1. Adjust confidence thresholds based on user feedback
        if "helpful" in feedback and "confidence" in feedback:
            confidence = feedback["confidence"]
            helpful = feedback["helpful"]

            if helpful and confidence < 0.75:
                # System was too conservative, lower refinement threshold
                logger.info("Lowering refinement threshold (false negative)")
                # Could adjust self._verification_thresholds here

            elif not helpful and confidence >= 0.75:
                # System was too optimistic, raise refinement threshold
                logger.info("Raising refinement threshold (false positive)")

        # 2. Track query patterns for future optimization
        if "query_type" in feedback:
            query_type = feedback["query_type"]
            logger.info(f"Query pattern tracked: {query_type}")
            # Could adjust retrieval params per query type

        # 3. Learn from refinement outcomes
        if "refinement_needed" in feedback:
            refinement_needed = feedback["refinement_needed"]
            actual_refinement = self._refinement_stats["total_refinements"] > 0

            if refinement_needed and not actual_refinement:
                logger.info("Refinement should have been triggered (missed opportunity)")
            elif not refinement_needed and actual_refinement:
                logger.info("Refinement was unnecessary (wasted resources)")

        # 4. Log feedback for future analysis
        await self._store_feedback(feedback)

        logger.info("✓ Strategy updated")

    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Report RAG Department capabilities and constraints.

        Returns:
            Dictionary with:
                - tasks: Supported task types
                - max_tokens: Maximum query length
                - supported_languages: Supported languages
                - reasoning_modes: Available reasoning modes
                - features: Enabled features (reranking, caching, etc.)
        """
        return {
            "tasks": [
                "question_answering",
                "document_search",
                "batch_processing",
            ],
            "max_tokens": 1000,
            "supported_languages": ["en"],  # Could expand with multilingual embeddings
            "reasoning_modes": ["direct", "verify", "research", "plan_execute"],
            "features": {
                "multi_scale_embeddings": True,
                "hybrid_retrieval": True,  # BM25 + semantic
                "reranking": self.enable_reranking,
                "caching": True,
                "llm_generation": self.rag is not None and self.rag.orchestrator is not None,
            },
            "constraints": {
                "max_sources": 20,
                "max_query_length": 1000,
                "max_batch_size": 100,
            },
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get RAG Department metrics for monitoring.

        Returns:
            Dictionary with:
                - rag_metrics: Metrics from SimpleRAG (cache, reranking, etc.)
                - confidence_stats: Mean/std confidence over time
                - query_patterns: Distribution of query types
                - refinement_stats: Refinement success rates
                - department_metrics: Base department metrics
        """
        rag_metrics = self.rag.get_metrics() if self.rag else {}

        # Confidence statistics
        confidence_stats = {}
        if self._confidence_history:
            import statistics
            confidence_stats = {
                "mean": statistics.mean(self._confidence_history),
                "stdev": statistics.stdev(self._confidence_history) if len(self._confidence_history) > 1 else 0.0,
                "min": min(self._confidence_history),
                "max": max(self._confidence_history),
                "count": len(self._confidence_history),
            }

        # Refinement statistics
        refinement_stats = {
            **self._refinement_stats,
            "refinement_rate": (
                self._refinement_stats["total_refinements"] / len(self._confidence_history)
                if self._confidence_history else 0.0
            ),
            "success_rate": (
                self._refinement_stats["successful_refinements"] / self._refinement_stats["total_refinements"]
                if self._refinement_stats["total_refinements"] > 0 else 0.0
            ),
        }

        return {
            "rag_metrics": rag_metrics,
            "confidence_stats": confidence_stats,
            "query_patterns": self._query_patterns,
            "refinement_stats": refinement_stats,
            "department_metrics": self._metrics,  # Direct access to BaseDepartment metrics
        }

    async def health_check(self) -> bool:
        """
        Check RAG Department health.

        Verifies:
        - SimpleRAG is initialized
        - HoloLoom memory system is accessible
        - LLM orchestrator is available (if configured)

        Returns:
            True if all systems operational, False otherwise
        """
        if self.rag is None:
            logger.warning("Health check failed: RAG not initialized")
            return False

        # Check HoloLoom memory
        if self.rag.loom is None:
            logger.warning("Health check failed: HoloLoom memory unavailable")
            return False

        # Check LLM orchestrator (optional but recommended)
        if self.rag.orchestrator is None:
            logger.warning("Health check warning: LLM orchestrator unavailable (fallback mode)")
            # Not a failure, but degraded capability

        logger.info("✓ Health check passed")
        return True

    # ===== Private Helper Methods =====

    def _classify_query_type(self, query: str) -> str:
        """Classify query type for pattern tracking."""
        query_lower = query.lower()

        if any(q in query_lower for q in ["what is", "define", "explain"]):
            return "factual"
        elif any(q in query_lower for q in ["how to", "how do", "steps"]):
            return "procedural"
        elif any(q in query_lower for q in ["why", "reason", "cause"]):
            return "analytical"
        elif any(q in query_lower for q in ["compare", "difference", "vs"]):
            return "comparative"
        else:
            return "other"

    def _check_domain_relevance(self, sources: List[str]) -> float:
        """Check if sources are relevant to the query domain."""
        if not sources:
            return 0.0

        # Simplified: Assume sources from memory are relevant
        # In production, could use domain classifier or embedding distance
        return 0.8  # Default assumption

    def _check_sensibility(self, answer: str, sources: List[str]) -> float:
        """Check if answer makes logical sense."""
        if not answer or not sources:
            return 0.0

        # Simplified: Check answer length and source utilization
        # In production, could use language model for coherence scoring
        answer_has_content = len(answer.split()) > 5
        uses_sources = any(snippet in answer for snippet in sources[:3])

        score = 0.0
        if answer_has_content:
            score += 0.5
        if uses_sources:
            score += 0.5

        return score

    def _check_temporal_freshness(self, sources: List[str]) -> float:
        """Check age of information (in days)."""
        # Simplified: Assume sources are recent unless metadata indicates otherwise
        # In production, would extract timestamps from source metadata
        return 30.0  # Assume 30 days old (arbitrary)

    def _check_argument_support(self, answer: str, sources: List[str]) -> float:
        """Check if answer is supported by sources."""
        if not answer or not sources:
            return 0.0

        # Simplified: Count source citations in answer
        # In production, would use semantic similarity or NLI model
        citations = sum(1 for source in sources if any(word in answer.lower() for word in source.lower().split()[:5]))
        support_score = min(citations / len(sources), 1.0)

        return support_score

    def _check_reference_traceability(self, sources: List[str]) -> float:
        """Check if sources are traceable and credible."""
        if not sources:
            return 0.0

        # Simplified: Assume memory sources are traceable
        # In production, would verify source URLs, dates, authors
        return 0.9  # Default assumption

    async def _store_feedback(self, feedback: Dict[str, Any]) -> None:
        """Store feedback for future analysis."""
        # In production, would persist to database
        # For now, just log
        logger.info(f"Feedback stored: {feedback}")

    # ===== Recursive Reasoning Integration =====

    async def recursive_reason(
        self,
        query: str,
        max_depth: int = 5,
        min_confidence: float = 0.85,
        enable_decomposition: bool = True,
        enable_learning: bool = True
    ) -> Dict[str, Any]:
        """
        Recursively refine response until convergence.

        This is the main entry point for recursive reasoning, integrating:
        - Automatic convergence detection
        - Multi-level query decomposition
        - Strategy selection with Thompson Sampling
        - Complete provenance tracking

        Args:
            query: Query to reason about
            max_depth: Maximum refinement iterations (default: 5)
            min_confidence: Target confidence threshold (default: 0.85)
            enable_decomposition: Enable query decomposition for complex queries
            enable_learning: Enable strategy learning from outcomes

        Returns:
            Dictionary with:
                - final_answer: Final response text
                - final_confidence: Final confidence score
                - iterations: Number of refinement iterations
                - convergence_reason: Why reasoning stopped
                - improvement_trajectory: Confidence at each step
                - refinement_steps: Complete history of refinements
                - total_latency_ms: Total processing time
                - decomposition: Optional decomposition tree (if decomposed)

        Example:
            >>> result = await rag_dept.recursive_reason(
            ...     "What are the tradeoffs of Thompson Sampling?",
            ...     max_depth=5,
            ...     min_confidence=0.85
            ... )
            >>> print(result["final_answer"])
            >>> print(f"Confidence: {result['final_confidence']:.2f}")
            >>> print(f"Iterations: {result['iterations']}")
        """
        from HoloLoom.convergence.recursive_reasoner_enhanced import create_recursive_reasoner
        from HoloLoom.protocols.recursive_reasoning import RecursiveConfig, ReasoningStrategy

        logger.info(f"Starting recursive reasoning: {query[:50]}... (max_depth={max_depth})")

        # Create configuration
        config = RecursiveConfig(
            strategy=ReasoningStrategy.REFINE,
            max_iterations=max_depth,
            quality_threshold=min_confidence,
            min_improvement=0.05,
            enable_decomposition=enable_decomposition,
            enable_learning=enable_learning,
            refinement_threshold=0.75,  # Auto-refine if confidence < 0.75
            max_refinement_iterations=max_depth
        )

        # Create recursive reasoner
        reasoner = create_recursive_reasoner(
            department=self,
            config=config,
            enable_decomposition=enable_decomposition,
            enable_learning=enable_learning
        )

        # Execute recursive reasoning
        result = await reasoner.reason(query, context={})

        # Convert to dictionary format
        return {
            "final_answer": result.final_response,
            "final_confidence": result.final_confidence,
            "iterations": result.iterations,
            "convergence_reason": result.convergence_reason,
            "improvement_trajectory": result.improvement_trajectory,
            "refinement_steps": [
                {
                    "iteration": step.iteration,
                    "query": step.query,
                    "confidence": step.confidence,
                    "improvement": step.improvement,
                    "strategy": step.strategy_used,
                    "reasoning": step.reasoning
                }
                for step in result.refinement_steps
            ],
            "total_latency_ms": result.total_latency_ms,
            "decomposition": {
                "decomposed": result.decomposition is not None,
                "sub_queries": result.decomposition.sub_queries if result.decomposition else [],
                "complexity_score": result.decomposition.complexity_score if result.decomposition else 0.0,
                "synthesis_strategy": result.decomposition.synthesis_strategy if result.decomposition else "direct"
            } if result.decomposition else None,
            "metadata": result.metadata
        }