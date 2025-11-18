#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-Integrated Context Packer

Extends SmartContextPacker with:
- LLM generation (complete the feedback loop)
- Quality scoring (coherence, completeness, relevance)
- Token efficiency tracking (quality / tokens)
- Context utilization analysis (which elements were used?)
- Learning system (adapt packing strategy based on outcomes)

This completes Phase 1: Feedback Loop

Usage:
    packer = LLMContextPacker(
        llm_provider="anthropic",
        llm_model="claude-3-5-sonnet-20241022"
    )

    result = await packer.pack_and_generate(
        query="What is quantum tunneling?",
        awareness_ctx=awareness_context,
        memory_results=memories
    )

    # Result includes:
    # - packed_context: PackedContext
    # - llm_response: LLMResponse
    # - quality_score: QualityScore
    # - token_efficiency: float
    # - context_utilization: ContextUtilization

    # Packer learns automatically from each interaction
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Import base context packer
from .context_packer import (
    SmartContextPacker,
    PackedContext,
    TokenBudget,
    ContextElement
)

# Import LLM providers
from .llm_providers import (
    get_llm_provider,
    BaseLLMProvider,
    LLMResponse,
    LLMGenerationConfig
)

# Import adaptive budgeting (Phase 2)
try:
    from .adaptive_budget import AdaptiveBudgetCalculator, AdaptiveBudget
    from .query_complexity import ComplexityLevel
    ADAPTIVE_BUDGET_AVAILABLE = True
except ImportError:
    ADAPTIVE_BUDGET_AVAILABLE = False
    # Provide dummy classes for type hints
    AdaptiveBudget = None
    ComplexityLevel = None

# Import outcome tracking and adaptive tuning (Phase 3)
try:
    from .outcome_tracker import OutcomeTracker, CompressionLevel
    from .adaptive_threshold_tuner import AdaptiveThresholdTuner, TuningStrategy
    from .performance_analyzer import PerformanceAnalyzer
    OUTCOME_TRACKING_AVAILABLE = True
except ImportError:
    OUTCOME_TRACKING_AVAILABLE = False
    OutcomeTracker = None
    AdaptiveThresholdTuner = None
    PerformanceAnalyzer = None
    TuningStrategy = None
    CompressionLevel = None

# Import context compression (Phase 4)
try:
    from .semantic_deduplicator import (
        SemanticDeduplicator,
        DeduplicationStrategy,
        SimilarityMethod
    )
    from .compression_analyzer import CompressionAnalyzer
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False
    SemanticDeduplicator = None
    DeduplicationStrategy = None
    SimilarityMethod = None
    CompressionAnalyzer = None

# Import multi-pass refinement (Phase 5)
try:
    from .refinement_engine import (
        RefinementEngine,
        RefinementStrategy,
        RefinementResult
    )
    REFINEMENT_AVAILABLE = True
except ImportError:
    REFINEMENT_AVAILABLE = False
    RefinementEngine = None
    RefinementStrategy = None
    RefinementResult = None


@dataclass
class QualityScore:
    """
    LLM response quality score.

    Combines multiple quality dimensions:
    - Coherence: Is the response well-structured and logical?
    - Completeness: Does it fully answer the query?
    - Relevance: Is it on-topic and useful?
    - Accuracy: Is it factually correct? (if verifiable)
    """

    # Individual dimensions (0.0-1.0)
    coherence: float
    completeness: float
    relevance: float
    accuracy: Optional[float] = None

    # Overall score (weighted average)
    @property
    def overall(self) -> float:
        """Calculate overall quality score"""
        scores = [self.coherence, self.completeness, self.relevance]
        weights = [0.3, 0.4, 0.3]  # Completeness weighted highest

        if self.accuracy is not None:
            scores.append(self.accuracy)
            weights = [0.25, 0.3, 0.25, 0.2]  # Add accuracy weight

        return sum(s * w for s, w in zip(scores, weights))

    @property
    def grade(self) -> str:
        """Letter grade based on overall score"""
        score = self.overall
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"


@dataclass
class ContextUtilization:
    """
    Analysis of which context elements were actually used in the response.

    This helps identify:
    - Which elements are valuable (high utilization)
    - Which elements are wasted (low/zero utilization)
    - Optimal importance thresholds
    """

    # Element-level utilization
    element_mentions: Dict[str, int]  # element_id → mention_count

    # Source-level utilization
    source_utilization: Dict[str, float]  # "awareness" → 0.8 (80% used)

    # Overall metrics
    total_elements_packed: int
    total_elements_utilized: int  # At least 1 mention

    @property
    def utilization_rate(self) -> float:
        """Overall utilization rate (0.0-1.0)"""
        if self.total_elements_packed == 0:
            return 0.0
        return self.total_elements_utilized / self.total_elements_packed

    @property
    def wasted_elements(self) -> int:
        """Number of packed elements not used at all"""
        return self.total_elements_packed - self.total_elements_utilized


@dataclass
class PackedGeneration:
    """
    Complete result of pack_and_generate().

    Includes:
    - Packed context (what was sent to LLM)
    - LLM response (what LLM generated)
    - Quality score (how good was the response)
    - Token efficiency (quality per token)
    - Context utilization (which elements were used)
    """

    # Core outputs
    packed_context: PackedContext
    llm_response: LLMResponse

    # Quality metrics
    quality_score: QualityScore
    token_efficiency: float  # quality_score / total_tokens

    # Utilization analysis
    context_utilization: ContextUtilization

    # Query metadata
    query: str
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        """Human-readable summary"""
        return (
            f"PackedGeneration(\n"
            f"  Query: {self.query[:60]}...\n"
            f"  Quality: {self.quality_score.overall:.2f} ({self.quality_score.grade})\n"
            f"  Token Efficiency: {self.token_efficiency:.4f}\n"
            f"  Context Utilization: {self.context_utilization.utilization_rate:.1%}\n"
            f"  Response Length: {len(self.llm_response.text)} chars\n"
            f"  Total Tokens: {self.llm_response.total_tokens}\n"
            f"  Cost: ${self.llm_response.cost_estimate_usd:.4f}\n"
            f"  Latency: {self.llm_response.latency_ms + self.packed_context.packing_time_ms:.0f}ms\n"
            f")"
        )


class LLMContextPacker(SmartContextPacker):
    """
    Smart Context Packer with LLM integration and learning.

    Extends SmartContextPacker with:
    1. LLM generation (pack → generate → feedback)
    2. Quality scoring (multi-dimensional quality assessment)
    3. Token efficiency tracking (quality / tokens)
    4. Context utilization analysis (which elements were used)
    5. Learning system (adapt packing strategy based on outcomes)

    Phase 1: Feedback Loop Complete
    """

    def __init__(
        self,
        token_budget: Optional[TokenBudget] = None,
        min_importance_threshold: float = 0.2,
        use_memory_fusion: bool = True,
        memory_backend=None,
        # LLM integration
        llm_provider: str = "ollama",
        llm_model: Optional[str] = None,
        llm_config: Optional[LLMGenerationConfig] = None,
        # Learning
        enable_learning: bool = True,
        learning_rate: float = 0.1,  # How fast to adapt strategies
        # Phase 2: Adaptive budgeting
        enable_adaptive_budgeting: bool = False,
        adaptive_budget_min: int = 2_000,
        adaptive_budget_max: int = 32_000,
        # Phase 3: Outcome tracking and adaptive tuning
        enable_outcome_tracking: bool = False,
        enable_adaptive_tuning: bool = False,
        tuning_strategy: Optional[str] = "balanced",
        # Phase 4: Context compression
        enable_compression: bool = False,
        compression_strategy: str = "merge_similar",  # "merge_similar", "keep_best", "summarize", "remove_exact"
        compression_similarity_threshold: float = 0.8,  # 0.8 = 80% similar
        compression_similarity_method: str = "ngram",  # "embeddings", "ngram", "jaccard", "exact"
        compression_min_savings: int = 50  # Minimum token savings to justify compression
    ):
        """
        Initialize LLM-integrated context packer.

        Args:
            token_budget: Token budget constraints (ignored if adaptive_budgeting=True)
            min_importance_threshold: Minimum importance to include
            use_memory_fusion: Enable multipass memory fusion
            memory_backend: Backend for memory fusion
            llm_provider: LLM provider ("anthropic", "openai", "ollama")
            llm_model: Model identifier (optional, uses default)
            llm_config: LLM generation config
            enable_learning: Enable learning from outcomes
            learning_rate: How fast to adapt (0.0-1.0)
            enable_adaptive_budgeting: Enable Phase 2 adaptive budgeting
            adaptive_budget_min: Minimum budget (if adaptive)
            adaptive_budget_max: Maximum budget (if adaptive)
            enable_outcome_tracking: Enable Phase 3 outcome tracking
            enable_adaptive_tuning: Enable Phase 3 adaptive threshold tuning
            tuning_strategy: Tuning strategy ("conservative", "balanced", "aggressive")
            enable_compression: Enable Phase 4 context compression
            compression_strategy: Deduplication strategy
            compression_similarity_threshold: Similarity threshold for deduplication (0.0-1.0)
            compression_similarity_method: Method for computing similarity
            compression_min_savings: Minimum token savings to justify compression
        """
        # Initialize base packer
        super().__init__(
            token_budget=token_budget,
            min_importance_threshold=min_importance_threshold,
            use_memory_fusion=use_memory_fusion,
            memory_backend=memory_backend
        )

        # LLM integration
        self.llm_provider_name = llm_provider
        self.llm_model = llm_model
        self.llm_config = llm_config or LLMGenerationConfig(
            max_tokens=1000,
            temperature=0.7
        )

        # Phase 2: Adaptive budgeting
        self.enable_adaptive_budgeting = enable_adaptive_budgeting and ADAPTIVE_BUDGET_AVAILABLE
        self.adaptive_budget_calculator = None

        if self.enable_adaptive_budgeting:
            if not ADAPTIVE_BUDGET_AVAILABLE:
                logger.warning("Adaptive budgeting requested but not available (import failed)")
            else:
                self.adaptive_budget_calculator = AdaptiveBudgetCalculator(
                    model_name=llm_model or self._get_default_model(llm_provider),
                    min_budget=adaptive_budget_min,
                    max_budget=adaptive_budget_max,
                    enable_learning=enable_learning
                )

        # Phase 3: Outcome tracking and adaptive tuning
        self.enable_outcome_tracking = enable_outcome_tracking and OUTCOME_TRACKING_AVAILABLE
        self.enable_adaptive_tuning = enable_adaptive_tuning and OUTCOME_TRACKING_AVAILABLE
        self.outcome_tracker = None
        self.adaptive_tuner = None
        self.performance_analyzer = None

        if self.enable_outcome_tracking or self.enable_adaptive_tuning:
            if not OUTCOME_TRACKING_AVAILABLE:
                logger.warning("Outcome tracking requested but not available (import failed)")
            else:
                # Create outcome tracker
                self.outcome_tracker = OutcomeTracker(
                    min_samples_for_recommendation=10,
                    enable_persistence=False  # TODO: Add persistence support
                )

                # Create adaptive threshold tuner (if enabled)
                if self.enable_adaptive_tuning and tuning_strategy:
                    strategy_map = {
                        "conservative": TuningStrategy.CONSERVATIVE,
                        "balanced": TuningStrategy.BALANCED,
                        "aggressive": TuningStrategy.AGGRESSIVE
                    }
                    strategy = strategy_map.get(tuning_strategy, TuningStrategy.BALANCED)

                    self.adaptive_tuner = AdaptiveThresholdTuner(
                        outcome_tracker=self.outcome_tracker,
                        enable_auto_tuning=True,
                        tuning_strategy=strategy
                    )

                # Create performance analyzer
                self.performance_analyzer = PerformanceAnalyzer(
                    outcome_tracker=self.outcome_tracker
                )

        # Phase 4: Context compression
        self.enable_compression = enable_compression and COMPRESSION_AVAILABLE
        self.semantic_deduplicator = None
        self.compression_analyzer = None

        if self.enable_compression:
            if not COMPRESSION_AVAILABLE:
                logger.warning("Compression requested but not available (import failed)")
            else:
                # Map strategy name to enum
                strategy_map = {
                    "merge_similar": DeduplicationStrategy.MERGE_SIMILAR,
                    "keep_best": DeduplicationStrategy.KEEP_BEST,
                    "summarize": DeduplicationStrategy.SUMMARIZE,
                    "remove_exact": DeduplicationStrategy.REMOVE_EXACT
                }
                self.compression_strategy = strategy_map.get(
                    compression_strategy,
                    DeduplicationStrategy.MERGE_SIMILAR
                )

                # Map similarity method name to enum
                method_map = {
                    "embeddings": SimilarityMethod.EMBEDDINGS,
                    "ngram": SimilarityMethod.NGRAM,
                    "jaccard": SimilarityMethod.JACCARD,
                    "exact": SimilarityMethod.EXACT
                }
                similarity_method = method_map.get(
                    compression_similarity_method,
                    SimilarityMethod.NGRAM
                )

                # Create semantic deduplicator
                self.semantic_deduplicator = SemanticDeduplicator(
                    similarity_threshold=compression_similarity_threshold,
                    similarity_method=similarity_method,
                    min_tokens_to_deduplicate=compression_min_savings,
                    preserve_importance=True,  # Never compress CRITICAL elements
                    embeddings=None  # TODO: Pass embeddings if available
                )

                # Create compression analyzer
                self.compression_analyzer = CompressionAnalyzer(
                    track_quality=True,
                    track_latency=True
                )

        # Lazy-load LLM provider
        self._llm_provider: Optional[BaseLLMProvider] = None

        # Learning system
        self.enable_learning = enable_learning
        self.learning_rate = learning_rate

        # Learning statistics (track correlations)
        self._learning_stats = {
            # importance_threshold → quality correlation
            "importance_quality": defaultdict(list),  # threshold → [quality_scores]

            # compression_level → quality correlation
            "compression_quality": defaultdict(list),  # level → [quality_scores]

            # memory_count → quality correlation
            "memory_count_quality": defaultdict(list),  # count → [quality_scores]

            # token_budget → quality correlation
            "budget_quality": defaultdict(list),  # budget → [quality_scores]

            # Total interactions
            "total_interactions": 0
        }

    def _get_llm_provider(self) -> BaseLLMProvider:
        """Lazy-load LLM provider"""
        if self._llm_provider is None:
            self._llm_provider = get_llm_provider(
                self.llm_provider_name,
                model=self.llm_model
            )
        return self._llm_provider

    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider"""
        defaults = {
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4-turbo-preview",
            "ollama": "llama3.2:3b"
        }
        return defaults.get(provider, "claude-3-5-sonnet-20241022")

    async def pack_and_generate(
        self,
        query: str,
        awareness_context,
        memory_results: Optional[List[Any]] = None,
        max_memories: int = 10,
        use_fusion: Optional[bool] = None,
        # LLM overrides
        llm_config: Optional[LLMGenerationConfig] = None
    ) -> PackedGeneration:
        """
        Complete pipeline: Pack context → Generate with LLM → Extract feedback → Learn

        Args:
            query: User query
            awareness_context: UnifiedAwarenessContext from awareness layer
            memory_results: Optional memory retrieval results
            max_memories: Maximum memories to include
            use_fusion: Override memory fusion setting
            llm_config: Override LLM generation config

        Returns:
            PackedGeneration with complete results and feedback
        """
        # Phase 2: Calculate adaptive budget if enabled
        if self.enable_adaptive_budgeting and self.adaptive_budget_calculator:
            adaptive_budget = self.adaptive_budget_calculator.calculate_budget(
                query,
                awareness_context,
                available_memories=len(memory_results) if memory_results else 0
            )
            # Override base budget with adaptive budget
            self.budget = adaptive_budget.to_token_budget()

        # 1. Pack context (use base class)
        packed = await self.pack_context(
            query,
            awareness_context,
            memory_results,
            max_memories,
            use_fusion
        )

        # 1.5. Compress context (Phase 4) - NEW STEP
        compression_result = None
        compression_latency_ms = 0.0

        if self.enable_compression and self.semantic_deduplicator:
            import time
            compression_start = time.time()

            # Extract context elements from packed context
            context_elements = self._extract_elements_from_packed(packed)

            # Apply compression
            compression_result = self.semantic_deduplicator.deduplicate(
                context_elements,
                strategy=self.compression_strategy
            )

            # Replace packed context with compressed version
            packed = self._rebuild_packed_with_compressed(
                packed,
                compression_result.deduplicated_elements
            )

            compression_latency_ms = (time.time() - compression_start) * 1000

        # 2. Generate with LLM
        llm_response = await self._generate_with_llm(
            packed,
            llm_config or self.llm_config
        )

        # 3. Extract quality score
        quality_score = self._score_quality(
            query,
            llm_response,
            packed
        )

        # 4. Calculate token efficiency
        token_efficiency = quality_score.overall / max(llm_response.total_tokens, 1)

        # 5. Analyze context utilization
        context_utilization = self._analyze_utilization(
            packed,
            llm_response
        )

        # 6. Track compression metrics (Phase 4)
        if compression_result and self.compression_analyzer:
            # Track compression with before/after quality
            # Note: We don't have "before compression" quality, so we estimate
            # by assuming quality was similar (this is a simplification)
            self.compression_analyzer.track_compression(
                original_tokens=compression_result.original_tokens,
                compressed_tokens=compression_result.deduplicated_tokens,
                strategy=self.compression_strategy.value if hasattr(self.compression_strategy, 'value') else str(self.compression_strategy),
                quality_before=None,  # Unknown (could track in future)
                quality_after=quality_score.overall,
                latency_ms=compression_latency_ms,
                duplicate_groups_found=len(compression_result.duplicate_groups),
                elements_removed=compression_result.elements_removed,
                elements_merged=compression_result.elements_merged
            )

        # 7. Learn from outcome (if enabled)
        if self.enable_learning:
            self._learn_from_outcome(
                packed,
                quality_score,
                token_efficiency,
                context_utilization
            )

        # 8. Build result
        result = PackedGeneration(
            packed_context=packed,
            llm_response=llm_response,
            quality_score=quality_score,
            token_efficiency=token_efficiency,
            context_utilization=context_utilization,
            query=query
        )

        # Add compression metadata if compression was applied
        if compression_result:
            result.packed_context.metadata["compression"] = {
                "enabled": True,
                "strategy": str(self.compression_strategy),
                "original_tokens": compression_result.original_tokens,
                "compressed_tokens": compression_result.deduplicated_tokens,
                "token_savings": compression_result.token_savings,
                "compression_ratio": compression_result.compression_ratio,
                "elements_removed": compression_result.elements_removed,
                "elements_merged": compression_result.elements_merged,
                "latency_ms": compression_latency_ms
            }

        return result

    async def _generate_with_llm(
        self,
        packed: PackedContext,
        config: LLMGenerationConfig
    ) -> LLMResponse:
        """Generate LLM response from packed context"""
        # Get provider
        provider = self._get_llm_provider()

        # Format prompt for LLM
        prompt = packed.format_for_llm(include_metadata=False)

        # Generate
        response = await provider.generate(prompt, config)

        return response

    def _score_quality(
        self,
        query: str,
        response: LLMResponse,
        packed: PackedContext
    ) -> QualityScore:
        """
        Score response quality across multiple dimensions.

        This is a heuristic-based scoring system. For production,
        you could use:
        - LLM-based evaluation (GPT-4 as judge)
        - Human feedback (RLHF)
        - Reference-based metrics (BLEU, ROUGE, BERTScore)
        """
        text = response.text

        # 1. Coherence (0.0-1.0)
        # Heuristics:
        # - Has proper sentence structure
        # - No repetition
        # - Logical flow
        coherence = self._score_coherence(text)

        # 2. Completeness (0.0-1.0)
        # Heuristics:
        # - Response length (not too short, not too long)
        # - Addresses all parts of query
        # - Has conclusion/summary
        completeness = self._score_completeness(query, text)

        # 3. Relevance (0.0-1.0)
        # Heuristics:
        # - Contains query keywords
        # - Mentions concepts from packed context
        # - Stays on-topic
        relevance = self._score_relevance(query, text, packed)

        # 4. Accuracy (optional - requires ground truth)
        # For now, leave as None (could add verification step)
        accuracy = None

        return QualityScore(
            coherence=coherence,
            completeness=completeness,
            relevance=relevance,
            accuracy=accuracy
        )

    def _score_coherence(self, text: str) -> float:
        """Score coherence (0.0-1.0)"""
        score = 1.0

        # Check for sentence structure
        sentences = text.split('.')
        if len(sentences) < 2:
            score -= 0.2  # Too short/choppy

        # Check for repetition (simple check)
        words = text.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            uniqueness = len(unique_words) / len(words)
            if uniqueness < 0.5:
                score -= 0.3  # High repetition

        # Check for proper capitalization
        if not text[0].isupper():
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _score_completeness(self, query: str, text: str) -> float:
        """Score completeness (0.0-1.0)"""
        score = 0.0

        # Length-based scoring
        # Too short (<50 chars) = incomplete
        # Too long (>2000 chars) = rambling
        # Optimal: 200-800 chars
        length = len(text)

        if length < 50:
            score = 0.3
        elif length < 200:
            score = 0.6
        elif length <= 800:
            score = 1.0
        elif length <= 2000:
            score = 0.8
        else:
            score = 0.6

        # Check for question words in query
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        query_lower = query.lower()

        for word in question_words:
            if word in query_lower:
                # Check if response addresses it
                # (Simple heuristic: mentions the question word or related concepts)
                if word in text.lower():
                    score += 0.1

        return max(0.0, min(1.0, score))

    def _score_relevance(self, query: str, text: str, packed: PackedContext) -> float:
        """Score relevance (0.0-1.0)"""
        score = 0.0

        # Extract keywords from query
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        query_words = {w for w in query_words if len(w) > 3}  # Filter short words

        # Count keyword mentions in response
        text_lower = text.lower()
        keyword_mentions = sum(1 for word in query_words if word in text_lower)

        if len(query_words) > 0:
            keyword_coverage = keyword_mentions / len(query_words)
            score += keyword_coverage * 0.5  # 50% weight

        # Check for concepts from packed context
        # Extract key phrases from memory section
        memory_section = packed.memory_section
        if memory_section:
            memory_phrases = re.findall(r'\b\w+\b', memory_section.lower())
            memory_phrases = {w for w in memory_phrases if len(w) > 4}

            concept_mentions = sum(1 for phrase in memory_phrases if phrase in text_lower)
            if len(memory_phrases) > 0:
                concept_coverage = min(concept_mentions / len(memory_phrases), 1.0)
                score += concept_coverage * 0.5  # 50% weight

        return max(0.0, min(1.0, score))

    def _analyze_utilization(
        self,
        packed: PackedContext,
        response: LLMResponse
    ) -> ContextUtilization:
        """
        Analyze which context elements were actually used in the response.

        Strategy:
        - Check if element content appears in response
        - Count mentions per element
        - Calculate utilization rates by source
        """
        element_mentions = {}
        source_counts = defaultdict(lambda: {"total": 0, "utilized": 0})

        # Parse sections to identify elements
        # (In production, you'd track elements explicitly during packing)

        # For now, simple heuristic: check if section content appears in response
        response_lower = response.text.lower()

        # Check awareness section
        if packed.awareness_section:
            awareness_phrases = re.findall(r'\b\w{4,}\b', packed.awareness_section.lower())
            awareness_used = any(phrase in response_lower for phrase in awareness_phrases)
            source_counts["awareness"]["total"] = 1
            if awareness_used:
                source_counts["awareness"]["utilized"] = 1

        # Check memory section
        if packed.memory_section:
            # Split by bullet points (memories)
            memories = [m.strip() for m in packed.memory_section.split('-') if m.strip()]
            source_counts["memory"]["total"] = len(memories)

            for i, memory in enumerate(memories):
                memory_phrases = re.findall(r'\b\w{4,}\b', memory.lower())
                memory_used = any(phrase in response_lower for phrase in memory_phrases)

                if memory_used:
                    source_counts["memory"]["utilized"] += 1
                    element_mentions[f"memory_{i}"] = 1

        # Calculate source utilization rates
        source_utilization = {}
        for source, counts in source_counts.items():
            if counts["total"] > 0:
                source_utilization[source] = counts["utilized"] / counts["total"]
            else:
                source_utilization[source] = 0.0

        # Total stats
        total_packed = sum(counts["total"] for counts in source_counts.values())
        total_utilized = sum(counts["utilized"] for counts in source_counts.values())

        return ContextUtilization(
            element_mentions=element_mentions,
            source_utilization=source_utilization,
            total_elements_packed=total_packed,
            total_elements_utilized=total_utilized
        )

    def _learn_from_outcome(
        self,
        packed: PackedContext,
        quality: QualityScore,
        token_efficiency: float,
        utilization: ContextUtilization
    ) -> None:
        """
        Learn from outcome and adjust packing strategy.

        Tracks correlations:
        - Importance threshold → Quality
        - Compression level → Quality
        - Memory count → Quality
        - Token budget → Quality

        Adjusts:
        - self.min_importance (if low-importance elements hurt quality)
        - Compression thresholds (if over-compression hurts quality)
        """
        # Track importance → quality
        min_importance = packed.min_importance
        self._learning_stats["importance_quality"][min_importance].append(quality.overall)

        # Track compression → quality
        compression_rate = (
            packed.elements_compressed / max(packed.elements_included, 1)
        )
        self._learning_stats["compression_quality"][compression_rate].append(quality.overall)

        # Track memory count → quality (estimate from total elements)
        memory_count = packed.elements_included
        self._learning_stats["memory_count_quality"][memory_count].append(quality.overall)

        # Track budget → quality
        budget_used = packed.total_tokens
        self._learning_stats["budget_quality"][budget_used].append(quality.overall)

        # Phase 2: Feed budget performance to adaptive budget calculator
        if self.enable_adaptive_budgeting and self.adaptive_budget_calculator:
            # Create adaptive budget from packed context
            adaptive_budget_used = AdaptiveBudget(
                total=packed.total_tokens,
                reserved_for_query=0,  # Not tracked
                reserved_for_response=0,  # Not tracked
                model_max=self.adaptive_budget_calculator.model_info.context_window,
                recommended_max=self.adaptive_budget_calculator.model_info.recommended_max,
                reasoning=[],
                adjustments={},
                query_complexity=ComplexityLevel.MODERATE,  # Default
                complexity_score=0.5
            )
            self.adaptive_budget_calculator.learn_from_outcome(
                adaptive_budget_used,
                quality.overall
            )

        # Increment total interactions
        self._learning_stats["total_interactions"] += 1

        # Adapt strategies (every 10 interactions)
        if self._learning_stats["total_interactions"] % 10 == 0:
            self._adapt_strategies()

    def _adapt_strategies(self) -> None:
        """
        Adapt packing strategies based on learned correlations.

        Strategy:
        - If low-importance elements correlate with low quality → increase threshold
        - If high compression correlates with low quality → reduce compression
        - If more memories correlate with higher quality → increase memory limit
        """
        # Analyze importance threshold
        importance_stats = self._learning_stats["importance_quality"]
        if len(importance_stats) >= 2:
            # Find optimal importance threshold
            avg_qualities = {
                threshold: sum(qualities) / len(qualities)
                for threshold, qualities in importance_stats.items()
            }

            best_threshold = max(avg_qualities, key=avg_qualities.get)

            # Gradually move toward best threshold
            current = self.min_importance
            target = best_threshold
            self.min_importance = current + (target - current) * self.learning_rate

            # Clamp to reasonable range
            self.min_importance = max(0.1, min(0.8, self.min_importance))

        # (Could add more strategy adaptations here)

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning system statistics"""
        stats = {
            "total_interactions": self._learning_stats["total_interactions"],
            "current_importance_threshold": self.min_importance,
            "learning_enabled": self.enable_learning,
            "learning_rate": self.learning_rate
        }

        # Add average qualities by dimension
        for dimension, dimension_stats in self._learning_stats.items():
            if dimension == "total_interactions":
                continue

            if len(dimension_stats) > 0:
                # Calculate averages
                avg_by_value = {
                    value: sum(qualities) / len(qualities)
                    for value, qualities in dimension_stats.items()
                }

                stats[f"{dimension}_averages"] = avg_by_value

        return stats

    def get_provider_statistics(self) -> Dict[str, Any]:
        """Get LLM provider statistics"""
        if self._llm_provider is None:
            return {"provider": "not_initialized"}

        return self._llm_provider.get_statistics()

    # ========================================================================
    # Phase 4: Compression Helper Methods
    # ========================================================================

    def _extract_elements_from_packed(self, packed: PackedContext) -> List[ContextElement]:
        """
        Extract context elements from packed context.

        This is a reconstruction step - we extract elements from the
        packed sections to enable compression.
        """
        elements = []

        # Extract from awareness section
        if packed.awareness_section:
            elements.append(ContextElement(
                content=packed.awareness_section,
                importance=1.0,  # High importance
                token_count=len(packed.awareness_section) // 4,
                source="awareness",
                metadata={"section": "awareness"}
            ))

        # Extract from memory section (split by newlines)
        if packed.memory_section:
            # Split memories (simple heuristic: split by double newlines)
            memory_chunks = packed.memory_section.split('\n\n')
            for i, chunk in enumerate(memory_chunks):
                if chunk.strip():
                    elements.append(ContextElement(
                        content=chunk.strip(),
                        importance=0.8,  # Medium-high importance
                        token_count=len(chunk) // 4,
                        source="memory",
                        metadata={"section": "memory", "index": i}
                    ))

        # Extract from pattern section
        if packed.pattern_section:
            elements.append(ContextElement(
                content=packed.pattern_section,
                importance=0.6,  # Medium importance
                token_count=len(packed.pattern_section) // 4,
                source="pattern",
                metadata={"section": "pattern"}
            ))

        return elements

    def _rebuild_packed_with_compressed(
        self,
        original_packed: PackedContext,
        compressed_elements: List[ContextElement]
    ) -> PackedContext:
        """
        Rebuild PackedContext with compressed elements.

        Reconstructs the packed context sections from compressed elements.
        """
        # Separate elements by source
        awareness_elements = [e for e in compressed_elements if e.source == "awareness"]
        memory_elements = [e for e in compressed_elements if e.source == "memory"]
        pattern_elements = [e for e in compressed_elements if e.source == "pattern"]

        # Rebuild sections
        awareness_section = "\n\n".join(e.content for e in awareness_elements) if awareness_elements else ""
        memory_section = "\n\n".join(e.content for e in memory_elements) if memory_elements else ""
        pattern_section = "\n\n".join(e.content for e in pattern_elements) if pattern_elements else ""

        # Calculate new token count
        total_tokens = sum(e.token_count for e in compressed_elements)
        total_tokens += len(original_packed.query_section) // 4  # Add query tokens

        # Create new packed context
        return PackedContext(
            awareness_section=awareness_section,
            memory_section=memory_section,
            pattern_section=pattern_section,
            query_section=original_packed.query_section,  # Keep original query
            total_tokens=total_tokens,
            elements_included=len(compressed_elements),
            packing_time_ms=original_packed.packing_time_ms,  # Keep original packing time
            metadata=original_packed.metadata.copy()  # Copy metadata
        )

    def get_compression_statistics(self) -> Optional[Dict[str, Any]]:
        """Get compression statistics (Phase 4)"""
        if not self.enable_compression or not self.compression_analyzer:
            return None

        return self.compression_analyzer.get_statistics()

    def get_compression_insights(self) -> Optional[Any]:
        """Get compression insights (Phase 4)"""
        if not self.enable_compression or not self.compression_analyzer:
            return None

        return self.compression_analyzer.get_insights()

    def get_compression_recommendations(self) -> Optional[List[str]]:
        """Get compression recommendations (Phase 4)"""
        if not self.enable_compression or not self.compression_analyzer:
            return None

        return self.compression_analyzer.get_recommendations()

    # ========================================================================
    # Phase 5: Multi-Pass Refinement
    # ========================================================================

    async def pack_and_generate_with_refinement(
        self,
        query: str,
        awareness_context,
        memory_results: Optional[List[Any]] = None,
        max_memories: int = 10,
        # Refinement configuration
        quality_threshold: float = 0.85,  # Refine if quality < 0.85
        max_passes: int = 3,
        refinement_strategy: str = "adaptive",  # "adaptive", "depth_first", "breadth_first", "focused"
        force_refinement: bool = False  # Force refinement even if quality acceptable
    ) -> Any:  # Returns RefinementResult
        """
        Pack, generate, and iteratively refine until quality acceptable.

        This is the simplest way to use Phase 5 refinement. It handles:
        - Initial generation
        - Quality checking
        - Automatic refinement passes
        - Intelligent stopping criteria

        Args:
            query: User query
            awareness_context: Awareness context
            memory_results: Memory retrieval results
            max_memories: Maximum memories to include
            quality_threshold: Minimum quality to accept (default: 0.85)
            max_passes: Maximum refinement passes (default: 3)
            refinement_strategy: Refinement strategy
            force_refinement: Force refinement even if quality acceptable

        Returns:
            RefinementResult with complete refinement history

        Example:
            result = await packer.pack_and_generate_with_refinement(
                query="Complex research question",
                awareness_ctx=ctx,
                memory_results=memories,
                quality_threshold=0.90,  # High quality bar
                max_passes=3
            )

            print(f"Quality: {result.initial_quality:.2f} → {result.final_quality:.2f}")
            print(f"Passes: {result.passes_executed}")
        """
        if not REFINEMENT_AVAILABLE:
            logger.warning("Refinement requested but not available (import failed)")
            # Fall back to single pass
            generation = await self.pack_and_generate(
                query, awareness_context, memory_results, max_memories
            )

            # Wrap in RefinementResult-like structure
            class FallbackResult:
                def __init__(self, gen):
                    self.query = query
                    self.initial_generation = gen
                    self.final_generation = gen
                    self.passes = []
                    self.initial_quality = gen.quality_score.overall
                    self.final_quality = gen.quality_score.overall
                    self.total_improvement = 0.0
                    self.passes_executed = 0
                    self.stopping_criterion = "quality_threshold"

                def summary(self):
                    return f"Refinement unavailable, single pass: quality {self.final_quality:.2f}"

            return FallbackResult(generation)

        # Create refinement engine
        strategy_map = {
            "adaptive": RefinementStrategy.ADAPTIVE,
            "depth_first": RefinementStrategy.DEPTH_FIRST,
            "breadth_first": RefinementStrategy.BREADTH_FIRST,
            "focused": RefinementStrategy.FOCUSED
        }

        strategy = strategy_map.get(refinement_strategy, RefinementStrategy.ADAPTIVE)

        refiner = RefinementEngine(
            packer=self,
            quality_threshold=quality_threshold,
            max_passes=max_passes,
            strategy=strategy,
            enable_compression_disable=True,
            enable_budget_increase=True
        )

        # Execute refinement
        result = await refiner.refine(
            query=query,
            awareness_ctx=awareness_context,
            memory_results=memory_results,
            max_memories=max_memories,
            force_refinement=force_refinement
        )

        return result

    # ========================================================================
    # Phase 6.1: USER_FEEDBACK Refinement
    # ========================================================================

    async def pack_and_generate_with_feedback_learning(
        self,
        query: str,
        awareness_context,
        memory_results: Optional[List[Any]] = None,
        max_memories: int = 10,
        # Refinement configuration
        quality_threshold: float = 0.85,
        max_passes: int = 3,
        # Feedback configuration
        feedback_tracker: Optional[Any] = None,  # FeedbackTracker instance
        enable_feedback_learning: bool = True
    ) -> Any:  # Returns FeedbackAwareResult
        """
        Pack, generate, and refine with feedback-based learning.

        Phase 6.1 extension that learns from user feedback to improve
        future refinements. The system:
        - Classifies query type
        - Selects strategy based on past feedback for similar queries
        - Tracks feedback to improve over time
        - Adapts to user preferences

        Args:
            query: User query
            awareness_context: Awareness context
            memory_results: Memory retrieval results
            max_memories: Maximum memories to include
            quality_threshold: Minimum quality to accept (default: 0.85)
            max_passes: Maximum refinement passes (default: 3)
            feedback_tracker: Optional shared feedback tracker
            enable_feedback_learning: Enable feedback-based learning

        Returns:
            FeedbackAwareResult with query type and recommendations

        Example:
            # First query
            result1 = await packer.pack_and_generate_with_feedback_learning(
                query="Explain Thompson Sampling",
                awareness_ctx=ctx
            )

            # Track user feedback
            from HoloLoom.awareness.feedback_tracker import FeedbackSignal, FeedbackType
            feedback = FeedbackSignal(
                feedback_type=FeedbackType.THUMBS_UP,
                helpful=True
            )

            # This requires accessing the refiner (see Phase 6.1 docs for full example)
            # For now, user should use UserFeedbackRefiner directly for feedback tracking

            # Second similar query - automatically uses learned strategy
            result2 = await packer.pack_and_generate_with_feedback_learning(
                query="Explain Epsilon-Greedy",  # Similar to Thompson Sampling
                awareness_ctx=ctx,
                feedback_tracker=result1.feedback_tracker  # Shared learning
            )

        Note:
            For full feedback tracking, use UserFeedbackRefiner directly.
            This method provides simplified access with automatic strategy selection.
        """
        if not REFINEMENT_AVAILABLE:
            # Graceful fallback
            return await self.pack_and_generate(
                query=query,
                awareness_context=awareness_context,
                memory_results=memory_results,
                max_memories=max_memories
            )

        # Import Phase 6.1 components
        try:
            from .user_feedback_refiner import UserFeedbackRefiner
        except ImportError:
            # Phase 6.1 not available, fall back to Phase 5
            return await self.pack_and_generate_with_refinement(
                query=query,
                awareness_context=awareness_context,
                memory_results=memory_results,
                max_memories=max_memories,
                quality_threshold=quality_threshold,
                max_passes=max_passes
            )

        # Create feedback-aware refiner
        refiner = UserFeedbackRefiner(
            packer=self,
            quality_threshold=quality_threshold,
            max_passes=max_passes,
            enable_feedback_learning=enable_feedback_learning,
            feedback_tracker=feedback_tracker
        )

        # Execute refinement with feedback learning
        result = await refiner.refine(
            query=query,
            awareness_ctx=awareness_context,
            memory_results=memory_results,
            max_memories=max_memories
        )

        # Attach refiner for feedback tracking (optional)
        result.refiner = refiner

        return result

    async def pack_and_generate_with_consensus(
        self,
        query: str,
        awareness_context,
        memory_results: Optional[List[Any]] = None,
        max_memories: int = 10,
        strategies: Optional[List[Any]] = None,
        voting_method: str = "quality_weighted",
        quality_threshold: float = 0.85,
        max_passes: int = 3,
        require_unanimity: bool = False,
        unanimity_threshold: float = 0.8,
        enable_disagreement_detection: bool = True
    ) -> Any:  # Returns ConsensusResult
        """
        Pack, generate, and refine with parallel consensus (Phase 6.2).

        Executes multiple refinement strategies in parallel and uses
        ensemble voting to select the best result.

        Args:
            query: Query text
            awareness_context: Awareness context
            memory_results: Memory retrieval results
            max_memories: Maximum memories to include
            strategies: List of strategies to run in parallel
                       (default: [DEPTH_FIRST, BREADTH_FIRST, FOCUSED])
            voting_method: Ensemble voting method
                          ('quality_weighted', 'diversity', 'unanimous', 'best_of_n')
            quality_threshold: Quality threshold for refinement
            max_passes: Maximum refinement passes per strategy
            require_unanimity: Whether to require unanimous agreement
            unanimity_threshold: Agreement level required (0-1)
            enable_disagreement_detection: Whether to detect disagreements

        Returns:
            ConsensusResult with selected result and consensus metadata

        Raises:
            ImportError: If Phase 6.2 (consensus) not available
            ValueError: If all strategies fail

        Example:
            result = await packer.pack_and_generate_with_consensus(
                query="Explain Thompson Sampling",
                awareness_context=ctx,
                memory_results=memories,
                voting_method="quality_weighted"
            )

            print(f"Consensus: {result.consensus_confidence:.2f}")
            print(f"Agreement: {result.agreement_level:.1%}")
            print(f"Speedup: {result.parallel_speedup:.1f}x")
        """
        # Fall back to Phase 5 if consensus not available
        if not REFINEMENT_AVAILABLE:
            return await self.pack_and_generate(
                query=query,
                awareness_context=awareness_context,
                memory_results=memory_results,
                max_memories=max_memories
            )

        # Try to import consensus refiner (Phase 6.2)
        try:
            from .consensus_refiner import ConsensusRefiner, VotingMethod
        except ImportError:
            # Fall back to Phase 6.1 (feedback) or Phase 5 (refinement)
            return await self.pack_and_generate_with_refinement(
                query=query,
                awareness_context=awareness_context,
                memory_results=memory_results,
                max_memories=max_memories,
                quality_threshold=quality_threshold,
                max_passes=max_passes
            )

        # Convert voting method string to enum
        try:
            voting_enum = VotingMethod(voting_method)
        except ValueError:
            voting_enum = VotingMethod.QUALITY_WEIGHTED

        # Create consensus refiner
        refiner = ConsensusRefiner(
            packer=self,
            strategies=strategies,
            voting_method=voting_enum,
            quality_threshold=quality_threshold,
            max_passes=max_passes,
            require_unanimity=require_unanimity,
            unanimity_threshold=unanimity_threshold,
            enable_disagreement_detection=enable_disagreement_detection
        )

        # Execute consensus refinement
        result = await refiner.refine(
            query=query,
            awareness_ctx=awareness_context,
            memory_results=memory_results,
            max_memories=max_memories
        )

        return result
