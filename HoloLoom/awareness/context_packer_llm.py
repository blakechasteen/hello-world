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
from collections import defaultdict

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
        learning_rate: float = 0.1  # How fast to adapt strategies
    ):
        """
        Initialize LLM-integrated context packer.

        Args:
            token_budget: Token budget constraints
            min_importance_threshold: Minimum importance to include
            use_memory_fusion: Enable multipass memory fusion
            memory_backend: Backend for memory fusion
            llm_provider: LLM provider ("anthropic", "openai", "ollama")
            llm_model: Model identifier (optional, uses default)
            llm_config: LLM generation config
            enable_learning: Enable learning from outcomes
            learning_rate: How fast to adapt (0.0-1.0)
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
        # 1. Pack context (use base class)
        packed = await self.pack_context(
            query,
            awareness_context,
            memory_results,
            max_memories,
            use_fusion
        )

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

        # 6. Learn from outcome (if enabled)
        if self.enable_learning:
            self._learn_from_outcome(
                packed,
                quality_score,
                token_efficiency,
                context_utilization
            )

        # 7. Build result
        return PackedGeneration(
            packed_context=packed,
            llm_response=llm_response,
            quality_score=quality_score,
            token_efficiency=token_efficiency,
            context_utilization=context_utilization,
            query=query
        )

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
