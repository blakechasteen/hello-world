from __future__ import annotations
"""
Dark Trace SAE: Auto-Label Engine
=================================

LLM-powered automatic feature labeling using UnifiedLLMClient.
Extends the base FeatureLabeler with multi-provider support, batch processing,
quality assessment, and verification workflows.

Features:
- Multi-provider LLM support (Ollama, Anthropic, OpenAI, Google)
- Automatic fallback on failure
- Batch processing for efficiency
- Quality assessment and filtering
- Verification/refinement workflow
- Integration with FeatureRegistry

Created: 2025-12-28
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hololoom.dark_trace.sae.label_prompts import (
    LabelPromptBuilder,
    LabelPromptContext,
    LabelResponseParser,
)
from hololoom.dark_trace.sae.labeler import (
    ActivationPatternAnalyzer,
    FeatureLabel,
    LabelCache,
    SemanticAlignmentScorer,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class LabelingStrategy(Enum):
    """Strategy for labeling features."""
    SINGLE = "single"          # Label one feature at a time
    BATCH = "batch"            # Batch multiple features
    HIERARCHICAL = "hierarchical"  # Group similar features first


class QualityLevel(Enum):
    """Quality level for labels."""
    HIGH = "high"          # Confidence >= 0.8
    MEDIUM = "medium"      # Confidence 0.5-0.8
    LOW = "low"            # Confidence < 0.5
    FAILED = "failed"      # Labeling failed


@dataclass
class AutoLabelConfig:
    """
    Configuration for auto-labeling engine.

    Args:
        llm_provider: Primary LLM provider (ollama/anthropic/openai/google)
        llm_model: Model name for the provider
        fallback_provider: Fallback provider (optional)
        fallback_model: Fallback model (optional)
        strategy: Labeling strategy (single/batch/hierarchical)
        batch_size: Features per batch (for batch strategy)
        max_tokens: Max tokens for LLM response
        temperature: LLM sampling temperature
        min_confidence: Minimum confidence to accept a label
        verify_low_confidence: Re-verify labels below this threshold
        max_retries: Max retry attempts per feature
        cache_dir: Directory for label cache
    """
    # LLM settings
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2:3b"
    fallback_provider: str | None = None
    fallback_model: str | None = None

    # Labeling strategy
    strategy: LabelingStrategy = LabelingStrategy.SINGLE
    batch_size: int = 5
    max_tokens: int = 500
    temperature: float = 0.3  # Lower for more consistent labels

    # Quality control
    min_confidence: float = 0.3
    verify_low_confidence: float = 0.5
    max_retries: int = 2

    # Storage
    cache_dir: Path | None = None

    # Examples
    top_k_examples: int = 10
    max_example_length: int = 200

    # Concurrency
    max_concurrent_batches: int = 3

    @classmethod
    def fast(cls) -> "AutoLabelConfig":
        """Fast configuration with local models."""
        return cls(
            llm_provider="ollama",
            llm_model="llama3.2:3b",
            strategy=LabelingStrategy.BATCH,
            batch_size=10,
            max_tokens=300,
            temperature=0.2,
        )

    @classmethod
    def quality(cls) -> "AutoLabelConfig":
        """Quality configuration with Claude."""
        return cls(
            llm_provider="anthropic",
            llm_model="claude-3-5-sonnet-20241022",
            fallback_provider="ollama",
            fallback_model="llama3.2:3b",
            strategy=LabelingStrategy.SINGLE,
            batch_size=5,
            max_tokens=500,
            temperature=0.3,
            verify_low_confidence=0.6,
        )


# =============================================================================
# Quality Assessment
# =============================================================================

@dataclass
class QualityAssessment:
    """Assessment of a label's quality."""
    quality_level: QualityLevel
    overall_score: float  # 0.0-1.0
    confidence_score: float
    evidence_score: float
    coherence_score: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class LabelQualityAssessor:
    """
    Assess the quality of feature labels.

    Evaluates:
    - Confidence: LLM's self-reported confidence
    - Evidence: Number and quality of supporting examples
    - Coherence: Semantic alignment correlation
    - Specificity: Is the label specific enough?
    """

    def __init__(
        self,
        min_examples: int = 3,
        min_alignment_score: float = 0.2,
    ):
        self.min_examples = min_examples
        self.min_alignment_score = min_alignment_score

    def assess(
        self,
        label: FeatureLabel,
        context: LabelPromptContext | None = None,
    ) -> QualityAssessment:
        """
        Assess the quality of a feature label.

        Args:
            label: The feature label to assess
            context: Optional context with additional data

        Returns:
            QualityAssessment with scores and feedback
        """
        issues = []
        suggestions = []

        # 1. Confidence score (from LLM)
        confidence_score = label.confidence

        # 2. Evidence score (based on examples)
        n_examples = len(label.top_activating_examples)
        if n_examples >= self.min_examples * 2:
            evidence_score = 1.0
        elif n_examples >= self.min_examples:
            evidence_score = 0.7
        elif n_examples > 0:
            evidence_score = 0.4
            issues.append(f"Few supporting examples ({n_examples})")
            suggestions.append("Collect more activation examples")
        else:
            evidence_score = 0.1
            issues.append("No supporting examples")
            suggestions.append("Feature may be dead or very sparse")

        # 3. Coherence score (semantic alignment)
        coherence_score = label.semantic_alignment_score
        if coherence_score < self.min_alignment_score:
            issues.append(f"Low semantic alignment ({coherence_score:.2f})")
            suggestions.append("Feature may not align with interpretable dimensions")

        # 4. Specificity check
        description_lower = label.description.lower()
        generic_patterns = [
            "no strongly activating",
            "not available",
            "unclear",
            "unknown",
            "[llm",
        ]
        if any(pattern in description_lower for pattern in generic_patterns):
            confidence_score *= 0.5
            issues.append("Generic or failed description")
            suggestions.append("Retry with different prompt or more examples")

        # Calculate overall score
        overall_score = (
            0.4 * confidence_score +
            0.3 * evidence_score +
            0.3 * coherence_score
        )

        # Determine quality level
        if overall_score >= 0.8:
            quality_level = QualityLevel.HIGH
        elif overall_score >= 0.5:
            quality_level = QualityLevel.MEDIUM
        elif overall_score >= 0.3:
            quality_level = QualityLevel.LOW
        else:
            quality_level = QualityLevel.FAILED

        return QualityAssessment(
            quality_level=quality_level,
            overall_score=overall_score,
            confidence_score=confidence_score,
            evidence_score=evidence_score,
            coherence_score=coherence_score,
            issues=issues,
            suggestions=suggestions,
        )

    def filter_by_quality(
        self,
        labels: dict[int, FeatureLabel],
        min_level: QualityLevel = QualityLevel.LOW,
    ) -> dict[int, FeatureLabel]:
        """
        Filter labels by quality level.

        Args:
            labels: Dict of feature_idx -> FeatureLabel
            min_level: Minimum quality level to include

        Returns:
            Filtered dict of labels meeting quality threshold
        """
        level_order = [QualityLevel.FAILED, QualityLevel.LOW, QualityLevel.MEDIUM, QualityLevel.HIGH]
        min_idx = level_order.index(min_level)

        filtered = {}
        for idx, label in labels.items():
            assessment = self.assess(label)
            if level_order.index(assessment.quality_level) >= min_idx:
                filtered[idx] = label

        return filtered


# =============================================================================
# Auto-Label Engine
# =============================================================================

class AutoLabelEngine:
    """
    LLM-powered automatic feature labeling engine.

    Extends the base FeatureLabeler with:
    - Multi-provider LLM support via UnifiedLLMClient
    - Batch processing for efficiency
    - Quality assessment and filtering
    - Verification/refinement workflow
    - Integration with FeatureRegistry

    Usage:
        config = AutoLabelConfig.quality()
        engine = AutoLabelEngine(config)

        # Collect examples
        engine.collect_examples(sae, activations, texts)

        # Label features
        labels = await engine.label_features(
            sae,
            feature_indices=[0, 1, 2, 3, 4],
        )

        # Verify low-confidence labels
        verified = await engine.verify_labels(labels)

        # Get quality report
        report = engine.get_quality_report(verified)
    """

    def __init__(
        self,
        config: AutoLabelConfig | None = None,
        registry: Any | None = None,  # FeatureRegistry
    ):
        """
        Initialize auto-labeling engine.

        Args:
            config: Configuration settings
            registry: Optional FeatureRegistry for storing labels
        """
        self.config = config or AutoLabelConfig()
        self.registry = registry

        # Initialize LLM client
        self._llm_client = None
        self._init_llm_client()

        # Components
        self.analyzer = ActivationPatternAnalyzer(
            top_k=self.config.top_k_examples
        )
        self.alignment_scorer = SemanticAlignmentScorer()
        self.cache = LabelCache(cache_dir=self.config.cache_dir)

        # Prompt building and parsing
        self.prompt_builder = LabelPromptBuilder()
        self.response_parser = LabelResponseParser()

        # Quality assessment
        self.quality_assessor = LabelQualityAssessor()

        # Statistics
        self._stats = {
            "total_labeled": 0,
            "batch_calls": 0,
            "single_calls": 0,
            "retries": 0,
            "failures": 0,
            "verifications": 0,
        }

        logger.info(
            f"AutoLabelEngine initialized: provider={self.config.llm_provider}, "
            f"strategy={self.config.strategy.value}"
        )

    def _init_llm_client(self) -> None:
        """Initialize the LLM client."""
        try:
            from hololoom.llm import LLMConfig, UnifiedLLMClient

            primary = LLMConfig(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
            )

            fallbacks = []
            if self.config.fallback_provider and self.config.fallback_model:
                fallbacks.append(LLMConfig(
                    provider=self.config.fallback_provider,
                    model=self.config.fallback_model,
                ))

            self._llm_client = UnifiedLLMClient(
                primary=primary,
                fallbacks=fallbacks,
                enable_cost_tracking=True,
            )

            logger.info(f"LLM client initialized: {primary.full_name}")

        except ImportError:
            logger.warning("UnifiedLLMClient not available, falling back to direct ollama")
            self._llm_client = None

    def set_semantic_axes(self, axes: dict[str, np.ndarray]) -> None:
        """Set semantic axes for alignment scoring."""
        self.alignment_scorer.set_semantic_axes(axes)

    def collect_examples(
        self,
        sae: torch.nn.Module,
        activations: torch.Tensor,
        example_texts: list[str],
    ) -> None:
        """
        Collect activation examples for labeling.

        Args:
            sae: The SAE model
            activations: Input activations [batch, input_dim]
            example_texts: Text corresponding to each activation
        """
        sae.eval()
        with torch.no_grad():
            _, latents, _ = sae(activations)

        # Collect for both analyzer and alignment scorer
        self.analyzer.collect_activations(sae, activations, example_texts)
        self.alignment_scorer.collect_projections(activations, latents)

    async def label_features(
        self,
        sae: torch.nn.Module,
        feature_indices: list[int] | None = None,
        force_relabel: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[int, FeatureLabel]:
        """
        Label features using LLM.

        Args:
            sae: The SAE model
            feature_indices: Which features to label (None = all)
            force_relabel: Relabel even if cached
            progress_callback: Callback(current, total) for progress

        Returns:
            Dict mapping feature index to FeatureLabel
        """
        # Determine features to label
        n_features = sae.latent_dim if hasattr(sae, 'latent_dim') else sae.decoder.weight.shape[0]

        if feature_indices is None:
            feature_indices = list(range(n_features))

        # Filter cached unless force_relabel
        if not force_relabel:
            feature_indices = [
                idx for idx in feature_indices
                if not self.cache.has(idx)
            ]

        if not feature_indices:
            logger.info("All features already cached")
            return self.cache.get_all()

        logger.info(f"Labeling {len(feature_indices)} features...")

        # Choose strategy
        if self.config.strategy == LabelingStrategy.BATCH:
            results = await self._label_batch(
                feature_indices, progress_callback
            )
        elif self.config.strategy == LabelingStrategy.HIERARCHICAL:
            results = await self._label_hierarchical(
                feature_indices, progress_callback
            )
        else:  # SINGLE
            results = await self._label_single(
                feature_indices, progress_callback
            )

        # Cache results
        for idx, label in results.items():
            self.cache.set(label)
        self.cache.save()

        # Update registry if available
        if self.registry:
            self._update_registry(results)

        # Merge with cached
        all_labels = self.cache.get_all()
        all_labels.update(results)

        self._stats["total_labeled"] += len(results)

        return all_labels

    async def _label_single(
        self,
        feature_indices: list[int],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[int, FeatureLabel]:
        """Label features one at a time."""
        results = {}
        total = len(feature_indices)

        for i, idx in enumerate(feature_indices):
            try:
                label = await self._label_single_feature(idx)
                results[idx] = label
                self._stats["single_calls"] += 1
            except Exception as e:
                logger.warning(f"Failed to label feature {idx}: {e}")
                self._stats["failures"] += 1

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    async def _label_batch(
        self,
        feature_indices: list[int],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[int, FeatureLabel]:
        """Label features in batches."""
        results = {}
        batch_size = self.config.batch_size
        total = len(feature_indices)

        batches = [
            feature_indices[i:i + batch_size]
            for i in range(0, len(feature_indices), batch_size)
        ]

        labeled_count = 0
        for batch in batches:
            try:
                batch_results = await self._label_feature_batch(batch)
                results.update(batch_results)
                self._stats["batch_calls"] += 1
            except Exception as e:
                logger.warning(f"Batch labeling failed: {e}")
                # Fallback to single labeling
                for idx in batch:
                    try:
                        label = await self._label_single_feature(idx)
                        results[idx] = label
                    except Exception as e2:
                        logger.warning(f"Single label fallback failed for {idx}: {e2}")
                        self._stats["failures"] += 1

            labeled_count += len(batch)
            if progress_callback:
                progress_callback(labeled_count, total)

        return results

    async def _label_hierarchical(
        self,
        feature_indices: list[int],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[int, FeatureLabel]:
        """Label features hierarchically (group similar first)."""
        # First, get activation patterns to cluster
        patterns = {}
        for idx in feature_indices:
            _, act_values = self.analyzer.get_top_activating_examples(idx)
            if act_values:
                patterns[idx] = act_values[:5]  # Top 5 activation values

        # Simple grouping by activation magnitude
        groups = {"high": [], "medium": [], "low": []}
        for idx, vals in patterns.items():
            max_val = max(vals) if vals else 0
            if max_val > 0.7:
                groups["high"].append(idx)
            elif max_val > 0.3:
                groups["medium"].append(idx)
            else:
                groups["low"].append(idx)

        # Add ungrouped features
        ungrouped = set(feature_indices) - set(patterns.keys())
        groups["low"].extend(ungrouped)

        # Label each group
        results = {}
        total = len(feature_indices)
        labeled_count = 0

        for group_name, group_indices in groups.items():
            if not group_indices:
                continue

            logger.info(f"Labeling {len(group_indices)} {group_name}-activation features")

            # Use batch for groups
            group_results = await self._label_batch(group_indices, None)
            results.update(group_results)

            labeled_count += len(group_indices)
            if progress_callback:
                progress_callback(labeled_count, total)

        return results

    async def _label_single_feature(
        self,
        feature_idx: int,
        retry_count: int = 0,
    ) -> FeatureLabel:
        """Label a single feature."""
        # Get top activating examples
        examples, act_values = self.analyzer.get_top_activating_examples(feature_idx)
        activation_pattern = self.analyzer.get_activation_pattern_summary(feature_idx)

        # Get semantic alignments
        alignments = self.alignment_scorer.score_feature_alignment(feature_idx)
        top_alignments = self.alignment_scorer.get_top_alignments(feature_idx)

        primary_axis = top_alignments[0][0] if top_alignments else None
        alignment_score = abs(top_alignments[0][1]) if top_alignments else 0.0

        # Build prompt context
        context = LabelPromptContext(
            feature_idx=feature_idx,
            examples=examples,
            activation_values=act_values,
            activation_pattern=activation_pattern,
            semantic_alignments=alignments,
            primary_semantic_axis=primary_axis,
            alignment_score=alignment_score,
            max_example_length=self.config.max_example_length,
        )

        # Build and send prompt
        prompt = self.prompt_builder.build_single_feature_prompt(context)
        response_text = await self._call_llm(prompt)

        # Parse response
        parsed = self.response_parser.parse_single_response(response_text)

        # Check confidence and maybe retry
        if parsed.confidence < self.config.min_confidence:
            if retry_count < self.config.max_retries:
                self._stats["retries"] += 1
                return await self._label_single_feature(feature_idx, retry_count + 1)

        return FeatureLabel(
            feature_idx=feature_idx,
            description=parsed.description,
            confidence=parsed.confidence,
            top_activating_examples=examples[:5],
            activation_pattern=activation_pattern,
            semantic_alignments=alignments,
            primary_semantic_axis=primary_axis,
            semantic_alignment_score=alignment_score,
            labeled_at=datetime.now().isoformat(),
            labeler_model=f"{self.config.llm_provider}/{self.config.llm_model}",
        )

    async def _label_feature_batch(
        self,
        feature_indices: list[int],
    ) -> dict[int, FeatureLabel]:
        """Label a batch of features."""
        contexts = []

        for idx in feature_indices:
            examples, act_values = self.analyzer.get_top_activating_examples(idx)
            activation_pattern = self.analyzer.get_activation_pattern_summary(idx)
            top_alignments = self.alignment_scorer.get_top_alignments(idx)

            context = LabelPromptContext(
                feature_idx=idx,
                examples=examples,
                activation_values=act_values,
                activation_pattern=activation_pattern,
                primary_semantic_axis=top_alignments[0][0] if top_alignments else None,
                alignment_score=abs(top_alignments[0][1]) if top_alignments else 0.0,
            )
            contexts.append(context)

        # Build batch prompt
        prompt = self.prompt_builder.build_batch_prompt(contexts)
        response_text = await self._call_llm(prompt)

        # Parse response
        parsed_list = self.response_parser.parse_batch_response(response_text)

        # Convert to FeatureLabels
        results = {}
        for parsed in parsed_list:
            idx = parsed["feature_idx"]
            if idx in feature_indices:
                # Get additional context
                context = next((c for c in contexts if c.feature_idx == idx), None)

                results[idx] = FeatureLabel(
                    feature_idx=idx,
                    description=parsed["description"],
                    confidence=parsed["confidence"],
                    top_activating_examples=context.examples[:5] if context else [],
                    activation_pattern=context.activation_pattern if context else "",
                    semantic_alignments=self.alignment_scorer.score_feature_alignment(idx),
                    primary_semantic_axis=context.primary_semantic_axis if context else None,
                    semantic_alignment_score=context.alignment_score if context else 0.0,
                    labeled_at=datetime.now().isoformat(),
                    labeler_model=f"{self.config.llm_provider}/{self.config.llm_model}",
                )

        return results

    async def verify_labels(
        self,
        labels: dict[int, FeatureLabel],
        confidence_threshold: float | None = None,
    ) -> dict[int, FeatureLabel]:
        """
        Verify and refine low-confidence labels.

        Args:
            labels: Labels to verify
            confidence_threshold: Verify labels below this threshold

        Returns:
            Dict with verified/refined labels
        """
        threshold = confidence_threshold or self.config.verify_low_confidence

        to_verify = [
            (idx, label) for idx, label in labels.items()
            if label.confidence < threshold
        ]

        if not to_verify:
            logger.info("No labels need verification")
            return labels

        logger.info(f"Verifying {len(to_verify)} low-confidence labels...")

        verified = labels.copy()

        for idx, label in to_verify:
            try:
                refined = await self._verify_single_label(idx, label)
                verified[idx] = refined
                self._stats["verifications"] += 1
            except Exception as e:
                logger.warning(f"Verification failed for {idx}: {e}")

        # Update cache
        for idx, label in verified.items():
            self.cache.set(label)
        self.cache.save()

        return verified

    async def _verify_single_label(
        self,
        feature_idx: int,
        current_label: FeatureLabel,
    ) -> FeatureLabel:
        """Verify/refine a single label."""
        examples, act_values = self.analyzer.get_top_activating_examples(feature_idx)
        activation_pattern = self.analyzer.get_activation_pattern_summary(feature_idx)
        alignments = self.alignment_scorer.score_feature_alignment(feature_idx)

        context = LabelPromptContext(
            feature_idx=feature_idx,
            examples=examples,
            activation_values=act_values,
            activation_pattern=activation_pattern,
            semantic_alignments=alignments,
            current_description=current_label.description,
            current_confidence=current_label.confidence,
        )

        prompt = self.prompt_builder.build_verification_prompt(context)
        response_text = await self._call_llm(prompt)

        parsed = self.response_parser.parse_verification_response(response_text)

        if parsed.verdict == "VERIFIED":
            # Keep original with boosted confidence
            return FeatureLabel(
                feature_idx=feature_idx,
                description=current_label.description,
                confidence=min(1.0, current_label.confidence + 0.1),
                top_activating_examples=current_label.top_activating_examples,
                activation_pattern=current_label.activation_pattern,
                semantic_alignments=current_label.semantic_alignments,
                primary_semantic_axis=current_label.primary_semantic_axis,
                semantic_alignment_score=current_label.semantic_alignment_score,
                labeled_at=datetime.now().isoformat(),
                labeler_model=f"{self.config.llm_provider}/{self.config.llm_model}",
                version="1.1",  # Mark as verified
            )
        elif parsed.verdict == "REFINED":
            return FeatureLabel(
                feature_idx=feature_idx,
                description=parsed.description,
                confidence=parsed.confidence,
                top_activating_examples=examples[:5],
                activation_pattern=activation_pattern,
                semantic_alignments=alignments,
                primary_semantic_axis=current_label.primary_semantic_axis,
                semantic_alignment_score=current_label.semantic_alignment_score,
                labeled_at=datetime.now().isoformat(),
                labeler_model=f"{self.config.llm_provider}/{self.config.llm_model}",
                version="1.1",
            )
        else:  # UNCLEAR
            # Keep original but mark as uncertain
            return FeatureLabel(
                feature_idx=feature_idx,
                description=current_label.description + " [UNVERIFIED]",
                confidence=current_label.confidence * 0.8,
                top_activating_examples=current_label.top_activating_examples,
                activation_pattern=current_label.activation_pattern,
                semantic_alignments=current_label.semantic_alignments,
                primary_semantic_axis=current_label.primary_semantic_axis,
                semantic_alignment_score=current_label.semantic_alignment_score,
                labeled_at=datetime.now().isoformat(),
                labeler_model=f"{self.config.llm_provider}/{self.config.llm_model}",
            )

    async def compare_features(
        self,
        feature_a: int,
        feature_b: int,
        label_a: FeatureLabel,
        label_b: FeatureLabel,
    ) -> dict[str, Any]:
        """
        Compare two features and their relationship.

        Args:
            feature_a: First feature index
            feature_b: Second feature index
            label_a: Label for first feature
            label_b: Label for second feature

        Returns:
            Dict with comparison results
        """
        examples_a, act_a = self.analyzer.get_top_activating_examples(feature_a)
        examples_b, act_b = self.analyzer.get_top_activating_examples(feature_b)

        context_a = LabelPromptContext(
            feature_idx=feature_a,
            examples=examples_a,
            activation_values=act_a,
        )

        context_b = LabelPromptContext(
            feature_idx=feature_b,
            examples=examples_b,
            activation_values=act_b,
        )

        prompt = self.prompt_builder.build_comparison_prompt(
            context_a, context_b,
            label_a.description, label_b.description,
        )

        response_text = await self._call_llm(prompt)
        parsed = self.response_parser.parse_comparison_response(response_text)

        return {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "relationship": parsed.relationship,
            "similarity_score": parsed.similarity_score,
            "explanation": parsed.description,
        }

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt."""
        if self._llm_client:
            response = await self._llm_client.complete(
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            return response.content
        else:
            # Fallback to direct ollama
            return await self._direct_ollama_call(prompt)

    async def _direct_ollama_call(self, prompt: str) -> str:
        """Direct ollama call as fallback."""
        try:
            import ollama
            response = ollama.chat(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except ImportError:
            return "[LLM not available - install ollama package]"
        except Exception as e:
            return f"[LLM error: {e}]"

    def get_quality_report(
        self,
        labels: dict[int, FeatureLabel],
    ) -> dict[str, Any]:
        """
        Generate quality report for labels.

        Args:
            labels: Dict of feature_idx -> FeatureLabel

        Returns:
            Dict with quality statistics and breakdown
        """
        assessments = {}
        quality_counts = dict.fromkeys(QualityLevel, 0)
        total_score = 0.0

        for idx, label in labels.items():
            assessment = self.quality_assessor.assess(label)
            assessments[idx] = assessment
            quality_counts[assessment.quality_level] += 1
            total_score += assessment.overall_score

        return {
            "total_labels": len(labels),
            "average_quality_score": total_score / len(labels) if labels else 0.0,
            "quality_distribution": {
                level.value: count
                for level, count in quality_counts.items()
            },
            "high_quality_count": quality_counts[QualityLevel.HIGH],
            "low_quality_count": quality_counts[QualityLevel.LOW] + quality_counts[QualityLevel.FAILED],
            "stats": self._stats.copy(),
            "assessments": {
                idx: {
                    "level": a.quality_level.value,
                    "score": a.overall_score,
                    "issues": a.issues,
                }
                for idx, a in assessments.items()
            },
        }

    def _update_registry(self, labels: dict[int, FeatureLabel]) -> None:
        """Update feature registry with new labels."""
        if not self.registry:
            return

        try:
            for idx, label in labels.items():
                self.registry.update_feature_label(
                    feature_idx=idx,
                    label=label.description,
                    confidence=label.confidence,
                    metadata={
                        "semantic_axis": label.primary_semantic_axis,
                        "alignment_score": label.semantic_alignment_score,
                        "labeled_at": label.labeled_at,
                    }
                )
            logger.info(f"Updated registry with {len(labels)} labels")
        except Exception as e:
            logger.warning(f"Failed to update registry: {e}")

    def clear_collected_data(self) -> None:
        """Clear all collected activation data."""
        self.analyzer.clear()
        self.alignment_scorer.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get labeling statistics."""
        return {
            **self._stats,
            "cache_size": self.cache.size,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_autolabel_engine(
    provider: str = "ollama",
    model: str = "llama3.2:3b",
    **kwargs,
) -> AutoLabelEngine:
    """
    Create an auto-label engine with specified provider.

    Args:
        provider: LLM provider (ollama/anthropic/openai/google)
        model: Model name
        **kwargs: Additional config options

    Returns:
        Configured AutoLabelEngine
    """
    config = AutoLabelConfig(
        llm_provider=provider,
        llm_model=model,
        **kwargs,
    )
    return AutoLabelEngine(config)


async def label_sae_features(
    sae: torch.nn.Module,
    activations: torch.Tensor,
    example_texts: list[str],
    feature_indices: list[int] | None = None,
    provider: str = "ollama",
    model: str = "llama3.2:3b",
) -> dict[int, FeatureLabel]:
    """
    Convenience function to label SAE features.

    Args:
        sae: The SAE model
        activations: Input activations [n_examples, input_dim]
        example_texts: Text for each example
        feature_indices: Which features to label (None = all)
        provider: LLM provider
        model: Model name

    Returns:
        Dict of feature_idx -> FeatureLabel
    """
    engine = create_autolabel_engine(provider=provider, model=model)
    engine.collect_examples(sae, activations, example_texts)
    labels = await engine.label_features(sae, feature_indices=feature_indices)
    return labels
