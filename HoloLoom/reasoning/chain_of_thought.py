#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chain-of-Thought Generator
===========================
Generates multi-step reasoning chains for STANDARD mode.

Author: Claude Code
Date: 2025-11-15
Phase: 1 (Foundation)
"""

import logging
from typing import List

from HoloLoom.documentation.types import Query, Features, Context
from HoloLoom.reasoning.types import (
    ReasoningStep,
    StepType,
    QueryIntent,
    EvidencePiece,
    Synthesis,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Evidence Extraction
# ============================================================================

class ChainOfThought:
    """
    Generates multi-step reasoning chains.

    Phase 1: Rule-based chain generation
    Future: LLM-based reasoning step generation
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ChainOfThought")

    def extract_evidence(
        self,
        context: Context,
        intent: QueryIntent
    ) -> List[EvidencePiece]:
        """
        Extract relevant evidence from context.

        Args:
            context: Retrieved context
            intent: Query intent

        Returns:
            List of evidence pieces
        """
        evidence = []

        # Extract from shards
        for i, shard in enumerate(context.shards[:10]):  # Limit to top 10
            # Calculate relevance based on key concepts
            relevance = self._calculate_relevance(shard, intent)

            if relevance > 0.3:  # Threshold for inclusion
                # Extract text snippet
                text = shard.text[:200]  # Limit length
                if len(shard.text) > 200:
                    text += "..."

                evidence.append(EvidencePiece(
                    text=text,
                    source=f"shard_{i}",
                    relevance=relevance,
                    confidence=0.8  # Base confidence
                ))

        # Sort by relevance
        evidence.sort(key=lambda e: e.relevance, reverse=True)

        return evidence[:5]  # Return top 5

    def _calculate_relevance(
        self,
        shard,
        intent: QueryIntent
    ) -> float:
        """
        Calculate shard relevance to query intent.

        Args:
            shard: Memory shard
            intent: Query intent

        Returns:
            Relevance score [0.0, 1.0]
        """
        text = shard.text.lower()
        relevance = 0.0

        # Check for key concept matches
        for concept in intent.key_concepts:
            if concept.lower() in text:
                relevance += 0.3

        # Check for entity matches
        if hasattr(shard, 'entities'):
            for entity in shard.entities:
                if entity.lower() in intent.key_concepts:
                    relevance += 0.2

        return min(1.0, relevance)

    def synthesize(
        self,
        intent: QueryIntent,
        evidence: List[EvidencePiece]
    ) -> Synthesis:
        """
        Synthesize reasoning from intent and evidence.

        Args:
            intent: Query intent
            evidence: Extracted evidence

        Returns:
            Synthesis with reasoning and key points
        """
        if not evidence:
            return Synthesis(
                reasoning="Insufficient evidence to answer query",
                key_points=["No relevant context found"],
                confidence=0.3,
                evidence_used=0
            )

        # Build reasoning based on query type
        key_points = []
        reasoning_parts = []

        # Add evidence-based points
        for i, ev in enumerate(evidence[:3]):  # Top 3 pieces
            key_points.append(f"Point {i+1}: {ev.text[:100]}")
            reasoning_parts.append(ev.text)

        # Synthesize based on query type
        if intent.type.value == "factual":
            reasoning = f"Based on the evidence, {reasoning_parts[0]}"
        elif intent.type.value == "comparative":
            reasoning = f"Comparing the evidence: {' versus '.join(reasoning_parts[:2])}"
        elif intent.type.value == "analytical":
            reasoning = f"Analyzing the relationship: {reasoning_parts[0]}"
        else:
            reasoning = f"Synthesizing from evidence: {' '.join(reasoning_parts)}"

        # Calculate synthesis confidence
        avg_relevance = sum(e.relevance for e in evidence) / len(evidence)
        confidence = min(0.95, avg_relevance * 1.1)  # Boost slightly

        return Synthesis(
            reasoning=reasoning,
            key_points=key_points,
            confidence=confidence,
            evidence_used=len(evidence)
        )

    def generate_standard_chain(
        self,
        query: Query,
        intent: QueryIntent,
        context: Context
    ) -> List[ReasoningStep]:
        """
        Generate standard chain-of-thought (3-5 steps).

        Args:
            query: Original query
            intent: Analyzed intent
            context: Retrieved context

        Returns:
            List of reasoning steps
        """
        chain = []

        # Step 1: Understanding
        understanding_confidence = intent.confidence
        chain.append(ReasoningStep(
            thought=f"Query type: {intent.type.value}, requires: {', '.join(intent.requirements[:2])}",
            evidence=f"Key concepts: {', '.join(intent.key_concepts[:3])}",
            confidence=understanding_confidence,
            step_type=StepType.UNDERSTANDING
        ))

        # Step 2: Evidence gathering
        evidence = self.extract_evidence(context, intent)
        evidence_confidence = (
            sum(e.relevance for e in evidence) / len(evidence)
            if evidence else 0.3
        )

        chain.append(ReasoningStep(
            thought=f"Found {len(evidence)} relevant pieces of evidence",
            evidence="; ".join([e.text[:80] for e in evidence[:3]]),
            confidence=evidence_confidence,
            step_type=StepType.EVIDENCE
        ))

        # Step 3: Synthesis
        synthesis = self.synthesize(intent, evidence)
        chain.append(ReasoningStep(
            thought=synthesis.reasoning,
            evidence="; ".join(synthesis.key_points),
            confidence=synthesis.confidence,
            step_type=StepType.SYNTHESIS
        ))

        return chain

    def estimate_confidence(
        self,
        features: Features,
        context: Context
    ) -> float:
        """
        Quick confidence estimation for FAST mode.

        Args:
            features: Extracted features
            context: Retrieved context

        Returns:
            Confidence estimate [0.0, 1.0]
        """
        # Base confidence on context quality
        if not context.shards:
            return 0.2

        # More shards = higher confidence (up to a point)
        shard_factor = min(1.0, len(context.shards) / 5.0)

        # Motifs indicate good understanding
        motif_factor = min(1.0, len(features.motifs) / 3.0) if features.motifs else 0.5

        # Combine factors
        confidence = 0.6 * shard_factor + 0.4 * motif_factor

        return min(0.95, max(0.2, confidence))
