#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtracker Component
=====================
Detects contradictions and revises reasoning chains for DEEP mode.

Author: Claude Code
Date: 2025-11-15
Phase: 3 (DEEP Mode)
"""

import logging
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass

from HoloLoom.reasoning.types import (
    ReasoningStep,
    StepType,
    VerificationResult,
    MultiPassVerification,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Contradiction Detection Types
# ============================================================================

@dataclass
class Contradiction:
    """
    Detected contradiction in reasoning chain.

    Attributes:
        step_indices: Indices of contradicting steps
        description: Human-readable description
        severity: How severe is the contradiction [0.0, 1.0]
        suggested_revision: Suggested fix
    """
    step_indices: Tuple[int, int]
    description: str
    severity: float
    suggested_revision: str

    def __post_init__(self):
        """Validate ranges."""
        assert 0.0 <= self.severity <= 1.0, "Severity must be [0.0, 1.0]"


@dataclass
class BacktrackResult:
    """
    Result of backtracking operation.

    Attributes:
        revised_chain: Revised reasoning chain
        revisions_made: Number of revisions
        contradictions_resolved: Number of contradictions resolved
        revision_history: List of revision descriptions
    """
    revised_chain: List[ReasoningStep]
    revisions_made: int
    contradictions_resolved: int
    revision_history: List[str]


# ============================================================================
# Backtracker
# ============================================================================

class Backtracker:
    """
    Detects contradictions and revises reasoning chains.

    Phase 3: Rule-based contradiction detection
    Future: Semantic contradiction detection with embeddings
    """

    def __init__(self, max_revisions: int = 3):
        """
        Initialize backtracker.

        Args:
            max_revisions: Maximum number of revision iterations
        """
        self.max_revisions = max_revisions
        self.logger = logging.getLogger(f"{__name__}.Backtracker")

        # Contradiction patterns (keyword-based for Phase 3)
        self.negation_pairs = [
            ("always", "never"),
            ("all", "none"),
            ("every", "no"),
            ("impossible", "possible"),
            ("true", "false"),
            ("correct", "incorrect"),
            ("increase", "decrease"),
            ("improve", "worsen"),
            ("higher", "lower"),
            ("more", "less"),
        ]

        # Confidence contradiction threshold
        self.confidence_contradiction_threshold = 0.4

    async def detect_contradictions(
        self,
        chain: List[ReasoningStep]
    ) -> List[Contradiction]:
        """
        Detect contradictions in reasoning chain.

        Args:
            chain: Reasoning chain to analyze

        Returns:
            List of detected contradictions
        """
        contradictions = []

        # Check 1: Keyword-based contradictions
        keyword_contradictions = self._detect_keyword_contradictions(chain)
        contradictions.extend(keyword_contradictions)

        # Check 2: Confidence contradictions
        confidence_contradictions = self._detect_confidence_contradictions(chain)
        contradictions.extend(confidence_contradictions)

        # Check 3: Logical flow contradictions
        flow_contradictions = self._detect_flow_contradictions(chain)
        contradictions.extend(flow_contradictions)

        self.logger.info(f"Detected {len(contradictions)} contradictions")

        return contradictions

    def _detect_keyword_contradictions(
        self,
        chain: List[ReasoningStep]
    ) -> List[Contradiction]:
        """
        Detect contradictions using keyword patterns.

        Args:
            chain: Reasoning chain

        Returns:
            List of keyword-based contradictions
        """
        contradictions = []

        for i, step_i in enumerate(chain):
            for j, step_j in enumerate(chain[i+1:], start=i+1):
                thought_i = step_i.thought.lower()
                thought_j = step_j.thought.lower()

                # Check for negation pairs
                for word1, word2 in self.negation_pairs:
                    if word1 in thought_i and word2 in thought_j:
                        contradictions.append(Contradiction(
                            step_indices=(i, j),
                            description=f"Keyword contradiction: step {i} uses '{word1}' but step {j} uses '{word2}'",
                            severity=0.7,
                            suggested_revision=f"Reconcile the use of '{word1}' and '{word2}' in reasoning"
                        ))
                    elif word2 in thought_i and word1 in thought_j:
                        contradictions.append(Contradiction(
                            step_indices=(i, j),
                            description=f"Keyword contradiction: step {i} uses '{word2}' but step {j} uses '{word1}'",
                            severity=0.7,
                            suggested_revision=f"Reconcile the use of '{word1}' and '{word2}' in reasoning"
                        ))

        return contradictions

    def _detect_confidence_contradictions(
        self,
        chain: List[ReasoningStep]
    ) -> List[Contradiction]:
        """
        Detect contradictions from confidence patterns.

        Args:
            chain: Reasoning chain

        Returns:
            List of confidence-based contradictions
        """
        contradictions = []

        for i in range(len(chain) - 1):
            curr_conf = chain[i].confidence
            next_conf = chain[i+1].confidence

            # Large confidence drop suggests contradiction or error
            if curr_conf - next_conf > self.confidence_contradiction_threshold:
                contradictions.append(Contradiction(
                    step_indices=(i, i+1),
                    description=f"Confidence drop from {curr_conf:.2f} to {next_conf:.2f}",
                    severity=(curr_conf - next_conf),
                    suggested_revision=f"Review reasoning at step {i} or {i+1} for errors"
                ))

        return contradictions

    def _detect_flow_contradictions(
        self,
        chain: List[ReasoningStep]
    ) -> List[Contradiction]:
        """
        Detect logical flow contradictions.

        Args:
            chain: Reasoning chain

        Returns:
            List of logical flow contradictions
        """
        contradictions = []

        # Check for synthesis before evidence
        evidence_idx = None
        synthesis_idx = None

        for i, step in enumerate(chain):
            if step.step_type == StepType.EVIDENCE and evidence_idx is None:
                evidence_idx = i
            if step.step_type == StepType.SYNTHESIS and synthesis_idx is None:
                synthesis_idx = i

        if synthesis_idx is not None and evidence_idx is not None:
            if synthesis_idx < evidence_idx:
                contradictions.append(Contradiction(
                    step_indices=(synthesis_idx, evidence_idx),
                    description="Synthesis appears before evidence gathering",
                    severity=0.6,
                    suggested_revision="Move synthesis step after evidence gathering"
                ))

        # Check for backtrack loops (backtrack → backtrack)
        for i in range(len(chain) - 1):
            if (chain[i].step_type == StepType.BACKTRACK and
                chain[i+1].step_type == StepType.BACKTRACK):
                contradictions.append(Contradiction(
                    step_indices=(i, i+1),
                    description="Consecutive backtrack steps detected (possible loop)",
                    severity=0.8,
                    suggested_revision="Break backtrack loop by adding corrective reasoning"
                ))

        return contradictions

    async def revise(
        self,
        chain: List[ReasoningStep],
        contradictions: List[Contradiction]
    ) -> BacktrackResult:
        """
        Revise reasoning chain to resolve contradictions.

        Args:
            chain: Original reasoning chain
            contradictions: Detected contradictions

        Returns:
            BacktrackResult with revised chain
        """
        if not contradictions:
            return BacktrackResult(
                revised_chain=chain,
                revisions_made=0,
                contradictions_resolved=0,
                revision_history=[]
            )

        revised_chain = chain.copy()
        revision_history = []
        revisions_made = 0

        # Sort contradictions by severity (highest first)
        sorted_contradictions = sorted(
            contradictions,
            key=lambda c: c.severity,
            reverse=True
        )

        for contradiction in sorted_contradictions[:self.max_revisions]:
            # Apply revision
            result = self._apply_revision(
                revised_chain,
                contradiction
            )

            if result:
                revised_chain, revision_desc = result
                revision_history.append(revision_desc)
                revisions_made += 1

        self.logger.info(
            f"Revised chain: {revisions_made} revisions made, "
            f"{len(contradictions)} contradictions addressed"
        )

        return BacktrackResult(
            revised_chain=revised_chain,
            revisions_made=revisions_made,
            contradictions_resolved=min(revisions_made, len(contradictions)),
            revision_history=revision_history
        )

    def _apply_revision(
        self,
        chain: List[ReasoningStep],
        contradiction: Contradiction
    ) -> Optional[Tuple[List[ReasoningStep], str]]:
        """
        Apply a single revision to resolve contradiction.

        Args:
            chain: Current chain
            contradiction: Contradiction to resolve

        Returns:
            Tuple of (revised_chain, description) or None if no revision possible
        """
        i, j = contradiction.step_indices

        # Strategy: Insert a backtrack step between contradicting steps
        backtrack_step = ReasoningStep(
            thought=f"REVISION: {contradiction.description}",
            evidence=contradiction.suggested_revision,
            confidence=0.6,
            step_type=StepType.BACKTRACK,
            metadata={
                "contradiction": contradiction.description,
                "revised_steps": [i, j],
                "severity": contradiction.severity
            }
        )

        # Insert backtrack step after the first contradicting step
        insert_position = j  # Insert before step j
        new_chain = chain[:insert_position] + [backtrack_step] + chain[insert_position:]

        revision_desc = (
            f"Inserted backtrack at position {insert_position} to resolve: "
            f"{contradiction.description}"
        )

        return new_chain, revision_desc

    async def resolve_contradictions(
        self,
        chain: List[ReasoningStep],
        verification: MultiPassVerification
    ) -> List[ReasoningStep]:
        """
        Resolve contradictions found during verification.

        Args:
            chain: Original reasoning chain
            verification: Multi-pass verification result

        Returns:
            Revised reasoning chain
        """
        if not verification.has_contradictions:
            return chain

        # Detect all contradictions
        contradictions = await self.detect_contradictions(chain)

        # Add verification-detected contradictions
        for contradiction_desc in verification.contradictions:
            # Parse contradiction description to find step indices
            # For Phase 3, create a general contradiction
            contradictions.append(Contradiction(
                step_indices=(0, len(chain) - 1),
                description=contradiction_desc,
                severity=0.7,
                suggested_revision="Review overall reasoning for consistency"
            ))

        # Revise chain
        result = await self.revise(chain, contradictions)

        self.logger.info(
            f"Contradiction resolution: {result.contradictions_resolved} resolved, "
            f"{result.revisions_made} revisions made"
        )

        return result.revised_chain

    def prevent_infinite_loops(
        self,
        chain: List[ReasoningStep],
        max_backtracks: int = 2
    ) -> bool:
        """
        Check if chain has too many backtrack steps (infinite loop prevention).

        Args:
            chain: Reasoning chain
            max_backtracks: Maximum allowed backtrack steps

        Returns:
            True if chain is safe, False if loop detected
        """
        backtrack_count = sum(
            1 for step in chain
            if step.step_type == StepType.BACKTRACK
        )

        if backtrack_count > max_backtracks:
            self.logger.warning(
                f"Infinite loop detected: {backtrack_count} backtrack steps "
                f"(max allowed: {max_backtracks})"
            )
            return False

        return True

    def detect_cycles(
        self,
        chain: List[ReasoningStep],
        window_size: int = 3
    ) -> List[Tuple[int, int]]:
        """
        Detect reasoning cycles (repeated patterns).

        Args:
            chain: Reasoning chain
            window_size: Size of pattern window

        Returns:
            List of (start, end) indices for detected cycles
        """
        cycles = []

        # Simple cycle detection: look for repeated thought patterns
        thoughts = [step.thought.lower() for step in chain]

        for i in range(len(thoughts) - window_size):
            pattern = thoughts[i:i+window_size]

            # Search for pattern repetition
            for j in range(i + window_size, len(thoughts) - window_size + 1):
                candidate = thoughts[j:j+window_size]

                # Check for similarity (simple string matching)
                if pattern == candidate:
                    cycles.append((i, j))
                    self.logger.warning(
                        f"Cycle detected: steps {i}-{i+window_size} "
                        f"repeat at steps {j}-{j+window_size}"
                    )

        return cycles
