#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Provenance Tracker
==============================
Converts Layer 6 reasoning chains into scratchpad format for provenance tracking.

This module bridges the Reasoning Engine with the Scratchpad system, enabling
complete audit trails of reasoning processes.

Each reasoning step becomes a scratchpad entry:
- Thought: The reasoning step's thought
- Action: "reasoning_step_{i}" + step type
- Observation: Evidence collected
- Score: Confidence of this step

Author: Claude Code
Date: 2025-11-15
Phase: 2 (WeavingOrchestrator Integration)
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# Reasoning types
from HoloLoom.reasoning.types import ReasoningResult, ReasoningStep, ReasoningMode, StepType

# Scratchpad integration
try:
    from Promptly.promptly.recursive_loops import Scratchpad, ScratchpadEntry
    SCRATCHPAD_AVAILABLE = True
except ImportError:
    SCRATCHPAD_AVAILABLE = False
    # Create stub classes for type hints
    @dataclass
    class ScratchpadEntry:
        thought: str
        action: str
        observation: str
        score: float
        iteration: int
        metadata: Dict[str, Any]

    class Scratchpad:
        pass

logger = logging.getLogger(__name__)


# ============================================================================
# Provenance Tracker
# ============================================================================

class ReasoningProvenanceTracker:
    """
    Extracts provenance from reasoning chains into scratchpad format.

    Converts ReasoningResult (reasoning chain) into ScratchpadEntry list,
    preserving the complete thought process for learning and debugging.

    Usage:
        tracker = ReasoningProvenanceTracker()
        reasoning_result = await reasoning_engine.reason(query, features, context)
        entries = tracker.extract_reasoning_provenance(reasoning_result)

        # Add to scratchpad
        for entry in entries:
            scratchpad.add_entry(entry)
    """

    def __init__(self):
        """Initialize provenance tracker."""
        self.logger = logging.getLogger(f"{__name__}.ReasoningProvenanceTracker")

        if not SCRATCHPAD_AVAILABLE:
            self.logger.warning(
                "Promptly Scratchpad not available. "
                "Provenance tracking will use stub entries."
            )

    def extract_reasoning_provenance(
        self,
        reasoning_result: ReasoningResult,
        base_iteration: int = 1
    ) -> List[ScratchpadEntry]:
        """
        Convert reasoning chain to scratchpad entries.

        Each reasoning step becomes an entry:
        - Thought: The reasoning step's thought
        - Action: "reasoning_step_{i}_{step_type}"
        - Observation: Evidence collected
        - Score: Confidence of this step

        Args:
            reasoning_result: Result from reasoning engine
            base_iteration: Starting iteration number

        Returns:
            List of ScratchpadEntry objects
        """
        entries = []

        for i, step in enumerate(reasoning_result.chain):
            entry = self._step_to_entry(
                step=step,
                step_number=i + 1,
                iteration=base_iteration + i,
                reasoning_mode=reasoning_result.mode
            )
            entries.append(entry)

        self.logger.debug(
            f"Extracted {len(entries)} scratchpad entries from reasoning chain "
            f"(mode={reasoning_result.mode.value})"
        )

        return entries

    def _step_to_entry(
        self,
        step: ReasoningStep,
        step_number: int,
        iteration: int,
        reasoning_mode: ReasoningMode
    ) -> ScratchpadEntry:
        """
        Convert single reasoning step to scratchpad entry.

        Args:
            step: Reasoning step
            step_number: Step number in chain (1-indexed)
            iteration: Overall iteration number
            reasoning_mode: Reasoning mode used

        Returns:
            ScratchpadEntry
        """
        # THOUGHT: Main reasoning thought
        thought = f"[{step.step_type.value.upper()}] {step.thought}"

        # ACTION: Step identifier
        action = f"reasoning_step_{step_number}_{step.step_type.value}"

        # OBSERVATION: Evidence and confidence
        observation_parts = [step.evidence]

        if step.metadata:
            # Add key metadata to observation
            metadata_str = ", ".join(
                f"{k}={v}" for k, v in step.metadata.items()
                if k not in ['timestamp']  # Exclude timestamp (already in entry)
            )
            if metadata_str:
                observation_parts.append(f"[{metadata_str}]")

        observation = " | ".join(observation_parts)

        # SCORE: Confidence
        score = step.confidence

        # METADATA: Full step details
        metadata = {
            'step_type': step.step_type.value,
            'step_number': step_number,
            'reasoning_mode': reasoning_mode.value,
            'timestamp': step.timestamp.isoformat(),
            **step.metadata  # Include original metadata
        }

        # Create entry
        entry = ScratchpadEntry(
            thought=thought,
            action=action,
            observation=observation,
            score=score,
            iteration=iteration,
            metadata=metadata
        )

        return entry

    def summarize_reasoning_provenance(
        self,
        reasoning_result: ReasoningResult
    ) -> str:
        """
        Generate human-readable summary of reasoning provenance.

        Args:
            reasoning_result: Result from reasoning engine

        Returns:
            Summary string
        """
        lines = [
            f"Reasoning Provenance Summary",
            f"===========================",
            f"Mode: {reasoning_result.mode.value}",
            f"Steps: {len(reasoning_result.chain)}",
            f"Confidence: {reasoning_result.total_confidence:.2f}",
            f"Duration: {reasoning_result.duration_ms:.1f}ms",
            f"",
            f"Reasoning Chain:",
        ]

        for i, step in enumerate(reasoning_result.chain):
            lines.append(
                f"  {i+1}. [{step.step_type.value}] {step.thought[:60]}... "
                f"(conf: {step.confidence:.2f})"
            )

        return "\n".join(lines)

    def extract_quality_signals(
        self,
        reasoning_result: ReasoningResult
    ) -> Dict[str, Any]:
        """
        Extract quality signals from reasoning chain for learning.

        These signals can be used by the reflection buffer to improve
        future reasoning processes.

        Args:
            reasoning_result: Result from reasoning engine

        Returns:
            Dict of quality signals
        """
        signals = {
            # Overall quality
            'mode': reasoning_result.mode.value,
            'total_steps': len(reasoning_result.chain),
            'avg_confidence': reasoning_result.total_confidence,
            'duration_ms': reasoning_result.duration_ms,

            # Step-level analysis
            'step_types': [step.step_type.value for step in reasoning_result.chain],
            'confidence_trajectory': [step.confidence for step in reasoning_result.chain],
            'confidence_variance': self._compute_variance(
                [step.confidence for step in reasoning_result.chain]
            ),

            # Quality indicators
            'had_verification': any(
                step.step_type == StepType.VERIFICATION
                for step in reasoning_result.chain
            ),
            'had_backtracking': any(
                step.step_type == StepType.BACKTRACK
                for step in reasoning_result.chain
            ),
            'had_correction': any(
                step.step_type == StepType.CORRECTION
                for step in reasoning_result.chain
            ),

            # Metadata
            'metadata': reasoning_result.metadata
        }

        return signals

    def _compute_variance(self, values: List[float]) -> float:
        """Compute variance of values."""
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance


# ============================================================================
# Integration with Scratchpad Orchestrator
# ============================================================================

class ReasoningAwareScratchpadOrchestrator:
    """
    Enhanced scratchpad orchestrator that tracks reasoning chains.

    This is a helper class that combines:
    - Base WeavingOrchestrator (or ReasoningOrchestrator)
    - Scratchpad for provenance tracking
    - ReasoningProvenanceTracker for reasoning chain integration

    Usage:
        async with ReasoningAwareScratchpadOrchestrator(
            cfg=config,
            shards=shards,
            enable_reasoning=True
        ) as orchestrator:
            spacetime = await orchestrator.weave(query)

            # Scratchpad automatically populated with reasoning chain
            history = orchestrator.scratchpad.get_history()
    """

    def __init__(
        self,
        cfg,
        shards: Optional[List] = None,
        memory=None,
        enable_reasoning: bool = True,
        scratchpad_capacity: int = 1000
    ):
        """
        Initialize reasoning-aware scratchpad orchestrator.

        Args:
            cfg: HoloLoom configuration
            shards: Memory shards (optional)
            memory: Memory backend (optional)
            enable_reasoning: Enable reasoning layer
            scratchpad_capacity: Maximum scratchpad entries
        """
        self.cfg = cfg
        self.shards = shards
        self.memory = memory
        self.enable_reasoning = enable_reasoning

        # Create scratchpad
        if SCRATCHPAD_AVAILABLE:
            self.scratchpad = Scratchpad(capacity=scratchpad_capacity)
        else:
            self.scratchpad = None
            logger.warning("Scratchpad not available - provenance tracking disabled")

        # Create provenance tracker
        self.provenance_tracker = ReasoningProvenanceTracker()

        # Create orchestrator (will be set in __aenter__)
        self.orchestrator = None

        self.logger = logging.getLogger(f"{__name__}.ReasoningAwareScratchpadOrchestrator")

    async def __aenter__(self):
        """Async context manager entry."""
        from HoloLoom.weaving_orchestrator_reasoning import ReasoningOrchestrator

        # Create orchestrator
        self.orchestrator = ReasoningOrchestrator(
            cfg=self.cfg,
            shards=self.shards,
            memory=self.memory,
            enable_reasoning=self.enable_reasoning
        )

        # Initialize orchestrator (async context manager)
        await self.orchestrator.__aenter__()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.orchestrator:
            await self.orchestrator.__aexit__(exc_type, exc_val, exc_tb)

    async def weave(self, query, **kwargs):
        """
        Weave with automatic scratchpad provenance tracking.

        Args:
            query: User query
            **kwargs: Additional weave parameters

        Returns:
            Spacetime with reasoning chain
        """
        # Call orchestrator
        spacetime = await self.orchestrator.weave(query, **kwargs)

        # Extract reasoning provenance if available
        if self.scratchpad and 'reasoning_chain' in spacetime.metadata:
            try:
                # Reconstruct ReasoningResult from metadata
                from HoloLoom.reasoning.types import ReasoningResult, ReasoningStep, ReasoningMode, StepType
                from datetime import datetime

                chain = [
                    ReasoningStep(
                        thought=step['thought'],
                        evidence=step['evidence'],
                        confidence=step['confidence'],
                        step_type=StepType(step['step_type']),
                        timestamp=datetime.fromisoformat(step['timestamp']),
                        metadata={}
                    )
                    for step in spacetime.metadata['reasoning_chain']
                ]

                reasoning_result = ReasoningResult(
                    chain=chain,
                    mode=ReasoningMode(spacetime.metadata['reasoning_mode']),
                    total_confidence=spacetime.metadata['reasoning_confidence'],
                    duration_ms=spacetime.metadata['reasoning_duration_ms']
                )

                # Extract provenance
                entries = self.provenance_tracker.extract_reasoning_provenance(reasoning_result)

                # Add to scratchpad
                for entry in entries:
                    self.scratchpad.add_entry(entry)

                self.logger.info(f"Added {len(entries)} reasoning entries to scratchpad")

            except Exception as e:
                self.logger.warning(f"Failed to add reasoning provenance to scratchpad: {e}")

        return spacetime

    def get_reasoning_history(self) -> List[ScratchpadEntry]:
        """
        Get all reasoning-related entries from scratchpad.

        Returns:
            List of reasoning scratchpad entries
        """
        if not self.scratchpad:
            return []

        all_entries = self.scratchpad.get_history()
        reasoning_entries = [
            entry for entry in all_entries
            if 'reasoning_step' in entry.action
        ]

        return reasoning_entries
