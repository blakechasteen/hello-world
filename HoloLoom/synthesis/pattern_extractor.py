#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern Extractor
=================
Mines patterns from enriched memories for synthesis.

Extracts:
- Q&A pairs (question → answer)
- Reasoning chains (if X then Y, because Z)
- Causal relationships (X causes Y)
- Analogies (X is like Y)
- Decision patterns (when A, choose B)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    from .enriched_memory import EnrichedMemory, ReasoningType
except ImportError:
    # Fallback for standalone execution
    from enriched_memory import EnrichedMemory, ReasoningType


class PatternType(Enum):
    """Type of extracted pattern."""
    QA_PAIR = "qa_pair"                 # Question-Answer pair
    REASONING_CHAIN = "reasoning_chain" # Multi-step reasoning
    CAUSAL = "causal"                   # X causes Y
    ANALOGY = "analogy"                 # X is like Y
    DECISION = "decision"               # When X, choose Y
    COMPARISON = "comparison"           # X vs Y
    PROCEDURE = "procedure"             # Steps to do X
    DEFINITION = "definition"           # X is defined as Y


@dataclass
class Pattern:
    """
    Extracted pattern from memories.

    Represents a learnable unit (Q&A, reasoning, etc.)
    """
    pattern_type: PatternType
    content: Dict[str, Any]  # Pattern-specific structure
    source_memories: List[str] = field(default_factory=list)  # Memory IDs
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pattern_type': self.pattern_type.value,
            'content': self.content,
            'source_memories': self.source_memories,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class PatternExtractor:
    """
    Extracts learnable patterns from enriched memories.

    The gold mining operation: turns filtered signal into
    structured patterns ready for training.
    """

    def __init__(self, min_confidence: float = 0.0):
        """
        Initialize pattern extractor.

        Args:
            min_confidence: Minimum confidence threshold for patterns (0.0-1.0)
        """
        self.min_confidence = min_confidence

    def extract_patterns(self, memories: List[EnrichedMemory]) -> List[Pattern]:
        """
        Extract all patterns from a collection of memories.

        Args:
            memories: List of enriched memories

        Returns:
            List of extracted patterns
        """
        patterns = []

        for memory in memories:
            # Extract Q&A pairs
            if memory.reasoning_type in [ReasoningType.QUESTION, ReasoningType.ANSWER]:
                qa_patterns = self._extract_qa_pairs(memory)
                patterns.extend(qa_patterns)

            # Extract reasoning chains
            if memory.reasoning_type == ReasoningType.EXPLANATION:
                reasoning = self._extract_reasoning_chains(memory)
                patterns.extend(reasoning)

            # Extract causal relationships
            causal = self._extract_causal(memory)
            patterns.extend(causal)

            # Extract decisions
            if memory.reasoning_type == ReasoningType.DECISION:
                decisions = self._extract_decisions(memory)
                patterns.extend(decisions)

            # Extract comparisons
            if memory.reasoning_type == ReasoningType.COMPARISON:
                comparisons = self._extract_comparisons(memory)
                patterns.extend(comparisons)

            # Extract procedures
            if memory.reasoning_type == ReasoningType.PROCEDURE:
                procedures = self._extract_procedures(memory)
                patterns.extend(procedures)

            # Extract definitions
            definitions = self._extract_definitions(memory)
            patterns.extend(definitions)

        return patterns

    def _extract_qa_pairs(self, memory: EnrichedMemory) -> List[Pattern]:
        """Extract question-answer pairs."""
        patterns = []

        if memory.user_input and memory.system_output:
            # Direct Q&A from conversation
            if '?' in memory.user_input:
                pattern = Pattern(
                    pattern_type=PatternType.QA_PAIR,
                    content={
                        'question': memory.user_input.strip(),
                        'answer': memory.system_output.strip(),
                        'entities': memory.entities,
                        'topics': memory.topics
                    },
                    source_memories=[memory.id],
                    confidence=memory.importance,
                    metadata={'timestamp': memory.timestamp.isoformat()}
                )
                patterns.append(pattern)

        return patterns

    def _extract_reasoning_chains(self, memory: EnrichedMemory) -> List[Pattern]:
        """Extract multi-step reasoning chains."""
        patterns = []

        text = memory.system_output or memory.text

        # Look for reasoning indicators
        reasoning_markers = ['because', 'therefore', 'thus', 'so', 'hence']
        has_reasoning = any(marker in text.lower() for marker in reasoning_markers)

        if has_reasoning:
            # Split on reasoning markers
            steps = []
            current_step = []

            for sentence in text.split('.'):
                current_step.append(sentence.strip())
                if any(marker in sentence.lower() for marker in reasoning_markers):
                    if current_step:
                        steps.append('. '.join(current_step))
                        current_step = []

            if current_step:
                steps.append('. '.join(current_step))

            if len(steps) > 1:
                pattern = Pattern(
                    pattern_type=PatternType.REASONING_CHAIN,
                    content={
                        'steps': steps,
                        'premise': memory.user_input or steps[0],
                        'conclusion': steps[-1] if steps else '',
                        'entities': memory.entities
                    },
                    source_memories=[memory.id],
                    confidence=memory.importance * 0.9,  # Slightly lower confidence
                    metadata={'step_count': len(steps)}
                )
                patterns.append(pattern)

        return patterns

    def _extract_causal(self, memory: EnrichedMemory) -> List[Pattern]:
        """Extract causal relationships (X causes Y)."""
        patterns = []

        text = memory.text.lower()

        # Causal patterns
        import re
        causal_patterns = [
            r'(.+?)\s+causes?\s+(.+)',
            r'(.+?)\s+leads? to\s+(.+)',
            r'(.+?)\s+results? in\s+(.+)',
            r'if\s+(.+?)\s+then\s+(.+)',
            r'when\s+(.+?),\s+(.+)'
        ]

        for pattern_str in causal_patterns:
            matches = re.finditer(pattern_str, text)
            for match in matches:
                cause = match.group(1).strip()
                effect = match.group(2).strip()

                if len(cause) > 5 and len(effect) > 5:  # Filter tiny matches
                    pattern = Pattern(
                        pattern_type=PatternType.CAUSAL,
                        content={
                            'cause': cause,
                            'effect': effect,
                            'entities': memory.entities
                        },
                        source_memories=[memory.id],
                        confidence=memory.importance * 0.8,
                        metadata={}
                    )
                    patterns.append(pattern)

        return patterns

    def _extract_decisions(self, memory: EnrichedMemory) -> List[Pattern]:
        """Extract decision patterns."""
        patterns = []

        text = memory.text.lower()

        # Decision indicators
        import re
        decision_patterns = [
            r'(?:should|recommend|choose|prefer)\s+(.+?)\s+(?:because|when|if)',
            r'decided to\s+(.+?)\s+because',
            r'best to\s+(.+?)\s+when'
        ]

        for pattern_str in decision_patterns:
            matches = re.finditer(pattern_str, text)
            for match in matches:
                decision = match.group(1).strip()

                if len(decision) > 5:
                    pattern = Pattern(
                        pattern_type=PatternType.DECISION,
                        content={
                            'decision': decision,
                            'context': memory.user_input or '',
                            'reasoning': memory.system_output or '',
                            'entities': memory.entities
                        },
                        source_memories=[memory.id],
                        confidence=memory.importance * 0.85,
                        metadata={}
                    )
                    patterns.append(pattern)

        return patterns

    def _extract_comparisons(self, memory: EnrichedMemory) -> List[Pattern]:
        """Extract comparisons (X vs Y)."""
        patterns = []

        text = memory.text.lower()

        # Comparison patterns
        import re
        comparison_patterns = [
            r'(.+?)\s+(?:vs|versus)\s+(.+)',
            r'difference between\s+(.+?)\s+and\s+(.+)',
            r'(.+?)\s+compared to\s+(.+)',
            r'(.+?)\s+while\s+(.+)'
        ]

        for pattern_str in comparison_patterns:
            matches = re.finditer(pattern_str, text)
            for match in matches:
                item_a = match.group(1).strip()
                item_b = match.group(2).strip()

                if len(item_a) > 3 and len(item_b) > 3:
                    pattern = Pattern(
                        pattern_type=PatternType.COMPARISON,
                        content={
                            'item_a': item_a,
                            'item_b': item_b,
                            'comparison': memory.system_output or memory.text,
                            'entities': memory.entities
                        },
                        source_memories=[memory.id],
                        confidence=memory.importance * 0.85,
                        metadata={}
                    )
                    patterns.append(pattern)

        return patterns

    def _extract_procedures(self, memory: EnrichedMemory) -> List[Pattern]:
        """Extract step-by-step procedures."""
        patterns = []

        text = memory.system_output or memory.text

        # Look for step indicators
        step_markers = ['first', 'second', 'third', 'then', 'next', 'finally', 'step']
        has_steps = any(marker in text.lower() for marker in step_markers)

        if has_steps:
            # Try to extract steps
            import re
            # Numbered steps
            numbered = re.findall(r'\d+[\.\)]\s*([^\n]+)', text)

            # Bullet steps
            bullets = re.findall(r'[-•]\s*([^\n]+)', text)

            # Word-based steps
            words = []
            for marker in ['first', 'second', 'third', 'then', 'next', 'finally']:
                pattern_str = rf'{marker}[,:]?\s*([^\.]+)'
                found = re.findall(pattern_str, text, re.IGNORECASE)
                words.extend(found)

            steps = numbered or bullets or words

            if len(steps) >= 2:
                pattern = Pattern(
                    pattern_type=PatternType.PROCEDURE,
                    content={
                        'task': memory.user_input or 'procedure',
                        'steps': steps,
                        'entities': memory.entities
                    },
                    source_memories=[memory.id],
                    confidence=memory.importance * 0.9,
                    metadata={'step_count': len(steps)}
                )
                patterns.append(pattern)

        return patterns

    def _extract_definitions(self, memory: EnrichedMemory) -> List[Pattern]:
        """Extract definitions (X is defined as Y)."""
        patterns = []

        text = memory.text

        # Definition patterns
        import re
        definition_patterns = [
            r'(\w+)\s+is\s+(?:a|an)\s+(.+?)(?:\.|,)',
            r'(\w+)\s+means\s+(.+?)(?:\.|,)',
            r'(\w+)\s+refers to\s+(.+?)(?:\.|,)',
            r'(\w+)\s+(?:is )?defined as\s+(.+?)(?:\.|,)'
        ]

        for pattern_str in definition_patterns:
            matches = re.finditer(pattern_str, text, re.IGNORECASE)
            for match in matches:
                term = match.group(1).strip()
                definition = match.group(2).strip()

                # Filter short or generic
                if len(term) > 2 and len(definition) > 10:
                    # Check if term is in entities or keywords
                    if term.title() in memory.entities or term.lower() in memory.keywords:
                        pattern = Pattern(
                            pattern_type=PatternType.DEFINITION,
                            content={
                                'term': term,
                                'definition': definition,
                                'context': memory.system_output or memory.text,
                                'entities': memory.entities
                            },
                            source_memories=[memory.id],
                            confidence=memory.importance * 0.95,
                            metadata={}
                        )
                        patterns.append(pattern)

        return patterns

    def cluster_patterns(self, patterns: List[Pattern],
                        by: str = 'pattern_type') -> Dict[str, List[Pattern]]:
        """
        Cluster patterns by type, topic, or other criteria.

        Args:
            patterns: List of patterns
            by: Clustering key ('pattern_type', 'topic', etc.)

        Returns:
            Dict mapping cluster key to patterns
        """
        clusters = {}

        for pattern in patterns:
            if by == 'pattern_type':
                key = pattern.pattern_type.value
            elif by == 'topic':
                # Use first topic from metadata
                topics = pattern.content.get('topics', ['general'])
                key = topics[0] if topics else 'general'
            else:
                key = 'all'

            if key not in clusters:
                clusters[key] = []
            clusters[key].append(pattern)

        return clusters
