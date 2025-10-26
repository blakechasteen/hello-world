#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Synthesizer
================
Synthesizes training data from extracted patterns.

Transforms patterns into:
- Instruction-response pairs (for fine-tuning)
- Few-shot examples (for prompting)
- Chain-of-thought demonstrations
- Domain-specific datasets

The alchemy: Signal → Training Data → Intelligence
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

try:
    from .pattern_extractor import Pattern, PatternType
except ImportError:
    # Fallback for standalone execution
    from pattern_extractor import Pattern, PatternType


@dataclass
class TrainingExample:
    """
    Single training example for fine-tuning or prompting.

    Formats:
    - Alpaca (instruction, input, output)
    - ChatML (messages format)
    - Raw (custom format)
    """
    instruction: str
    input: str = ""
    output: str = ""
    system: str = ""  # System prompt
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_alpaca(self) -> Dict[str, str]:
        """Convert to Alpaca format."""
        return {
            'instruction': self.instruction,
            'input': self.input,
            'output': self.output
        }

    def to_chatml(self) -> List[Dict[str, str]]:
        """Convert to ChatML format."""
        messages = []

        if self.system:
            messages.append({'role': 'system', 'content': self.system})

        if self.input:
            user_content = f"{self.instruction}\n\n{self.input}"
        else:
            user_content = self.instruction

        messages.append({'role': 'user', 'content': user_content})
        messages.append({'role': 'assistant', 'content': self.output})

        return messages

    def to_raw(self) -> Dict[str, Any]:
        """Convert to raw format with all fields."""
        return {
            'instruction': self.instruction,
            'input': self.input,
            'output': self.output,
            'system': self.system,
            'metadata': self.metadata
        }


@dataclass
class SynthesisConfig:
    """Configuration for data synthesis."""
    include_reasoning: bool = True       # Include chain-of-thought
    include_context: bool = True         # Include entities/topics as context
    min_confidence: float = 0.4          # Minimum pattern confidence
    max_examples_per_pattern: int = 1   # Max examples per pattern
    system_prompt: str = "You are a helpful AI assistant trained on high-quality conversations."


class DataSynthesizer:
    """
    Synthesizes training data from patterns.

    The money maker: Turns your filtered signal into
    training examples that capture YOUR reasoning.
    """

    def __init__(self, config: Optional[SynthesisConfig] = None):
        """
        Initialize synthesizer.

        Args:
            config: Synthesis configuration
        """
        self.config = config or SynthesisConfig()

    def synthesize(self, patterns: List[Pattern]) -> List[TrainingExample]:
        """
        Synthesize training examples from patterns.

        Args:
            patterns: List of extracted patterns

        Returns:
            List of training examples ready for fine-tuning
        """
        examples = []

        for pattern in patterns:
            # Filter by confidence
            if pattern.confidence < self.config.min_confidence:
                continue

            # Synthesize based on pattern type
            if pattern.pattern_type == PatternType.QA_PAIR:
                ex = self._synthesize_qa(pattern)
                examples.extend(ex)

            elif pattern.pattern_type == PatternType.REASONING_CHAIN:
                ex = self._synthesize_reasoning(pattern)
                examples.extend(ex)

            elif pattern.pattern_type == PatternType.CAUSAL:
                ex = self._synthesize_causal(pattern)
                examples.extend(ex)

            elif pattern.pattern_type == PatternType.DECISION:
                ex = self._synthesize_decision(pattern)
                examples.extend(ex)

            elif pattern.pattern_type == PatternType.COMPARISON:
                ex = self._synthesize_comparison(pattern)
                examples.extend(ex)

            elif pattern.pattern_type == PatternType.PROCEDURE:
                ex = self._synthesize_procedure(pattern)
                examples.extend(ex)

            elif pattern.pattern_type == PatternType.DEFINITION:
                ex = self._synthesize_definition(pattern)
                examples.extend(ex)

        return examples

    def _synthesize_qa(self, pattern: Pattern) -> List[TrainingExample]:
        """Synthesize Q&A training example."""
        content = pattern.content

        instruction = content['question']
        output = content['answer']

        # Add context if enabled
        input_context = ""
        if self.config.include_context:
            if content.get('entities'):
                input_context += f"Entities: {', '.join(content['entities'])}\n"
            if content.get('topics'):
                input_context += f"Topics: {', '.join(content['topics'])}\n"

        example = TrainingExample(
            instruction=instruction,
            input=input_context,
            output=output,
            system=self.config.system_prompt,
            metadata={
                'pattern_type': 'qa_pair',
                'confidence': pattern.confidence,
                'source': pattern.source_memories
            }
        )

        return [example]

    def _synthesize_reasoning(self, pattern: Pattern) -> List[TrainingExample]:
        """Synthesize reasoning chain example."""
        content = pattern.content

        instruction = f"Explain the reasoning: {content['premise']}"

        # Build chain-of-thought output
        if self.config.include_reasoning:
            reasoning_steps = "\n\n".join([
                f"Step {i+1}: {step}"
                for i, step in enumerate(content['steps'])
            ])
            output = f"{reasoning_steps}\n\nConclusion: {content['conclusion']}"
        else:
            output = content['conclusion']

        example = TrainingExample(
            instruction=instruction,
            output=output,
            system=self.config.system_prompt,
            metadata={
                'pattern_type': 'reasoning_chain',
                'step_count': len(content['steps']),
                'confidence': pattern.confidence
            }
        )

        return [example]

    def _synthesize_causal(self, pattern: Pattern) -> List[TrainingExample]:
        """Synthesize causal relationship example."""
        content = pattern.content

        instruction = f"What happens if {content['cause']}?"
        output = f"If {content['cause']}, then {content['effect']}."

        example = TrainingExample(
            instruction=instruction,
            output=output,
            system=self.config.system_prompt,
            metadata={
                'pattern_type': 'causal',
                'confidence': pattern.confidence
            }
        )

        return [example]

    def _synthesize_decision(self, pattern: Pattern) -> List[TrainingExample]:
        """Synthesize decision pattern example."""
        content = pattern.content

        instruction = f"Given the context: {content['context']}, what should be done?"
        output = f"Based on the situation, {content['decision']}.\n\nReasoning: {content['reasoning']}"

        example = TrainingExample(
            instruction=instruction,
            output=output,
            system=self.config.system_prompt,
            metadata={
                'pattern_type': 'decision',
                'confidence': pattern.confidence
            }
        )

        return [example]

    def _synthesize_comparison(self, pattern: Pattern) -> List[TrainingExample]:
        """Synthesize comparison example."""
        content = pattern.content

        instruction = f"Compare {content['item_a']} and {content['item_b']}"
        output = content['comparison']

        example = TrainingExample(
            instruction=instruction,
            output=output,
            system=self.config.system_prompt,
            metadata={
                'pattern_type': 'comparison',
                'confidence': pattern.confidence
            }
        )

        return [example]

    def _synthesize_procedure(self, pattern: Pattern) -> List[TrainingExample]:
        """Synthesize procedure example."""
        content = pattern.content

        instruction = f"How to: {content['task']}"

        # Format steps
        steps_formatted = "\n".join([
            f"{i+1}. {step}"
            for i, step in enumerate(content['steps'])
        ])

        output = f"Here are the steps:\n\n{steps_formatted}"

        example = TrainingExample(
            instruction=instruction,
            output=output,
            system=self.config.system_prompt,
            metadata={
                'pattern_type': 'procedure',
                'step_count': len(content['steps']),
                'confidence': pattern.confidence
            }
        )

        return [example]

    def _synthesize_definition(self, pattern: Pattern) -> List[TrainingExample]:
        """Synthesize definition example."""
        content = pattern.content

        instruction = f"What is {content['term']}?"
        output = f"{content['term']} is {content['definition']}."

        # Add context if available
        if self.config.include_context and content.get('context'):
            output += f"\n\nContext: {content['context']}"

        example = TrainingExample(
            instruction=instruction,
            output=output,
            system=self.config.system_prompt,
            metadata={
                'pattern_type': 'definition',
                'confidence': pattern.confidence
            }
        )

        return [example]

    def export_jsonl(self, examples: List[TrainingExample],
                    output_file: str, format: str = 'alpaca'):
        """
        Export training examples to JSONL file.

        Args:
            examples: List of training examples
            output_file: Output file path
            format: Export format ('alpaca', 'chatml', 'raw')
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                if format == 'alpaca':
                    data = example.to_alpaca()
                elif format == 'chatml':
                    data = {'messages': example.to_chatml()}
                else:
                    data = example.to_raw()

                f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def export_statistics(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """
        Get statistics about synthesized data.

        Args:
            examples: List of training examples

        Returns:
            Statistics dict
        """
        pattern_types = {}
        total_length = 0
        confidences = []

        for example in examples:
            # Count by pattern type
            ptype = example.metadata.get('pattern_type', 'unknown')
            pattern_types[ptype] = pattern_types.get(ptype, 0) + 1

            # Track length
            total_length += len(example.instruction) + len(example.output)

            # Track confidence
            if 'confidence' in example.metadata:
                confidences.append(example.metadata['confidence'])

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        avg_length = total_length / len(examples) if examples else 0

        return {
            'total_examples': len(examples),
            'pattern_types': pattern_types,
            'avg_length': avg_length,
            'avg_confidence': avg_confidence,
            'high_confidence_count': sum(1 for c in confidences if c >= 0.7)
        }
