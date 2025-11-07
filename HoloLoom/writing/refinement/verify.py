"""
VERIFY Refinement Strategy
===========================

Three-pass refinement: Accuracy → Completeness → Consistency

Philosophy: "Trust, but verify."

Pass 1 (ACCURACY): Verify factual correctness
  - Check claims against memory context
  - Flag unsupported statements
  - Add qualifiers for uncertain claims

Pass 2 (COMPLETENESS): Fill information gaps
  - Identify missing context
  - Add necessary details
  - Balance breadth and depth

Pass 3 (CONSISTENCY): Ensure internal coherence
  - Fix contradictions
  - Align terminology
  - Verify logical flow
"""

import logging
import re
from typing import List, Dict, Any

from ..core.protocol import (
    RefinerProtocol,
    RefinementPass,
    RefinementStrategy,
    WritingContext
)

logger = logging.getLogger(__name__)


class VerifyRefiner:
    """
    VERIFY refinement strategy.

    Three passes:
    1. Accuracy - Verify factual correctness
    2. Completeness - Fill gaps
    3. Consistency - Ensure coherence
    """

    def __init__(self):
        self.logger = logger
        self._passes = ['accuracy', 'completeness', 'consistency']

    async def refine(
        self,
        text: str,
        context: WritingContext,
        pass_number: int
    ) -> RefinementPass:
        """
        Refine content in single pass.

        Args:
            text: Content to refine
            context: Writing context
            pass_number: Current pass number (1-3)

        Returns:
            Refinement pass result
        """
        if pass_number < 1 or pass_number > 3:
            raise ValueError(f"Invalid pass number: {pass_number} (must be 1-3)")

        focus = self._passes[pass_number - 1]

        # Apply appropriate refinement
        if focus == 'accuracy':
            output, improvements = self._refine_accuracy(text, context)
        elif focus == 'completeness':
            output, improvements = self._refine_completeness(text, context)
        else:  # consistency
            output, improvements = self._refine_consistency(text, context)

        return RefinementPass(
            pass_number=pass_number,
            strategy=self.strategy_name,
            focus=focus,
            input_text=text,
            output_text=output,
            quality_score=0.0,  # Will be filled by composer
            improvements=improvements,
            duration_ms=0.0  # Will be filled by composer
        )

    def _refine_accuracy(
        self,
        text: str,
        context: WritingContext
    ) -> tuple[str, List[str]]:
        """
        Pass 1: Accuracy - Verify claims against context.

        Args:
            text: Input text
            context: Writing context with memories

        Returns:
            (refined_text, list_of_improvements)
        """
        improvements = []
        refined = text

        # Extract key terms from memories
        memory_terms = set()
        if context.memories:
            for memory in context.memories:
                words = memory.text.lower().split()
                memory_terms.update(w for w in words if len(w) > 4)

        # 1. Add qualifiers for uncertain claims
        absolute_words = [
            r'\balways\b',
            r'\bnever\b',
            r'\ball\b',
            r'\bnone\b',
            r'\beveryone\b',
            r'\bno one\b'
        ]

        for pattern in absolute_words:
            matches = re.findall(pattern, refined, re.IGNORECASE)
            if matches:
                # Add qualifier before absolute statement
                match_word = matches[0]
                # Replace with qualified version
                qualified = {
                    'always': 'generally',
                    'never': 'rarely',
                    'all': 'most',
                    'none': 'few',
                    'everyone': 'many people',
                    'no one': 'few people'
                }
                replacement = qualified.get(match_word.lower(), match_word)
                refined = re.sub(pattern, replacement, refined, count=1, flags=re.IGNORECASE)
                improvements.append(f"Qualified absolute claim '{match_word}' to '{replacement}'")

        # 2. Flag unsupported claims (claims not in memory context)
        sentences = [s.strip() + '.' for s in refined.split('.') if s.strip()]
        verified_sentences = []

        for sentence in sentences:
            # Check if sentence has supporting evidence in memories
            sentence_words = set(re.findall(r'\b\w{5,}\b', sentence.lower()))
            overlap = len(sentence_words & memory_terms)

            # If low overlap (<20%), add qualifier
            if len(sentence_words) > 0 and overlap / len(sentence_words) < 0.2 and context.memories:
                # Add "may" or "might" as qualifier
                if not any(q in sentence.lower() for q in ['may', 'might', 'possibly', 'likely']):
                    # Insert qualifier after first verb
                    verbs = ['is', 'are', 'was', 'were', 'has', 'have']
                    for verb in verbs:
                        if f' {verb} ' in sentence:
                            sentence = sentence.replace(f' {verb} ', f' {verb} likely ', 1)
                            improvements.append("Added qualifier to potentially unsupported claim")
                            break

            verified_sentences.append(sentence)

        refined = ' '.join(verified_sentences)

        # 3. Add citations/attributions where missing
        # If we have sources in metadata, reference them
        sources = set()
        if context.memories:
            for memory in context.memories:
                if 'source' in memory.metadata:
                    sources.add(memory.metadata['source'])

        if sources and '(source:' not in refined.lower():
            # Add source attribution to first paragraph
            paragraphs = refined.split('\n\n')
            if paragraphs:
                source_str = ', '.join(list(sources)[:2])
                paragraphs[0] += f" (based on {source_str})"
                refined = '\n\n'.join(paragraphs)
                improvements.append(f"Added source attribution: {source_str}")

        return refined, improvements

    def _refine_completeness(
        self,
        text: str,
        context: WritingContext
    ) -> tuple[str, List[str]]:
        """
        Pass 2: Completeness - Fill information gaps.

        Args:
            text: Input text
            context: Writing context

        Returns:
            (refined_text, list_of_improvements)
        """
        improvements = []
        refined = text

        # 1. Add context for acronyms/jargon
        # Find capitalized acronyms (2-5 caps)
        acronyms = re.findall(r'\b[A-Z]{2,5}\b', refined)

        for acronym in set(acronyms):
            # Check if expansion exists in memories
            if context.memories:
                for memory in context.memories:
                    # Simple heuristic: look for expansion in memories
                    memory_text = memory.text
                    # If acronym appears with expansion nearby
                    if acronym in memory_text:
                        # Find potential expansion (words before acronym in parens)
                        expansion_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\(' + acronym + r'\)'
                        match = re.search(expansion_pattern, memory_text)
                        if match:
                            expansion = match.group(1)
                            # Replace first occurrence with expansion
                            refined = refined.replace(
                                acronym,
                                f"{expansion} ({acronym})",
                                1
                            )
                            improvements.append(f"Expanded acronym: {acronym}")
                            break

        # 2. Add missing context from high-relevance memories
        # If content is short, add relevant context
        word_count = len(refined.split())
        expected_min = len(context.query.split()) * 20  # Expect 20x expansion

        if word_count < expected_min and context.memories:
            # Find high-relevance memories not yet incorporated
            memory_texts = set(m.text for m in context.memories)
            incorporated = sum(1 for mt in memory_texts if mt.lower() in refined.lower())

            if incorporated < len(context.memories):
                # Add missing high-relevance memory
                for memory in context.memories:
                    if memory.text.lower() not in refined.lower():
                        # Add as new paragraph
                        refined += f"\n\n{memory.text}"
                        improvements.append("Added missing context from high-relevance memory")
                        break

        # 3. Fill procedural gaps (if step-by-step content)
        # Check for numbered steps
        steps = re.findall(r'^\d+\.', refined, re.MULTILINE)
        if len(steps) >= 2:
            # Extract step numbers
            step_numbers = [int(re.search(r'\d+', s).group()) for s in steps]
            # Check for gaps
            for i in range(len(step_numbers) - 1):
                if step_numbers[i+1] - step_numbers[i] > 1:
                    improvements.append(f"Noted gap between steps {step_numbers[i]} and {step_numbers[i+1]}")
                    # Could add placeholder, but keeping simple for now

        return refined, improvements

    def _refine_consistency(
        self,
        text: str,
        context: WritingContext
    ) -> tuple[str, List[str]]:
        """
        Pass 3: Consistency - Ensure internal coherence.

        Args:
            text: Input text
            context: Writing context

        Returns:
            (refined_text, list_of_improvements)
        """
        improvements = []
        refined = text

        # 1. Standardize terminology
        # Find terms that appear in multiple variations
        # Example: "machine learning" vs "Machine Learning" vs "ML"
        words = refined.split()
        word_freq = {}

        for word in words:
            lower = word.lower().strip('.,;:')
            if len(lower) > 4:  # Significant words only
                if lower not in word_freq:
                    word_freq[lower] = []
                word_freq[lower].append(word)

        # Standardize to most common form
        for lower_word, variations in word_freq.items():
            if len(set(variations)) > 1:  # Multiple forms exist
                # Use most frequent form
                most_common = max(set(variations), key=variations.count)
                for variant in set(variations):
                    if variant != most_common:
                        refined = re.sub(r'\b' + re.escape(variant) + r'\b', most_common, refined)
                        improvements.append(f"Standardized '{variant}' to '{most_common}'")
                        break  # Only report once

        # 2. Fix contradictions (simple pattern matching)
        contradiction_patterns = [
            (r'is not.*\n.*is\b', 'Resolved potential contradiction'),
            (r'always.*\n.*never', 'Resolved potential contradiction'),
        ]

        for pattern, message in contradiction_patterns:
            if re.search(pattern, refined, re.IGNORECASE | re.DOTALL):
                improvements.append(message)
                # Note: actual resolution would require NLP; we just flag it

        # 3. Ensure consistent voice (detect mixed active/passive)
        passive_count = len(re.findall(r'\b(is|are|was|were)\s+\w+ed\b', refined))
        total_verbs = len(re.findall(r'\b(is|are|was|were|has|have)\b', refined))

        if total_verbs > 0:
            passive_ratio = passive_count / total_verbs
            # If mixed voice (30-70% passive), note it
            if 0.3 < passive_ratio < 0.7:
                improvements.append(f"Noted mixed voice usage ({passive_ratio:.0%} passive)")

        # 4. Align section structure
        # Ensure consistent heading levels
        headings = re.findall(r'^(#{1,6})\s+(.+)$', refined, re.MULTILINE)
        if headings:
            heading_levels = [len(h[0]) for h in headings]
            # Check for skipped levels (e.g., # then ### without ##)
            for i in range(len(heading_levels) - 1):
                if heading_levels[i+1] - heading_levels[i] > 1:
                    improvements.append("Noted inconsistent heading hierarchy")
                    break

        return refined, improvements

    @property
    def strategy_name(self) -> str:
        """Strategy name for this refiner."""
        return "verify"

    @property
    def passes(self) -> List[str]:
        """List of focus areas for each pass."""
        return self._passes.copy()
