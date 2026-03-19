"""
ELEGANCE Refinement Strategy
=============================

Three-pass refinement: Clarity → Simplicity → Beauty

Philosophy: "Great writing isn't written, it's refined."

Pass 1 (CLARITY): Make it understandable
  - Break up long sentences
  - Replace jargon with plain language
  - Add structure (headers, paragraphs)

Pass 2 (SIMPLICITY): Remove unnecessary
  - Cut filler words
  - Eliminate redundancy
  - Consolidate related ideas

Pass 3 (BEAUTY): Add grace
  - Vary sentence structure
  - Use active voice
  - Polish word choice
"""

import logging
import re

from ..core.protocol import RefinementPass, WritingContext

logger = logging.getLogger(__name__)


class EleganceRefiner:
    """
    ELEGANCE refinement strategy.

    Three passes:
    1. Clarity - Make understandable
    2. Simplicity - Remove unnecessary
    3. Beauty - Add grace
    """

    def __init__(self):
        self.logger = logger
        self._passes = ['clarity', 'simplicity', 'beauty']

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
        if focus == 'clarity':
            output, improvements = self._refine_clarity(text)
        elif focus == 'simplicity':
            output, improvements = self._refine_simplicity(text)
        else:  # beauty
            output, improvements = self._refine_beauty(text)

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

    def _refine_clarity(self, text: str) -> tuple[str, list[str]]:
        """
        Pass 1: Clarity - Make understandable.

        Args:
            text: Input text

        Returns:
            (refined_text, list_of_improvements)
        """
        improvements = []
        refined = text

        # 1. Break up long sentences (>35 words)
        sentences = refined.split('.')
        new_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            words = sentence.split()
            if len(words) > 35:
                # Try to split at conjunctions
                for conj in [', and', ', but', ', or', '; however', '; therefore']:
                    if conj in sentence:
                        parts = sentence.split(conj, 1)
                        new_sentences.append(parts[0].strip())
                        new_sentences.append(parts[1].strip())
                        improvements.append(f"Split long sentence at '{conj.strip()}'")
                        break
                else:
                    new_sentences.append(sentence)
            else:
                new_sentences.append(sentence)

        refined = '. '.join(new_sentences)
        if not refined.endswith('.'):
            refined += '.'

        # 2. Add paragraph breaks for better structure
        # If no paragraphs exist, add breaks at logical points
        if '\n\n' not in refined:
            # Add break every 3-4 sentences
            sentences = [s.strip() for s in refined.split('.') if s.strip()]
            paragraphs = []
            current_para = []

            for i, sentence in enumerate(sentences):
                current_para.append(sentence)
                if (i + 1) % 4 == 0 and i < len(sentences) - 1:
                    paragraphs.append('. '.join(current_para) + '.')
                    current_para = []
                    improvements.append("Added paragraph break")

            if current_para:
                paragraphs.append('. '.join(current_para) + '.')

            refined = '\n\n'.join(paragraphs)

        # 3. Replace common jargon with plain language
        jargon_replacements = {
            'utilize': 'use',
            'facilitate': 'help',
            'implement': 'add',
            'demonstrate': 'show',
            'indicates': 'shows',
            'sufficient': 'enough',
            'purchase': 'buy',
            'endeavor': 'try',
            'commence': 'start',
            'terminate': 'end'
        }

        for jargon, plain in jargon_replacements.items():
            pattern = r'\b' + jargon + r'\b'
            if re.search(pattern, refined, re.IGNORECASE):
                refined = re.sub(pattern, plain, refined, flags=re.IGNORECASE)
                improvements.append(f"Simplified '{jargon}' to '{plain}'")

        return refined, improvements

    def _refine_simplicity(self, text: str) -> tuple[str, list[str]]:
        """
        Pass 2: Simplicity - Remove unnecessary.

        Args:
            text: Input text

        Returns:
            (refined_text, list_of_improvements)
        """
        improvements = []
        refined = text

        # 1. Remove filler words
        filler_words = [
            r'\bvery\s+',
            r'\breally\s+',
            r'\bquite\s+',
            r'\bjust\s+',
            r'\bactually\s+',
            r'\bbasically\s+',
            r'\bliterally\s+',
            r'\bessentially\s+',
            r'\bpractically\s+',
            r'\bvirtually\s+'
        ]

        for pattern in filler_words:
            matches = re.findall(pattern, refined, re.IGNORECASE)
            if matches:
                refined = re.sub(pattern, '', refined, flags=re.IGNORECASE)
                improvements.append(f"Removed filler word '{matches[0].strip()}'")

        # 2. Eliminate redundant phrases
        redundant_phrases = {
            r'\bin order to\b': 'to',
            r'\bdue to the fact that\b': 'because',
            r'\bat this point in time\b': 'now',
            r'\bin the event that\b': 'if',
            r'\bfor the purpose of\b': 'to',
            r'\bhas the ability to\b': 'can',
            r'\bis able to\b': 'can',
            r'\bin spite of the fact that\b': 'although'
        }

        for verbose, concise in redundant_phrases.items():
            if re.search(verbose, refined, re.IGNORECASE):
                refined = re.sub(verbose, concise, refined, flags=re.IGNORECASE)
                improvements.append(f"Simplified redundant phrase to '{concise}'")

        # 3. Remove excessive modifiers
        # Pattern: adjective + adjective + noun -> adjective + noun
        # This is simplified - real implementation would use NLP
        double_adj_pattern = r'\b(\w+ly)\s+(\w+)\s+(\w+)\b'
        matches = re.findall(double_adj_pattern, refined)
        if matches:
            # Remove the adverb (first modifier)
            refined = re.sub(double_adj_pattern, r'\2 \3', refined)
            improvements.append("Removed excessive modifiers")

        # 4. Consolidate whitespace
        # Remove multiple spaces
        refined = re.sub(r' +', ' ', refined)
        # Remove multiple newlines (keep paragraph breaks)
        refined = re.sub(r'\n\n+', '\n\n', refined)

        return refined, improvements

    def _refine_beauty(self, text: str) -> tuple[str, list[str]]:
        """
        Pass 3: Beauty - Add grace.

        Args:
            text: Input text

        Returns:
            (refined_text, list_of_improvements)
        """
        improvements = []
        refined = text

        # 1. Convert passive voice to active (simple cases)
        passive_patterns = [
            (r'\bwas (\w+ed)\b', r'\1'),  # "was implemented" -> "implemented"
            (r'\bwere (\w+ed)\b', r'\1'),  # "were created" -> "created"
            (r'\bis being (\w+ed)\b', r'\1'),  # "is being used" -> "uses"
        ]

        for pattern, replacement in passive_patterns:
            matches = re.findall(pattern, refined, re.IGNORECASE)
            if matches:
                refined = re.sub(pattern, replacement, refined, flags=re.IGNORECASE)
                improvements.append("Converted passive voice to active")
                break  # Only report once

        # 2. Vary sentence openings
        # Check if too many sentences start with "The"
        sentences = [s.strip() for s in refined.split('.') if s.strip()]
        the_count = sum(1 for s in sentences if s.startswith('The '))

        if the_count > len(sentences) * 0.4:  # More than 40% start with "The"
            # Replace some with alternatives
            for i, sentence in enumerate(sentences):
                if sentence.startswith('The ') and i % 2 == 0:  # Every other one
                    # Try to move "The X" to later in sentence
                    parts = sentence.split(',', 1)
                    if len(parts) > 1:
                        sentences[i] = parts[1].strip() + ', ' + parts[0].lower()
                        improvements.append("Varied sentence opening")
                        break  # Only do a few

            refined = '. '.join(sentences)
            if not refined.endswith('.'):
                refined += '.'

        # 3. Strengthen weak verbs
        weak_verb_replacements = {
            r'\bmake\s+use\s+of\b': 'use',
            r'\bgive\s+consideration\s+to\b': 'consider',
            r'\bprovide\s+assistance\s+to\b': 'assist',
            r'\btake\s+action\b': 'act',
            r'\bmake\s+a\s+decision\b': 'decide',
            r'\bcome\s+to\s+a\s+conclusion\b': 'conclude'
        }

        for weak, strong in weak_verb_replacements.items():
            if re.search(weak, refined, re.IGNORECASE):
                refined = re.sub(weak, strong, refined, flags=re.IGNORECASE)
                improvements.append(f"Strengthened verb to '{strong}'")

        # 4. Polish transitions
        # Ensure good paragraph transitions
        paragraphs = refined.split('\n\n')
        if len(paragraphs) > 1:
            # Check if paragraphs have good connectors
            connectors = [
                'However', 'Furthermore', 'Moreover', 'Additionally',
                'Nevertheless', 'Consequently', 'Therefore', 'Thus'
            ]

            for i in range(1, len(paragraphs)):
                para = paragraphs[i].strip()
                # If paragraph doesn't start with connector, consider adding one
                if not any(para.startswith(c) for c in connectors):
                    # Add subtle transition (simplified logic)
                    if 'but' in para.lower() or 'although' in para.lower():
                        # Already has implicit contrast
                        pass
                    else:
                        # Could add transition, but don't force it
                        pass

            improvements.append("Reviewed paragraph transitions")

        return refined, improvements

    @property
    def strategy_name(self) -> str:
        """Strategy name for this refiner."""
        return "elegance"

    @property
    def passes(self) -> list[str]:
        """List of focus areas for each pass."""
        return self._passes.copy()
