"""
Ernest Refinement Patterns
==========================

Hem

ingway-enhanced 3-pass refinement: Clarity → Simplicity → Beauty (Grace)

Based on HoloLoom/writing/refinement/elegance.py with Hemingway-specific rules.

Philosophy: "The first draft of anything is shit." - Ernest Hemingway

Pass 1 (CLARITY): Short sentences, plain language
Pass 2 (SIMPLICITY): Cut filler, strengthen verbs
Pass 3 (BEAUTY/GRACE): Active voice, precise nouns, rhythm
"""

import re
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from enum import Enum


class HemingwayMode(Enum):
    """Writing modes inspired by Hemingway."""
    SPARSE = "sparse"      # Iceberg maximum - 90% subtext
    DIRECT = "direct"      # Journalistic precision
    GRACE = "grace"        # Economical elegance
    LOST_GEN = "lost_gen"  # Disaffected truth


@dataclass
class ProseMetrics:
    """Hemingway-style prose quality metrics."""
    # Conciseness
    avg_words_per_sentence: float
    total_sentences: int
    total_words: int

    # Voice
    active_voice_pct: float
    passive_voice_count: int

    # Verb strength
    strong_verb_pct: float
    weak_verb_count: int

    # Readability
    flesch_kincaid_grade: float

    # Hemingway specific
    iceberg_ratio: float  # % subtext (showing vs telling)
    filler_word_count: int
    emotion_word_count: int  # Direct emotion statements

    # Overall
    hemingway_score: float  # 0-100 composite score


@dataclass
class RefinementResult:
    """Result of refinement pass."""
    refined_text: str
    improvements: List[str]
    before_metrics: ProseMetrics
    after_metrics: ProseMetrics


class HemingwayPatterns:
    """
    Hemingway-specific prose patterns and refinement rules.

    Based on Hemingway principles:
    - Iceberg Theory (show, don't tell)
    - Short, declarative sentences
    - Active voice dominance
    - Strong, precise verbs
    - Concrete nouns over abstractions
    - Minimal adverbs and adjectives
    """

    # Hemingway targets (stricter than standard elegance)
    TARGET_AVG_SENTENCE_LENGTH = 16  # words (Hemingway avg: 15-20)
    MAX_SENTENCE_LENGTH = 25  # words (vs 35 in standard elegance)
    TARGET_ACTIVE_VOICE_PCT = 85  # % (vs general 70%)
    TARGET_STRONG_VERB_PCT = 70  # %
    TARGET_FLESCH_GRADE = 7  # Grade level (6-8 is ideal)
    TARGET_ICEBERG_RATIO = 70  # % showing vs telling

    # Filler words to eliminate (Hemingway hated these)
    FILLER_WORDS = {
        'very', 'really', 'quite', 'just', 'actually', 'basically',
        'literally', 'essentially', 'practically', 'virtually',
        'somewhat', 'rather', 'fairly', 'pretty',  # as adverbs
        'totally', 'absolutely', 'completely', 'extremely'
    }

    # Weak verbs to replace
    WEAK_VERBS = {
        'is', 'are', 'was', 'were', 'been', 'being',
        'have', 'has', 'had',
        'do', 'does', 'did',
        'get', 'gets', 'got', 'gotten',
        'make', 'makes', 'made',
        'go', 'goes', 'went', 'gone'
    }

    # Emotion words (telling, not showing)
    EMOTION_WORDS = {
        'happy', 'sad', 'angry', 'scared', 'afraid', 'nervous',
        'excited', 'bored', 'tired', 'frustrated', 'annoyed',
        'depressed', 'anxious', 'worried', 'jealous', 'proud',
        'ashamed', 'embarrassed', 'confused', 'surprised'
    }

    # Telling verbs (iceberg violations)
    TELLING_VERBS = {
        'felt', 'thought', 'knew', 'realized', 'understood',
        'believed', 'supposed', 'imagined', 'wondered',
        'seemed', 'appeared'
    }

    # Redundant phrases
    REDUNDANT_PHRASES = {
        'in order to': 'to',
        'due to the fact that': 'because',
        'at this point in time': 'now',
        'for the purpose of': 'to',
        'in the event that': 'if',
        'with the exception of': 'except',
        'has the ability to': 'can',
        'is able to': 'can',
        'make use of': 'use',
        'give consideration to': 'consider',
        'take action': 'act',
        'come to the conclusion': 'conclude',
        'at the present time': 'now',
        'in spite of the fact': 'despite',
        'on account of': 'because'
    }

    # Jargon to plain language
    JARGON_MAP = {
        'utilize': 'use',
        'facilitate': 'help',
        'implement': 'do',
        'optimize': 'improve',
        'leverage': 'use',
        'paradigm': 'pattern',
        'synergy': 'teamwork',
        'ideate': 'think',
        'actualize': 'make real',
        'prioritize': 'rank'
    }

    def __init__(self, mode: HemingwayMode = HemingwayMode.GRACE):
        """
        Initialize with specific Hemingway mode.

        Args:
            mode: Writing mode (SPARSE/DIRECT/GRACE/LOST_GEN)
        """
        self.mode = mode

        # Adjust targets based on mode
        if mode == HemingwayMode.SPARSE:
            self.TARGET_AVG_SENTENCE_LENGTH = 10  # Ultra-short
            self.TARGET_ICEBERG_RATIO = 90  # Maximum subtext
        elif mode == HemingwayMode.DIRECT:
            self.TARGET_AVG_SENTENCE_LENGTH = 14  # Journalistic
            self.TARGET_ACTIVE_VOICE_PCT = 95  # Almost all active

    def analyze_prose(self, text: str) -> ProseMetrics:
        """
        Analyze prose quality using Hemingway metrics.

        Args:
            text: Text to analyze

        Returns:
            ProseMetrics with all scores
        """
        # Basic counts
        sentences = self._split_sentences(text)
        total_sentences = len(sentences)
        total_words = len(text.split())
        avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0

        # Active vs passive voice
        passive_count = self._count_passive_voice(text)
        active_pct = ((total_sentences - passive_count) / total_sentences * 100) if total_sentences > 0 else 0

        # Verb strength
        weak_verbs = self._count_weak_verbs(text)
        total_verbs = self._count_total_verbs(text)
        strong_verb_pct = ((total_verbs - weak_verbs) / total_verbs * 100) if total_verbs > 0 else 0

        # Readability (simplified Flesch-Kincaid)
        fk_grade = self._flesch_kincaid_grade(text)

        # Hemingway-specific
        iceberg_ratio = self._iceberg_ratio(text)
        filler_count = self._count_filler_words(text)
        emotion_count = self._count_emotion_words(text)

        # Composite Hemingway score
        hemingway_score = self._calculate_hemingway_score(
            avg_words_per_sentence,
            active_pct,
            strong_verb_pct,
            fk_grade,
            iceberg_ratio
        )

        return ProseMetrics(
            avg_words_per_sentence=avg_words_per_sentence,
            total_sentences=total_sentences,
            total_words=total_words,
            active_voice_pct=active_pct,
            passive_voice_count=passive_count,
            strong_verb_pct=strong_verb_pct,
            weak_verb_count=weak_verbs,
            flesch_kincaid_grade=fk_grade,
            iceberg_ratio=iceberg_ratio,
            filler_word_count=filler_count,
            emotion_word_count=emotion_count,
            hemingway_score=hemingway_score
        )

    def refine_pass1_clarity(self, text: str) -> RefinementResult:
        """
        Pass 1: CLARITY - Short sentences, plain language.

        Hemingway targets:
        - Break sentences >25 words (not 35)
        - Replace jargon with Anglo-Saxon words
        - Paragraph every 3-4 sentences

        Args:
            text: Input text

        Returns:
            Refined text with improvements list
        """
        before_metrics = self.analyze_prose(text)
        improvements = []
        refined = text

        # 1. Break up long sentences (>25 words for Hemingway)
        sentences = self._split_sentences(refined)
        new_sentences = []

        for sentence in sentences:
            words = sentence.split()
            if len(words) > self.MAX_SENTENCE_LENGTH:
                # Try to split at conjunctions
                split = False
                for conj in [', and', ', but', ', or', '; however', '; therefore', '; yet']:
                    if conj in sentence:
                        parts = sentence.split(conj, 1)
                        new_sentences.append(parts[0].strip() + '.')
                        new_sentences.append(parts[1].strip())
                        improvements.append(f"Split {len(words)}-word sentence at '{conj.strip()}'")
                        split = True
                        break

                if not split:
                    # Try to split at semicolon
                    if ';' in sentence:
                        parts = sentence.split(';', 1)
                        new_sentences.append(parts[0].strip() + '.')
                        new_sentences.append(parts[1].strip())
                        improvements.append(f"Split {len(words)}-word sentence at semicolon")
                    else:
                        new_sentences.append(sentence)
            else:
                new_sentences.append(sentence)

        refined = ' '.join(new_sentences)

        # 2. Replace jargon with plain language
        for jargon, plain in self.JARGON_MAP.items():
            pattern = r'\b' + jargon + r'\b'
            if re.search(pattern, refined, re.IGNORECASE):
                refined = re.sub(pattern, plain, refined, flags=re.IGNORECASE)
                improvements.append(f"Replaced '{jargon}' with '{plain}'")

        # 3. Add paragraph breaks
        if '\n\n' not in refined:
            sentences = self._split_sentences(refined)
            paragraphs = []
            current_para = []

            for i, sentence in enumerate(sentences):
                current_para.append(sentence)
                if (i + 1) % 4 == 0 and i < len(sentences) - 1:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
                    improvements.append("Added paragraph break")

            if current_para:
                paragraphs.append(' '.join(current_para))

            refined = '\n\n'.join(paragraphs)

        after_metrics = self.analyze_prose(refined)

        return RefinementResult(
            refined_text=refined,
            improvements=improvements,
            before_metrics=before_metrics,
            after_metrics=after_metrics
        )

    def refine_pass2_simplicity(self, text: str) -> RefinementResult:
        """
        Pass 2: SIMPLICITY - Cut filler, strengthen verbs.

        Hemingway targets:
        - Remove all filler words
        - Replace weak verbs with strong verbs
        - Eliminate redundant phrases

        Args:
            text: Input text

        Returns:
            Refined text with improvements list
        """
        before_metrics = self.analyze_prose(text)
        improvements = []
        refined = text

        # 1. Remove filler words
        for filler in self.FILLER_WORDS:
            pattern = r'\b' + filler + r'\b'
            count = len(re.findall(pattern, refined, re.IGNORECASE))
            if count > 0:
                refined = re.sub(pattern, '', refined, flags=re.IGNORECASE)
                # Clean up double spaces
                refined = re.sub(r'\s+', ' ', refined)
                improvements.append(f"Removed '{filler}' ({count} occurrences)")

        # 2. Replace redundant phrases
        for redundant, replacement in self.REDUNDANT_PHRASES.items():
            if redundant in refined.lower():
                # Case-insensitive replacement
                pattern = re.compile(re.escape(redundant), re.IGNORECASE)
                refined = pattern.sub(replacement, refined)
                improvements.append(f"'{redundant}' → '{replacement}'")

        # 3. Flag weak verbs for manual replacement
        # (Auto-replacement too risky without context)
        weak_verbs_found = set()
        for weak_verb in self.WEAK_VERBS:
            pattern = r'\b' + weak_verb + r'\b'
            if re.search(pattern, refined, re.IGNORECASE):
                weak_verbs_found.add(weak_verb)

        if weak_verbs_found:
            improvements.append(f"Found weak verbs to consider replacing: {', '.join(sorted(weak_verbs_found))}")

        after_metrics = self.analyze_prose(refined)

        return RefinementResult(
            refined_text=refined,
            improvements=improvements,
            before_metrics=before_metrics,
            after_metrics=after_metrics
        )

    def refine_pass3_beauty(self, text: str) -> RefinementResult:
        """
        Pass 3: BEAUTY (Grace) - Active voice, precise nouns, rhythm.

        Hemingway targets:
        - Convert passive → active voice
        - Vary sentence openings
        - Polish rhythm (short-medium-long pattern)

        Args:
            text: Input text

        Returns:
            Refined text with improvements list
        """
        before_metrics = self.analyze_prose(text)
        improvements = []
        refined = text

        # 1. Detect passive voice (flag for manual fix)
        passive_patterns = [
            r'\b(was|were|is|are|been|being)\s+\w+ed\b',  # was caught
            r'\b(was|were|is|are|been|being)\s+\w+en\b',  # was written
        ]

        passive_count = 0
        for pattern in passive_patterns:
            passive_count += len(re.findall(pattern, refined, re.IGNORECASE))

        if passive_count > 0:
            improvements.append(f"Found {passive_count} passive voice constructions to consider converting to active")

        # 2. Detect repeated sentence openings
        sentences = self._split_sentences(refined)
        openings = {}
        for sentence in sentences:
            first_word = sentence.split()[0] if sentence.split() else ''
            openings[first_word] = openings.get(first_word, 0) + 1

        repeated = {word: count for word, count in openings.items() if count > len(sentences) * 0.4}
        if repeated:
            improvements.append(f"Consider varying sentence openings: {', '.join(repeated.keys())} used frequently")

        # 3. Check rhythm variety
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(set(sentence_lengths)) < len(sentence_lengths) / 3:
            improvements.append("Consider varying sentence length for natural rhythm")

        after_metrics = self.analyze_prose(refined)

        return RefinementResult(
            refined_text=refined,
            improvements=improvements,
            before_metrics=before_metrics,
            after_metrics=after_metrics
        )

    def refine_all_passes(self, text: str) -> List[RefinementResult]:
        """
        Run all 3 refinement passes in sequence.

        Args:
            text: Input text

        Returns:
            List of results for each pass
        """
        results = []

        # Pass 1: Clarity
        result1 = self.refine_pass1_clarity(text)
        results.append(result1)

        # Pass 2: Simplicity (on Pass 1 output)
        result2 = self.refine_pass2_simplicity(result1.refined_text)
        results.append(result2)

        # Pass 3: Beauty (on Pass 2 output)
        result3 = self.refine_pass3_beauty(result2.refined_text)
        results.append(result3)

        return results

    # ===== Helper Methods =====

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple split on period, exclamation, question mark
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_passive_voice(self, text: str) -> int:
        """Count passive voice constructions."""
        passive_patterns = [
            r'\b(was|were|is|are|been|being)\s+\w+ed\b',
            r'\b(was|were|is|are|been|being)\s+\w+en\b',
        ]
        count = 0
        for pattern in passive_patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count

    def _count_weak_verbs(self, text: str) -> int:
        """Count weak verbs."""
        count = 0
        for weak_verb in self.WEAK_VERBS:
            pattern = r'\b' + weak_verb + r'\b'
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count

    def _count_total_verbs(self, text: str) -> int:
        """Estimate total verb count (simplified)."""
        # This is a rough estimate - in production, use POS tagging
        words = text.split()
        # Assume ~15% of words are verbs (English average)
        return max(1, int(len(words) * 0.15))

    def _count_filler_words(self, text: str) -> int:
        """Count filler words."""
        count = 0
        for filler in self.FILLER_WORDS:
            pattern = r'\b' + filler + r'\b'
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count

    def _count_emotion_words(self, text: str) -> int:
        """Count direct emotion words (telling, not showing)."""
        count = 0
        for emotion in self.EMOTION_WORDS:
            pattern = r'\b' + emotion + r'\b'
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count

    def _iceberg_ratio(self, text: str) -> float:
        """
        Calculate iceberg ratio (showing vs telling).

        Simplified algorithm:
        - Telling: Uses felt/thought/knew/realized + emotion words
        - Showing: Action verbs, dialogue, sensory details

        Returns:
            Percentage showing (0-100)
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return 0.0

        telling_count = 0
        for sentence in sentences:
            # Check for telling verbs
            for telling_verb in self.TELLING_VERBS:
                if re.search(r'\b' + telling_verb + r'\b', sentence, re.IGNORECASE):
                    telling_count += 1
                    break

            # Check for emotion words
            for emotion in self.EMOTION_WORDS:
                if re.search(r'\b' + emotion + r'\b', sentence, re.IGNORECASE):
                    telling_count += 1
                    break

        showing_count = len(sentences) - telling_count
        showing_pct = (showing_count / len(sentences) * 100) if sentences else 0

        return showing_pct

    def _flesch_kincaid_grade(self, text: str) -> float:
        """Calculate Flesch-Kincaid grade level (simplified)."""
        sentences = self._split_sentences(text)
        words = text.split()

        if not sentences or not words:
            return 0.0

        # Count syllables (very rough approximation)
        syllables = sum(self._count_syllables(word) for word in words)

        # Flesch-Kincaid formula
        grade = (0.39 * (len(words) / len(sentences))) + (11.8 * (syllables / len(words))) - 15.59

        return max(0, grade)

    def _count_syllables(self, word: str) -> int:
        """Count syllables in word (rough approximation)."""
        word = word.lower()
        vowels = 'aeiou'
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Adjust for silent 'e'
        if word.endswith('e'):
            syllable_count -= 1

        # Every word has at least one syllable
        return max(1, syllable_count)

    def _calculate_hemingway_score(
        self,
        avg_sentence_len: float,
        active_voice_pct: float,
        strong_verb_pct: float,
        fk_grade: float,
        iceberg_ratio: float
    ) -> float:
        """
        Calculate composite Hemingway score (0-100).

        Weights:
        - Conciseness: 25%
        - Active voice: 20%
        - Verb strength: 20%
        - Readability: 15%
        - Iceberg ratio: 20%
        """
        # Conciseness score (closer to target = higher score)
        conciseness_diff = abs(avg_sentence_len - self.TARGET_AVG_SENTENCE_LENGTH)
        conciseness_score = max(0, 100 - (conciseness_diff * 5))

        # Active voice score
        active_voice_score = (active_voice_pct / self.TARGET_ACTIVE_VOICE_PCT) * 100
        active_voice_score = min(100, active_voice_score)

        # Verb strength score
        verb_strength_score = (strong_verb_pct / self.TARGET_STRONG_VERB_PCT) * 100
        verb_strength_score = min(100, verb_strength_score)

        # Readability score (closer to grade 7 = higher score)
        readability_diff = abs(fk_grade - self.TARGET_FLESCH_GRADE)
        readability_score = max(0, 100 - (readability_diff * 10))

        # Iceberg ratio score
        iceberg_score = (iceberg_ratio / self.TARGET_ICEBERG_RATIO) * 100
        iceberg_score = min(100, iceberg_score)

        # Weighted average
        composite = (
            conciseness_score * 0.25 +
            active_voice_score * 0.20 +
            verb_strength_score * 0.20 +
            readability_score * 0.15 +
            iceberg_score * 0.20
        )

        return round(composite, 1)


# ===== Quick Analysis Functions =====

def quick_analyze(text: str, mode: HemingwayMode = HemingwayMode.GRACE) -> ProseMetrics:
    """
    Quick prose analysis with Hemingway metrics.

    Args:
        text: Text to analyze
        mode: Hemingway mode

    Returns:
        ProseMetrics
    """
    analyzer = HemingwayPatterns(mode=mode)
    return analyzer.analyze_prose(text)


def quick_refine(text: str, mode: HemingwayMode = HemingwayMode.GRACE) -> str:
    """
    Quick 3-pass refinement.

    Args:
        text: Text to refine
        mode: Hemingway mode

    Returns:
        Refined text (after all 3 passes)
    """
    analyzer = HemingwayPatterns(mode=mode)
    results = analyzer.refine_all_passes(text)
    return results[-1].refined_text  # Return final pass output


def hemingway_score(text: str) -> float:
    """
    Get Hemingway score (0-100) for text.

    Args:
        text: Text to score

    Returns:
        Score (0-100)
    """
    metrics = quick_analyze(text)
    return metrics.hemingway_score
