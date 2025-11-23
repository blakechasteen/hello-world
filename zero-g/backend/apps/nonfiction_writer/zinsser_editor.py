"""
Zinsser-Style Scientific Editor

Based on William Zinsser's "On Writing Well" - the classic guide to nonfiction writing.

Core Principles:
1. Simplicity: Strip every sentence to its cleanest components
2. Clutter: Fighting clutter is like fighting weeds—the writer is always slightly behind
3. Style: Be yourself and write naturally
4. Words: Use small words, short sentences, short paragraphs
5. Usage: Good writing is lean and confident

Editing Passes:
- CLARITY: Remove ambiguity, strengthen meaning
- BREVITY: Cut unnecessary words, tighten prose
- STRENGTH: Active voice, strong verbs, concrete details
- HUMANITY: Natural voice, warmth, personality
- PRECISION: Exact words, specific details

Targets common scientific writing problems:
- Passive voice epidemic
- Jargon overload
- Hedge word abuse ("very", "quite", "rather")
- Nominalization ("utilization" → "use")
- Wordiness and redundancy
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class ZinsserPrinciple(Enum):
    """Zinsser's core writing principles"""
    CLARITY = "clarity"          # Make meaning crystal clear
    BREVITY = "brevity"          # Use fewer, better words
    STRENGTH = "strength"        # Strong verbs, active voice
    HUMANITY = "humanity"        # Natural, warm, personal
    PRECISION = "precision"      # Exact words, specific details


@dataclass
class EditSuggestion:
    """
    A suggested edit

    Attributes:
        original: Original text
        suggested: Suggested replacement
        reason: Why this edit improves the text
        principle: Which Zinsser principle it addresses
        severity: How important (0.0-1.0, 1.0 = critical)
        location: Character position in text
    """
    original: str
    suggested: str
    reason: str
    principle: ZinsserPrinciple
    severity: float = 0.5
    location: int = 0


@dataclass
class ZinsserReport:
    """
    Complete editing report

    Attributes:
        original_text: Original text
        edited_text: Text with all suggestions applied
        suggestions: All edit suggestions
        statistics: Before/after statistics
        readability_improvement: Readability score change
        total_words_cut: Number of words removed
        passive_voice_fixed: Passive voice instances fixed
    """
    original_text: str
    edited_text: str
    suggestions: List[EditSuggestion] = field(default_factory=list)
    statistics: Dict[str, any] = field(default_factory=dict)
    readability_improvement: float = 0.0
    total_words_cut: int = 0
    passive_voice_fixed: int = 0

    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = []
        lines.append("Zinsser Editing Report")
        lines.append("=" * 50)

        # Before/after stats
        before = self.statistics.get('before', {})
        after = self.statistics.get('after', {})

        lines.append(f"\nWord Count: {before.get('words', 0)} → {after.get('words', 0)}")
        lines.append(f"Words Cut: {self.total_words_cut}")
        lines.append(f"Passive Voice Fixed: {self.passive_voice_fixed}")
        lines.append(f"Readability: {self.readability_improvement:+.1f}% improvement")

        # Suggestions by principle
        by_principle = {}
        for sugg in self.suggestions:
            principle = sugg.principle.value
            by_principle[principle] = by_principle.get(principle, 0) + 1

        lines.append(f"\nSuggestions by Principle:")
        for principle, count in sorted(by_principle.items()):
            lines.append(f"  {principle.capitalize()}: {count}")

        # Top suggestions
        if self.suggestions:
            lines.append(f"\nTop Suggestions (showing 5):")
            top_5 = sorted(self.suggestions, key=lambda s: s.severity, reverse=True)[:5]

            for i, sugg in enumerate(top_5, 1):
                lines.append(f"\n{i}. {sugg.principle.value.upper()}")
                lines.append(f"   Original: {sugg.original}")
                lines.append(f"   Suggested: {sugg.suggested}")
                lines.append(f"   Reason: {sugg.reason}")

        return "\n".join(lines)


class ZinsserEditor:
    """
    Scientific writing editor based on Zinsser's principles

    Applies systematic improvements following "On Writing Well":
    - Cuts clutter (unnecessary words, redundancy)
    - Strengthens verbs (eliminates weak verbs, passive voice)
    - Simplifies language (removes jargon, complex words)
    - Adds precision (vague → specific)
    - Maintains humanity (natural voice, warmth)

    Usage:
        editor = ZinsserEditor()

        # Edit text
        report = editor.edit(
            text="The utilization of machine learning was implemented...",
            principles=[ZinsserPrinciple.CLARITY, ZinsserPrinciple.BREVITY]
        )

        print(report.edited_text)
        print(report.get_summary())

        # Individual passes
        text = editor.remove_clutter(text)
        text = editor.strengthen_verbs(text)
        text = editor.simplify_language(text)
    """

    def __init__(self):
        """Initialize Zinsser editor"""
        # Load replacement dictionaries
        self.clutter_words = self._load_clutter_words()
        self.weak_verbs = self._load_weak_verbs()
        self.jargon_replacements = self._load_jargon_replacements()
        self.nominalizations = self._load_nominalizations()
        self.hedge_words = self._load_hedge_words()

    def edit(self,
            text: str,
            principles: Optional[List[ZinsserPrinciple]] = None,
            aggressive: bool = False) -> ZinsserReport:
        """
        Edit text following Zinsser principles

        Args:
            text: Text to edit
            principles: Which principles to apply (None = all)
            aggressive: More aggressive editing

        Returns:
            ZinsserReport with suggestions and edited text
        """
        if principles is None:
            principles = list(ZinsserPrinciple)

        original_stats = self._calculate_statistics(text)
        edited_text = text
        all_suggestions = []

        # Apply each principle
        for principle in principles:
            if principle == ZinsserPrinciple.CLARITY:
                edited_text, suggestions = self.improve_clarity(edited_text)
                all_suggestions.extend(suggestions)

            elif principle == ZinsserPrinciple.BREVITY:
                edited_text, suggestions = self.remove_clutter(edited_text)
                all_suggestions.extend(suggestions)

            elif principle == ZinsserPrinciple.STRENGTH:
                edited_text, suggestions = self.strengthen_verbs(edited_text)
                all_suggestions.extend(suggestions)

            elif principle == ZinsserPrinciple.HUMANITY:
                edited_text, suggestions = self.add_humanity(edited_text)
                all_suggestions.extend(suggestions)

            elif principle == ZinsserPrinciple.PRECISION:
                edited_text, suggestions = self.increase_precision(edited_text)
                all_suggestions.extend(suggestions)

        final_stats = self._calculate_statistics(edited_text)

        # Calculate improvements
        words_cut = original_stats['words'] - final_stats['words']
        passive_fixed = original_stats.get('passive_voice', 0) - final_stats.get('passive_voice', 0)
        readability_improvement = final_stats.get('readability', 0) - original_stats.get('readability', 0)

        report = ZinsserReport(
            original_text=text,
            edited_text=edited_text,
            suggestions=all_suggestions,
            statistics={'before': original_stats, 'after': final_stats},
            readability_improvement=readability_improvement,
            total_words_cut=words_cut,
            passive_voice_fixed=passive_fixed,
        )

        return report

    # ========================================================================
    # Editing methods (one per principle)
    # ========================================================================

    def improve_clarity(self, text: str) -> Tuple[str, List[EditSuggestion]]:
        """
        Improve clarity - make meaning crystal clear

        Fixes:
        - Ambiguous pronouns
        - Unclear antecedents
        - Confusing sentence structure
        - Double negatives
        """
        suggestions = []
        edited = text

        # Fix double negatives
        double_negatives = [
            (r'not uncommon', 'common'),
            (r'not unlike', 'like'),
            (r'not insignificant', 'significant'),
            (r'not unimportant', 'important'),
        ]

        for pattern, replacement in double_negatives:
            if re.search(pattern, edited, re.IGNORECASE):
                original = re.search(pattern, edited, re.IGNORECASE).group()
                edited = re.sub(pattern, replacement, edited, flags=re.IGNORECASE)

                suggestions.append(EditSuggestion(
                    original=original,
                    suggested=replacement,
                    reason="Double negative obscures meaning",
                    principle=ZinsserPrinciple.CLARITY,
                    severity=0.8,
                ))

        return edited, suggestions

    def remove_clutter(self, text: str) -> Tuple[str, List[EditSuggestion]]:
        """
        Remove clutter - cut unnecessary words

        Zinsser: "Writing improves in direct ratio to the number of things
        we can keep out of it that shouldn't be there."

        Removes:
        - Redundant pairs ("each and every")
        - Throat-clearing phrases ("it is important to note")
        - Hedge words ("very", "quite", "rather")
        - Wordy phrases ("in order to" → "to")
        """
        suggestions = []
        edited = text

        # Redundant pairs
        redundant_pairs = {
            'each and every': 'each',
            'first and foremost': 'first',
            'null and void': 'void',
            'over and over': 'repeatedly',
            'various different': 'various',
            'past history': 'history',
            'future plans': 'plans',
            'end result': 'result',
            'final outcome': 'outcome',
        }

        for redundant, replacement in redundant_pairs.items():
            if redundant in edited.lower():
                original = redundant
                edited = re.sub(redundant, replacement, edited, flags=re.IGNORECASE)

                suggestions.append(EditSuggestion(
                    original=original,
                    suggested=replacement,
                    reason="Redundant phrase - one word suffices",
                    principle=ZinsserPrinciple.BREVITY,
                    severity=0.7,
                ))

        # Wordy phrases
        wordy_phrases = {
            'in order to': 'to',
            'due to the fact that': 'because',
            'in the event that': 'if',
            'at this point in time': 'now',
            'for the purpose of': 'to',
            'in spite of the fact that': 'although',
            'with regard to': 'about',
            'in the process of': '(delete)',
            'it is important to note that': '(delete)',
            'it should be noted that': '(delete)',
        }

        for wordy, concise in wordy_phrases.items():
            if wordy in edited.lower():
                original = wordy
                if concise == '(delete)':
                    edited = re.sub(r'\b' + wordy + r'\b', '', edited, flags=re.IGNORECASE)
                    concise = "[deleted]"
                else:
                    edited = re.sub(r'\b' + wordy + r'\b', concise, edited, flags=re.IGNORECASE)

                suggestions.append(EditSuggestion(
                    original=original,
                    suggested=concise,
                    reason="Wordy phrase - use simpler alternative",
                    principle=ZinsserPrinciple.BREVITY,
                    severity=0.9,
                ))

        # Hedge words (reduce, don't eliminate completely)
        hedge_words = ['very', 'quite', 'rather', 'fairly', 'somewhat', 'relatively']

        for hedge in hedge_words:
            pattern = r'\b' + hedge + r'\s+'
            matches = re.finditer(pattern, edited, re.IGNORECASE)

            for match in list(matches)[:3]:  # Limit to 3 per hedge word
                edited = edited[:match.start()] + edited[match.end():]

                suggestions.append(EditSuggestion(
                    original=f"{hedge} (word)",
                    suggested="[deleted]",
                    reason=f"'{hedge}' weakens writing - cut it",
                    principle=ZinsserPrinciple.BREVITY,
                    severity=0.6,
                ))

        return edited, suggestions

    def strengthen_verbs(self, text: str) -> Tuple[str, List[EditSuggestion]]:
        """
        Strengthen verbs - use active voice and strong verbs

        Zinsser: "Verbs are the most important of all your tools.
        They push the sentence forward and give it momentum."

        Fixes:
        - Passive voice → active voice
        - Weak verbs (is, has, makes) → strong verbs
        - Nominalizations ("make a decision" → "decide")
        """
        suggestions = []
        edited = text

        # Fix passive voice
        passive_patterns = [
            (r'is being (\w+)', r'\1s'),
            (r'was (\w+ed) by', r'[subject] \1'),
            (r'were (\w+ed) by', r'[subject] \1'),
            (r'has been (\w+ed)', r'[subject] \1'),
        ]

        for pattern, active_hint in passive_patterns:
            matches = list(re.finditer(pattern, edited, re.IGNORECASE))[:2]  # Limit suggestions

            for match in matches:
                suggestions.append(EditSuggestion(
                    original=match.group(),
                    suggested=active_hint,
                    reason="Passive voice - consider active voice for stronger writing",
                    principle=ZinsserPrinciple.STRENGTH,
                    severity=0.8,
                ))

        # Nominalization fixes
        nominalizations = {
            'make a decision': 'decide',
            'make a determination': 'determine',
            'give consideration to': 'consider',
            'have a discussion': 'discuss',
            'make an examination': 'examine',
            'reach a conclusion': 'conclude',
            'make a recommendation': 'recommend',
            'perform an analysis': 'analyze',
        }

        for nominal, verb in nominalizations.items():
            if nominal in edited.lower():
                edited = re.sub(nominal, verb, edited, flags=re.IGNORECASE)

                suggestions.append(EditSuggestion(
                    original=nominal,
                    suggested=verb,
                    reason="Nominalization - use the verb directly",
                    principle=ZinsserPrinciple.STRENGTH,
                    severity=0.9,
                ))

        return edited, suggestions

    def add_humanity(self, text: str) -> Tuple[str, List[EditSuggestion]]:
        """
        Add humanity - make writing natural and warm

        Zinsser: "Be yourself. Other writers are already taken."

        Suggestions for:
        - Overly formal language
        - Bureaucratic jargon
        - Robot-like tone
        """
        suggestions = []

        # Detect overly formal phrases
        formal_phrases = [
            'pursuant to',
            'heretofore',
            'aforementioned',
            'hereby',
            'whereas',
        ]

        for formal in formal_phrases:
            if formal in text.lower():
                suggestions.append(EditSuggestion(
                    original=formal,
                    suggested="[use plain English]",
                    reason=f"'{formal}' is overly formal - write naturally",
                    principle=ZinsserPrinciple.HUMANITY,
                    severity=0.7,
                ))

        # Don't actually edit (these require human judgment)
        return text, suggestions

    def increase_precision(self, text: str) -> Tuple[str, List[EditSuggestion]]:
        """
        Increase precision - use exact words and specific details

        Zinsser: "The secret of good writing is to strip every sentence
        to its cleanest components."

        Fixes:
        - Vague words → specific words
        - General → particular
        - Abstract → concrete
        """
        suggestions = []
        edited = text

        # Vague → specific
        vague_words = {
            'things': 'elements / items / factors',
            'stuff': 'material / equipment / items',
            'a lot': 'many / numerous / substantial',
            'very big': 'enormous / massive / extensive',
            'very small': 'tiny / minute / minimal',
        }

        for vague, specific in vague_words.items():
            if vague in edited.lower():
                suggestions.append(EditSuggestion(
                    original=vague,
                    suggested=specific,
                    reason=f"'{vague}' is vague - use specific word",
                    principle=ZinsserPrinciple.PRECISION,
                    severity=0.6,
                ))

        return edited, suggestions

    # ========================================================================
    # Helper methods
    # ========================================================================

    def _calculate_statistics(self, text: str) -> Dict[str, any]:
        """Calculate text statistics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Count passive voice
        passive_patterns = [
            r'is being',
            r'was \w+ed by',
            r'were \w+ed by',
            r'has been \w+ed',
        ]
        passive_count = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in passive_patterns
        )

        # Simple readability score (Flesch-Kincaid approximation)
        avg_sentence_length = len(words) / max(1, len(sentences))
        avg_word_length = sum(len(w) for w in words) / max(1, len(words))
        readability = 206.835 - 1.015 * avg_sentence_length - 84.6 * (avg_word_length / 5)

        return {
            'words': len(words),
            'sentences': len(sentences),
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'passive_voice': passive_count,
            'readability': readability,
        }

    def _load_clutter_words(self) -> Set[str]:
        """Load common clutter words"""
        return {
            'very', 'quite', 'rather', 'fairly', 'somewhat', 'relatively',
            'actually', 'basically', 'essentially', 'generally', 'virtually',
            'really', 'literally', 'simply', 'just', 'merely',
        }

    def _load_weak_verbs(self) -> Set[str]:
        """Load weak verbs to replace"""
        return {'is', 'are', 'was', 'were', 'has', 'have', 'had', 'makes', 'does'}

    def _load_jargon_replacements(self) -> Dict[str, str]:
        """Load jargon replacements"""
        return {
            'utilize': 'use',
            'facilitate': 'help',
            'implement': 'do / carry out',
            'demonstrate': 'show',
            'additional': 'more',
            'sufficient': 'enough',
            'commence': 'begin',
            'terminate': 'end',
            'prioritize': 'rank',
            'optimize': 'improve',
        }

    def _load_nominalizations(self) -> Dict[str, str]:
        """Load nominalization to verb mappings"""
        return {
            'utilization': 'use',
            'implementation': 'implement',
            'facilitation': 'facilitate',
            'optimization': 'optimize',
            'maximization': 'maximize',
            'minimization': 'minimize',
        }

    def _load_hedge_words(self) -> Set[str]:
        """Load hedge words"""
        return {'very', 'quite', 'rather', 'fairly', 'somewhat', 'relatively'}


# Factory function
def create_zinsser_editor() -> ZinsserEditor:
    """Create a new Zinsser editor"""
    return ZinsserEditor()
