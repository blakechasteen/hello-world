"""
BossPig Main Detector

Implements TOP 5 detection categories for MVP:
1. Corporate Jargon Detection
2. Vague Commitments Detection
3. Missing Dates/Deadlines Detection
4. AI Hallucination Markers Detection
5. Passive Voice Detection

Created: 2025-11-22
Status: Production Ready (Week 1-3)
"""

from pathlib import Path
from typing import List, Optional, Union
import re

# Import core detection infrastructure
from .core import (
    Finding,
    BossPigFindings,
    QualityMetrics,
    FindingCategory,
    Severity,
    PatternMatcher,
    DocumentStats
)
from .jargon_dict import load_jargon_dictionary, JARGON_REPLACEMENTS


class BossPigDetector:
    """
    Main BossPig detection engine.

    Detects 5 categories of business slop (MVP):
    1. Corporate jargon
    2. Vague commitments
    3. Missing dates/deadlines
    4. AI hallucination markers
    5. Passive voice
    """

    def __init__(
        self,
        jargon_dict_path: Optional[Path] = None,
        enable_nlp: bool = False
    ):
        """
        Initialize BossPig detector.

        Args:
            jargon_dict_path: Optional path to custom jargon dictionary JSON
            enable_nlp: Enable spaCy for NLP analysis (passive voice detection)
        """
        self.jargon_dict = load_jargon_dictionary(jargon_dict_path)
        self.matcher = PatternMatcher(case_sensitive=False)
        self.enable_nlp = enable_nlp

        # Try to import spaCy for passive voice detection
        self.nlp = None
        if enable_nlp:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                print("Warning: spaCy not available. Passive voice detection will use regex fallback.")
                self.nlp = None

    def analyze(self, text_or_file: Union[str, Path]) -> BossPigFindings:
        """
        Analyze text or file for business slop.

        Args:
            text_or_file: Text string or path to file

        Returns:
            BossPigFindings with all detected issues and quality score
        """
        # Load text
        if isinstance(text_or_file, Path):
            with open(text_or_file, 'r', encoding='utf-8') as f:
                text = f.read()
        elif isinstance(text_or_file, str) and text_or_file and Path(text_or_file).is_file():
            with open(text_or_file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = text_or_file

        # Calculate document stats
        doc_stats = DocumentStats.calculate_stats(text)

        # Run all detectors
        findings: List[Finding] = []
        findings.extend(self.detect_jargon(text))
        findings.extend(self.detect_vague_commitments(text))
        findings.extend(self.detect_missing_dates(text))
        findings.extend(self.detect_ai_hallucinations(text))
        findings.extend(self.detect_passive_voice(text))

        # Calculate quality metrics
        metrics = self.calculate_quality_metrics(text, findings, doc_stats)

        # Create findings object
        result = BossPigFindings(
            findings=findings,
            quality_metrics=metrics,
            document_stats=doc_stats
        )

        return result

    # ==================== DETECTOR 1: Corporate Jargon ====================

    def detect_jargon(self, text: str) -> List[Finding]:
        """
        Detect corporate jargon using jargon dictionary.

        Matches jargon phrases case-insensitively and provides
        plain language replacements.
        """
        findings = []

        for jargon_phrase, info in self.jargon_dict.items():
            # Only detect jargon in corporate_buzzwords category
            if info.get("category") != "corporate_buzzwords":
                continue

            # Escape special regex characters in phrase
            escaped_phrase = re.escape(jargon_phrase)
            # Word boundary pattern
            pattern = r'\b' + escaped_phrase + r'\b'

            matches = self.matcher.find_all_matches(text, pattern)

            for match in matches:
                finding = Finding(
                    category=FindingCategory.CORPORATE_JARGON,
                    severity=Severity[info.get("severity", "warning").upper()],
                    line_number=match["line_number"],
                    column_number=match["column_number"],
                    text=match["text"],
                    message=f"Corporate jargon: '{match['text']}'. {info.get('explanation', '')}",
                    suggestion=f"Replace with '{info.get('replacement', '')}'",
                    context=match["context"]
                )
                findings.append(finding)

        return findings

    # ==================== DETECTOR 2: Vague Commitments ====================

    def detect_vague_commitments(self, text: str) -> List[Finding]:
        """
        Detect vague commitment language.

        Patterns:
        - "we will try to"
        - "hopefully we can"
        - "it would be nice if"
        - "we should probably"
        """
        findings = []

        vague_patterns = [
            (r'\bwe\s+will\s+try\s+to\b', "Weak commitment. State specific action and deadline."),
            (r'\bhopefully\s+we\s+can\b', "No commitment. State specific action and owner."),
            (r'\bit\s+would\s+be\s+nice\s+if\b', "Wishful thinking. Convert to requirement with owner."),
            (r'\bwe\s+should\s+probably\b', "Indecision. Assign owner and deadline."),
            (r'\bwe\s+might\b', "Uncertain. Set decision deadline."),
            (r'\bwe\s+could\b', "Possibility, not commitment. Evaluate and decide by specific date."),
            (r'\bsomeone\s+should\b', "No ownership. Assign specific person."),
            (r'\bthe\s+team\s+will\b', "Vague ownership. Name specific team members."),
        ]

        for pattern, explanation in vague_patterns:
            matches = self.matcher.find_all_matches(text, pattern)

            for match in matches:
                finding = Finding(
                    category=FindingCategory.VAGUE_COMMITMENTS,
                    severity=Severity.CRITICAL,
                    line_number=match["line_number"],
                    column_number=match["column_number"],
                    text=match["text"],
                    message=f"Vague commitment: '{match['text']}'. {explanation}",
                    suggestion="Replace with specific action, owner, and date (e.g., '@alice will complete X by 2025-12-01')",
                    context=match["context"]
                )
                findings.append(finding)

        return findings

    # ==================== DETECTOR 3: Missing Dates/Deadlines ====================

    def detect_missing_dates(self, text: str) -> List[Finding]:
        """
        Detect vague time references and missing deadlines.

        Patterns:
        - "soon", "shortly", "ASAP"
        - "pending", "TBD", "to be determined"
        - "in the near future", "eventually"
        """
        findings = []

        vague_date_patterns = [
            (r'\bsoon\b', "Ambiguous deadline. Specify exact date (YYYY-MM-DD)."),
            (r'\bshortly\b', "Vague timeline. Provide specific date."),
            (r'\basap\b', "Unclear urgency. Set specific date and time."),
            (r'\bas\s+soon\s+as\s+possible\b', "Vague deadline. Specify exact date and time."),
            (r'\bpending\b', "Unclear status. Specify what's pending and expected date."),
            (r'\btbd\b', "Incomplete. Set date to determine decision."),
            (r'\bto\s+be\s+determined\b', "Missing decision. Set decision deadline."),
            (r'\bin\s+the\s+near\s+future\b', "Vague timeline. Define with specific date."),
            (r'\beventually\b', "No timeline. Commit to date or mark as low priority."),
            (r'\bat\s+some\s+point\b', "Missing deadline. Provide specific timeline."),
            (r'\bdown\s+the\s+road\b', "Vague future reference. Specify quarter/year (Q1 2026, etc.)."),
            (r'\blater\b', "Undefined timeline. Set specific date."),
            (r'\bwhen\s+we\s+have\s+time\b', "No commitment. Schedule specific date."),
        ]

        for pattern, explanation in vague_date_patterns:
            matches = self.matcher.find_all_matches(text, pattern)

            for match in matches:
                finding = Finding(
                    category=FindingCategory.MISSING_DATES,
                    severity=Severity.WARNING,
                    line_number=match["line_number"],
                    column_number=match["column_number"],
                    text=match["text"],
                    message=f"Vague date/deadline: '{match['text']}'. {explanation}",
                    suggestion="Replace with specific date (YYYY-MM-DD format)",
                    context=match["context"]
                )
                findings.append(finding)

        return findings

    # ==================== DETECTOR 4: AI Hallucination Markers ====================

    def detect_ai_hallucinations(self, text: str) -> List[Finding]:
        """
        Detect AI hallucination markers and incomplete content.

        Patterns:
        - "As an AI language model..."
        - "I don't have personal opinions but..."
        - "[INSERT X HERE]", "TBD", "[TO BE DETERMINED]"
        """
        findings = []

        # AI boilerplate patterns
        ai_patterns = [
            (r'\bas\s+an\s+ai\s+language\s+model\b', "LLM boilerplate. Remove entirely."),
            (r'\bi\s+don\'t\s+have\s+personal\s+opinions\b', "AI disclaimer. Remove."),
            (r'\bi\s+cannot\s+provide\s+personal\s+opinions\b', "AI limitation statement. Remove."),
            (r'\bas\s+an\s+ai\b', "AI self-reference. Remove."),
        ]

        for pattern, explanation in ai_patterns:
            matches = self.matcher.find_all_matches(text, pattern)

            for match in matches:
                finding = Finding(
                    category=FindingCategory.AI_HALLUCINATION,
                    severity=Severity.CRITICAL,
                    line_number=match["line_number"],
                    column_number=match["column_number"],
                    text=match["text"],
                    message=f"AI hallucination marker: '{match['text']}'. {explanation}",
                    suggestion="Remove LLM boilerplate entirely and provide actual content",
                    context=match["context"]
                )
                findings.append(finding)

        # Placeholder patterns
        placeholder_patterns = [
            (r'\[INSERT\s+[\w\s]+\s+HERE\]', "Placeholder text. Fill in actual content."),
            (r'\[TO\s+BE\s+DETERMINED\]', "Incomplete. Provide actual information."),
            (r'\[TBD\]', "Placeholder. Complete section."),
            (r'\[TODO:.*?\]', "TODO placeholder. Complete task."),
            (r'\[XXX\]', "Placeholder. Fill in content."),
        ]

        for pattern, explanation in placeholder_patterns:
            matches = self.matcher.find_all_matches(text, pattern)

            for match in matches:
                finding = Finding(
                    category=FindingCategory.AI_HALLUCINATION,
                    severity=Severity.CRITICAL,
                    line_number=match["line_number"],
                    column_number=match["column_number"],
                    text=match["text"],
                    message=f"Placeholder text: '{match['text']}'. {explanation}",
                    suggestion="Replace placeholder with actual content",
                    context=match["context"]
                )
                findings.append(finding)

        return findings

    # ==================== DETECTOR 5: Passive Voice ====================

    def detect_passive_voice(self, text: str) -> List[Finding]:
        """
        Detect passive voice constructions.

        Uses spaCy if available, otherwise falls back to regex patterns.

        Common patterns:
        - "was made", "were identified", "has been determined"
        - "is being considered", "will be completed"
        """
        findings = []

        if self.nlp:
            # Use spaCy for accurate passive voice detection
            findings.extend(self._detect_passive_voice_nlp(text))
        else:
            # Fallback to regex patterns
            findings.extend(self._detect_passive_voice_regex(text))

        return findings

    def _detect_passive_voice_nlp(self, text: str) -> List[Finding]:
        """Detect passive voice using spaCy NLP"""
        findings = []
        doc = self.nlp(text)

        for sent in doc.sents:
            # Check for passive voice indicators
            has_passive = False
            passive_tokens = []

            for token in sent:
                # Check if token is a passive auxiliary or past participle
                if token.dep_ == "auxpass" or (token.tag_ == "VBN" and any(child.dep_ == "auxpass" for child in token.children)):
                    has_passive = True
                    passive_tokens.append(token.text)

            if has_passive:
                # Calculate line number
                line_num = text[:sent.start_char].count('\n') + 1

                finding = Finding(
                    category=FindingCategory.PASSIVE_VOICE,
                    severity=Severity.WARNING,
                    line_number=line_num,
                    column_number=None,
                    text=sent.text.strip(),
                    message="Passive voice detected. Ownership unclear.",
                    suggestion="Convert to active voice with specific owner (e.g., '@alice completed the analysis')",
                    context=sent.text[:100]  # Limit context length
                )
                findings.append(finding)

        return findings

    def _detect_passive_voice_regex(self, text: str) -> List[Finding]:
        """Detect passive voice using regex patterns (fallback)"""
        findings = []

        passive_patterns = [
            r'\bwas\s+\w+ed\b',       # "was completed", "was identified"
            r'\bwere\s+\w+ed\b',      # "were made", "were found"
            r'\bhas\s+been\s+\w+ed\b', # "has been determined"
            r'\bhave\s+been\s+\w+ed\b', # "have been completed"
            r'\bis\s+being\s+\w+ed\b', # "is being considered"
            r'\bare\s+being\s+\w+ed\b', # "are being reviewed"
            r'\bwill\s+be\s+\w+ed\b',  # "will be completed"
        ]

        for pattern in passive_patterns:
            matches = self.matcher.find_all_matches(text, pattern)

            for match in matches:
                finding = Finding(
                    category=FindingCategory.PASSIVE_VOICE,
                    severity=Severity.WARNING,
                    line_number=match["line_number"],
                    column_number=match["column_number"],
                    text=match["text"],
                    message=f"Passive voice: '{match['text']}'. Ownership unclear.",
                    suggestion="Convert to active voice with specific owner",
                    context=match["context"]
                )
                findings.append(finding)

        return findings

    # ==================== Quality Metrics Calculation ====================

    def calculate_quality_metrics(
        self,
        text: str,
        findings: List[Finding],
        doc_stats: dict
    ) -> QualityMetrics:
        """
        Calculate quality metrics based on findings.

        Returns:
            QualityMetrics with all scores calculated
        """
        total_words = max(doc_stats.get("word_count", 1), 1)  # Avoid division by zero
        total_sentences = max(doc_stats.get("sentence_count", 1), 1)
        total_paragraphs = max(doc_stats.get("paragraph_count", 1), 1)
        total_sections = max(doc_stats.get("section_count", 1), 1)

        # Count findings by category
        jargon_count = len([f for f in findings if f.category == FindingCategory.CORPORATE_JARGON])
        vague_count = len([f for f in findings if f.category == FindingCategory.VAGUE_COMMITMENTS])
        passive_count = len([f for f in findings if f.category == FindingCategory.PASSIVE_VOICE])
        ai_hallucination_count = len([f for f in findings if f.category == FindingCategory.AI_HALLUCINATION])
        missing_dates_count = len([f for f in findings if f.category == FindingCategory.MISSING_DATES])

        # Clarity score: 1.0 - (jargon_count / total_words)
        clarity_score = max(0.0, 1.0 - (jargon_count / total_words))

        # Specificity score: 1.0 - (vague_phrases / total_sentences)
        specificity_score = max(0.0, 1.0 - ((vague_count + missing_dates_count) / total_sentences))

        # Actionability score: Assume 1 action item per paragraph is ideal
        # Deduct for passive voice (unclear ownership)
        actionability_score = max(0.0, 1.0 - (passive_count / total_paragraphs))

        # Professionalism score: 1.0 - (hallucinations / total_sections)
        professionalism_score = max(0.0, 1.0 - (ai_hallucination_count / total_sections))

        # Completeness score: 1.0 if no placeholders, otherwise deduct
        completeness_score = max(0.0, 1.0 - (ai_hallucination_count / total_sections))

        return QualityMetrics(
            clarity_score=clarity_score,
            specificity_score=specificity_score,
            actionability_score=actionability_score,
            professionalism_score=professionalism_score,
            completeness_score=completeness_score
        )


if __name__ == "__main__":
    # Test detector
    detector = BossPigDetector()

    test_text = """
    # Q4 Strategic Initiative

    We need to synergize our core competencies to leverage the low-hanging fruit
    and move the needle on our key performance indicators. This paradigm shift
    will be a game changer for our thought leadership in the space.

    Action Items:
    - Circle back on the deep dive
    - Touch base with stakeholders
    - Reach out to the team

    We will try to finish this soon. Hopefully we can get it done before
    the end of the quarter. It would be nice if we could avoid downtime.

    As an AI language model, I cannot provide specific recommendations, but
    here are some general suggestions: [INSERT SPECIFIC PLAN HERE]

    The work was completed by the team. Mistakes were made during the process.
    """

    print("Analyzing test document...")
    results = detector.analyze(test_text)

    print(results.summary())
    print(f"\nTotal findings: {len(results.findings)}")
    print(f"\nSample findings:")
    for finding in results.findings[:5]:
        print(f"  Line {finding.line_number}: {finding.category.value} - {finding.text}")
        print(f"    Suggestion: {finding.suggestion}")
