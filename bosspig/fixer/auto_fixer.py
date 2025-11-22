"""
BossPig Auto-Fixer

Automatically fixes detected business slop issues:
1. Corporate jargon replacement
2. Vague commitment fixes
3. Missing date fixes
4. AI hallucination cleanup
5. Passive voice flagging

Created: 2025-11-22
Week: 4
Status: Production Implementation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import re
from pathlib import Path

# Import detector components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from detector.core import Finding, BossPigFindings, FindingCategory, Severity
from detector.detector import BossPigDetector
from detector.jargon_dict import JARGON_REPLACEMENTS


class FixStrategy(Enum):
    """Strategy for applying fixes"""
    INTERACTIVE = "interactive"  # Prompt user for each fix
    BATCH = "batch"              # Apply all automatic fixes
    DRY_RUN = "dry_run"         # Preview changes without applying


@dataclass
class FixResult:
    """
    Result of auto-fix operation.

    Attributes:
        original_text: Original document text
        fixed_text: Text after fixes applied
        fixes_applied: Number of fixes successfully applied
        fixes_failed: Number of fixes that failed
        quality_improvement: Quality score delta (after - before)
        findings_before: Findings detected before fixes
        findings_after: Findings detected after fixes
        fix_summary: Human-readable summary of changes
    """
    original_text: str
    fixed_text: str
    fixes_applied: int
    fixes_failed: int
    quality_improvement: float
    findings_before: BossPigFindings
    findings_after: BossPigFindings
    fix_summary: List[str] = field(default_factory=list)

    def get_diff(self) -> str:
        """Generate unified diff showing changes"""
        import difflib
        diff = difflib.unified_diff(
            self.original_text.splitlines(keepends=True),
            self.fixed_text.splitlines(keepends=True),
            fromfile='original',
            tofile='fixed',
            lineterm=''
        )
        return ''.join(diff)


class AutoFixer:
    """
    Automatic fixer for business slop issues.

    Applies fixes based on detected findings from BossPigDetector.
    Supports interactive, batch, and dry-run modes.
    """

    def __init__(
        self,
        detector: Optional[BossPigDetector] = None,
        strategy: FixStrategy = FixStrategy.BATCH
    ):
        """
        Initialize auto-fixer.

        Args:
            detector: BossPigDetector instance (creates new if None)
            strategy: Fix strategy (interactive, batch, dry_run)
        """
        self.detector = detector or BossPigDetector()
        self.strategy = strategy
        self.jargon_replacements = JARGON_REPLACEMENTS

    def fix(
        self,
        text: str,
        categories: Optional[List[FindingCategory]] = None,
        interactive: bool = False
    ) -> FixResult:
        """
        Fix business slop issues in text.

        Args:
            text: Original text to fix
            categories: Categories to fix (None = all)
            interactive: Override strategy to use interactive mode

        Returns:
            FixResult with original, fixed text and metadata
        """
        # Override strategy if interactive flag set
        strategy = FixStrategy.INTERACTIVE if interactive else self.strategy

        # Analyze original text
        findings_before = self.detector.analyze(text)

        # Apply fixes
        fixed_text = text
        fix_summary = []
        fixes_applied = 0
        fixes_failed = 0

        # Filter findings by category if specified
        findings_to_fix = findings_before.findings
        if categories:
            findings_to_fix = [
                f for f in findings_to_fix
                if f.category in categories
            ]

        # Group findings by category for organized fixing
        findings_by_category = {}
        for finding in findings_to_fix:
            cat = finding.category
            if cat not in findings_by_category:
                findings_by_category[cat] = []
            findings_by_category[cat].append(finding)

        # Apply fixes in priority order
        fix_order = [
            FindingCategory.CORPORATE_JARGON,
            FindingCategory.VAGUE_COMMITMENTS,
            FindingCategory.MISSING_DATES,
            FindingCategory.AI_HALLUCINATION,
            FindingCategory.PASSIVE_VOICE
        ]

        for category in fix_order:
            if category not in findings_by_category:
                continue

            findings = findings_by_category[category]

            if category == FindingCategory.CORPORATE_JARGON:
                result = self._fix_jargon(fixed_text, findings, strategy)
                fixed_text, applied, failed, summary = result
                fixes_applied += applied
                fixes_failed += failed
                fix_summary.extend(summary)

            elif category == FindingCategory.VAGUE_COMMITMENTS:
                result = self._fix_vague_commitments(fixed_text, findings, strategy)
                fixed_text, applied, failed, summary = result
                fixes_applied += applied
                fixes_failed += failed
                fix_summary.extend(summary)

            elif category == FindingCategory.MISSING_DATES:
                result = self._fix_missing_dates(fixed_text, findings, strategy)
                fixed_text, applied, failed, summary = result
                fixes_applied += applied
                fixes_failed += failed
                fix_summary.extend(summary)

            elif category == FindingCategory.AI_HALLUCINATION:
                result = self._fix_ai_hallucinations(fixed_text, findings, strategy)
                fixed_text, applied, failed, summary = result
                fixes_applied += applied
                fixes_failed += failed
                fix_summary.extend(summary)

            elif category == FindingCategory.PASSIVE_VOICE:
                result = self._fix_passive_voice(fixed_text, findings, strategy)
                fixed_text, applied, failed, summary = result
                fixes_applied += applied
                fixes_failed += failed
                fix_summary.extend(summary)

        # Re-analyze fixed text
        findings_after = self.detector.analyze(fixed_text)

        # Calculate quality improvement
        quality_improvement = (
            findings_after.quality_metrics.overall_score() -
            findings_before.quality_metrics.overall_score()
        )

        return FixResult(
            original_text=text,
            fixed_text=fixed_text,
            fixes_applied=fixes_applied,
            fixes_failed=fixes_failed,
            quality_improvement=quality_improvement,
            findings_before=findings_before,
            findings_after=findings_after,
            fix_summary=fix_summary
        )

    def _fix_jargon(
        self,
        text: str,
        findings: List[Finding],
        strategy: FixStrategy
    ) -> Tuple[str, int, int, List[str]]:
        """
        Fix corporate jargon by replacing with plain language.

        Args:
            text: Text to fix
            findings: Jargon findings
            strategy: Fix strategy

        Returns:
            (fixed_text, applied_count, failed_count, summary)
        """
        fixed_text = text
        applied = 0
        failed = 0
        summary = []

        # Sort findings by position (reverse order to preserve positions)
        sorted_findings = sorted(
            findings,
            key=lambda f: (f.line_number, f.column_number or 0),
            reverse=True
        )

        for finding in sorted_findings:
            jargon = finding.text.strip()
            suggestion = finding.suggestion

            # Extract replacement from suggestion
            # Format: "Replace with 'plain language'" or "Replace with: 'plain language'"
            match = re.search(r"Replace with:? ['\"](.+?)['\"]", suggestion)
            if not match:
                # Try alternative format: "Use 'plain language'"
                match = re.search(r"[Uu]se ['\"](.+?)['\"]", suggestion)
            if not match:
                # Try alternative format: "Say 'plain language'"
                match = re.search(r"[Ss]ay ['\"](.+?)['\"]", suggestion)

            if not match:
                failed += 1
                continue

            replacement = match.group(1)

            # Interactive mode: ask user
            if strategy == FixStrategy.INTERACTIVE:
                print(f"\nJargon detected: '{jargon}'")
                print(f"Suggested replacement: '{replacement}'")
                response = input("Apply fix? (y/n/custom): ").strip().lower()

                if response == 'y':
                    pass  # Use suggested replacement
                elif response == 'n':
                    failed += 1
                    continue
                elif response.startswith('custom'):
                    custom = input("Enter custom replacement: ").strip()
                    replacement = custom if custom else replacement
                else:
                    failed += 1
                    continue

            # Apply replacement (case-preserving)
            try:
                fixed_text = self._replace_preserving_case(
                    fixed_text,
                    jargon,
                    replacement
                )
                applied += 1
                summary.append(f"Replaced '{jargon}' → '{replacement}'")
            except Exception as e:
                failed += 1
                print(f"Failed to replace '{jargon}': {e}")

        return fixed_text, applied, failed, summary

    def _fix_vague_commitments(
        self,
        text: str,
        findings: List[Finding],
        strategy: FixStrategy
    ) -> Tuple[str, int, int, List[str]]:
        """
        Fix vague commitments by adding specific actions/dates.

        Args:
            text: Text to fix
            findings: Vague commitment findings
            strategy: Fix strategy

        Returns:
            (fixed_text, applied_count, failed_count, summary)
        """
        fixed_text = text
        applied = 0
        failed = 0
        summary = []

        for finding in findings:
            vague_phrase = finding.text.strip()

            # Determine replacement based on phrase
            if "will try to" in vague_phrase.lower():
                if strategy == FixStrategy.INTERACTIVE:
                    print(f"\nVague commitment: '{vague_phrase}'")
                    action = input("What specific action will be taken? ").strip()
                    date = input("By when? (YYYY-MM-DD or 'in X days'): ").strip()
                    owner = input("Who is responsible? (optional): ").strip()

                    replacement = f"will {action} by {date}"
                    if owner:
                        replacement += f" ({owner})"
                else:
                    # Batch mode: use placeholders
                    replacement = "will [ACTION] by [DATE]"

            elif "best effort" in vague_phrase.lower():
                if strategy == FixStrategy.INTERACTIVE:
                    print(f"\nVague commitment: '{vague_phrase}'")
                    commitment = input("What specific commitment can be made? ").strip()
                    replacement = commitment
                else:
                    replacement = "[SPECIFIC_COMMITMENT]"

            elif "as soon as possible" in vague_phrase.lower() or "asap" in vague_phrase.lower():
                if strategy == FixStrategy.INTERACTIVE:
                    print(f"\nVague deadline: '{vague_phrase}'")
                    date = input("Specific date? (YYYY-MM-DD or 'in X days'): ").strip()
                    replacement = f"by {date}"
                else:
                    # Suggest 7 days from now as default
                    default_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
                    replacement = f"by {default_date}"

            else:
                replacement = "[SPECIFIC_COMMITMENT]"

            try:
                fixed_text = fixed_text.replace(vague_phrase, replacement)
                applied += 1
                summary.append(f"Fixed vague commitment: '{vague_phrase}' → '{replacement}'")
            except Exception as e:
                failed += 1
                print(f"Failed to fix vague commitment '{vague_phrase}': {e}")

        return fixed_text, applied, failed, summary

    def _fix_missing_dates(
        self,
        text: str,
        findings: List[Finding],
        strategy: FixStrategy
    ) -> Tuple[str, int, int, List[str]]:
        """
        Fix missing dates by adding specific deadlines.

        Args:
            text: Text to fix
            findings: Missing date findings
            strategy: Fix strategy

        Returns:
            (fixed_text, applied_count, failed_count, summary)
        """
        fixed_text = text
        applied = 0
        failed = 0
        summary = []

        for finding in findings:
            vague_date = finding.text.strip()

            if strategy == FixStrategy.INTERACTIVE:
                print(f"\nVague date: '{vague_date}'")
                date = input("Specific date? (YYYY-MM-DD): ").strip()
                replacement = date
            else:
                # Batch mode: use placeholder or suggest dates
                if "soon" in vague_date.lower():
                    # Suggest 14 days from now
                    replacement = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
                elif "asap" in vague_date.lower() or "tbd" in vague_date.lower():
                    # Suggest 7 days from now
                    replacement = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
                else:
                    replacement = "[DATE: YYYY-MM-DD]"

            try:
                fixed_text = fixed_text.replace(vague_date, replacement)
                applied += 1
                summary.append(f"Added specific date: '{vague_date}' → '{replacement}'")
            except Exception as e:
                failed += 1
                print(f"Failed to fix date '{vague_date}': {e}")

        return fixed_text, applied, failed, summary

    def _fix_ai_hallucinations(
        self,
        text: str,
        findings: List[Finding],
        strategy: FixStrategy
    ) -> Tuple[str, int, int, List[str]]:
        """
        Fix AI hallucination markers by removing or highlighting them.

        Args:
            text: Text to fix
            findings: AI hallucination findings
            strategy: Fix strategy

        Returns:
            (fixed_text, applied_count, failed_count, summary)
        """
        fixed_text = text
        applied = 0
        failed = 0
        summary = []

        for finding in findings:
            hallucination = finding.text.strip()

            # LLM boilerplate: Remove entirely
            if "as an ai" in hallucination.lower() or "language model" in hallucination.lower():
                try:
                    # Remove the entire sentence
                    fixed_text = fixed_text.replace(hallucination, "")
                    applied += 1
                    summary.append(f"Removed AI boilerplate: '{hallucination[:50]}...'")
                except Exception as e:
                    failed += 1

            # Placeholder content: Flag for manual review
            elif "[insert" in hallucination.lower() or "[fill in" in hallucination.lower():
                # Extract what needs to be filled
                match = re.search(r'\[(?:insert|fill in|add)\s+(.+?)\]', hallucination, re.IGNORECASE)
                if match:
                    placeholder_type = match.group(1)
                    replacement = f"**[FILL IN: {placeholder_type.upper()}]**"
                else:
                    replacement = "**[INCOMPLETE CONTENT]**"

                try:
                    fixed_text = fixed_text.replace(hallucination, replacement)
                    applied += 1
                    summary.append(f"Flagged placeholder: '{hallucination}' → '{replacement}'")
                except Exception as e:
                    failed += 1

            else:
                # Unknown hallucination type: flag it
                replacement = f"**[REVIEW: {hallucination}]**"
                try:
                    fixed_text = fixed_text.replace(hallucination, replacement)
                    applied += 1
                    summary.append(f"Flagged suspicious content: '{hallucination[:30]}...'")
                except Exception as e:
                    failed += 1

        return fixed_text, applied, failed, summary

    def _fix_passive_voice(
        self,
        text: str,
        findings: List[Finding],
        strategy: FixStrategy
    ) -> Tuple[str, int, int, List[str]]:
        """
        Flag passive voice for manual review (auto-conversion is complex).

        Args:
            text: Text to fix
            findings: Passive voice findings
            strategy: Fix strategy

        Returns:
            (fixed_text, applied_count, failed_count, summary)
        """
        fixed_text = text
        applied = 0
        failed = 0
        summary = []

        # For MVP: Just flag passive voice, don't auto-convert
        # Auto-conversion requires NLP and is error-prone

        for finding in findings:
            passive = finding.text.strip()

            if strategy == FixStrategy.INTERACTIVE:
                print(f"\nPassive voice detected: '{passive}'")
                active = input("Suggest active voice replacement (or press Enter to skip): ").strip()

                if active:
                    try:
                        fixed_text = fixed_text.replace(passive, active)
                        applied += 1
                        summary.append(f"Converted to active voice: '{passive}' → '{active}'")
                    except Exception as e:
                        failed += 1
                else:
                    failed += 1
            else:
                # Batch mode: Just flag it
                # We don't auto-convert because it's error-prone
                summary.append(f"Passive voice detected (not auto-fixed): '{passive}'")
                failed += 1

        return fixed_text, applied, failed, summary

    def _replace_preserving_case(
        self,
        text: str,
        old: str,
        new: str
    ) -> str:
        """
        Replace text while preserving original case.

        Args:
            text: Text to search
            old: Text to replace
            new: Replacement text

        Returns:
            Text with replacements applied
        """
        # Find all occurrences with word boundaries
        pattern = r'\b' + re.escape(old) + r'\b'

        def replace_match(match):
            original = match.group(0)

            # Preserve case
            if original.isupper():
                return new.upper()
            elif original[0].isupper():
                return new.capitalize()
            else:
                return new.lower()

        return re.sub(pattern, replace_match, text, flags=re.IGNORECASE)
