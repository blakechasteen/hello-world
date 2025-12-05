"""
MRF-Powered Annotation Refinement System.

Applies Metaprompting Refinement Framework strategies to improve
annotation quality in real-time collaborative sessions.

Phase 3: Collaborative Intelligence - MRF Integration.

Created: December 2025
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class AnnotationType(Enum):
    """Types of annotations with different refinement strategies."""
    NOTE = "note"           # General notes -> ELEGANCE (clarity, simplicity)
    QUESTION = "question"   # Questions -> VERIFY (ensure well-formed, answerable)
    INSIGHT = "insight"     # Insights -> CRITIQUE (challenge assumptions)
    CONNECTION = "connection"  # Relationship links -> HOFSTADTER (recursive discovery)
    SUMMARY = "summary"     # Summaries -> REFINE (expand and improve)
    CORRECTION = "correction"  # Corrections -> VERIFY (accuracy check)


class RefinementStrategy(Enum):
    """MRF refinement strategies mapped to annotation types."""
    ELEGANCE = "elegance"      # Clarity -> Simplicity -> Beauty (3 passes)
    VERIFY = "verify"          # Accuracy -> Completeness -> Consistency (3 passes)
    CRITIQUE = "critique"      # Challenge assumptions, find gaps (1 pass)
    REFINE = "refine"          # Iterative context expansion
    HOFSTADTER = "hofstadter"  # Recursive self-reference, meta-analysis


# Default strategy mapping by annotation type
ANNOTATION_STRATEGY_MAP: Dict[AnnotationType, RefinementStrategy] = {
    AnnotationType.NOTE: RefinementStrategy.ELEGANCE,
    AnnotationType.QUESTION: RefinementStrategy.VERIFY,
    AnnotationType.INSIGHT: RefinementStrategy.CRITIQUE,
    AnnotationType.CONNECTION: RefinementStrategy.HOFSTADTER,
    AnnotationType.SUMMARY: RefinementStrategy.REFINE,
    AnnotationType.CORRECTION: RefinementStrategy.VERIFY,
}


@dataclass
class RefinementSuggestion:
    """A single refinement suggestion for an annotation."""
    original_text: str
    refined_text: str
    suggestion_type: str  # "clarity", "specificity", "structure", "tone", etc.
    confidence: float  # 0.0-1.0
    rationale: str
    strategy_used: RefinementStrategy
    pass_number: int = 1  # Which pass in multi-pass refinement

    def improvement_delta(self) -> float:
        """Estimate improvement from original to refined."""
        # Simple heuristic based on changes
        if self.original_text == self.refined_text:
            return 0.0

        len_change = abs(len(self.refined_text) - len(self.original_text))
        len_original = max(len(self.original_text), 1)

        # Moderate changes are often improvements
        change_ratio = len_change / len_original
        if 0.1 <= change_ratio <= 0.5:
            return self.confidence * 0.8
        elif change_ratio < 0.1:
            return self.confidence * 0.5  # Minor polish
        else:
            return self.confidence * 0.6  # Major rewrite

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "refined_text": self.refined_text,
            "suggestion_type": self.suggestion_type,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "strategy_used": self.strategy_used.value,
            "pass_number": self.pass_number,
            "improvement_delta": self.improvement_delta()
        }


@dataclass
class RefinementResult:
    """Result of annotation refinement process."""
    annotation_id: str
    annotation_type: AnnotationType
    original_text: str
    final_text: str
    suggestions: List[RefinementSuggestion]
    strategy_used: RefinementStrategy
    total_passes: int
    quality_before: float  # 0.0-1.0
    quality_after: float   # 0.0-1.0
    refinement_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def quality_improvement(self) -> float:
        """Calculate quality improvement."""
        return self.quality_after - self.quality_before

    @property
    def was_improved(self) -> bool:
        """Check if refinement improved the annotation."""
        return self.quality_after > self.quality_before

    def best_suggestion(self) -> Optional[RefinementSuggestion]:
        """Get highest confidence suggestion."""
        if not self.suggestions:
            return None
        return max(self.suggestions, key=lambda s: s.confidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "annotation_type": self.annotation_type.value,
            "original_text": self.original_text,
            "final_text": self.final_text,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "strategy_used": self.strategy_used.value,
            "total_passes": self.total_passes,
            "quality_before": self.quality_before,
            "quality_after": self.quality_after,
            "quality_improvement": self.quality_improvement,
            "was_improved": self.was_improved,
            "refinement_time_ms": self.refinement_time_ms,
            "metadata": self.metadata
        }


class AnnotationQualityScorer:
    """
    Scores annotation quality on multiple dimensions.

    Dimensions:
    - Clarity: Is the meaning clear?
    - Specificity: Is it specific enough to be actionable?
    - Completeness: Does it capture the full thought?
    - Tone: Is the tone appropriate for collaboration?
    - Structure: Is it well-organized?
    """

    def __init__(self):
        # Quality patterns for detection
        self.vague_patterns = [
            r'\b(stuff|things|etc|whatever)\b',
            r'\b(maybe|probably|might|perhaps|possibly)\b',
            r'\b(some|many|few|several)\b(?!\s+\w+\s+\w+)',  # vague without specifics
        ]

        self.clarity_enhancers = [
            r'\b(specifically|namely|for example|such as)\b',
            r'\b(because|therefore|consequently|thus)\b',
            r':\s*\d',  # numbered lists
        ]

        self.question_patterns = [
            r'\?$',
            r'^(what|why|how|when|where|who|which)\b',
        ]

        self.insight_patterns = [
            r'\b(implies|suggests|indicates|reveals)\b',
            r'\b(pattern|trend|correlation|relationship)\b',
            r'\b(key|important|critical|essential)\b',
        ]

    def score(self, text: str, annotation_type: AnnotationType) -> Dict[str, float]:
        """
        Score annotation on multiple quality dimensions.

        Returns dict with dimension scores (0.0-1.0) and overall score.
        """
        if not text or not text.strip():
            return {
                "clarity": 0.0,
                "specificity": 0.0,
                "completeness": 0.0,
                "tone": 0.5,  # neutral
                "structure": 0.0,
                "overall": 0.0
            }

        text_lower = text.lower()
        word_count = len(text.split())

        # Clarity score
        clarity = self._score_clarity(text, text_lower, word_count)

        # Specificity score
        specificity = self._score_specificity(text, text_lower, word_count)

        # Completeness score
        completeness = self._score_completeness(text, word_count, annotation_type)

        # Tone score
        tone = self._score_tone(text_lower)

        # Structure score
        structure = self._score_structure(text, word_count)

        # Overall weighted score (weights vary by annotation type)
        weights = self._get_weights(annotation_type)
        overall = (
            clarity * weights["clarity"] +
            specificity * weights["specificity"] +
            completeness * weights["completeness"] +
            tone * weights["tone"] +
            structure * weights["structure"]
        )

        return {
            "clarity": round(clarity, 3),
            "specificity": round(specificity, 3),
            "completeness": round(completeness, 3),
            "tone": round(tone, 3),
            "structure": round(structure, 3),
            "overall": round(overall, 3)
        }

    def _score_clarity(self, text: str, text_lower: str, word_count: int) -> float:
        """Score clarity dimension."""
        score = 0.7  # Base score

        # Penalize vague language
        vague_count = sum(
            len(re.findall(p, text_lower))
            for p in self.vague_patterns
        )
        score -= min(vague_count * 0.1, 0.3)

        # Reward clarity enhancers
        enhancer_count = sum(
            len(re.findall(p, text_lower))
            for p in self.clarity_enhancers
        )
        score += min(enhancer_count * 0.1, 0.2)

        # Sentence length affects clarity
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = word_count / max(len([s for s in sentences if s.strip()]), 1)

        if avg_sentence_length > 30:
            score -= 0.15  # Too long
        elif avg_sentence_length < 5:
            score -= 0.1   # Too fragmented

        return max(0.0, min(1.0, score))

    def _score_specificity(self, text: str, text_lower: str, word_count: int) -> float:
        """Score specificity dimension."""
        score = 0.5  # Base score

        # Numbers add specificity
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        score += min(len(numbers) * 0.1, 0.2)

        # Proper nouns/names add specificity
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        score += min(len(proper_nouns) * 0.05, 0.15)

        # Technical terms add specificity
        technical = re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', text_lower)  # snake_case
        score += min(len(technical) * 0.1, 0.15)

        # Very short text is often unspecific
        if word_count < 5:
            score *= 0.7

        return max(0.0, min(1.0, score))

    def _score_completeness(self, text: str, word_count: int, annotation_type: AnnotationType) -> float:
        """Score completeness dimension."""
        # Different types have different completeness expectations
        min_words = {
            AnnotationType.NOTE: 5,
            AnnotationType.QUESTION: 4,
            AnnotationType.INSIGHT: 10,
            AnnotationType.CONNECTION: 3,
            AnnotationType.SUMMARY: 15,
            AnnotationType.CORRECTION: 5,
        }

        expected_min = min_words.get(annotation_type, 5)

        if word_count < expected_min:
            return 0.3 + (word_count / expected_min) * 0.4
        elif word_count < expected_min * 3:
            return 0.7 + (word_count - expected_min) / (expected_min * 2) * 0.3
        else:
            return 1.0

    def _score_tone(self, text_lower: str) -> float:
        """Score tone appropriateness."""
        score = 0.8  # Assume neutral/good

        # Negative patterns
        negative = re.findall(r'\b(wrong|bad|stupid|terrible|awful|hate)\b', text_lower)
        score -= len(negative) * 0.15

        # Overly casual
        casual = re.findall(r'\b(lol|lmao|omg|wtf|haha)\b', text_lower)
        score -= len(casual) * 0.1

        # Collaborative language is good
        collaborative = re.findall(r'\b(we|our|together|consider|suggest)\b', text_lower)
        score += min(len(collaborative) * 0.05, 0.1)

        return max(0.0, min(1.0, score))

    def _score_structure(self, text: str, word_count: int) -> float:
        """Score structural organization."""
        score = 0.6  # Base

        # Punctuation indicates structure
        has_punctuation = bool(re.search(r'[.!?;:]', text))
        if has_punctuation:
            score += 0.1

        # Lists are well-structured
        has_list = bool(re.search(r'(\d+[.)]\s|\*\s|-\s|•)', text))
        if has_list:
            score += 0.15

        # Very short = no structure needed
        if word_count < 10:
            score = max(score, 0.7)

        # Paragraphs for longer text
        if word_count > 50:
            paragraphs = text.count('\n\n') + 1
            if paragraphs > 1:
                score += 0.1

        return max(0.0, min(1.0, score))

    def _get_weights(self, annotation_type: AnnotationType) -> Dict[str, float]:
        """Get dimension weights by annotation type."""
        weights = {
            AnnotationType.NOTE: {
                "clarity": 0.3, "specificity": 0.2, "completeness": 0.2,
                "tone": 0.1, "structure": 0.2
            },
            AnnotationType.QUESTION: {
                "clarity": 0.35, "specificity": 0.25, "completeness": 0.2,
                "tone": 0.1, "structure": 0.1
            },
            AnnotationType.INSIGHT: {
                "clarity": 0.25, "specificity": 0.3, "completeness": 0.25,
                "tone": 0.1, "structure": 0.1
            },
            AnnotationType.CONNECTION: {
                "clarity": 0.35, "specificity": 0.35, "completeness": 0.15,
                "tone": 0.05, "structure": 0.1
            },
            AnnotationType.SUMMARY: {
                "clarity": 0.25, "specificity": 0.15, "completeness": 0.35,
                "tone": 0.1, "structure": 0.15
            },
            AnnotationType.CORRECTION: {
                "clarity": 0.3, "specificity": 0.3, "completeness": 0.2,
                "tone": 0.15, "structure": 0.05
            },
        }
        return weights.get(annotation_type, weights[AnnotationType.NOTE])


class AnnotationRefiner:
    """
    Refines annotations using MRF strategies.

    Applies strategy-specific refinement passes to improve annotation quality:
    - ELEGANCE: Clarity -> Simplicity -> Beauty (3 passes)
    - VERIFY: Accuracy -> Completeness -> Consistency (3 passes)
    - CRITIQUE: Challenge assumptions (1 pass)
    - REFINE: Iterative improvement
    - HOFSTADTER: Recursive meta-analysis
    """

    def __init__(
        self,
        scorer: Optional[AnnotationQualityScorer] = None,
        strategy_map: Optional[Dict[AnnotationType, RefinementStrategy]] = None,
        enable_learning: bool = True
    ):
        self.scorer = scorer or AnnotationQualityScorer()
        self.strategy_map = strategy_map or ANNOTATION_STRATEGY_MAP.copy()
        self.enable_learning = enable_learning

        # Learning: track which refinements are accepted
        self.refinement_history: List[Dict[str, Any]] = []
        self.strategy_success_rates: Dict[str, Dict[str, float]] = {}

        # Refinement handlers by strategy
        self._strategy_handlers: Dict[RefinementStrategy, Callable] = {
            RefinementStrategy.ELEGANCE: self._apply_elegance,
            RefinementStrategy.VERIFY: self._apply_verify,
            RefinementStrategy.CRITIQUE: self._apply_critique,
            RefinementStrategy.REFINE: self._apply_refine,
            RefinementStrategy.HOFSTADTER: self._apply_hofstadter,
        }

    def refine(
        self,
        annotation_id: str,
        text: str,
        annotation_type: AnnotationType,
        context: Optional[Dict[str, Any]] = None,
        strategy: Optional[RefinementStrategy] = None,
        max_passes: int = 3
    ) -> RefinementResult:
        """
        Refine an annotation using the appropriate MRF strategy.

        Args:
            annotation_id: Unique annotation identifier
            text: Original annotation text
            annotation_type: Type of annotation
            context: Optional context (node info, session info, etc.)
            strategy: Override default strategy (uses type mapping if None)
            max_passes: Maximum refinement passes

        Returns:
            RefinementResult with suggestions and quality metrics
        """
        import time
        start_time = time.perf_counter()

        # Determine strategy
        if strategy is None:
            strategy = self.strategy_map.get(annotation_type, RefinementStrategy.ELEGANCE)

        # Score original
        quality_before = self.scorer.score(text, annotation_type)

        # Apply strategy
        handler = self._strategy_handlers.get(strategy)
        if handler is None:
            handler = self._apply_elegance  # Fallback

        suggestions, final_text = handler(
            text=text,
            annotation_type=annotation_type,
            context=context or {},
            max_passes=max_passes
        )

        # Score refined
        quality_after = self.scorer.score(final_text, annotation_type)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result = RefinementResult(
            annotation_id=annotation_id,
            annotation_type=annotation_type,
            original_text=text,
            final_text=final_text,
            suggestions=suggestions,
            strategy_used=strategy,
            total_passes=len(suggestions),
            quality_before=quality_before["overall"],
            quality_after=quality_after["overall"],
            refinement_time_ms=elapsed_ms,
            metadata={
                "quality_dimensions_before": quality_before,
                "quality_dimensions_after": quality_after,
                "context": context
            }
        )

        # Log for learning
        if self.enable_learning:
            self._log_refinement(result)

        return result

    def _apply_elegance(
        self,
        text: str,
        annotation_type: AnnotationType,
        context: Dict[str, Any],
        max_passes: int
    ) -> Tuple[List[RefinementSuggestion], str]:
        """
        ELEGANCE strategy: Clarity -> Simplicity -> Beauty

        Pass 1: Improve clarity (remove ambiguity)
        Pass 2: Simplify (remove unnecessary words)
        Pass 3: Polish (improve flow and rhythm)
        """
        suggestions = []
        current_text = text

        # Pass 1: Clarity
        if max_passes >= 1:
            clarity_text, clarity_suggestion = self._clarity_pass(current_text, annotation_type)
            if clarity_suggestion:
                clarity_suggestion.pass_number = 1
                suggestions.append(clarity_suggestion)
                current_text = clarity_text

        # Pass 2: Simplicity
        if max_passes >= 2:
            simple_text, simple_suggestion = self._simplicity_pass(current_text, annotation_type)
            if simple_suggestion:
                simple_suggestion.pass_number = 2
                suggestions.append(simple_suggestion)
                current_text = simple_text

        # Pass 3: Beauty (polish)
        if max_passes >= 3:
            polished_text, polish_suggestion = self._polish_pass(current_text, annotation_type)
            if polish_suggestion:
                polish_suggestion.pass_number = 3
                suggestions.append(polish_suggestion)
                current_text = polished_text

        return suggestions, current_text

    def _apply_verify(
        self,
        text: str,
        annotation_type: AnnotationType,
        context: Dict[str, Any],
        max_passes: int
    ) -> Tuple[List[RefinementSuggestion], str]:
        """
        VERIFY strategy: Accuracy -> Completeness -> Consistency

        Pass 1: Check for accuracy/specificity issues
        Pass 2: Ensure completeness
        Pass 3: Check internal consistency
        """
        suggestions = []
        current_text = text

        # Pass 1: Accuracy (for questions, ensure answerable)
        if max_passes >= 1:
            if annotation_type == AnnotationType.QUESTION:
                verified_text, verify_suggestion = self._verify_question(current_text)
            else:
                verified_text, verify_suggestion = self._verify_accuracy(current_text, annotation_type)

            if verify_suggestion:
                verify_suggestion.pass_number = 1
                suggestions.append(verify_suggestion)
                current_text = verified_text

        # Pass 2: Completeness
        if max_passes >= 2:
            complete_text, complete_suggestion = self._completeness_pass(current_text, annotation_type)
            if complete_suggestion:
                complete_suggestion.pass_number = 2
                suggestions.append(complete_suggestion)
                current_text = complete_text

        # Pass 3: Consistency
        if max_passes >= 3:
            consistent_text, consistent_suggestion = self._consistency_pass(current_text, annotation_type)
            if consistent_suggestion:
                consistent_suggestion.pass_number = 3
                suggestions.append(consistent_suggestion)
                current_text = consistent_text

        return suggestions, current_text

    def _apply_critique(
        self,
        text: str,
        annotation_type: AnnotationType,
        context: Dict[str, Any],
        max_passes: int
    ) -> Tuple[List[RefinementSuggestion], str]:
        """
        CRITIQUE strategy: Challenge assumptions (single pass)

        Identifies unsupported claims, missing evidence, logical gaps.
        """
        suggestions = []
        current_text = text

        # Find assumptions and unsupported claims
        assumptions = self._identify_assumptions(text)

        if assumptions:
            refined_text = text
            for assumption in assumptions[:3]:  # Limit to top 3
                refined_text = self._address_assumption(refined_text, assumption)

            if refined_text != text:
                suggestions.append(RefinementSuggestion(
                    original_text=text,
                    refined_text=refined_text,
                    suggestion_type="critique",
                    confidence=0.7,
                    rationale=f"Addressed {len(assumptions)} assumption(s): {', '.join(assumptions[:3])}",
                    strategy_used=RefinementStrategy.CRITIQUE,
                    pass_number=1
                ))
                current_text = refined_text

        return suggestions, current_text

    def _apply_refine(
        self,
        text: str,
        annotation_type: AnnotationType,
        context: Dict[str, Any],
        max_passes: int
    ) -> Tuple[List[RefinementSuggestion], str]:
        """
        REFINE strategy: Iterative expansion and improvement.

        Expands brief annotations with additional context.
        """
        suggestions = []
        current_text = text
        word_count = len(text.split())

        # If too brief, suggest expansion
        if word_count < 10 and annotation_type in [AnnotationType.SUMMARY, AnnotationType.INSIGHT]:
            expanded = self._expand_annotation(text, annotation_type, context)
            if expanded != text:
                suggestions.append(RefinementSuggestion(
                    original_text=text,
                    refined_text=expanded,
                    suggestion_type="expansion",
                    confidence=0.65,
                    rationale="Expanded brief annotation with additional context",
                    strategy_used=RefinementStrategy.REFINE,
                    pass_number=1
                ))
                current_text = expanded

        # Then apply elegance for polish
        elegance_suggestions, elegance_text = self._apply_elegance(
            current_text, annotation_type, context, max_passes=2
        )

        for s in elegance_suggestions:
            s.pass_number += len(suggestions)
        suggestions.extend(elegance_suggestions)

        return suggestions, elegance_text

    def _apply_hofstadter(
        self,
        text: str,
        annotation_type: AnnotationType,
        context: Dict[str, Any],
        max_passes: int
    ) -> Tuple[List[RefinementSuggestion], str]:
        """
        HOFSTADTER strategy: Recursive self-reference and meta-analysis.

        For connections, discovers deeper relationships.
        """
        suggestions = []
        current_text = text

        # Meta-analysis: what is this annotation really saying?
        meta_insight = self._meta_analyze(text, annotation_type)

        if meta_insight:
            # Enrich with meta-insight
            enriched = f"{text}\n\n[Meta-insight: {meta_insight}]"
            suggestions.append(RefinementSuggestion(
                original_text=text,
                refined_text=enriched,
                suggestion_type="meta_analysis",
                confidence=0.6,
                rationale=f"Added recursive meta-analysis: {meta_insight[:50]}...",
                strategy_used=RefinementStrategy.HOFSTADTER,
                pass_number=1
            ))
            current_text = enriched

        # For connections, identify implicit relationships
        if annotation_type == AnnotationType.CONNECTION:
            implicit_rels = self._find_implicit_relationships(text, context)
            if implicit_rels:
                rel_text = f"{current_text}\n\nImplicit connections: {', '.join(implicit_rels)}"
                suggestions.append(RefinementSuggestion(
                    original_text=current_text,
                    refined_text=rel_text,
                    suggestion_type="relationship_discovery",
                    confidence=0.55,
                    rationale=f"Discovered {len(implicit_rels)} implicit relationship(s)",
                    strategy_used=RefinementStrategy.HOFSTADTER,
                    pass_number=2
                ))
                current_text = rel_text

        return suggestions, current_text

    # --- Helper methods for refinement passes ---

    def _clarity_pass(self, text: str, annotation_type: AnnotationType) -> Tuple[str, Optional[RefinementSuggestion]]:
        """Improve clarity by removing ambiguity."""
        refined = text
        changes = []

        # Replace vague words
        replacements = [
            (r'\bstuff\b', 'items'),
            (r'\bthings\b', 'elements'),
            (r'\betc\b\.?', ''),
            (r'\bwhatever\b', ''),
        ]

        for pattern, replacement in replacements:
            if re.search(pattern, refined, re.IGNORECASE):
                refined = re.sub(pattern, replacement, refined, flags=re.IGNORECASE)
                changes.append(f"replaced '{pattern}' with '{replacement}'")

        # Clean up extra whitespace
        refined = re.sub(r'\s+', ' ', refined).strip()

        if refined != text and changes:
            return refined, RefinementSuggestion(
                original_text=text,
                refined_text=refined,
                suggestion_type="clarity",
                confidence=0.75,
                rationale=f"Improved clarity: {', '.join(changes[:3])}",
                strategy_used=RefinementStrategy.ELEGANCE
            )

        return text, None

    def _simplicity_pass(self, text: str, annotation_type: AnnotationType) -> Tuple[str, Optional[RefinementSuggestion]]:
        """Simplify by removing unnecessary words."""
        refined = text
        changes = []

        # Remove filler words
        fillers = [
            r'\b(basically|essentially|actually|really|very|quite)\b\s*',
            r'\b(in order to)\b',
            r'\b(the fact that)\b',
        ]

        for pattern in fillers:
            match = re.search(pattern, refined, re.IGNORECASE)
            if match:
                refined = re.sub(pattern, '', refined, flags=re.IGNORECASE)
                changes.append(f"removed '{match.group()}'")

        # Simplify verbose phrases
        verbose = [
            (r'at this point in time', 'now'),
            (r'due to the fact that', 'because'),
            (r'in the event that', 'if'),
        ]

        for pattern, replacement in verbose:
            if re.search(pattern, refined, re.IGNORECASE):
                refined = re.sub(pattern, replacement, refined, flags=re.IGNORECASE)
                changes.append(f"simplified '{pattern}' to '{replacement}'")

        refined = re.sub(r'\s+', ' ', refined).strip()

        if refined != text and changes:
            return refined, RefinementSuggestion(
                original_text=text,
                refined_text=refined,
                suggestion_type="simplicity",
                confidence=0.7,
                rationale=f"Simplified: {', '.join(changes[:3])}",
                strategy_used=RefinementStrategy.ELEGANCE
            )

        return text, None

    def _polish_pass(self, text: str, annotation_type: AnnotationType) -> Tuple[str, Optional[RefinementSuggestion]]:
        """Polish for flow and rhythm."""
        refined = text
        changes = []

        # Ensure proper capitalization
        if refined and refined[0].islower():
            refined = refined[0].upper() + refined[1:]
            changes.append("capitalized first letter")

        # Ensure ending punctuation
        if refined and refined[-1] not in '.!?':
            if annotation_type == AnnotationType.QUESTION:
                refined += '?'
                changes.append("added question mark")
            else:
                refined += '.'
                changes.append("added period")

        if refined != text and changes:
            return refined, RefinementSuggestion(
                original_text=text,
                refined_text=refined,
                suggestion_type="polish",
                confidence=0.8,
                rationale=f"Polished: {', '.join(changes)}",
                strategy_used=RefinementStrategy.ELEGANCE
            )

        return text, None

    def _verify_question(self, text: str) -> Tuple[str, Optional[RefinementSuggestion]]:
        """Verify question is well-formed and answerable."""
        refined = text
        issues = []

        # Ensure it's actually a question
        if '?' not in text:
            if not re.match(r'^(what|why|how|when|where|who|which|is|are|do|does|can|could|should)', text.lower()):
                issues.append("not clearly a question")
                refined = f"Question: {refined}?"

        # Check for specificity
        vague_question_words = ['something', 'anything', 'stuff']
        for word in vague_question_words:
            if word in text.lower():
                issues.append(f"vague word '{word}'")

        if issues:
            return refined, RefinementSuggestion(
                original_text=text,
                refined_text=refined,
                suggestion_type="question_verification",
                confidence=0.65,
                rationale=f"Question issues: {', '.join(issues)}",
                strategy_used=RefinementStrategy.VERIFY
            )

        return text, None

    def _verify_accuracy(self, text: str, annotation_type: AnnotationType) -> Tuple[str, Optional[RefinementSuggestion]]:
        """Verify accuracy of claims."""
        # For now, just flag absolute claims that may need hedging
        absolute_words = re.findall(r'\b(always|never|all|none|every|no one|everyone)\b', text.lower())

        if absolute_words:
            hedged = text
            for word in set(absolute_words):
                hedged = re.sub(rf'\b{word}\b', f'typically {word}', hedged, flags=re.IGNORECASE)

            return hedged, RefinementSuggestion(
                original_text=text,
                refined_text=hedged,
                suggestion_type="accuracy_hedging",
                confidence=0.6,
                rationale=f"Added hedging to absolute claims: {', '.join(set(absolute_words))}",
                strategy_used=RefinementStrategy.VERIFY
            )

        return text, None

    def _completeness_pass(self, text: str, annotation_type: AnnotationType) -> Tuple[str, Optional[RefinementSuggestion]]:
        """Check for completeness."""
        word_count = len(text.split())

        # Type-specific completeness checks
        if annotation_type == AnnotationType.INSIGHT and word_count < 10:
            expanded = f"{text} This suggests further investigation may be valuable."
            return expanded, RefinementSuggestion(
                original_text=text,
                refined_text=expanded,
                suggestion_type="completeness",
                confidence=0.55,
                rationale="Added follow-up suggestion to brief insight",
                strategy_used=RefinementStrategy.VERIFY
            )

        return text, None

    def _consistency_pass(self, text: str, annotation_type: AnnotationType) -> Tuple[str, Optional[RefinementSuggestion]]:
        """Check internal consistency."""
        # Look for contradictions
        contradictions = [
            (r'\bbut\s+however\b', 'but'),
            (r'\balthough\s+but\b', 'although'),
        ]

        refined = text
        found = []

        for pattern, replacement in contradictions:
            if re.search(pattern, refined, re.IGNORECASE):
                refined = re.sub(pattern, replacement, refined, flags=re.IGNORECASE)
                found.append(pattern)

        if found:
            return refined, RefinementSuggestion(
                original_text=text,
                refined_text=refined,
                suggestion_type="consistency",
                confidence=0.7,
                rationale="Fixed redundant/contradictory conjunctions",
                strategy_used=RefinementStrategy.VERIFY
            )

        return text, None

    def _identify_assumptions(self, text: str) -> List[str]:
        """Identify unstated assumptions in text."""
        assumptions = []

        # Causal claims without evidence
        if re.search(r'\b(because|since|therefore|thus)\b', text.lower()):
            if not re.search(r'\b(evidence|data|research|study|found|shows)\b', text.lower()):
                assumptions.append("causal claim without cited evidence")

        # Generalizations
        if re.search(r'\b(always|never|all|everyone|no one)\b', text.lower()):
            assumptions.append("universal generalization")

        # Implicit knowledge assumptions
        if re.search(r'\b(obviously|clearly|of course|naturally)\b', text.lower()):
            assumptions.append("assumes shared knowledge")

        return assumptions

    def _address_assumption(self, text: str, assumption: str) -> str:
        """Suggest addressing an assumption."""
        if assumption == "causal claim without cited evidence":
            return re.sub(r'\b(because|since)\b', r'(potentially \1', text, count=1)
        elif assumption == "universal generalization":
            text = re.sub(r'\balways\b', 'often', text, flags=re.IGNORECASE)
            text = re.sub(r'\bnever\b', 'rarely', text, flags=re.IGNORECASE)
            return text
        elif assumption == "assumes shared knowledge":
            text = re.sub(r'\b(obviously|clearly)\b', 'arguably', text, flags=re.IGNORECASE)
            return text
        return text

    def _expand_annotation(self, text: str, annotation_type: AnnotationType, context: Dict[str, Any]) -> str:
        """Expand a brief annotation with context."""
        node_name = context.get("node_name", "this concept")

        if annotation_type == AnnotationType.SUMMARY:
            return f"{text} This summary covers the key aspects of {node_name}."
        elif annotation_type == AnnotationType.INSIGHT:
            return f"{text} This insight about {node_name} may have broader implications."

        return text

    def _meta_analyze(self, text: str, annotation_type: AnnotationType) -> Optional[str]:
        """Generate meta-analysis of the annotation."""
        word_count = len(text.split())

        # Simple meta-analysis based on structure
        if annotation_type == AnnotationType.CONNECTION:
            if ' -> ' in text or ' → ' in text or ' relates to ' in text.lower():
                return "This describes a directional relationship between concepts."
            elif ' and ' in text.lower():
                return "This connects multiple concepts in a non-hierarchical relationship."

        if word_count > 20:
            return "This is a detailed annotation that may benefit from summarization."

        return None

    def _find_implicit_relationships(self, text: str, context: Dict[str, Any]) -> List[str]:
        """Find implicit relationships in connection annotations."""
        implicit = []

        # Look for transitive relationships
        if ' -> ' in text:
            parts = text.split(' -> ')
            if len(parts) >= 3:
                implicit.append(f"{parts[0]} -> {parts[-1]} (transitive)")

        # Look for category membership
        if ' is a ' in text.lower():
            implicit.append("hierarchical (IS_A)")

        if ' uses ' in text.lower() or ' requires ' in text.lower():
            implicit.append("dependency (USES)")

        return implicit

    def _log_refinement(self, result: RefinementResult):
        """Log refinement for learning."""
        self.refinement_history.append({
            "annotation_id": result.annotation_id,
            "annotation_type": result.annotation_type.value,
            "strategy": result.strategy_used.value,
            "quality_improvement": result.quality_improvement,
            "was_improved": result.was_improved,
            "timestamp": datetime.now().isoformat()
        })

        # Keep history bounded
        if len(self.refinement_history) > 1000:
            self.refinement_history = self.refinement_history[-500:]

    def get_strategy_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """Calculate effectiveness of each strategy by annotation type."""
        from collections import defaultdict

        stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        for record in self.refinement_history:
            key = f"{record['annotation_type']}_{record['strategy']}"
            stats[record['annotation_type']][record['strategy']].append(
                record['quality_improvement']
            )

        effectiveness = {}
        for ann_type, strategies in stats.items():
            effectiveness[ann_type] = {}
            for strategy, improvements in strategies.items():
                if improvements:
                    effectiveness[ann_type][strategy] = sum(improvements) / len(improvements)

        return effectiveness

    def suggest_strategy(self, annotation_type: AnnotationType, context: Optional[Dict[str, Any]] = None) -> RefinementStrategy:
        """
        Suggest best strategy based on learning history.

        Uses Thompson Sampling-like approach if enough history exists.
        """
        effectiveness = self.get_strategy_effectiveness()
        type_key = annotation_type.value

        if type_key in effectiveness and effectiveness[type_key]:
            # Pick strategy with best historical performance
            best_strategy = max(effectiveness[type_key].items(), key=lambda x: x[1])
            try:
                return RefinementStrategy(best_strategy[0])
            except ValueError:
                pass

        # Fall back to default mapping
        return self.strategy_map.get(annotation_type, RefinementStrategy.ELEGANCE)


# Factory functions
def create_annotation_refiner(
    enable_learning: bool = True,
    custom_strategy_map: Optional[Dict[AnnotationType, RefinementStrategy]] = None
) -> AnnotationRefiner:
    """Create an annotation refiner with MRF strategies."""
    return AnnotationRefiner(
        scorer=AnnotationQualityScorer(),
        strategy_map=custom_strategy_map,
        enable_learning=enable_learning
    )


def get_default_strategy(annotation_type: AnnotationType) -> RefinementStrategy:
    """Get the default MRF strategy for an annotation type."""
    return ANNOTATION_STRATEGY_MAP.get(annotation_type, RefinementStrategy.ELEGANCE)
