#!/usr/bin/env python3
"""
Auto-Scoring System for Prompts
================================
Automatically evaluate prompt quality using multiple criteria.

Features:
- Clarity scoring (how clear is the prompt?)
- Completeness scoring (does it include all necessary elements?)
- Effectiveness scoring (likelihood of good results)
- Overall quality score (weighted average)

Can use LLM-based evaluation or rule-based heuristics.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re
from enum import Enum


class ScoringMethod(Enum):
    """Scoring method types"""
    HEURISTIC = "heuristic"  # Rule-based scoring
    LLM = "llm"  # LLM-based evaluation
    HYBRID = "hybrid"  # Combination of both


@dataclass
class QualityScore:
    """Quality score breakdown"""
    clarity: float  # 0.0-1.0
    completeness: float  # 0.0-1.0
    effectiveness: float  # 0.0-1.0
    overall: float  # Weighted average
    method: str  # Scoring method used
    feedback: List[str]  # Improvement suggestions


class PromptAutoScorer:
    """
    Automatically score prompt quality.

    Provides both heuristic and LLM-based scoring.
    """

    def __init__(self, method: ScoringMethod = ScoringMethod.HEURISTIC):
        """
        Initialize auto-scorer.

        Args:
            method: Scoring method to use
        """
        self.method = method

    def score_prompt(self, content: str, context: Optional[Dict[str, Any]] = None) -> QualityScore:
        """
        Score a prompt's quality.

        Args:
            content: Prompt content
            context: Optional context (tags, usage history, etc.)

        Returns:
            QualityScore with breakdown
        """
        if self.method == ScoringMethod.HEURISTIC:
            return self._heuristic_score(content, context)
        elif self.method == ScoringMethod.LLM:
            return self._llm_score(content, context)
        else:  # HYBRID
            heuristic = self._heuristic_score(content, context)
            # Could blend with LLM if available
            return heuristic

    def _heuristic_score(self, content: str, context: Optional[Dict[str, Any]]) -> QualityScore:
        """
        Rule-based heuristic scoring.

        Criteria:
        - Clarity: Length, structure, specificity
        - Completeness: Has context, examples, constraints
        - Effectiveness: Uses good practices (role, format, etc.)
        """
        feedback = []

        # === Clarity Scoring ===
        clarity_score = 0.0
        clarity_factors = 0

        # Length check (20-500 words is ideal)
        word_count = len(content.split())
        if 20 <= word_count <= 500:
            clarity_score += 1.0
            clarity_factors += 1
        elif word_count < 20:
            clarity_score += 0.3
            clarity_factors += 1
            feedback.append("Prompt is too short. Add more context or detail.")
        else:  # > 500
            clarity_score += 0.7
            clarity_factors += 1
            feedback.append("Prompt is quite long. Consider breaking into sections.")

        # Has clear structure (paragraphs, lists, sections)
        has_structure = ('\n\n' in content or  # Paragraphs
                         '\n-' in content or    # Lists
                         '\n*' in content or
                         '\n1.' in content or   # Numbered lists
                         '##' in content)       # Headings
        if has_structure:
            clarity_score += 1.0
            clarity_factors += 1
        else:
            clarity_score += 0.4
            clarity_factors += 1
            feedback.append("Add structure (lists, sections) for better clarity.")

        # Uses specific language (not too vague)
        vague_words = ['something', 'anything', 'stuff', 'things', 'maybe', 'kinda', 'sorta']
        vague_count = sum(1 for word in vague_words if word in content.lower())
        if vague_count == 0:
            clarity_score += 1.0
            clarity_factors += 1
        elif vague_count <= 2:
            clarity_score += 0.6
            clarity_factors += 1
            feedback.append("Try to be more specific (avoid vague language).")
        else:
            clarity_score += 0.3
            clarity_factors += 1
            feedback.append("Too many vague words. Be more specific.")

        clarity = clarity_score / clarity_factors if clarity_factors > 0 else 0.5

        # === Completeness Scoring ===
        completeness_score = 0.0
        completeness_factors = 0

        # Has examples or specifics
        has_examples = ('{' in content or  # Variables/placeholders
                        'example' in content.lower() or
                        'e.g.' in content.lower() or
                        'for instance' in content.lower())
        if has_examples:
            completeness_score += 1.0
            completeness_factors += 1
        else:
            completeness_score += 0.4
            completeness_factors += 1
            feedback.append("Add examples or placeholders to clarify expected input.")

        # Has context or background
        has_context = ('context:' in content.lower() or
                       'background:' in content.lower() or
                       'about:' in content.lower() or
                       len(content) > 100)  # Longer prompts usually have context
        if has_context:
            completeness_score += 1.0
            completeness_factors += 1
        else:
            completeness_score += 0.5
            completeness_factors += 1
            feedback.append("Add context or background information.")

        # Has constraints or requirements
        has_constraints = ('must' in content.lower() or
                           'should' in content.lower() or
                           'required' in content.lower() or
                           'limit' in content.lower() or
                           'constraint' in content.lower())
        if has_constraints:
            completeness_score += 1.0
            completeness_factors += 1
        else:
            completeness_score += 0.6
            completeness_factors += 1
            feedback.append("Specify constraints or requirements.")

        completeness = completeness_score / completeness_factors if completeness_factors > 0 else 0.5

        # === Effectiveness Scoring ===
        effectiveness_score = 0.0
        effectiveness_factors = 0

        # Uses role-playing ("You are...", "Act as...")
        has_role = ('you are' in content.lower() or
                    'act as' in content.lower() or
                    'role:' in content.lower())
        if has_role:
            effectiveness_score += 1.0
            effectiveness_factors += 1
        else:
            effectiveness_score += 0.6
            effectiveness_factors += 1
            feedback.append("Consider specifying a role (e.g., 'You are an expert...')")

        # Specifies output format
        has_format = ('format:' in content.lower() or
                      'output:' in content.lower() or
                      'return' in content.lower() or
                      'json' in content.lower() or
                      'markdown' in content.lower() or
                      'list' in content.lower())
        if has_format:
            effectiveness_score += 1.0
            effectiveness_factors += 1
        else:
            effectiveness_score += 0.5
            effectiveness_factors += 1
            feedback.append("Specify the desired output format.")

        # Uses action verbs (analyze, generate, create, etc.)
        action_verbs = ['analyze', 'generate', 'create', 'design', 'write', 'summarize',
                        'explain', 'compare', 'evaluate', 'optimize', 'review']
        has_action = any(verb in content.lower() for verb in action_verbs)
        if has_action:
            effectiveness_score += 1.0
            effectiveness_factors += 1
        else:
            effectiveness_score += 0.4
            effectiveness_factors += 1
            feedback.append("Use clear action verbs (analyze, generate, etc.).")

        effectiveness = effectiveness_score / effectiveness_factors if effectiveness_factors > 0 else 0.5

        # === Overall Score (weighted average) ===
        # Weights: clarity 30%, completeness 35%, effectiveness 35%
        overall = (clarity * 0.30 + completeness * 0.35 + effectiveness * 0.35)

        # Add summary feedback based on overall score
        if overall > 0.8:
            feedback.insert(0, "Excellent prompt! Minor improvements possible.")
        elif overall > 0.6:
            feedback.insert(0, "Good prompt with room for improvement.")
        else:
            feedback.insert(0, "Prompt needs significant improvement.")

        return QualityScore(
            clarity=clarity,
            completeness=completeness,
            effectiveness=effectiveness,
            overall=overall,
            method="heuristic",
            feedback=feedback
        )

    def _llm_score(self, content: str, context: Optional[Dict[str, Any]]) -> QualityScore:
        """
        LLM-based scoring (placeholder for now).

        Would use an LLM to evaluate the prompt quality.
        Could integrate with Ollama, OpenAI, etc.
        """
        # Fallback to heuristic if LLM not available
        score = self._heuristic_score(content, context)
        score.method = "llm_fallback"
        score.feedback.append("Note: Using heuristic fallback (LLM scoring not yet implemented).")
        return score

    def score_batch(self, prompts: List[Dict[str, Any]]) -> List[QualityScore]:
        """
        Score multiple prompts at once.

        Args:
            prompts: List of prompt dicts with 'content' and optional 'context'

        Returns:
            List of QualityScore objects
        """
        scores = []
        for prompt in prompts:
            content = prompt.get('content', '')
            context = prompt.get('context', None)
            score = self.score_prompt(content, context)
            scores.append(score)
        return scores


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_score(content: str) -> float:
    """
    Get a quick quality score (0.0-1.0).

    Args:
        content: Prompt content

    Returns:
        Overall quality score
    """
    scorer = PromptAutoScorer()
    return scorer.score_prompt(content).overall


def detailed_score(content: str) -> QualityScore:
    """
    Get detailed quality breakdown.

    Args:
        content: Prompt content

    Returns:
        Full QualityScore object
    """
    scorer = PromptAutoScorer()
    return scorer.score_prompt(content)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Prompt Auto-Scoring System\n")

    # Test prompts
    test_prompts = [
        {
            "name": "Bad Example (vague)",
            "content": "Write something about AI"
        },
        {
            "name": "Good Example (structured)",
            "content": """
You are an expert technical writer.

Task: Explain how transformer models work in machine learning.

Requirements:
- Write for a technical audience with some ML background
- Include examples and diagrams where helpful
- Limit to 500 words
- Use clear, concise language

Format: Markdown with headings and code examples where appropriate.
"""
        },
        {
            "name": "Medium Example",
            "content": """
Analyze this code for bugs and suggest improvements.

Code: {code_snippet}

Focus on performance and security issues.
"""
        }
    ]

    scorer = PromptAutoScorer()

    for test in test_prompts:
        print(f"=== {test['name']} ===")
        score = scorer.score_prompt(test['content'])

        print(f"Overall: {score.overall:.2f}")
        print(f"  Clarity: {score.clarity:.2f}")
        print(f"  Completeness: {score.completeness:.2f}")
        print(f"  Effectiveness: {score.effectiveness:.2f}")

        print("\nFeedback:")
        for fb in score.feedback:
            print(f"  - {fb}")

        print("\n")
