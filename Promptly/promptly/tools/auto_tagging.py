#!/usr/bin/env python3
"""
Auto-Tagging System for Prompts
================================
Automatically extract and suggest tags from prompt content.

Features:
- Keyword extraction
- Topic detection
- Domain classification
- Tag suggestions based on content analysis
"""

from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import re
from collections import Counter


@dataclass
class TagSuggestion:
    """A suggested tag with confidence"""
    tag: str
    confidence: float  # 0.0-1.0
    reason: str  # Why this tag was suggested


class AutoTagger:
    """
    Automatically extract tags from prompt content.

    Uses keyword extraction and pattern matching.
    """

    def __init__(self):
        """Initialize auto-tagger with domain keywords."""
        # Domain-specific keywords
        self.domain_keywords = {
            'programming': [
                'code', 'function', 'class', 'variable', 'debug', 'compile',
                'programming', 'developer', 'software', 'algorithm', 'data structure'
            ],
            'database': [
                'sql', 'query', 'database', 'table', 'index', 'join', 'select',
                'postgres', 'mysql', 'mongodb', 'nosql'
            ],
            'ai-ml': [
                'model', 'training', 'neural', 'machine learning', 'deep learning',
                'ai', 'artificial intelligence', 'tensor', 'gradient', 'prediction'
            ],
            'writing': [
                'write', 'article', 'essay', 'content', 'blog', 'story',
                'documentation', 'technical writing', 'copywriting'
            ],
            'analysis': [
                'analyze', 'analysis', 'evaluate', 'assess', 'review', 'examine',
                'study', 'research', 'investigate'
            ],
            'optimization': [
                'optimize', 'optimization', 'performance', 'efficiency', 'improve',
                'enhance', 'speed up', 'faster'
            ],
            'security': [
                'security', 'secure', 'vulnerability', 'authentication', 'authorization',
                'encryption', 'password', 'attack', 'threat'
            ],
            'testing': [
                'test', 'testing', 'unit test', 'integration test', 'qa',
                'quality assurance', 'bug', 'debugging'
            ],
            'design': [
                'design', 'architecture', 'pattern', 'structure', 'blueprint',
                'layout', 'ui', 'ux', 'interface'
            ],
            'data': [
                'data', 'dataset', 'dataframe', 'statistics', 'analytics',
                'visualization', 'analysis', 'metrics'
            ]
        }

        # Common technical tags
        self.tech_tags = {
            'python', 'javascript', 'java', 'c++', 'rust', 'go',
            'react', 'vue', 'angular', 'django', 'flask',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp',
            'git', 'github', 'ci/cd', 'devops'
        }

        # Action-based tags
        self.action_tags = {
            'refactoring': ['refactor', 'restructure', 'reorganize'],
            'debugging': ['debug', 'fix', 'troubleshoot', 'diagnose'],
            'creation': ['create', 'generate', 'build', 'develop', 'implement'],
            'explanation': ['explain', 'describe', 'clarify', 'elaborate'],
            'comparison': ['compare', 'contrast', 'versus', 'vs', 'difference']
        }

    def extract_tags(
        self,
        content: str,
        max_tags: int = 10,
        min_confidence: float = 0.5
    ) -> List[TagSuggestion]:
        """
        Extract tags from prompt content.

        Args:
            content: Prompt content
            max_tags: Maximum number of tags to return
            min_confidence: Minimum confidence threshold

        Returns:
            List of TagSuggestion objects
        """
        suggestions = []
        content_lower = content.lower()

        # === Domain-based tags ===
        for domain, keywords in self.domain_keywords.items():
            matches = sum(1 for kw in keywords if kw in content_lower)
            if matches > 0:
                confidence = min(1.0, matches / len(keywords) * 3)
                if confidence >= min_confidence:
                    suggestions.append(TagSuggestion(
                        tag=domain,
                        confidence=confidence,
                        reason=f"Found {matches} related keyword(s)"
                    ))

        # === Technical tags ===
        for tech in self.tech_tags:
            if tech.lower() in content_lower:
                suggestions.append(TagSuggestion(
                    tag=tech,
                    confidence=1.0,
                    reason="Explicitly mentioned"
                ))

        # === Action-based tags ===
        for action, keywords in self.action_tags.items():
            if any(kw in content_lower for kw in keywords):
                suggestions.append(TagSuggestion(
                    tag=action,
                    confidence=0.8,
                    reason="Action verb detected"
                ))

        # === Extract key nouns (simple approach) ===
        # Look for capitalized words or technical terms
        words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', content)  # CamelCase
        words += re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', content)  # snake_case

        # Count word frequency
        word_counts = Counter(w.lower() for w in words)
        common_words = word_counts.most_common(3)

        for word, count in common_words:
            if len(word) > 3 and count > 1:  # At least 4 chars, mentioned twice
                suggestions.append(TagSuggestion(
                    tag=word.replace('_', '-'),
                    confidence=0.6,
                    reason=f"Frequently mentioned ({count}x)"
                ))

        # === Deduplicate and sort ===
        seen = set()
        unique_suggestions = []

        for sug in sorted(suggestions, key=lambda x: x.confidence, reverse=True):
            if sug.tag not in seen:
                seen.add(sug.tag)
                unique_suggestions.append(sug)

        return unique_suggestions[:max_tags]

    def suggest_additional_tags(
        self,
        content: str,
        existing_tags: List[str],
        limit: int = 5
    ) -> List[TagSuggestion]:
        """
        Suggest tags not already present.

        Args:
            content: Prompt content
            existing_tags: Tags already applied
            limit: Maximum new tags to suggest

        Returns:
            List of new TagSuggestion objects
        """
        all_suggestions = self.extract_tags(content, max_tags=20)
        existing_set = set(tag.lower() for tag in existing_tags)

        # Filter out existing tags
        new_suggestions = [
            sug for sug in all_suggestions
            if sug.tag.lower() not in existing_set
        ]

        return new_suggestions[:limit]

    def validate_tags(self, tags: List[str]) -> List[Dict[str, any]]:
        """
        Validate and normalize tag list.

        Args:
            tags: List of tags to validate

        Returns:
            List of dicts with normalized tags and issues
        """
        results = []

        for tag in tags:
            issues = []

            # Normalize
            normalized = tag.lower().strip()
            normalized = re.sub(r'[^a-z0-9-]', '-', normalized)  # Replace special chars
            normalized = re.sub(r'-+', '-', normalized)  # Remove duplicate hyphens
            normalized = normalized.strip('-')  # Remove leading/trailing hyphens

            # Check for issues
            if len(normalized) < 2:
                issues.append("Too short (min 2 chars)")
            if len(normalized) > 30:
                issues.append("Too long (max 30 chars)")
            if normalized != tag:
                issues.append(f"Normalized to: {normalized}")

            results.append({
                'original': tag,
                'normalized': normalized,
                'valid': len(issues) == 0 or (len(issues) == 1 and 'Normalized' in issues[0]),
                'issues': issues
            })

        return results

    def merge_similar_tags(self, tags: List[str]) -> Dict[str, List[str]]:
        """
        Find and group similar tags.

        Args:
            tags: List of tags

        Returns:
            Dict mapping canonical tag to similar variants
        """
        groups = {}
        processed = set()

        for tag in tags:
            if tag in processed:
                continue

            # Find similar tags (simple string similarity)
            similar = [tag]
            tag_lower = tag.lower()

            for other_tag in tags:
                if other_tag == tag or other_tag in processed:
                    continue

                other_lower = other_tag.lower()

                # Check similarity
                if (tag_lower in other_lower or other_lower in tag_lower or
                    tag_lower.replace('-', '') == other_lower.replace('-', '')):
                    similar.append(other_tag)
                    processed.add(other_tag)

            if len(similar) > 1:
                # Use shortest tag as canonical
                canonical = min(similar, key=len)
                groups[canonical] = similar

            processed.add(tag)

        return groups


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_tag(content: str, max_tags: int = 5) -> List[str]:
    """
    Quick tag extraction (just tag names).

    Args:
        content: Prompt content
        max_tags: Maximum tags

    Returns:
        List of tag strings
    """
    tagger = AutoTagger()
    suggestions = tagger.extract_tags(content, max_tags=max_tags)
    return [sug.tag for sug in suggestions]


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Auto-Tagging System\n")

    # Test prompts
    test_prompts = [
        """
        Analyze this Python code for security vulnerabilities.
        Focus on SQL injection, XSS, and authentication issues.
        """,
        """
        Create a React component for user authentication with JWT tokens.
        Include login, logout, and token refresh functionality.
        Use TypeScript and follow best practices.
        """,
        """
        Optimize this database query for better performance.
        The query joins multiple tables and has a slow execution time.
        """,
        """
        Write a technical blog post explaining how neural networks work.
        Target audience: developers with no ML background.
        Include diagrams and code examples in Python.
        """
    ]

    tagger = AutoTagger()

    for i, prompt in enumerate(test_prompts, 1):
        print(f"=== Prompt {i} ===")
        print(prompt[:60] + "...")
        print("\nSuggested Tags:")

        suggestions = tagger.extract_tags(prompt, max_tags=7)

        for sug in suggestions:
            confidence_bar = "#" * int(sug.confidence * 10)
            print(f"  {confidence_bar:10} {sug.tag:20} ({sug.reason})")

        print("\n")
