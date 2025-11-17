"""
Text Tools - Text Processing Utilities
=======================================
Tools for text manipulation and analysis.
"""

import re
from typing import List, Dict, Any


def register_text_tools(server):
    """Register text processing tools with MCP server."""

    @server.tool(
        "count_words",
        "Count words in text",
        category="text",
        timeout=5.0
    )
    async def count_words(text: str, unique_only: bool = False) -> Dict[str, int]:
        """
        Count words in text.

        Args:
            text: Input text
            unique_only: Count only unique words

        Returns:
            Dictionary with word counts
        """
        # Split into words
        words = re.findall(r'\b\w+\b', text.lower())

        if unique_only:
            return {
                "unique_words": len(set(words)),
                "total_words": len(words)
            }
        else:
            # Word frequency
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

            return {
                "total_words": len(words),
                "unique_words": len(word_freq),
                "most_common": sorted(
                    word_freq.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }

    @server.tool(
        "extract_emails",
        "Extract email addresses from text",
        category="text",
        timeout=5.0
    )
    async def extract_emails(text: str) -> List[str]:
        """
        Extract email addresses.

        Args:
            text: Input text

        Returns:
            List of email addresses
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return list(set(emails))  # Remove duplicates

    @server.tool(
        "summarize_text",
        "Generate text summary (extractive)",
        category="text",
        timeout=10.0
    )
    async def summarize_text(text: str, num_sentences: int = 3) -> str:
        """
        Generate extractive summary.

        Args:
            text: Input text
            num_sentences: Number of sentences in summary

        Returns:
            Summary text

        Note:
            This is a simple implementation. For production, use
            transformer models or more sophisticated algorithms.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Simple scoring: prefer sentences with more words
        scored = [(s, len(s.split())) for s in sentences]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top N sentences
        top_sentences = [s for s, _ in scored[:num_sentences]]

        # Preserve original order
        summary_sentences = [s for s in sentences if s in top_sentences]

        return '. '.join(summary_sentences[:num_sentences]) + '.'


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("="*80)
        print("Text Tools Demo")
        print("="*80 + "\n")

        # Mock server
        class MockServer:
            def tool(self, name, desc, **kwargs):
                def decorator(func):
                    return func
                return decorator

        server = MockServer()
        register_text_tools(server)

        # Test text
        sample_text = """
        Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.
        It balances exploration and exploitation by sampling from posterior distributions.
        This algorithm is widely used in online advertising and recommendation systems.
        Contact us at info@example.com or support@hololoom.ai for more information.
        """

        print("1. Word Count:")
        word_stats = await count_words(sample_text)
        print(f"   Total words: {word_stats['total_words']}")
        print(f"   Unique words: {word_stats['unique_words']}")
        if 'most_common' in word_stats:
            print(f"   Most common: {word_stats['most_common'][:3]}\n")

        print("2. Extract Emails:")
        emails = await extract_emails(sample_text)
        for email in emails:
            print(f"   - {email}")
        print()

        print("3. Summarize:")
        summary = await summarize_text(sample_text, num_sentences=2)
        print(f"   {summary}\n")

        print("✓ Demo complete!")

    asyncio.run(demo())
