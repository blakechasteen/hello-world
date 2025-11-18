"""
Sentiment Analyzer Plugin - Example O2 Plugin

Analyzes sentiment of messages and proposals to detect community mood.

Features:
- Real-time sentiment analysis of messages
- Proposal sentiment tracking
- Community mood dashboard
- Alert on negative sentiment spikes
"""

import asyncio
from typing import Dict, Optional
import re

from bot.plugin_system import PluginBase, PluginContext


class SentimentAnalyzerPlugin(PluginBase):
    """
    Example plugin that analyzes sentiment of messages and proposals.
    """

    def __init__(self):
        super().__init__()
        self.sentiment_history = []  # List of (timestamp, sentiment, message)

    async def on_load(self, context: PluginContext):
        """Initialize plugin."""
        self.context = context
        print(f"[SentimentAnalyzer] Plugin loaded!")

    async def on_enable(self):
        """Called when plugin is enabled."""
        print(f"[SentimentAnalyzer] Plugin enabled - monitoring sentiment...")

    async def on_disable(self):
        """Called when plugin is disabled."""
        print(f"[SentimentAnalyzer] Plugin disabled")

    async def on_message(self, room_id: str, sender: str, message: str) -> Optional[str]:
        """
        Analyze sentiment of incoming message.

        Args:
            room_id: Matrix room ID
            sender: Message sender
            message: Message content

        Returns:
            Sentiment analysis or None
        """
        # Simple sentiment analysis (would use ML in production)
        sentiment = self._analyze_sentiment(message)

        # Store in history
        import datetime
        self.sentiment_history.append({
            'timestamp': datetime.datetime.now(),
            'sentiment': sentiment,
            'message': message,
            'sender': sender
        })

        # Alert on very negative sentiment
        if sentiment < -0.7:
            return f"⚠️ **Sentiment Alert**: Message has very negative sentiment ({sentiment:.2f}). Community mood may need attention."

        return None

    async def on_proposal_created(self, proposal_id: int, title: str, author: str):
        """Analyze sentiment of proposal title."""
        sentiment = self._analyze_sentiment(title)

        print(f"[SentimentAnalyzer] Proposal #{proposal_id} sentiment: {sentiment:.2f}")

        # Could send message to room if very negative
        if sentiment < -0.5:
            # Would need room_id context to send message
            print(f"[SentimentAnalyzer] Warning: Proposal has negative sentiment")

    def _analyze_sentiment(self, text: str) -> float:
        """
        Simple sentiment analysis.

        Returns:
            Sentiment score from -1.0 (very negative) to +1.0 (very positive)
        """
        # Simple keyword-based sentiment (production would use BERT, etc.)
        positive_words = {
            'good', 'great', 'excellent', 'awesome', 'love', 'happy',
            'yes', 'agree', 'support', 'approve', 'thanks', 'wonderful'
        }
        negative_words = {
            'bad', 'terrible', 'awful', 'hate', 'angry', 'sad',
            'no', 'disagree', 'reject', 'oppose', 'disappointed', 'frustrated'
        }

        # Tokenize
        words = re.findall(r'\w+', text.lower())

        # Count sentiment words
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)

        # Calculate score
        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return (pos_count - neg_count) / total
