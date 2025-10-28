#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversational AutoLoom
=======================
Automatically captures and spins conversation history into memory.

After each input/output exchange, the system:
1. Evaluates importance of the exchange
2. If important, spins it into a MemoryShard
3. Adds to knowledge base for future context

This creates an evolving memory that learns from the conversation.

Usage:
    conv = await ConversationalAutoLoom.from_text("Initial knowledge...")

    # Each exchange automatically becomes memory
    response1 = await conv.chat("What is HoloLoom?")
    # ^ This Q&A is now in memory for future questions

    response2 = await conv.chat("Tell me more about the policy engine")
    # ^ This one too! And it has context from previous exchange
"""

import asyncio
import re
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field

try:
    from holoLoom.documentation.types import Query, MemoryShard
    from holoLoom.config import Config
    from holoLoom.orchestrator import HoloLoomOrchestrator
    from holoLoom.spinningWheel import TextSpinner, TextSpinnerConfig
    from holoLoom.autospin import AutoSpinOrchestrator
except ImportError as e:
    print(f"Import error: {e}")
    print("\nMake sure you run from repository root with PYTHONPATH set")
    raise


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    turn_id: int
    timestamp: str
    user_input: str
    system_output: str
    importance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Convert to text representation for spinning."""
        return f"""
Conversation Turn {self.turn_id} ({self.timestamp})

User: {self.user_input}

System: {self.system_output}

Importance: {self.importance_score:.2f}
"""


class ImportanceScorer:
    """
    Determines if a conversation turn is important enough to remember.

    Signal vs Noise filtering:
    - HIGH importance: New information, decisions, facts, references
    - MEDIUM importance: Clarifications, explanations, meaningful Q&A
    - LOW importance: Greetings, acknowledgments, trivial exchanges
    """

    @staticmethod
    def score_turn(user_input: str, system_output: str, metadata: Dict = None) -> float:
        """
        Score importance of a conversation turn (0.0 to 1.0).

        Args:
            user_input: User's message
            system_output: System's response
            metadata: Optional metadata (tool used, confidence, etc.)

        Returns:
            Importance score (0.0 = noise, 1.0 = critical signal)
        """
        score = 0.5  # Start neutral

        # ================================================================
        # NOISE INDICATORS (reduce score)
        # ================================================================

        # Very short exchanges
        if len(user_input) < 10 and len(system_output) < 20:
            score -= 0.3

        # Greetings and pleasantries
        greetings = r'\b(hi|hello|hey|thanks|thank you|ok|okay|bye|goodbye)\b'
        if re.search(greetings, user_input.lower()) and len(user_input) < 30:
            score -= 0.4

        # Acknowledgments only
        acknowledgments = r'^(ok|okay|sure|yes|no|got it|i see|alright)[\.\!]?$'
        if re.match(acknowledgments, user_input.lower().strip()):
            score -= 0.4

        # Error messages (usually not worth remembering)
        if 'error' in system_output.lower() or 'failed' in system_output.lower():
            score -= 0.2

        # ================================================================
        # SIGNAL INDICATORS (increase score)
        # ================================================================

        # Questions (usually important)
        if '?' in user_input:
            score += 0.2

        # Substantive content (longer messages)
        if len(user_input) > 50:
            score += 0.1
        if len(system_output) > 100:
            score += 0.1

        # Information-dense words
        info_keywords = [
            'how', 'what', 'why', 'when', 'where', 'who',
            'explain', 'describe', 'tell me', 'show me',
            'define', 'meaning', 'example', 'difference'
        ]
        keyword_matches = sum(1 for kw in info_keywords if kw in user_input.lower())
        score += keyword_matches * 0.1

        # Domain-specific terms (high signal for technical conversations)
        domain_terms = [
            'policy', 'thompson', 'sampling', 'embedding', 'neural',
            'memory', 'shard', 'orchestrator', 'spinner', 'motif',
            'knowledge', 'graph', 'vector', 'retrieval', 'bandit'
        ]
        domain_matches = sum(1 for term in domain_terms if term in user_input.lower() or term in system_output.lower())
        score += domain_matches * 0.05

        # Metadata signals
        if metadata:
            # High confidence responses are usually important
            confidence = metadata.get('confidence', 0.5)
            score += (confidence - 0.5) * 0.2

            # Certain tools indicate important interactions
            important_tools = ['notion_write', 'calc', 'search']
            if metadata.get('tool') in important_tools:
                score += 0.15

        # References to specific entities or facts
        # Capitalized words often indicate proper nouns (names, places, concepts)
        caps_words = re.findall(r'\b[A-Z][a-z]+\b', user_input + ' ' + system_output)
        score += min(len(set(caps_words)) * 0.02, 0.2)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    @staticmethod
    def should_remember(score: float, threshold: float = 0.4) -> bool:
        """
        Determine if a turn should be remembered based on score.

        Args:
            score: Importance score
            threshold: Minimum score to remember (default: 0.4)

        Returns:
            True if should remember, False if noise
        """
        return score >= threshold


class ConversationalAutoLoom:
    """
    AutoLoom that automatically captures conversation history.

    After each exchange, it:
    1. Scores importance
    2. Spins important turns into MemoryShards
    3. Adds to knowledge base

    This creates a self-building episodic memory.
    """

    def __init__(
        self,
        orchestrator: AutoSpinOrchestrator,
        importance_threshold: float = 0.4,
        max_history_size: int = 100,
        custom_scorer: Optional[Callable] = None
    ):
        """
        Initialize conversational system.

        Args:
            orchestrator: Base AutoSpinOrchestrator
            importance_threshold: Min score to remember (0.0-1.0)
            max_history_size: Max conversation turns to keep
            custom_scorer: Optional custom importance scoring function
        """
        self.orchestrator = orchestrator
        self.importance_threshold = importance_threshold
        self.max_history_size = max_history_size
        self.scorer = custom_scorer or ImportanceScorer.score_turn

        self.conversation_history: List[ConversationTurn] = []
        self.turn_counter = 0

        # Stats
        self.stats = {
            'total_turns': 0,
            'remembered_turns': 0,
            'forgotten_turns': 0,
            'avg_importance': 0.0
        }

    @classmethod
    async def from_text(
        cls,
        text: str,
        config: Optional[Config] = None,
        importance_threshold: float = 0.4,
        **kwargs
    ) -> "ConversationalAutoLoom":
        """
        Create conversational loom from initial knowledge base.

        Args:
            text: Initial knowledge base
            config: HoloLoom config
            importance_threshold: Min importance to remember
            **kwargs: Additional args for AutoSpinOrchestrator

        Returns:
            ConversationalAutoLoom ready for chat
        """
        # Create base orchestrator
        orch = await AutoSpinOrchestrator.from_text(
            text=text,
            config=config,
            **kwargs
        )

        return cls(
            orchestrator=orch,
            importance_threshold=importance_threshold
        )

    @classmethod
    async def from_file(
        cls,
        file_path: str,
        config: Optional[Config] = None,
        importance_threshold: float = 0.4
    ) -> "ConversationalAutoLoom":
        """Create from file."""
        orch = await AutoSpinOrchestrator.from_file(
            file_path=file_path,
            config=config
        )

        return cls(
            orchestrator=orch,
            importance_threshold=importance_threshold
        )

    async def chat(
        self,
        user_input: str,
        auto_remember: bool = True
    ) -> Dict[str, Any]:
        """
        Chat with the system (automatically captures to memory).

        Args:
            user_input: User's message
            auto_remember: If True, automatically spin important turns

        Returns:
            Response dict with system output
        """
        # Process query
        query = Query(text=user_input)
        response = await self.orchestrator.process(query)

        # Extract system output
        system_output = response.get('result', str(response))

        # Create conversation turn
        turn = ConversationTurn(
            turn_id=self.turn_counter,
            timestamp=datetime.now().isoformat(),
            user_input=user_input,
            system_output=system_output,
            metadata={
                'tool': response.get('tool'),
                'confidence': response.get('confidence', 0.5),
                'status': response.get('status')
            }
        )

        # Score importance
        turn.importance_score = self.scorer(
            user_input,
            system_output,
            turn.metadata
        )

        # Update stats
        self.stats['total_turns'] += 1
        self.stats['avg_importance'] = (
            (self.stats['avg_importance'] * self.stats['total_turns'] + turn.importance_score) /
            (self.stats['total_turns'] + 1)
        )

        # Add to history
        self.conversation_history.append(turn)
        self.turn_counter += 1

        # Trim history if needed
        if len(self.conversation_history) > self.max_history_size:
            self.conversation_history = self.conversation_history[-self.max_history_size:]

        # Auto-remember if important
        if auto_remember and ImportanceScorer.should_remember(turn.importance_score, self.importance_threshold):
            await self._remember_turn(turn)
            self.stats['remembered_turns'] += 1
        else:
            self.stats['forgotten_turns'] += 1

        # Add metadata to response
        response['_meta'] = {
            'turn_id': turn.turn_id,
            'importance': turn.importance_score,
            'remembered': turn.importance_score >= self.importance_threshold,
            'timestamp': turn.timestamp
        }

        return response

    async def _remember_turn(self, turn: ConversationTurn) -> None:
        """
        Spin a conversation turn into memory.

        This adds it to the knowledge base so future queries can reference it.
        """
        # Convert turn to text
        turn_text = turn.to_text()

        # Spin it into memory
        source = f"conversation_turn_{turn.turn_id}"
        await self.orchestrator.add_text(turn_text, source=source)

    def get_history(self, limit: Optional[int] = None, min_importance: float = 0.0) -> List[ConversationTurn]:
        """
        Get conversation history.

        Args:
            limit: Max turns to return
            min_importance: Only return turns above this importance

        Returns:
            List of conversation turns
        """
        filtered = [t for t in self.conversation_history if t.importance_score >= min_importance]

        if limit:
            filtered = filtered[-limit:]

        return filtered

    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return {
            **self.stats,
            'current_memory_shards': self.orchestrator.get_shard_count(),
            'history_size': len(self.conversation_history),
            'remember_rate': (
                self.stats['remembered_turns'] / self.stats['total_turns']
                if self.stats['total_turns'] > 0
                else 0.0
            )
        }

    def print_history(self, limit: int = 10) -> None:
        """Print recent conversation history."""
        recent = self.get_history(limit=limit)

        print(f"\n{'='*70}")
        print(f"Conversation History (last {len(recent)} turns)")
        print(f"{'='*70}\n")

        for turn in recent:
            remembered = "✓" if turn.importance_score >= self.importance_threshold else "✗"
            print(f"Turn {turn.turn_id} [{remembered}] (Importance: {turn.importance_score:.2f})")
            print(f"  User: {turn.user_input[:60]}...")
            print(f"  System: {turn.system_output[:60]}...")
            print()

    def clear_history(self, keep_memory: bool = True) -> None:
        """
        Clear conversation history.

        Args:
            keep_memory: If True, keeps the spun memory shards,
                        only clears the conversation history list
        """
        self.conversation_history = []
        self.turn_counter = 0

        if not keep_memory:
            # Would need to rebuild orchestrator from scratch
            # (not implemented - usually you want to keep learned knowledge)
            pass


# ============================================================================
# Convenience Functions
# ============================================================================

async def conversational_loom(
    initial_knowledge: str = "",
    importance_threshold: float = 0.4,
    config: Optional[Config] = None
) -> ConversationalAutoLoom:
    """
    Quick function to create a conversational loom.

    Args:
        initial_knowledge: Starting knowledge base (optional)
        importance_threshold: Min score to remember (0.4 = balanced)
        config: HoloLoom config

    Returns:
        Ready-to-chat ConversationalAutoLoom

    Example:
        loom = await conversational_loom("HoloLoom is a neural system...")

        response = await loom.chat("What is HoloLoom?")
        # ^ This Q&A is now remembered!

        response = await loom.chat("Tell me more")
        # ^ Can reference previous context
    """
    if not initial_knowledge:
        initial_knowledge = "Conversational AI system initialized."

    return await ConversationalAutoLoom.from_text(
        text=initial_knowledge,
        config=config or Config.fast(),
        importance_threshold=importance_threshold
    )
