"""
Chat History Spinner
====================

Ingests conversation history from ConversationManager into HoloLoom's memory system.
Makes all past conversations queryable via the unified memory API.

Features:
- Converts conversation turns into MemoryShards
- Preserves temporal ordering and conversation context
- Importance scoring for relevance filtering
- Searchable by topic, entities, or conversation metadata

Usage:
    from HoloLoom.spinningWheel.chat_history import ChatHistorySpinner
    from HoloLoom.web_dashboard.conversation_manager import ConversationManager

    # Ingest all conversations
    conv_mgr = ConversationManager("./data/conversations.db")
    spinner = ChatHistorySpinner(conv_mgr)

    shards = await spinner.spin_all()  # All conversations
    # or
    shards = await spinner.spin_conversation(conversation_id=42)  # Specific conversation

    # Add to HoloLoom memory
    async with HoloLoom() as loom:
        for shard in shards:
            await loom.experience(shard.text)

        # Now you can query conversations!
        results = await loom.recall("What did we discuss about Thompson Sampling?")
"""

import asyncio
import time
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from HoloLoom.documentation.types import MemoryShard
from HoloLoom.web_dashboard.conversation_manager import ConversationManager, Message, Conversation


@dataclass
class ImportanceScore:
    """Importance score for a conversation turn."""
    score: float  # 0.0 to 1.0
    signals: Dict[str, float]  # Individual scoring signals
    reason: str  # Human-readable explanation


class ChatHistorySpinner:
    """
    Spins conversation history into queryable MemoryShards.

    Each conversation turn (user + assistant message pair) becomes a MemoryShard
    that can be searched and retrieved by HoloLoom's memory system.

    Conversation Structure:
    - Each turn is a MemoryShard with user query + assistant response
    - Metadata includes timestamps, conversation ID, importance score
    - Entities and motifs extracted from conversation content
    - Temporal relationships preserved via episode field
    """

    def __init__(
        self,
        conversation_manager: ConversationManager,
        importance_threshold: float = 0.3,
        extract_entities: bool = True,
        extract_motifs: bool = True
    ):
        """
        Initialize chat history spinner.

        Args:
            conversation_manager: ConversationManager instance
            importance_threshold: Minimum importance score to include (0.0-1.0)
            extract_entities: Whether to extract entities from conversations
            extract_motifs: Whether to extract motifs/topics
        """
        self.conv_mgr = conversation_manager
        self.importance_threshold = importance_threshold
        self.extract_entities = extract_entities
        self.extract_motifs = extract_motifs

    async def spin_all(
        self,
        limit: Optional[int] = None,
        min_importance: Optional[float] = None
    ) -> List[MemoryShard]:
        """
        Spin all conversations into MemoryShards.

        Args:
            limit: Maximum number of conversations to process
            min_importance: Override importance threshold

        Returns:
            List of MemoryShards from all conversations
        """
        start_time = time.time()

        # Get all conversations (list_conversations returns (list, has_more))
        conversations, _ = self.conv_mgr.list_conversations(limit=limit or 1000)

        all_shards = []
        for conv in conversations:
            shards = await self.spin_conversation(
                conv.id,
                min_importance=min_importance
            )
            all_shards.extend(shards)

        processing_time = (time.time() - start_time) * 1000
        print(f"ChatHistorySpinner: Processed {len(conversations)} conversations "
              f"into {len(all_shards)} shards in {processing_time:.1f}ms")

        return all_shards

    async def spin_conversation(
        self,
        conversation_id: int,
        min_importance: Optional[float] = None
    ) -> List[MemoryShard]:
        """
        Spin a single conversation into MemoryShards.

        Args:
            conversation_id: ID of conversation to spin
            min_importance: Override importance threshold

        Returns:
            List of MemoryShards from this conversation
        """
        # Get conversation metadata
        conv = self.conv_mgr.get_conversation(conversation_id)
        if not conv:
            return []

        # Get all messages
        messages = self.conv_mgr.get_messages(conversation_id)

        # Group messages into turns (user + assistant pairs)
        turns = self._group_into_turns(messages)

        # Convert turns to shards
        shards = []
        threshold = min_importance or self.importance_threshold

        for turn_idx, turn in enumerate(turns):
            # Score importance
            importance = self._score_importance(turn)

            if importance.score < threshold:
                continue  # Skip low-importance turns

            # Create shard
            shard = self._create_shard(
                turn=turn,
                turn_idx=turn_idx,
                conversation=conv,
                importance=importance
            )
            shards.append(shard)

        return shards

    async def spin_recent(
        self,
        hours: int = 24,
        min_importance: Optional[float] = None
    ) -> List[MemoryShard]:
        """
        Spin recent conversations (within last N hours).

        Args:
            hours: Number of hours to look back
            min_importance: Override importance threshold

        Returns:
            List of MemoryShards from recent conversations
        """
        # Get all conversations (ConversationManager doesn't have time filtering)
        # We'll filter by message timestamps instead
        conversations, _ = self.conv_mgr.list_conversations(limit=100)

        all_shards = []
        cutoff_time = datetime.now().timestamp() - (hours * 3600)

        for conv in conversations:
            messages = self.conv_mgr.get_messages(conv.id)

            # Filter recent messages
            recent_messages = [
                m for m in messages
                if datetime.fromisoformat(m.timestamp).timestamp() >= cutoff_time
            ]

            if not recent_messages:
                continue

            # Process this conversation
            shards = await self.spin_conversation(conv.id, min_importance)
            all_shards.extend(shards)

        return all_shards

    def _group_into_turns(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Group messages into conversation turns (user + assistant pairs).

        Args:
            messages: List of messages in chronological order

        Returns:
            List of turn dicts with user_msg and assistant_msg
        """
        turns = []

        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.role == 'user':
                # Look for assistant response
                assistant_msg = None
                if i + 1 < len(messages) and messages[i + 1].role == 'assistant':
                    assistant_msg = messages[i + 1]
                    i += 2
                else:
                    i += 1

                turns.append({
                    'user_msg': msg,
                    'assistant_msg': assistant_msg,
                    'timestamp': msg.timestamp
                })
            else:
                # Orphaned assistant message (shouldn't happen, but handle it)
                i += 1

        return turns

    def _score_importance(self, turn: Dict[str, Any]) -> ImportanceScore:
        """
        Score the importance of a conversation turn.

        Uses multiple signals:
        - Message length (longer = more substantive)
        - Question indicators (questions = seeking knowledge)
        - Technical terms (domain-specific = higher signal)
        - Metadata (confidence, reasoning steps)

        Args:
            turn: Turn dict with user_msg and assistant_msg

        Returns:
            ImportanceScore with score and reasoning
        """
        user_msg = turn['user_msg']
        assistant_msg = turn.get('assistant_msg')

        signals = {}

        # === LENGTH SIGNAL ===
        user_len = len(user_msg.content)
        assistant_len = len(assistant_msg.content) if assistant_msg else 0

        # Normalize to 0-1 (50 chars = 0.5, 200+ chars = 1.0)
        len_score = min((user_len + assistant_len) / 300, 1.0)
        signals['length'] = len_score * 0.2

        # === QUESTION SIGNAL ===
        has_question = '?' in user_msg.content
        signals['question'] = 0.3 if has_question else 0.0

        # === TECHNICAL TERMS SIGNAL ===
        technical_terms = [
            'policy', 'thompson', 'sampling', 'embedding', 'neural',
            'memory', 'shard', 'orchestrator', 'spinner', 'motif',
            'knowledge', 'graph', 'vector', 'retrieval', 'bandit',
            'matryoshka', 'spectral', 'warp', 'loom', 'weaving'
        ]

        combined_text = (user_msg.content + ' ' + (assistant_msg.content if assistant_msg else '')).lower()
        term_matches = sum(1 for term in technical_terms if term in combined_text)
        tech_score = min(term_matches * 0.1, 0.4)
        signals['technical'] = tech_score

        # === METADATA SIGNAL ===
        if assistant_msg and assistant_msg.metadata:
            confidence = assistant_msg.metadata.get('confidence', 0.5)
            signals['confidence'] = (confidence - 0.5) * 0.2

            reasoning_steps = assistant_msg.metadata.get('reasoning_steps', [])
            if len(reasoning_steps) > 1:
                signals['reasoning_depth'] = 0.15

        # === NOISE FILTERS ===
        # Greetings and acknowledgments
        greetings = r'\b(hi|hello|hey|thanks|thank you|ok|okay|bye)\b'
        if re.search(greetings, user_msg.content.lower()) and user_len < 30:
            signals['noise_penalty'] = -0.3

        # Very short exchanges
        if user_len < 10 and assistant_len < 20:
            signals['noise_penalty'] = signals.get('noise_penalty', 0) - 0.2

        # === COMPUTE TOTAL ===
        total_score = max(0.0, min(1.0, sum(signals.values())))

        # Generate reason
        top_signals = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        reason = ', '.join([f"{k}={v:.2f}" for k, v in top_signals])

        return ImportanceScore(
            score=total_score,
            signals=signals,
            reason=reason
        )

    def _create_shard(
        self,
        turn: Dict[str, Any],
        turn_idx: int,
        conversation: Conversation,
        importance: ImportanceScore
    ) -> MemoryShard:
        """
        Create a MemoryShard from a conversation turn.

        Args:
            turn: Turn dict with user_msg and assistant_msg
            turn_idx: Index of this turn in conversation
            conversation: Conversation metadata
            importance: ImportanceScore for this turn

        Returns:
            MemoryShard representing this conversation turn
        """
        user_msg = turn['user_msg']
        assistant_msg = turn.get('assistant_msg')

        # Build shard text
        text_parts = [f"[Conversation: {conversation.title}]"]
        text_parts.append(f"Turn {turn_idx + 1} ({turn['timestamp']})")
        text_parts.append(f"\nUser: {user_msg.content}")

        if assistant_msg:
            text_parts.append(f"\nAssistant: {assistant_msg.content}")

        shard_text = '\n'.join(text_parts)

        # Extract entities (capitalized words, proper nouns)
        entities = []
        if self.extract_entities:
            combined = user_msg.content + ' ' + (assistant_msg.content if assistant_msg else '')
            # Simple entity extraction: capitalized words
            cap_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', combined)
            entities = list(set(cap_words[:10]))  # Limit to 10 unique entities

        # Extract motifs (key phrases, questions)
        motifs = []
        if self.extract_motifs:
            # Extract questions as motifs
            questions = re.findall(r'([^.!?]*\?)', user_msg.content)
            motifs.extend([q.strip() for q in questions[:3]])

            # Extract key phrases (simplified)
            key_patterns = [
                r'what (?:is|are|was|were) (\w+(?:\s+\w+){0,3})',
                r'how (?:do|does|did) (\w+(?:\s+\w+){0,3})',
                r'explain (\w+(?:\s+\w+){0,3})',
                r'(?:tell me|show me) (?:about )?(\w+(?:\s+\w+){0,3})'
            ]
            for pattern in key_patterns:
                matches = re.findall(pattern, user_msg.content.lower())
                motifs.extend(matches[:2])

        # Create shard
        return MemoryShard(
            id=f"chat_conv{conversation.id}_turn{turn_idx}",
            text=shard_text,
            episode=conversation.title,
            entities=entities,
            motifs=motifs,
            metadata={
                'spinner': 'ChatHistorySpinner',
                'conversation_id': conversation.id,
                'conversation_title': conversation.title,
                'turn_index': turn_idx,
                'timestamp': turn['timestamp'],
                'importance_score': importance.score,
                'importance_signals': importance.signals,
                'importance_reason': importance.reason,
                'user_message_id': user_msg.id,
                'assistant_message_id': assistant_msg.id if assistant_msg else None,
                'user_metadata': user_msg.metadata,
                'assistant_metadata': assistant_msg.metadata if assistant_msg else None,
                'project_id': conversation.project_id,
                'is_favorite': conversation.is_favorite,
                'tags': conversation.tags.split(',') if conversation.tags else []
            }
        )

    async def spin_by_tag(self, tag: str) -> List[MemoryShard]:
        """
        Spin conversations with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of MemoryShards from tagged conversations
        """
        conversations, _ = self.conv_mgr.list_conversations(limit=1000)
        tagged_convs = [
            c for c in conversations
            if tag in (c.tags.split(',') if c.tags else [])
        ]

        all_shards = []
        for conv in tagged_convs:
            shards = await self.spin_conversation(conv.id)
            all_shards.extend(shards)

        return all_shards

    async def spin_by_project(self, project_id: int) -> List[MemoryShard]:
        """
        Spin conversations in a specific project.

        Args:
            project_id: Project ID to filter by

        Returns:
            List of MemoryShards from project conversations
        """
        conversations, _ = self.conv_mgr.list_conversations(limit=1000)
        project_convs = [c for c in conversations if c.project_id == project_id]

        all_shards = []
        for conv in project_convs:
            shards = await self.spin_conversation(conv.id)
            all_shards.extend(shards)

        return all_shards


# ============================================================================
# Auto-Capture Integration
# ============================================================================

class ChatHistoryAutoCapture:
    """
    Automatically captures new conversations to HoloLoom memory.

    Hooks into ConversationManager to automatically ingest conversations
    as they happen, keeping HoloLoom's memory in sync with chat history.

    Usage:
        from HoloLoom import HoloLoom
        from HoloLoom.web_dashboard.conversation_manager import ConversationManager
        from HoloLoom.spinningWheel.chat_history import ChatHistoryAutoCapture

        conv_mgr = ConversationManager()
        loom = await HoloLoom().__aenter__()

        # Enable auto-capture
        auto_capture = ChatHistoryAutoCapture(conv_mgr, loom)

        # Now all new conversations are automatically ingested!
        conv_mgr.add_message(conv_id, 'user', 'What is Thompson Sampling?')
        conv_mgr.add_message(conv_id, 'assistant', 'Thompson Sampling is...')
        # ^ These are automatically added to HoloLoom memory
    """

    def __init__(
        self,
        conversation_manager: ConversationManager,
        hololoom_instance,
        importance_threshold: float = 0.3,
        batch_size: int = 10,
        batch_interval: float = 5.0
    ):
        """
        Initialize auto-capture.

        Args:
            conversation_manager: ConversationManager to monitor
            hololoom_instance: HoloLoom instance to ingest into
            importance_threshold: Minimum importance to capture
            batch_size: Batch messages before ingesting
            batch_interval: Max seconds between batches
        """
        self.conv_mgr = conversation_manager
        self.loom = hololoom_instance
        self.spinner = ChatHistorySpinner(
            conversation_manager,
            importance_threshold=importance_threshold
        )

        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.pending_conversations = set()
        self.last_batch_time = time.time()

        # Background task tracking
        self._background_tasks: set[asyncio.Task] = set()

        # Patch ConversationManager.add_message to trigger capture
        self._original_add_message = conversation_manager.add_message
        conversation_manager.add_message = self._hooked_add_message

    def _spawn_background_task(self, coro):
        """Spawn a background task and track it."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._task_done_callback)
        return task

    def _task_done_callback(self, task: asyncio.Task):
        """Safe cleanup callback for completed tasks."""
        try:
            self._background_tasks.discard(task)
        except (KeyError, RuntimeError):
            # Task already removed or set being modified during iteration
            pass

    def _hooked_add_message(self, conversation_id: int, role: str, content: str, metadata=None):
        """Hooked version of add_message that triggers auto-capture."""
        # Call original method
        result = self._original_add_message(conversation_id, role, content, metadata)

        # Track for ingestion
        self.pending_conversations.add(conversation_id)

        # Ingest if batch full or time elapsed
        should_ingest = (
            len(self.pending_conversations) >= self.batch_size or
            (time.time() - self.last_batch_time) >= self.batch_interval
        )

        if should_ingest:
            self._spawn_background_task(self._ingest_pending())

        return result

    async def _ingest_pending(self):
        """Ingest all pending conversations."""
        if not self.pending_conversations:
            return

        conv_ids = list(self.pending_conversations)
        self.pending_conversations.clear()
        self.last_batch_time = time.time()

        for conv_id in conv_ids:
            shards = await self.spinner.spin_conversation(conv_id)

            for shard in shards:
                await self.loom.experience(shard.text)

        print(f"ChatHistoryAutoCapture: Ingested {len(conv_ids)} conversations "
              f"({sum(len(await self.spinner.spin_conversation(c)) for c in conv_ids)} shards)")

    async def close(self):
        """Cleanup resources and cancel background tasks."""
        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for tasks to complete (with timeout)
        if self._background_tasks:
            await asyncio.wait(self._background_tasks, timeout=5.0)

    def disable(self):
        """Disable auto-capture (restore original add_message)."""
        self.conv_mgr.add_message = self._original_add_message


# ============================================================================
# Convenience Functions
# ============================================================================

async def ingest_chat_history(
    conversation_manager: ConversationManager,
    hololoom_instance,
    limit: Optional[int] = None,
    importance_threshold: float = 0.3
) -> int:
    """
    Quick function to ingest all chat history into HoloLoom.

    Args:
        conversation_manager: ConversationManager instance
        hololoom_instance: HoloLoom instance
        limit: Max conversations to process (None = all)
        importance_threshold: Min importance to include

    Returns:
        Number of shards ingested
    """
    spinner = ChatHistorySpinner(
        conversation_manager,
        importance_threshold=importance_threshold
    )

    shards = await spinner.spin_all(limit=limit)

    for shard in shards:
        await hololoom_instance.experience(shard.text)

    print(f"Ingested {len(shards)} conversation shards into HoloLoom memory")

    return len(shards)