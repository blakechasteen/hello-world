"""
Unified Multithreaded Chat - Thread Management
===============================================

Manages conversation threads with automatic detection and awareness tracking.

Key Features:
- Auto-detect thread context from semantic similarity
- Thread-aware awareness tracking (confidence builds over thread)
- Semantic clustering (threads form clusters in 228D space)
- Smart thread merging (references to earlier topics)

Architecture:
    User Message
        ↓
    Awareness Analysis
        ↓
    Thread Detection (auto or explicit)
        ↓
    Retrieve Thread Context
        ↓
    Generate with Full Context
        ↓
    Update Thread + Awareness
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
import uuid
import numpy as np
import asyncio


class ThreadStatus(Enum):
    """Thread lifecycle status"""
    ACTIVE = "active"          # Currently being discussed
    DORMANT = "dormant"        # Inactive but available
    MERGED = "merged"          # Merged into another thread
    ARCHIVED = "archived"      # Archived, read-only


@dataclass
class Message:
    """Single message in a conversation thread"""
    id: str
    thread_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime

    # Awareness context at this message
    awareness_snapshot: Optional[Any] = None  # UnifiedAwarenessContext
    meta_reflection: Optional[Any] = None     # SelfReflectionResult

    # Thread position
    depth: int = 0  # How deep in thread (0 = root)
    parent_message_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        return {
            'id': self.id,
            'thread_id': self.thread_id,
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'depth': self.depth,
            'parent_message_id': self.parent_message_id,
            'metadata': self.metadata,
            # Awareness snapshot converted separately if needed
        }


@dataclass
class ConversationThread:
    """A conversation thread with full awareness tracking"""
    id: str
    root_message: Message
    messages: List[Message] = field(default_factory=list)

    # Awareness tracking
    awareness_trajectory: List[Any] = field(default_factory=list)  # List[UnifiedAwarenessContext]
    confidence_trend: List[float] = field(default_factory=list)

    # Semantic positioning
    semantic_cluster: Optional[np.ndarray] = None  # Position in 228D space
    dominant_topic: str = "General"

    # Thread metadata
    status: ThreadStatus = ThreadStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    # Thread relationships
    parent_thread_id: Optional[str] = None  # For sub-threads
    related_thread_ids: List[str] = field(default_factory=list)  # Semantic similarity
    merged_into: Optional[str] = None  # If merged into another thread

    def get_context_window(self, max_messages: int = 10) -> List[Message]:
        """Get recent messages for context"""
        return self.messages[-max_messages:]

    def add_message(self, message: Message):
        """Add message to thread"""
        self.messages.append(message)
        self.last_activity = datetime.now()

    def update_awareness(self, awareness_ctx: Any):
        """Update awareness trajectory"""
        self.awareness_trajectory.append(awareness_ctx)

        # Extract confidence
        if hasattr(awareness_ctx, 'confidence'):
            confidence = 1.0 - awareness_ctx.confidence.uncertainty_level
            self.confidence_trend.append(confidence)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        return {
            'id': self.id,
            'root_message': self.root_message.to_dict(),
            'message_count': len(self.messages),
            'messages': [m.to_dict() for m in self.messages],
            'dominant_topic': self.dominant_topic,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'confidence_trend': self.confidence_trend,
            'related_thread_ids': self.related_thread_ids,
        }


class ThreadManager:
    """
    Manages conversation threads with automatic detection.

    Features:
    - Auto-detect thread from semantic similarity
    - Track awareness across thread
    - Semantic clustering
    - Thread merging and relationships
    """

    def __init__(self, awareness_layer=None, llm_generator=None, memory_backend=None):
        """
        Initialize thread manager.

        Args:
            awareness_layer: CompositionalAwarenessLayer instance
            llm_generator: LLM for response generation
            memory_backend: Optional persistent memory backend (KGStore)
        """
        self.threads: Dict[str, ConversationThread] = {}
        self.active_thread_id: Optional[str] = None

        # Awareness integration
        self.awareness = awareness_layer
        self.llm = llm_generator

        # Persistent memory integration
        self.memory = memory_backend
        self.enable_persistence = memory_backend is not None

        # Semantic index for thread detection
        self.semantic_index: Dict[str, np.ndarray] = {}  # thread_id → embedding

        # Statistics
        self.total_messages = 0
        self.thread_count = 0

        # Background task tracking
        self._background_tasks = set()

    async def process_message(
        self,
        user_input: str,
        explicit_thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process user message and generate response.

        Args:
            user_input: User's message
            explicit_thread_id: Optional explicit thread selection

        Returns:
            Dict with response, thread info, and awareness context
        """

        # 1. Analyze with awareness
        if self.awareness:
            awareness_ctx = await self.awareness.get_unified_context(user_input)
        else:
            awareness_ctx = None

        # 2. Detect thread (auto or explicit)
        if explicit_thread_id and explicit_thread_id in self.threads:
            thread = self.threads[explicit_thread_id]
        else:
            thread = await self._detect_or_create_thread(user_input, awareness_ctx)

        # 3. Create user message
        user_msg = Message(
            id=str(uuid.uuid4()),
            thread_id=thread.id,
            role="user",
            content=user_input,
            timestamp=datetime.now(),
            awareness_snapshot=awareness_ctx,
            depth=len(thread.messages)
        )

        # 4. Retrieve thread context for LLM
        context_messages = thread.get_context_window(max_messages=10)

        # 5. Generate response with thread context
        response_content = await self._generate_response(
            user_input,
            awareness_ctx,
            context_messages
        )

        # 6. Create assistant message
        assistant_msg = Message(
            id=str(uuid.uuid4()),
            thread_id=thread.id,
            role="assistant",
            content=response_content,
            timestamp=datetime.now(),
            depth=len(thread.messages) + 1
        )

        # 7. Update thread
        thread.add_message(user_msg)
        thread.add_message(assistant_msg)

        if awareness_ctx:
            thread.update_awareness(awareness_ctx)

        # 8. Archive to persistent memory (background, non-blocking)
        if self.enable_persistence:
            await self._archive_to_memory(user_msg, thread)
            await self._archive_to_memory(assistant_msg, thread)

        # 9. Update active thread
        self.active_thread_id = thread.id
        self.total_messages += 2

        return {
            'thread_id': thread.id,
            'user_message': user_msg.to_dict(),
            'assistant_message': assistant_msg.to_dict(),
            'thread': thread.to_dict(),
            'awareness_context': self._serialize_awareness(awareness_ctx),
        }

    async def _detect_or_create_thread(
        self,
        query: str,
        awareness_ctx: Any
    ) -> ConversationThread:
        """
        Auto-detect which thread this message belongs to.

        Uses semantic similarity in 228D awareness space.
        """

        # If no threads exist, create first one
        if not self.threads:
            return self._create_new_thread(query, awareness_ctx)

        # Get query embedding from awareness
        if awareness_ctx and hasattr(awareness_ctx, 'semantic_position'):
            query_embedding = awareness_ctx.semantic_position
        else:
            # Fallback: create new thread
            return self._create_new_thread(query, awareness_ctx)

        # Find most similar thread
        best_thread = None
        best_similarity = 0.0

        for thread_id, thread_embedding in self.semantic_index.items():
            similarity = self._cosine_similarity(query_embedding, thread_embedding)

            # Only consider active threads
            if self.threads[thread_id].status != ThreadStatus.ACTIVE:
                continue

            if similarity > best_similarity:
                best_similarity = similarity
                best_thread = self.threads[thread_id]

        # Threshold for continuing vs new thread
        if best_similarity > 0.7:
            # High similarity - continue thread
            return best_thread
        elif best_similarity > 0.4:
            # Medium similarity - continue for now (could ask user later)
            return best_thread
        else:
            # Low similarity - new thread
            return self._create_new_thread(query, awareness_ctx)

    def _create_new_thread(
        self,
        first_message: str,
        awareness_ctx: Any
    ) -> ConversationThread:
        """Create a new conversation thread"""

        thread_id = str(uuid.uuid4())

        # Create root message placeholder
        root_msg = Message(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            role="system",
            content=f"Thread started: {first_message[:50]}...",
            timestamp=datetime.now(),
            depth=0
        )

        # Determine topic from awareness
        topic = "General"
        if awareness_ctx and hasattr(awareness_ctx, 'patterns'):
            topic = f"{awareness_ctx.patterns.domain}"

        # Create thread
        thread = ConversationThread(
            id=thread_id,
            root_message=root_msg,
            dominant_topic=topic,
        )

        # Store semantic embedding for thread detection
        if awareness_ctx and hasattr(awareness_ctx, 'semantic_position'):
            self.semantic_index[thread_id] = awareness_ctx.semantic_position
            thread.semantic_cluster = awareness_ctx.semantic_position

        self.threads[thread_id] = thread
        self.thread_count += 1

        return thread

    async def _generate_response(
        self,
        user_input: str,
        awareness_ctx: Any,
        context_messages: List[Message]
    ) -> str:
        """Generate response with thread context"""

        # If no LLM, return awareness summary
        if not self.llm:
            if awareness_ctx:
                return self._generate_awareness_summary(awareness_ctx)
            else:
                return f"Received: {user_input}"

        # Build context from thread history for the query
        context_str = "\n".join([
            f"{m.role}: {m.content}"
            for m in context_messages
        ])

        # Enhance user input with thread context
        if context_messages:
            enhanced_query = f"[Thread context: {len(context_messages)} previous messages]\n{user_input}"
        else:
            enhanced_query = user_input

        # Generate with DualStreamGenerator
        try:
            # self.llm is actually a DualStreamGenerator
            dual_stream = await self.llm.generate(
                query=enhanced_query,
                show_internal=False,
                use_llm=True
            )
            return dual_stream.external_stream
        except Exception as e:
            # Fallback to awareness summary if LLM fails
            if awareness_ctx:
                return self._generate_awareness_summary(awareness_ctx)
            return f"Error generating response: {str(e)[:100]}"

    def _generate_awareness_summary(self, awareness_ctx: Any) -> str:
        """Generate summary from awareness context (fallback if no LLM)"""
        lines = []

        if hasattr(awareness_ctx, 'patterns'):
            lines.append(f"Domain: {awareness_ctx.patterns.domain}")

        if hasattr(awareness_ctx, 'confidence'):
            conf = 1.0 - awareness_ctx.confidence.uncertainty_level
            lines.append(f"Confidence: {conf:.2f}")
            lines.append(f"Cache: {awareness_ctx.confidence.query_cache_status}")

        return "Awareness: " + ", ".join(lines) if lines else "Processing..."

    def _serialize_awareness(self, awareness_ctx: Any) -> Dict[str, Any]:
        """Convert awareness context to JSON-serializable dict"""
        if not awareness_ctx:
            return {}

        result = {}

        try:
            if hasattr(awareness_ctx, 'structural'):
                result['structural'] = {
                    'phrase_type': awareness_ctx.structural.phrase_type,
                    'is_question': awareness_ctx.structural.is_question,
                    'question_type': awareness_ctx.structural.question_type,
                }

            if hasattr(awareness_ctx, 'patterns'):
                result['patterns'] = {
                    'domain': awareness_ctx.patterns.domain,
                    'subdomain': awareness_ctx.patterns.subdomain,
                    'seen_count': awareness_ctx.patterns.seen_count,
                    'confidence': awareness_ctx.patterns.confidence,
                }

            if hasattr(awareness_ctx, 'confidence'):
                result['confidence'] = {
                    'query_cache_status': awareness_ctx.confidence.query_cache_status,
                    'uncertainty_level': awareness_ctx.confidence.uncertainty_level,
                    'knowledge_gap_detected': awareness_ctx.confidence.knowledge_gap_detected,
                }
        except Exception as e:
            result['error'] = str(e)

        return result

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a < 1e-10 or norm_b < 1e-10:
                return 0.0

            return float(dot_product / (norm_a * norm_b))
        except:
            return 0.0

    def get_all_threads(self) -> List[ConversationThread]:
        """Get all threads"""
        return list(self.threads.values())

    def get_active_threads(self) -> List[ConversationThread]:
        """Get only active threads"""
        return [t for t in self.threads.values() if t.status == ThreadStatus.ACTIVE]

    # ============================================================================
    # MEMORY PERSISTENCE
    # ============================================================================

    async def _archive_to_memory(self, message: Message, thread: ConversationThread):
        """
        Archive message to persistent memory (non-blocking).

        Creates a MemoryShard for semantic retrieval and stores thread metadata.

        Args:
            message: Message to archive
            thread: Thread containing the message
        """
        if not self.enable_persistence:
            return

        # Create background task for archiving (don't block chat response)
        task = asyncio.create_task(self._do_archive(message, thread))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _do_archive(self, message: Message, thread: ConversationThread):
        """Actually perform the archiving (runs in background)"""
        try:
            # Import here to avoid circular dependency
            from HoloLoom.documentation.types import MemoryShard

            # Create memory shard from message
            shard = MemoryShard(
                content=message.content,
                source=f"chat_{message.role}",
                timestamp=message.timestamp.isoformat(),
                embedding=None,  # Will be computed by memory backend
                metadata={
                    'message_id': message.id,
                    'thread_id': message.thread_id,
                    'role': message.role,
                    'depth': message.depth,
                    'thread_topic': thread.dominant_topic,
                    'thread_status': thread.status.value,
                }
            )

            # Store in memory backend
            # Note: This assumes memory backend has a store/add method
            if hasattr(self.memory, 'add_memory'):
                await self.memory.add_memory(shard)
            elif hasattr(self.memory, 'store'):
                await self.memory.store(shard)

            # Also store thread metadata as entity (first message only)
            if message.depth == 0 and hasattr(self.memory, 'add_entity'):
                await self.memory.add_entity(
                    name=f"thread_{thread.id[:8]}",
                    type="conversation_thread",
                    properties={
                        'thread_id': thread.id,
                        'topic': thread.dominant_topic,
                        'created_at': thread.created_at.isoformat(),
                        'message_count': len(thread.messages),
                    }
                )

        except Exception as e:
            # Don't crash if memory storage fails - chat should keep working
            print(f"Warning: Failed to archive message to memory: {e}")

    async def retrieve_past_threads(self, query: str = None, limit: int = 10) -> List[ConversationThread]:
        """
        Retrieve past conversation threads from persistent memory.

        Args:
            query: Optional semantic query to search for
            limit: Maximum number of threads to retrieve

        Returns:
            List of past conversation threads
        """
        if not self.enable_persistence:
            return []

        try:
            # Search memory for past chat messages
            if query and hasattr(self.memory, 'search'):
                results = await self.memory.search(query, k=limit * 3)  # Get extra for grouping
            elif hasattr(self.memory, 'get_entities'):
                # Get recent thread entities
                results = await self.memory.get_entities(type="conversation_thread", limit=limit)
            else:
                return []

            # Group results by thread_id and reconstruct threads
            thread_dict = {}
            for result in results:
                thread_id = result.get('metadata', {}).get('thread_id')
                if thread_id and thread_id not in thread_dict:
                    # Create placeholder thread (messages would need full reconstruction)
                    thread_dict[thread_id] = {
                        'id': thread_id,
                        'topic': result.get('metadata', {}).get('thread_topic', 'Unknown'),
                        'message_count': 1,
                    }

            return list(thread_dict.values())[:limit]

        except Exception as e:
            print(f"Warning: Failed to retrieve past threads: {e}")
            return []

    async def cleanup(self):
        """Cleanup background tasks and resources"""
        # Wait for all background archiving tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
