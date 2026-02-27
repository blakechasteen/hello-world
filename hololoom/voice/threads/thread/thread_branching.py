"""
Thread Branching

Enables forking conversations to explore ideas without losing context.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class BranchContext:
    """Context extracted from parent thread for inheritance"""
    messages: List[Dict[str, Any]]
    entities: List[str]
    timestamp: float
    lookback_seconds: float

    def __post_init__(self):
        """Validate context"""
        if not isinstance(self.messages, list):
            raise ValueError("messages must be a list")
        if not isinstance(self.entities, list):
            raise ValueError("entities must be a list")
        if self.lookback_seconds <= 0:
            raise ValueError("lookback_seconds must be positive")


@dataclass
class ThreadBranch:
    """Metadata about a thread branch"""
    branch_id: str
    branch_name: str
    parent_thread_id: str
    parent_thread_name: str
    created_at: float
    context: BranchContext
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'branch_id': self.branch_id,
            'branch_name': self.branch_name,
            'parent_thread_id': self.parent_thread_id,
            'parent_thread_name': self.parent_thread_name,
            'created_at': self.created_at,
            'context': {
                'messages': self.context.messages,
                'entities': self.context.entities,
                'timestamp': self.context.timestamp,
                'lookback_seconds': self.context.lookback_seconds
            },
            'metadata': self.metadata
        }


class ThreadBrancher:
    """
    Handles thread branching (forking) with context inheritance.

    Features:
    - Extract recent context from parent thread
    - Preserve entities and key information
    - Create BRANCHED_FROM edges in YarnGraph
    - Support custom context override
    """

    def __init__(
        self,
        thread_manager: Any,  # elle.ThreadManager
        yarn_graph: Optional[Any] = None,  # HoloLoom.memory.graph.KG
        default_lookback_seconds: float = 30.0,
        enable_entity_extraction: bool = True
    ):
        """
        Initialize thread brancher.

        Args:
            thread_manager: Elle ThreadManager instance
            yarn_graph: Optional YarnGraph for relationship tracking
            default_lookback_seconds: Default context window (30s)
            enable_entity_extraction: Extract entities from context
        """
        self.thread_manager = thread_manager
        self.yarn_graph = yarn_graph
        self.default_lookback_seconds = default_lookback_seconds
        self.enable_entity_extraction = enable_entity_extraction

        logger.info(
            f"ThreadBrancher initialized: "
            f"lookback={default_lookback_seconds}s, "
            f"entity_extraction={enable_entity_extraction}"
        )

    async def fork_thread(
        self,
        parent_thread_id: str,
        branch_name: str,
        custom_context: Optional[BranchContext] = None,
        lookback_seconds: Optional[float] = None
    ) -> ThreadBranch:
        """
        Fork a thread into new branch with context inheritance.

        Algorithm:
        1. Validate parent thread exists
        2. Extract recent context (last N seconds) or use custom
        3. Extract entities from context
        4. Create new thread with inherited data
        5. Add BRANCHED_FROM edge to YarnGraph
        6. Return branch metadata

        Args:
            parent_thread_id: ID of parent thread to fork from
            branch_name: Name for the new branch
            custom_context: Optional custom context (overrides extraction)
            lookback_seconds: Context window (default: 30s)

        Returns:
            ThreadBranch metadata

        Raises:
            ValueError: If parent thread not found
        """
        # Step 1: Validate parent thread
        parent = self.thread_manager.get_thread(parent_thread_id)
        if parent is None:
            raise ValueError(f"Parent thread not found: {parent_thread_id}")

        logger.info(
            f"Forking thread '{parent.name}' ({parent_thread_id}) "
            f"into '{branch_name}'"
        )

        # Step 2: Extract or use custom context
        if custom_context is not None:
            context = custom_context
            logger.debug("Using custom context")
        else:
            lookback = lookback_seconds or self.default_lookback_seconds
            context = await self._extract_context(parent, lookback)
            logger.debug(
                f"Extracted context: {len(context.messages)} messages, "
                f"{len(context.entities)} entities"
            )

        # Step 3: Entity extraction (if enabled and not in custom context)
        if self.enable_entity_extraction and custom_context is None:
            # Entities already extracted in _extract_context
            pass

        # Step 4: Create new thread with inherited data
        branch_id = await self._create_branch_thread(
            parent=parent,
            branch_name=branch_name,
            context=context
        )

        # Step 5: Add BRANCHED_FROM edge to YarnGraph
        if self.yarn_graph is not None:
            await self._add_branch_edge(
                parent_id=parent_thread_id,
                branch_id=branch_id,
                branch_name=branch_name
            )

        # Step 6: Return branch metadata
        branch = ThreadBranch(
            branch_id=branch_id,
            branch_name=branch_name,
            parent_thread_id=parent_thread_id,
            parent_thread_name=parent.name,
            created_at=datetime.now().timestamp(),
            context=context,
            metadata={
                'lookback_seconds': lookback_seconds or self.default_lookback_seconds,
                'message_count': len(context.messages),
                'entity_count': len(context.entities)
            }
        )

        logger.info(
            f"Created branch '{branch_name}' ({branch_id}) "
            f"from '{parent.name}' ({parent_thread_id})"
        )

        return branch

    async def _extract_context(
        self,
        parent: Any,  # Thread
        lookback_seconds: float
    ) -> BranchContext:
        """
        Extract recent context from parent thread.

        Args:
            parent: Parent Thread object
            lookback_seconds: Time window for context

        Returns:
            BranchContext with messages and entities
        """
        now = datetime.now().timestamp()
        cutoff = now - lookback_seconds

        # Extract recent messages
        recent_messages = [
            {
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp,
                'metadata': msg.metadata
            }
            for msg in parent.messages
            if msg.timestamp >= cutoff
        ]

        # Extract entities from recent messages
        entities = []
        if self.enable_entity_extraction:
            entities = await self._extract_entities(recent_messages)

        context = BranchContext(
            messages=recent_messages,
            entities=entities,
            timestamp=now,
            lookback_seconds=lookback_seconds
        )

        return context

    async def _extract_entities(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract entities from messages.

        Simple implementation using keyword extraction.
        Can be enhanced with NER (spaCy) or LLM extraction.

        Args:
            messages: List of message dictionaries

        Returns:
            List of entity strings
        """
        entities = set()

        # Simple extraction: capitalized words (basic NER)
        import re
        for msg in messages:
            content = msg.get('content', '')
            # Find capitalized words (basic entity detection)
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            entities.update(words)

        # Filter out common words
        stopwords = {'I', 'A', 'The', 'This', 'That', 'It'}
        entities = entities - stopwords

        return sorted(list(entities))

    async def _create_branch_thread(
        self,
        parent: Any,  # Thread
        branch_name: str,
        context: BranchContext
    ) -> str:
        """
        Create new thread with inherited context.

        Args:
            parent: Parent Thread object
            branch_name: Name for new thread
            context: Context to inherit

        Returns:
            ID of created thread
        """
        # Create thread via thread manager
        thread_id = self.thread_manager.create_thread(branch_name)
        thread = self.thread_manager.get_thread(thread_id)

        # Add context messages to new thread
        for msg_dict in context.messages:
            # Reconstruct Message object
            from dataclasses import dataclass

            @dataclass
            class Message:
                role: str
                content: str
                timestamp: float
                metadata: Dict[str, Any]

            msg = Message(
                role=msg_dict['role'],
                content=msg_dict['content'],
                timestamp=msg_dict['timestamp'],
                metadata=msg_dict.get('metadata', {})
            )
            thread.messages.append(msg)

        # Add metadata about branching
        thread.metadata = thread.metadata or {}
        thread.metadata['branched_from'] = parent.thread_id
        thread.metadata['branch_timestamp'] = context.timestamp
        thread.metadata['inherited_entities'] = context.entities

        return thread_id

    async def _add_branch_edge(
        self,
        parent_id: str,
        branch_id: str,
        branch_name: str
    ) -> None:
        """
        Add BRANCHED_FROM edge to YarnGraph.

        Args:
            parent_id: Parent thread ID
            branch_id: Branch thread ID
            branch_name: Branch name for edge label
        """
        if self.yarn_graph is None:
            return

        try:
            # Add edge: branch -> parent (BRANCHED_FROM)
            from HoloLoom.memory.graph import KGEdge

            edge = KGEdge(
                src=branch_id,
                dst=parent_id,
                type="BRANCHED_FROM",
                weight=1.0,
                metadata={
                    'branch_name': branch_name,
                    'timestamp': datetime.now().timestamp()
                }
            )

            self.yarn_graph.add_edge(edge)

            logger.debug(
                f"Added BRANCHED_FROM edge: {branch_id} -> {parent_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to add YarnGraph edge: {e}")

    def get_branch_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Get branch history for a thread.

        Args:
            thread_id: Thread ID to query

        Returns:
            List of branch events (parent, children, timestamps)
        """
        if self.yarn_graph is None:
            return []

        history = []

        # Find parent branches (BRANCHED_FROM edges)
        # This would use YarnGraph.get_edges() method
        # Simplified for now

        return history
