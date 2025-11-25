"""
Thread Merging

Enables combining multiple conversation threads with LLM-powered synthesis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    """Strategies for merging threads"""
    APPEND = auto()         # Simple concatenation (chronological)
    SYNTHESIZE = auto()     # LLM-generated synthesis of insights
    PRESERVE_ALL = auto()   # Keep everything with clear markers


@dataclass
class MergeResult:
    """Result of a thread merge operation"""
    target_thread_id: str
    source_thread_ids: List[str]
    strategy: MergeStrategy
    messages_merged: int
    synthesis: Optional[str] = None  # LLM synthesis (if SYNTHESIZE strategy)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'target_thread_id': self.target_thread_id,
            'source_thread_ids': self.source_thread_ids,
            'strategy': self.strategy.name,
            'messages_merged': self.messages_merged,
            'synthesis': self.synthesis,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


class ThreadMerger:
    """
    Handles thread merging with multiple strategies.

    Features:
    - APPEND: Simple chronological concatenation
    - SYNTHESIZE: LLM-generated synthesis of insights
    - PRESERVE_ALL: Keep all messages with thread markers
    - YarnGraph MERGED_INTO edges
    """

    def __init__(
        self,
        thread_manager: Any,  # elle.ThreadManager
        yarn_graph: Optional[Any] = None,  # HoloLoom.memory.graph.KG
        llm_client: Optional[Any] = None,  # LLM client for SYNTHESIZE
        default_strategy: MergeStrategy = MergeStrategy.APPEND
    ):
        """
        Initialize thread merger.

        Args:
            thread_manager: Elle ThreadManager instance
            yarn_graph: Optional YarnGraph for relationship tracking
            llm_client: Optional LLM client for synthesis
            default_strategy: Default merge strategy
        """
        self.thread_manager = thread_manager
        self.yarn_graph = yarn_graph
        self.llm_client = llm_client
        self.default_strategy = default_strategy

        logger.info(
            f"ThreadMerger initialized: "
            f"default_strategy={default_strategy.name}, "
            f"llm_available={llm_client is not None}"
        )

    async def merge_threads(
        self,
        target_thread_id: str,
        source_thread_ids: List[str],
        strategy: Optional[MergeStrategy] = None,
        custom_synthesis_prompt: Optional[str] = None
    ) -> MergeResult:
        """
        Merge source threads into target thread.

        Algorithm:
        1. Validate all threads exist
        2. Extract messages from source threads
        3. Apply merge strategy (APPEND/SYNTHESIZE/PRESERVE_ALL)
        4. Update target thread
        5. Add MERGED_INTO edges to YarnGraph
        6. Return merge result

        Args:
            target_thread_id: ID of target thread (receives merged content)
            source_thread_ids: IDs of source threads (to be merged)
            strategy: Merge strategy (default: self.default_strategy)
            custom_synthesis_prompt: Optional custom LLM prompt

        Returns:
            MergeResult with merge metadata

        Raises:
            ValueError: If any thread not found or target is in sources
        """
        strategy = strategy or self.default_strategy

        logger.info(
            f"Merging {len(source_thread_ids)} threads into '{target_thread_id}' "
            f"using {strategy.name} strategy"
        )

        # Step 1: Validate threads
        target = self.thread_manager.get_thread(target_thread_id)
        if target is None:
            raise ValueError(f"Target thread not found: {target_thread_id}")

        if target_thread_id in source_thread_ids:
            raise ValueError("Target thread cannot be in source threads")

        sources = []
        for source_id in source_thread_ids:
            source = self.thread_manager.get_thread(source_id)
            if source is None:
                raise ValueError(f"Source thread not found: {source_id}")
            sources.append(source)

        # Step 2: Extract messages
        all_messages = self._collect_messages(target, sources)

        # Step 3: Apply merge strategy
        synthesis = None
        messages_merged = 0

        if strategy == MergeStrategy.APPEND:
            messages_merged = await self._merge_append(target, sources, all_messages)

        elif strategy == MergeStrategy.SYNTHESIZE:
            synthesis, messages_merged = await self._merge_synthesize(
                target, sources, all_messages, custom_synthesis_prompt
            )

        elif strategy == MergeStrategy.PRESERVE_ALL:
            messages_merged = await self._merge_preserve_all(target, sources, all_messages)

        # Step 4: Add MERGED_INTO edges to YarnGraph
        if self.yarn_graph is not None:
            await self._add_merge_edges(target_thread_id, source_thread_ids)

        # Step 5: Return result
        result = MergeResult(
            target_thread_id=target_thread_id,
            source_thread_ids=source_thread_ids,
            strategy=strategy,
            messages_merged=messages_merged,
            synthesis=synthesis,
            metadata={
                'target_name': target.name,
                'source_names': [s.name for s in sources],
                'total_messages': len(all_messages)
            }
        )

        logger.info(
            f"Merge complete: {messages_merged} messages merged "
            f"({strategy.name} strategy)"
        )

        return result

    def _collect_messages(
        self,
        target: Any,
        sources: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Collect all messages from target and source threads.

        Args:
            target: Target thread
            sources: List of source threads

        Returns:
            List of message dicts with thread metadata
        """
        all_messages = []

        # Add target messages
        for msg in target.messages:
            all_messages.append({
                'thread_id': target.thread_id,
                'thread_name': target.name,
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp,
                'metadata': msg.metadata
            })

        # Add source messages
        for source in sources:
            for msg in source.messages:
                all_messages.append({
                    'thread_id': source.thread_id,
                    'thread_name': source.name,
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp,
                    'metadata': msg.metadata
                })

        # Sort by timestamp (chronological)
        all_messages.sort(key=lambda m: m['timestamp'])

        return all_messages

    async def _merge_append(
        self,
        target: Any,
        sources: List[Any],
        messages: List[Dict[str, Any]]
    ) -> int:
        """
        APPEND strategy: Simple chronological concatenation.

        Args:
            target: Target thread
            sources: Source threads
            messages: All messages (already sorted)

        Returns:
            Number of messages added to target
        """
        from dataclasses import dataclass

        @dataclass
        class Message:
            role: str
            content: str
            timestamp: float
            metadata: Dict[str, Any]

        # Clear target messages and rebuild chronologically
        original_count = len(target.messages)
        target.messages.clear()

        for msg_dict in messages:
            msg = Message(
                role=msg_dict['role'],
                content=msg_dict['content'],
                timestamp=msg_dict['timestamp'],
                metadata={
                    **msg_dict.get('metadata', {}),
                    'merged_from': msg_dict['thread_name']
                }
            )
            target.messages.append(msg)

        messages_added = len(target.messages) - original_count
        return messages_added

    async def _merge_synthesize(
        self,
        target: Any,
        sources: List[Any],
        messages: List[Dict[str, Any]],
        custom_prompt: Optional[str] = None
    ) -> tuple[str, int]:
        """
        SYNTHESIZE strategy: LLM-generated synthesis of insights.

        Uses metaprompt enhancement for quality.

        Args:
            target: Target thread
            sources: Source threads
            messages: All messages
            custom_prompt: Optional custom synthesis prompt

        Returns:
            Tuple of (synthesis text, messages added count)
        """
        if self.llm_client is None:
            logger.warning("No LLM client available, falling back to APPEND")
            messages_added = await self._merge_append(target, sources, messages)
            return "No synthesis available (LLM not configured)", messages_added

        # Build synthesis prompt
        prompt = custom_prompt or self._build_synthesis_prompt(target, sources, messages)

        # Enhance with metaprompt (if available)
        try:
            from HoloLoom.prompting import create_metaprompt
            from HoloLoom.config import Config

            config = Config.fast()
            config.llm_provider = "anthropic"  # Or configured provider
            enhanced_prompt = create_metaprompt(prompt, config=config)
            prompt = enhanced_prompt
            logger.debug("Using metaprompt-enhanced synthesis")
        except ImportError:
            logger.debug("Metaprompt not available, using basic prompt")

        # Generate synthesis
        synthesis = await self.llm_client.generate(prompt)

        # Add synthesis as single message to target
        from dataclasses import dataclass

        @dataclass
        class Message:
            role: str
            content: str
            timestamp: float
            metadata: Dict[str, Any]

        synthesis_message = Message(
            role="assistant",
            content=f"**Synthesis of {len(sources)} threads:**\n\n{synthesis}",
            timestamp=datetime.now().timestamp(),
            metadata={
                'is_synthesis': True,
                'merged_threads': [s.name for s in sources],
                'message_count': len(messages)
            }
        )

        target.messages.append(synthesis_message)

        return synthesis, 1  # 1 synthesis message added

    def _build_synthesis_prompt(
        self,
        target: Any,
        sources: List[Any],
        messages: List[Dict[str, Any]]
    ) -> str:
        """
        Build LLM prompt for synthesis.

        Args:
            target: Target thread
            sources: Source threads
            messages: All messages

        Returns:
            LLM prompt string
        """
        thread_names = [target.name] + [s.name for s in sources]

        prompt = f"""Synthesize insights from {len(sources) + 1} conversation threads.

Threads to merge:
{chr(10).join(f'- {name}' for name in thread_names)}

Total messages: {len(messages)}

Task:
1. Identify key themes and insights across all threads
2. Note any connections or contradictions between threads
3. Summarize actionable takeaways
4. Keep synthesis concise (3-5 paragraphs)

Format your response as a clear synthesis that captures the essential insights from all threads.
"""

        # Add sample messages for context (first 3 and last 3)
        if len(messages) > 6:
            sample = messages[:3] + messages[-3:]
        else:
            sample = messages

        prompt += "\n\nSample messages:\n"
        for msg in sample:
            prompt += f"\n[{msg['thread_name']}] {msg['role']}: {msg['content'][:100]}..."

        return prompt

    async def _merge_preserve_all(
        self,
        target: Any,
        sources: List[Any],
        messages: List[Dict[str, Any]]
    ) -> int:
        """
        PRESERVE_ALL strategy: Keep all messages with thread markers.

        Args:
            target: Target thread
            sources: Source threads
            messages: All messages (already sorted)

        Returns:
            Number of messages added to target
        """
        from dataclasses import dataclass

        @dataclass
        class Message:
            role: str
            content: str
            timestamp: float
            metadata: Dict[str, Any]

        original_count = len(target.messages)
        target.messages.clear()

        for msg_dict in messages:
            # Add thread marker to content
            thread_marker = f"[{msg_dict['thread_name']}]"
            content = f"{thread_marker} {msg_dict['content']}"

            msg = Message(
                role=msg_dict['role'],
                content=content,
                timestamp=msg_dict['timestamp'],
                metadata={
                    **msg_dict.get('metadata', {}),
                    'merged_from': msg_dict['thread_name'],
                    'original_thread_id': msg_dict['thread_id']
                }
            )
            target.messages.append(msg)

        messages_added = len(target.messages) - original_count
        return messages_added

    async def _add_merge_edges(
        self,
        target_id: str,
        source_ids: List[str]
    ) -> None:
        """
        Add MERGED_INTO edges to YarnGraph.

        Args:
            target_id: Target thread ID
            source_ids: Source thread IDs
        """
        if self.yarn_graph is None:
            return

        try:
            from HoloLoom.memory.graph import KGEdge

            for source_id in source_ids:
                edge = KGEdge(
                    src=source_id,
                    dst=target_id,
                    type="MERGED_INTO",
                    weight=1.0,
                    metadata={
                        'timestamp': datetime.now().timestamp()
                    }
                )

                self.yarn_graph.add_edge(edge)

                logger.debug(f"Added MERGED_INTO edge: {source_id} -> {target_id}")

        except Exception as e:
            logger.warning(f"Failed to add YarnGraph edges: {e}")

    def get_merge_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Get merge history for a thread.

        Args:
            thread_id: Thread ID to query

        Returns:
            List of merge events (sources, targets, timestamps)
        """
        if self.yarn_graph is None:
            return []

        history = []

        # Find MERGED_INTO edges
        # This would use YarnGraph.get_edges() method
        # Simplified for now

        return history
