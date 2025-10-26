#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thread-Aware Response Handler
==============================
Handles Matrix conversation threading with context-aware responses.

Features:
- Thread detection and tracking
- Parent message context retrieval
- Thread summarization
- Nested conversation support

Architecture:
    Matrix Thread Event
        ↓
    ThreadHandler
        ├→ Detect thread relationships
        ├→ Build thread context
        └→ Generate contextual response
        ↓
    Store in Knowledge Graph
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# Thread Data Structures
# ============================================================================

@dataclass
class ThreadNode:
    """
    A message in a thread tree.

    Represents a single message with its threading relationships.
    """
    message_id: str
    text: str
    sender: str
    timestamp: datetime
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreadContext:
    """
    Complete context for a threaded conversation.

    Contains the full thread tree with parent-child relationships.
    """
    thread_root_id: str
    conversation_id: str
    nodes: Dict[str, ThreadNode] = field(default_factory=dict)
    max_depth: int = 0

    def add_node(self, node: ThreadNode) -> None:
        """Add a node to the thread."""
        self.nodes[node.message_id] = node

        # Update max depth
        if node.depth > self.max_depth:
            self.max_depth = node.depth

        # Link to parent
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.message_id not in parent.children_ids:
                parent.children_ids.append(node.message_id)

    def get_thread_chain(self, message_id: str) -> List[ThreadNode]:
        """
        Get the chain of messages from root to this message.

        Args:
            message_id: Message to get chain for

        Returns:
            List of ThreadNodes from root to message
        """
        chain = []
        current_id = message_id

        while current_id:
            if current_id not in self.nodes:
                break

            node = self.nodes[current_id]
            chain.insert(0, node)  # Prepend to maintain order

            current_id = node.parent_id

        return chain

    def get_children(self, message_id: str) -> List[ThreadNode]:
        """Get immediate children of a message."""
        if message_id not in self.nodes:
            return []

        node = self.nodes[message_id]
        return [self.nodes[child_id] for child_id in node.children_ids if child_id in self.nodes]

    def to_context_string(self, message_id: str, include_siblings: bool = False) -> str:
        """
        Format thread context as string.

        Args:
            message_id: Message to get context for
            include_siblings: Include sibling messages

        Returns:
            Formatted context string
        """
        chain = self.get_thread_chain(message_id)

        lines = ["**Thread Context:**\n"]
        for node in chain:
            indent = "  " * node.depth
            sender = node.sender.split(":")[0][1:]  # Clean @user:server -> user
            lines.append(f"{indent}→ {sender}: {node.text}")

        return "\n".join(lines)


# ============================================================================
# Thread Handler
# ============================================================================

class ThreadHandler:
    """
    Manages threaded conversations in Matrix rooms.

    Features:
    - Detect thread relationships from Matrix events
    - Build thread context trees
    - Retrieve parent message context
    - Generate thread summaries
    - Support nested conversations

    Usage:
        handler = ThreadHandler()

        # Process new message
        thread_context = handler.process_message(
            message_id="msg_123",
            text="This is a reply",
            sender="@alice:matrix.org",
            parent_id="msg_100",  # Message being replied to
            conversation_id="room_abc"
        )

        # Get context for response generation
        context_string = thread_context.to_context_string("msg_123")
    """

    def __init__(self, max_thread_depth: int = 10):
        """
        Initialize thread handler.

        Args:
            max_thread_depth: Maximum thread nesting depth to track
        """
        self.max_thread_depth = max_thread_depth

        # Active threads: thread_root_id -> ThreadContext
        self.threads: Dict[str, ThreadContext] = {}

        # Message to thread mapping: message_id -> thread_root_id
        self.message_to_thread: Dict[str, str] = {}

        logger.info("ThreadHandler initialized")

    def process_message(
        self,
        message_id: str,
        text: str,
        sender: str,
        conversation_id: str,
        parent_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ) -> ThreadContext:
        """
        Process a message and update thread context.

        Args:
            message_id: Unique message ID
            text: Message text
            sender: Sender user ID
            conversation_id: Room/conversation ID
            parent_id: Optional parent message ID (for threads)
            timestamp: Message timestamp
            metadata: Additional metadata

        Returns:
            ThreadContext for this message
        """
        timestamp = timestamp or datetime.now()
        metadata = metadata or {}

        # Determine thread membership
        if parent_id:
            # This is a reply - find or create thread
            if parent_id in self.message_to_thread:
                # Parent is already in a thread
                thread_root_id = self.message_to_thread[parent_id]
            else:
                # Start new thread with parent as root
                thread_root_id = parent_id

            # Get or create thread context
            if thread_root_id not in self.threads:
                self.threads[thread_root_id] = ThreadContext(
                    thread_root_id=thread_root_id,
                    conversation_id=conversation_id
                )

            thread_context = self.threads[thread_root_id]

            # Calculate depth
            if parent_id in thread_context.nodes:
                parent_depth = thread_context.nodes[parent_id].depth
                depth = min(parent_depth + 1, self.max_thread_depth)
            else:
                depth = 0

        else:
            # No parent - standalone message or potential thread root
            thread_root_id = message_id
            thread_context = ThreadContext(
                thread_root_id=thread_root_id,
                conversation_id=conversation_id
            )
            self.threads[thread_root_id] = thread_context
            depth = 0

        # Create thread node
        node = ThreadNode(
            message_id=message_id,
            text=text,
            sender=sender,
            timestamp=timestamp,
            parent_id=parent_id,
            depth=depth,
            metadata=metadata
        )

        # Add to thread
        thread_context.add_node(node)

        # Update mapping
        self.message_to_thread[message_id] = thread_root_id

        logger.debug(f"Processed message in thread {thread_root_id} at depth {depth}")

        return thread_context

    def get_thread_context(self, message_id: str) -> Optional[ThreadContext]:
        """
        Get thread context for a message.

        Args:
            message_id: Message ID

        Returns:
            ThreadContext if message is in a thread
        """
        if message_id not in self.message_to_thread:
            return None

        thread_root_id = self.message_to_thread[message_id]
        return self.threads.get(thread_root_id)

    def get_parent_chain(self, message_id: str) -> List[str]:
        """
        Get chain of parent message IDs.

        Args:
            message_id: Message ID

        Returns:
            List of message IDs from root to parent
        """
        thread_context = self.get_thread_context(message_id)
        if not thread_context:
            return []

        chain = thread_context.get_thread_chain(message_id)
        return [node.message_id for node in chain[:-1]]  # Exclude the message itself

    def summarize_thread(
        self,
        thread_root_id: str,
        max_messages: int = 20
    ) -> Optional[str]:
        """
        Generate summary of a thread.

        Args:
            thread_root_id: Thread root message ID
            max_messages: Maximum messages to include in summary

        Returns:
            Summary string
        """
        if thread_root_id not in self.threads:
            return None

        thread = self.threads[thread_root_id]

        # Collect all messages in chronological order
        messages = sorted(
            thread.nodes.values(),
            key=lambda n: n.timestamp
        )[:max_messages]

        # Format summary
        lines = [f"**Thread Summary** ({len(messages)} messages, depth {thread.max_depth}):\n"]

        for node in messages:
            indent = "  " * node.depth
            sender = node.sender.split(":")[0][1:]
            timestamp = node.timestamp.strftime("%H:%M")
            text_preview = node.text[:100] + "..." if len(node.text) > 100 else node.text

            lines.append(f"{indent}[{timestamp}] {sender}: {text_preview}")

        return "\n".join(lines)

    def get_statistics(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get threading statistics.

        Args:
            conversation_id: Optional conversation to filter by

        Returns:
            Statistics dict
        """
        if conversation_id:
            threads = [
                t for t in self.threads.values()
                if t.conversation_id == conversation_id
            ]
        else:
            threads = list(self.threads.values())

        total_messages = sum(len(t.nodes) for t in threads)
        avg_depth = sum(t.max_depth for t in threads) / len(threads) if threads else 0

        return {
            "active_threads": len(threads),
            "total_messages": total_messages,
            "average_depth": round(avg_depth, 2),
            "deepest_thread": max((t.max_depth for t in threads), default=0)
        }


# ============================================================================
# Integration with Matrix Events
# ============================================================================

def extract_thread_info_from_event(event: Any) -> Optional[str]:
    """
    Extract thread parent ID from Matrix event.

    Matrix threading uses m.relates_to with rel_type=m.thread

    Args:
        event: Matrix message event

    Returns:
        Parent message ID if this is a threaded reply
    """
    try:
        # Check for thread relation
        content = getattr(event, 'source', {}).get('content', {})
        relates_to = content.get('m.relates_to', {})

        # Thread relationship
        if relates_to.get('rel_type') == 'm.thread':
            return relates_to.get('event_id')

        # Fallback: reply relationship
        if 'm.in_reply_to' in relates_to:
            return relates_to['m.in_reply_to'].get('event_id')

        return None

    except Exception as e:
        logger.debug(f"Error extracting thread info: {e}")
        return None


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("Thread Handler Demo")
    print("="*80)
    print()

    # Create handler
    handler = ThreadHandler(max_thread_depth=5)

    # Simulate a threaded conversation
    print("Simulating threaded conversation...\n")

    # Root message
    ctx1 = handler.process_message(
        message_id="msg_1",
        text="Should we implement threading support?",
        sender="@alice:matrix.org",
        conversation_id="room_test"
    )
    print(f"1. Alice: {ctx1.nodes['msg_1'].text}")

    # Reply to root
    ctx2 = handler.process_message(
        message_id="msg_2",
        text="Yes! Thread-aware responses are important",
        sender="@bob:matrix.org",
        conversation_id="room_test",
        parent_id="msg_1"
    )
    print(f"  2. Bob (reply): {ctx2.nodes['msg_2'].text}")

    # Another reply to root
    ctx3 = handler.process_message(
        message_id="msg_3",
        text="I agree. Let's track parent context",
        sender="@charlie:matrix.org",
        conversation_id="room_test",
        parent_id="msg_1"
    )
    print(f"  3. Charlie (reply): {ctx3.nodes['msg_3'].text}")

    # Nested reply
    ctx4 = handler.process_message(
        message_id="msg_4",
        text="We should also summarize long threads",
        sender="@alice:matrix.org",
        conversation_id="room_test",
        parent_id="msg_2"
    )
    print(f"    4. Alice (nested): {ctx4.nodes['msg_4'].text}")

    print()

    # Get thread context for nested message
    print("Thread context for msg_4:")
    print(ctx4.to_context_string("msg_4"))
    print()

    # Generate thread summary
    print("Thread summary:")
    summary = handler.summarize_thread("msg_1")
    print(summary)
    print()

    # Statistics
    stats = handler.get_statistics("room_test")
    print("Threading statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✓ Demo complete!")
