"""
Thread Management for Elle Voice Assistant

Supports conversational threading:
- Create threads for different topics
- Switch between active threads
- Summarize thread contents
- Persist thread state

Created: November 20, 2025
Part of: Voice UX Milestone 1
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class Message:
    """Single message in a thread"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = "user"  # "user" or "assistant"
    content: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(**data)


@dataclass
class Thread:
    """
    Conversation thread with messages and metadata

    Each thread represents a focused conversation on a specific topic.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    topic: str = ""
    messages: List[Message] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the thread"""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()

    def get_last_n_messages(self, n: int = 10) -> List[Message]:
        """Get last N messages"""
        return self.messages[-n:] if len(self.messages) > n else self.messages

    def get_summary(self, max_length: int = 500) -> str:
        """
        Generate a summary of the thread

        For now, returns last few messages concatenated.
        Future: Use LLM to generate proper summary.
        """
        if not self.messages:
            return f"Thread '{self.name}' - no messages yet."

        # Get last 5 messages
        recent = self.get_last_n_messages(5)

        summary_parts = [f"Thread: {self.name} ({self.topic})"]
        summary_parts.append(f"Messages: {len(self.messages)} total, showing last {len(recent)}")

        for msg in recent:
            role_label = "You" if msg.role == "user" else "Elle"
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary_parts.append(f"  {role_label}: {content_preview}")

        summary = "\n".join(summary_parts)

        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."

        return summary

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "topic": self.topic,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Thread":
        """Create Thread from dictionary"""
        messages = [Message.from_dict(msg) for msg in data.get("messages", [])]
        return cls(
            id=data["id"],
            name=data["name"],
            topic=data["topic"],
            messages=messages,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata=data.get("metadata", {})
        )


class ThreadManager:
    """
    Manages conversation threads

    Features:
    - Create, switch, list, delete threads
    - Persist threads to disk (JSON)
    - Track active thread
    - Generate thread summaries
    """

    def __init__(self, storage_path: str = "elle/data/threads.json"):
        """
        Initialize ThreadManager

        Args:
            storage_path: Path to JSON file for thread persistence
        """
        self.storage_path = Path(storage_path)
        self.threads: Dict[str, Thread] = {}
        self.active_thread_id: Optional[str] = None

        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing threads
        self.load()

    def create_thread(self, name: str, topic: str = "") -> Thread:
        """
        Create a new thread

        Args:
            name: Human-readable thread name
            topic: Optional topic description

        Returns:
            Created Thread object
        """
        thread = Thread(name=name, topic=topic or name)
        self.threads[thread.id] = thread
        self.active_thread_id = thread.id
        self.save()
        return thread

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get thread by ID"""
        return self.threads.get(thread_id)

    def get_thread_by_name(self, name: str) -> Optional[Thread]:
        """
        Get thread by name (case-insensitive partial match)

        Args:
            name: Thread name (can be partial)

        Returns:
            First matching thread, or None
        """
        name_lower = name.lower()

        # Exact match first
        for thread in self.threads.values():
            if thread.name.lower() == name_lower:
                return thread

        # Partial match
        for thread in self.threads.values():
            if name_lower in thread.name.lower():
                return thread

        return None

    def get_active_thread(self) -> Optional[Thread]:
        """Get currently active thread"""
        if self.active_thread_id:
            return self.threads.get(self.active_thread_id)
        return None

    def switch_thread(self, thread_id_or_name: str) -> Optional[Thread]:
        """
        Switch to a different thread

        Args:
            thread_id_or_name: Thread ID or name

        Returns:
            Switched-to Thread, or None if not found
        """
        # Try by ID first
        thread = self.get_thread(thread_id_or_name)

        # Try by name if ID failed
        if not thread:
            thread = self.get_thread_by_name(thread_id_or_name)

        if thread:
            self.active_thread_id = thread.id
            self.save()
            return thread

        return None

    def list_threads(self) -> List[Thread]:
        """List all threads, sorted by updated_at (most recent first)"""
        threads = list(self.threads.values())
        threads.sort(key=lambda t: t.updated_at, reverse=True)
        return threads

    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread

        Args:
            thread_id: Thread ID to delete

        Returns:
            True if deleted, False if not found
        """
        if thread_id in self.threads:
            del self.threads[thread_id]

            # If active thread was deleted, clear active
            if self.active_thread_id == thread_id:
                self.active_thread_id = None

                # Set active to most recent thread if any exist
                if self.threads:
                    most_recent = self.list_threads()[0]
                    self.active_thread_id = most_recent.id

            self.save()
            return True

        return False

    def add_message_to_active(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """
        Add message to active thread

        Creates a default thread if none active.

        Args:
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata
        """
        # Create default thread if none active
        if not self.active_thread_id or self.active_thread_id not in self.threads:
            self.create_thread("Default", "General conversation")

        thread = self.get_active_thread()
        if thread:
            thread.add_message(role, content, metadata)
            self.save()

    def get_thread_summaries(self) -> List[str]:
        """
        Get summaries of all threads

        Returns:
            List of summary strings
        """
        summaries = []
        for thread in self.list_threads():
            summaries.append(thread.get_summary())
        return summaries

    def save(self):
        """Save threads to disk (JSON)"""
        data = {
            "threads": {tid: thread.to_dict() for tid, thread in self.threads.items()},
            "active_thread_id": self.active_thread_id
        }

        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠ Warning: Failed to save threads: {e}")

    def load(self):
        """Load threads from disk (JSON)"""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load threads
            for tid, thread_data in data.get("threads", {}).items():
                self.threads[tid] = Thread.from_dict(thread_data)

            # Load active thread ID
            self.active_thread_id = data.get("active_thread_id")

            print(f"[OK] Loaded {len(self.threads)} threads from {self.storage_path}")

        except Exception as e:
            print(f"⚠ Warning: Failed to load threads: {e}")
            self.threads = {}
            self.active_thread_id = None


async def demo_thread_management():
    """Demo: Thread management"""
    print("\n" + "="*60)
    print("THREAD MANAGER DEMO")
    print("="*60)

    # Create manager
    manager = ThreadManager(storage_path="elle/data/test_threads.json")

    # Create threads
    print("\n1. Creating threads...")
    thread1 = manager.create_thread("Baking Bread", "Discussing sourdough bread recipes")
    print(f"[OK] Created: {thread1.name} (ID: {thread1.id[:8]}...)")

    thread2 = manager.create_thread("Biochar", "Learning about biochar production")
    print(f"[OK] Created: {thread2.name} (ID: {thread2.id[:8]}...)")

    thread3 = manager.create_thread("Greenhouse", "Planning greenhouse setup")
    print(f"[OK] Created: {thread3.name} (ID: {thread3.id[:8]}...)")

    # Add messages to active thread (Greenhouse)
    print(f"\n2. Adding messages to active thread '{thread3.name}'...")
    manager.add_message_to_active("user", "What's the ideal temperature for tomatoes?")
    manager.add_message_to_active("assistant", "Tomatoes thrive in 70-80°F during the day.")
    manager.add_message_to_active("user", "What about nighttime?")
    print(f"[OK] Added 3 messages to {thread3.name}")

    # Switch to thread 1
    print(f"\n3. Switching to thread 'Baking'...")
    switched = manager.switch_thread("Baking")
    if switched:
        print(f"[OK] Switched to: {switched.name}")

        # Add messages to Baking thread
        manager.add_message_to_active("user", "How long should I proof sourdough?")
        manager.add_message_to_active("assistant", "Bulk fermentation: 4-6 hours. Final proof: 2-4 hours.")
        print(f"[OK] Added 2 messages to {switched.name}")

    # List all threads
    print("\n4. Listing all threads...")
    threads = manager.list_threads()
    for i, thread in enumerate(threads, 1):
        active_marker = "*" if thread.id == manager.active_thread_id else " "
        print(f"{active_marker} {i}. {thread.name} - {len(thread.messages)} messages")

    # Get summaries
    print("\n5. Thread summaries...")
    for summary in manager.get_thread_summaries():
        print(summary)
        print()

    # Switch by partial name
    print("\n6. Switching by partial name 'Bio'...")
    switched = manager.switch_thread("Bio")
    if switched:
        print(f"[OK] Switched to: {switched.name}")

    # Active thread
    print("\n7. Current active thread:")
    active = manager.get_active_thread()
    if active:
        print(f"* {active.name} ({len(active.messages)} messages)")

    # Delete thread
    print(f"\n8. Deleting thread '{thread3.name}'...")
    deleted = manager.delete_thread(thread3.id)
    if deleted:
        print(f"[OK] Deleted {thread3.name}")
        print(f"  Active thread is now: {manager.get_active_thread().name}")

    print("\n" + "="*60)
    print(f"[OK] Demo complete. {len(manager.threads)} threads saved to {manager.storage_path}")


if __name__ == "__main__":
    asyncio.run(demo_thread_management())
