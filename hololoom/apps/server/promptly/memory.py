"""
ConversationMemory — delegated to kit.

This module exists for backward compatibility. All logic lives in kit/memory.py.
"""
from ..kit.memory import ConversationMemory, conversation
from .config import MAX_HISTORY_TURNS, MEMORY_DB_PATH

# Module-level singleton (matches original interface)
conv_memory = ConversationMemory(db_path=MEMORY_DB_PATH, max_turns=MAX_HISTORY_TURNS, label="promptly")

__all__ = ["ConversationMemory", "conversation", "conv_memory"]
