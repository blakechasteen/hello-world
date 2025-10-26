"""
Simple Mem0 Integration Pattern - Practical Approach
=====================================================

This shows a BETTER way to integrate mem0 into your workflows
without the complexity of the full HoloLoom hybrid system.

Key insight: Use mem0 as a lightweight "user preference layer"
that sits ALONGSIDE your existing code, not integrated into it.

Usage:
    from mem0_simple_integration import UserMemory

    # Initialize once
    memory = UserMemory()

    # Store interactions
    memory.remember("blake", "I prefer organic bee treatments")
    memory.remember("blake", "My hives are Jodi, Aurora, and Luna")

    # Retrieve when needed
    context = memory.recall("blake", "What treatments does Blake use?")
    print(context)  # Relevant memories as text
"""

from mem0 import Memory
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserMemory:
    """
    Simple wrapper around mem0 for user-specific memory.

    This is a BETTER approach than the complex hybrid system because:
    1. Works standalone - no HoloLoom dependencies
    2. Simple API - just remember() and recall()
    3. Automatic extraction - mem0 decides what's important
    4. User-specific - tracks per-user preferences
    5. Fallback gracefully - works even if mem0 fails
    """

    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize user memory.

        Args:
            provider: "openai" (default, needs API key) or "ollama" (local)
            api_key: OpenAI API key (only for provider="openai")
        """
        self.provider = provider
        self.enabled = False

        try:
            if provider == "openai":
                # Use OpenAI (simple, reliable, but needs API key)
                import os
                key = api_key or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise ValueError("OpenAI API key required")

                self.memory = Memory()  # Uses OpenAI by default
                self.enabled = True
                logger.info("UserMemory initialized with OpenAI")

            elif provider == "ollama":
                # Use Ollama (local, free, but may have compatibility issues)
                config = {
                    "llm": {
                        "provider": "ollama",
                        "config": {"model": "llama3.2:3b"}
                    },
                    "embedder": {
                        "provider": "ollama",
                        "config": {"model": "nomic-embed-text"}
                    },
                    "version": "v1.1"
                }
                self.memory = Memory.from_config(config)
                self.enabled = True
                logger.info("UserMemory initialized with Ollama")

            else:
                raise ValueError(f"Unknown provider: {provider}")

        except Exception as e:
            logger.warning(f"Could not initialize mem0: {e}")
            logger.warning("UserMemory will operate in fallback mode (no persistence)")
            self.memory = None
            self.enabled = False
            # Fallback: in-memory dict
            self._fallback_store: Dict[str, List[str]] = {}

    def remember(self, user_id: str, text: str, context: Optional[str] = None) -> bool:
        """
        Store a memory for a user.

        Args:
            user_id: User identifier
            text: Text to remember (user statement, preference, fact)
            context: Optional context (e.g., "conversation", "task")

        Returns:
            True if stored successfully

        Example:
            memory.remember("blake", "I prefer organic treatments for bees")
        """
        if not self.enabled or not self.memory:
            # Fallback mode
            if user_id not in self._fallback_store:
                self._fallback_store[user_id] = []
            self._fallback_store[user_id].append(text)
            logger.debug(f"Stored in fallback: {text[:50]}...")
            return True

        try:
            # Format as a simple message
            messages = [{"role": "user", "content": text}]
            result = self.memory.add(messages, user_id=user_id)

            extracted = len(result.get('results', []))
            logger.info(f"Stored memory for {user_id}: {extracted} facts extracted")
            return True

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            # Fallback
            if user_id not in self._fallback_store:
                self._fallback_store[user_id] = []
            self._fallback_store[user_id].append(text)
            return False

    def recall(self, user_id: str, query: str, limit: int = 5) -> str:
        """
        Retrieve relevant memories for a user.

        Args:
            user_id: User identifier
            query: Search query
            limit: Max number of memories to retrieve

        Returns:
            Formatted string of relevant memories

        Example:
            context = memory.recall("blake", "What are my hives?")
            print(context)  # "Your hives are Jodi, Aurora, and Luna"
        """
        if not self.enabled or not self.memory:
            # Fallback mode
            if user_id in self._fallback_store:
                mems = self._fallback_store[user_id][-limit:]
                return "\n".join(f"- {m}" for m in mems)
            return ""

        try:
            results = self.memory.search(query, user_id=user_id, limit=limit)
            memories = results.get('results', [])

            if not memories:
                return ""

            # Format as bullet list
            formatted = "\n".join(
                f"- {mem.get('memory', '')}"
                for mem in memories
            )
            return formatted

        except Exception as e:
            logger.error(f"Failed to recall: {e}")
            # Fallback
            if user_id in self._fallback_store:
                mems = self._fallback_store[user_id][-limit:]
                return "\n".join(f"- {m}" for m in mems)
            return ""

    def get_all(self, user_id: str) -> List[str]:
        """Get all memories for a user as a list."""
        if not self.enabled or not self.memory:
            return self._fallback_store.get(user_id, [])

        try:
            result = self.memory.get_all(user_id=user_id)
            return [
                mem.get('memory', '')
                for mem in result.get('results', [])
            ]
        except Exception as e:
            logger.error(f"Failed to get all: {e}")
            return self._fallback_store.get(user_id, [])


# ==============================================================================
# Example Usage
# ==============================================================================

def example_basic():
    """Basic usage example."""
    print("="*80)
    print("Example 1: Basic Usage")
    print("="*80 + "\n")

    # Initialize (will use fallback if no API key)
    memory = UserMemory(provider="openai")

    # Store some memories
    print("Storing memories...")
    memory.remember("blake", "I'm a beekeeper with three hives")
    memory.remember("blake", "My hives are named Jodi, Aurora, and Luna")
    memory.remember("blake", "I prefer organic treatments for varroa mites")
    print()

    # Retrieve relevant memories
    print("Recalling memories...")
    context = memory.recall("blake", "What are Blake's hives?")
    print(f"Query: What are Blake's hives?")
    print(f"Result:\n{context}")
    print()


def example_workflow_integration():
    """Example: Integrating into a workflow."""
    print("="*80)
    print("Example 2: Workflow Integration")
    print("="*80 + "\n")

    memory = UserMemory(provider="openai")

    # Simulate an agent workflow
    def process_user_query(user_id: str, query: str) -> str:
        """Simple agent that uses user memory."""
        # 1. Get user context
        user_context = memory.recall(user_id, query, limit=3)

        # 2. Use context in response
        if user_context:
            response = f"Based on what I know about you:\n{user_context}\n\n"
        else:
            response = ""

        # 3. Generate response (simplified)
        response += f"Processing query: {query}"

        # 4. Remember this interaction
        memory.remember(user_id, f"Asked about: {query}")

        return response

    # Test it
    query = "How should I prepare for winter?"
    response = process_user_query("blake", query)
    print(f"Query: {query}")
    print(f"Response:\n{response}")
    print()


def example_better_pattern():
    """
    Example: The BETTER integration pattern.

    Instead of complex hybrid systems, use mem0 as a simple
    "user preference layer" that enriches your existing code.
    """
    print("="*80)
    print("Example 3: Better Integration Pattern")
    print("="*80 + "\n")

    memory = UserMemory(provider="openai")

    # Your existing agent/workflow
    class SimpleAgent:
        def __init__(self, user_memory: UserMemory):
            self.memory = user_memory

        def answer(self, user_id: str, question: str) -> str:
            # Enrich with user-specific context
            user_prefs = self.memory.recall(user_id, question)

            # Your existing logic here
            answer = f"[Your existing answer logic]\n"

            if user_prefs:
                answer += f"\nPersonalized context:\n{user_prefs}"

            return answer

    agent = SimpleAgent(memory)

    # Use it
    response = agent.answer("blake", "What treatments should I use?")
    print(response)
    print()

    print("Key advantages of this pattern:")
    print("  1. Mem0 is optional - agent works without it")
    print("  2. No complex dependencies")
    print("  3. Easy to test and debug")
    print("  4. User-specific personalization")
    print("  5. Automatic fact extraction")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Mem0 Simple Integration Examples")
    print("="*80 + "\n")

    print("NOTE: These examples will use fallback mode unless you have:")
    print("  - OpenAI API key: export OPENAI_API_KEY=sk-...")
    print("  - Or Ollama running: ollama serve")
    print()

    input("Press Enter to continue...")
    print()

    example_basic()
    example_workflow_integration()
    example_better_pattern()

    print("="*80)
    print("Complete!")
    print("="*80)
