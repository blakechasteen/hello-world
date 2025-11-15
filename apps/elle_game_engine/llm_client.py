"""
LLM client abstraction for Elle Game Engine.

Provides a clean interface for calling LLM providers without coupling
to specific implementations. Supports:
- DummyClient (for testing)
- OpenAI (future)
- Anthropic (future)
- Local models (future)
"""

from typing import Protocol, Dict, Any, List, Optional
from abc import abstractmethod
import json


class BaseLLMClient(Protocol):
    """
    Protocol defining the LLM client interface.

    All LLM implementations must conform to this interface,
    allowing the policy to remain provider-agnostic.
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a completion from the LLM.

        Args:
            prompt: The user prompt to complete
            system_prompt: Optional system/instruction prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)

        Returns:
            Generated text completion
        """
        ...


class DummyLLMClient:
    """
    Dummy LLM client for testing.

    Returns deterministic, well-formed ElleGameAction responses
    without calling any real LLM API.

    Useful for:
    - Unit testing policy logic
    - Integration testing without API costs
    - Development without API keys
    """

    def __init__(self, *, response_mode: str = "npc_dialogue"):
        """
        Initialize dummy client.

        Args:
            response_mode: Default mode to return ("npc_dialogue", "hint", "world_reaction")
        """
        self.response_mode = response_mode
        self.call_count = 0
        self.last_prompt = None

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a dummy completion.

        Returns a well-formed JSON response based on response_mode.
        """
        self.call_count += 1
        self.last_prompt = prompt

        # Parse intent from prompt (simple keyword detection)
        if "talk_to_npc" in prompt.lower():
            return self._npc_dialogue_response(prompt)
        elif "enter_scene" in prompt.lower():
            return self._world_reaction_response(prompt)
        elif "request_hint" in prompt.lower():
            return self._hint_response(prompt)
        elif "debug_summary" in prompt.lower():
            return self._debug_response(prompt)

        # Default: return based on configured mode
        return self._default_response()

    def _npc_dialogue_response(self, prompt: str) -> str:
        """Generate NPC dialogue response."""
        # Try to extract NPC name from prompt
        npc_id = "npc"
        if "innkeeper" in prompt.lower():
            npc_id = "innkeeper"
        elif "guard" in prompt.lower():
            npc_id = "guard"
        elif "merchant" in prompt.lower():
            npc_id = "merchant"

        response = {
            "mode": "npc_dialogue",
            "priority": "medium",
            "dialogue": [
                {
                    "npc_id": npc_id,
                    "text": "Greetings, traveler. How can I help you?",
                    "tone": "neutral"
                }
            ],
            "hint_text": None,
            "world_reaction": None,
            "debug_notes": "Test NPC dialogue from DummyLLMClient"
        }
        return json.dumps(response, indent=2)

    def _world_reaction_response(self, prompt: str) -> str:
        """Generate world reaction response."""
        response = {
            "mode": "world_reaction",
            "priority": "low",
            "dialogue": [],
            "hint_text": None,
            "world_reaction": {
                "description": "The air is calm and peaceful.",
                "flag_changes": {"scene_entered": True}
            },
            "debug_notes": "Test world reaction from DummyLLMClient"
        }
        return json.dumps(response, indent=2)

    def _hint_response(self, prompt: str) -> str:
        """Generate hint response."""
        response = {
            "mode": "hint",
            "priority": "medium",
            "dialogue": [],
            "hint_text": "Try exploring the area more carefully. There might be something you missed.",
            "world_reaction": None,
            "debug_notes": "Test hint from DummyLLMClient"
        }
        return json.dumps(response, indent=2)

    def _debug_response(self, prompt: str) -> str:
        """Generate debug response."""
        response = {
            "mode": "dev_debug",
            "priority": "low",
            "dialogue": [],
            "hint_text": None,
            "world_reaction": None,
            "debug_notes": "This is a test debug summary from DummyLLMClient. Scene analysis shows potential narrative opportunities."
        }
        return json.dumps(response, indent=2)

    def _default_response(self) -> str:
        """Generate default response based on configured mode."""
        if self.response_mode == "npc_dialogue":
            return self._npc_dialogue_response("")
        elif self.response_mode == "hint":
            return self._hint_response("")
        elif self.response_mode == "world_reaction":
            return self._world_reaction_response("")
        else:
            return self._debug_response("")

    def reset(self):
        """Reset call tracking."""
        self.call_count = 0
        self.last_prompt = None


# Future: Add real LLM clients

class OpenAIClient:
    """OpenAI client implementation (stub for future)."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        raise NotImplementedError("OpenAI client not yet implemented")


class AnthropicClient:
    """Anthropic Claude client implementation (stub for future)."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        raise NotImplementedError("Anthropic client not yet implemented")


class LocalLLMClient:
    """Local LLM client (Ollama, etc.) implementation (stub for future)."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.base_url = base_url
        self.model = model
        raise NotImplementedError("Local LLM client not yet implemented")


def create_llm_client(provider: str = "dummy", **kwargs) -> BaseLLMClient:
    """
    Factory function to create LLM clients.

    Args:
        provider: "dummy", "openai", "anthropic", "local"
        **kwargs: Provider-specific arguments

    Returns:
        LLM client instance

    Examples:
        >>> client = create_llm_client("dummy")
        >>> client = create_llm_client("dummy", response_mode="hint")
    """
    if provider == "dummy":
        return DummyLLMClient(**kwargs)
    elif provider == "openai":
        raise NotImplementedError("OpenAI client not yet implemented")
    elif provider == "anthropic":
        raise NotImplementedError("Anthropic client not yet implemented")
    elif provider == "local":
        raise NotImplementedError("Local LLM client not yet implemented")
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
