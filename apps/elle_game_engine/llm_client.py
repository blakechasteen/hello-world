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


# Real LLM Clients

class AnthropicClient:
    """
    Anthropic Claude client implementation.

    Requires: pip install anthropic
    Environment: ANTHROPIC_API_KEY
    """

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key
            model: Model to use (default: claude-3-5-sonnet-20241022)
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate completion using Claude.

        Args:
            prompt: User prompt
            system_prompt: System/instruction prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=messages,
        )

        return response.content[0].text


class OpenAIClient:
    """
    OpenAI client implementation.

    Requires: pip install openai
    Environment: OPENAI_API_KEY
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini for cost efficiency)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate completion using OpenAI.

        Args:
            prompt: User prompt
            system_prompt: System/instruction prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content


class LocalLLMClient:
    """
    Local LLM client using Ollama.

    Requires: Ollama running locally (ollama.ai)
    No API key needed.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        """
        Initialize local LLM client.

        Args:
            base_url: Ollama API base URL
            model: Model name (must be pulled via ollama pull first)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate completion using local Ollama model.

        Args:
            prompt: User prompt
            system_prompt: System/instruction prompt
            max_tokens: Maximum tokens to generate (not all models respect this)
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        import httpx

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["response"]


def create_llm_client(provider: str = "dummy", **kwargs) -> BaseLLMClient:
    """
    Factory function to create LLM clients.

    Args:
        provider: "dummy", "openai", "anthropic", "local"
        **kwargs: Provider-specific arguments

    Returns:
        LLM client instance

    Examples:
        >>> # Dummy client for testing
        >>> client = create_llm_client("dummy")
        >>> client = create_llm_client("dummy", response_mode="hint")

        >>> # Real LLM clients
        >>> client = create_llm_client("anthropic", api_key="sk-...")
        >>> client = create_llm_client("openai", api_key="sk-...", model="gpt-4")
        >>> client = create_llm_client("local", model="llama3.2:3b")
    """
    if provider == "dummy":
        return DummyLLMClient(**kwargs)
    elif provider == "anthropic":
        return AnthropicClient(**kwargs)
    elif provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "local":
        return LocalLLMClient(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
