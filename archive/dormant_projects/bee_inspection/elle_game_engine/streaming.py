"""
Server-Sent Events (SSE) streaming for Elle Game Engine.

Provides token-by-token streaming for LLM responses, reducing perceived latency
and enabling progressive UI updates in game engines.

Performance Benefits:
- First token arrives faster (50-200ms vs blocking wait)
- Progressive UI updates (smoother UX)
- Better timeout handling (can cancel mid-stream)
- Lower perceived latency for users

Supports all LLM providers:
- Anthropic (Claude) - native streaming
- OpenAI (GPT) - native streaming
- Ollama (local) - native streaming
- Dummy - simulated streaming for testing
"""

from typing import Protocol, AsyncIterator, Optional, Dict, Any
from dataclasses import dataclass
import json
import asyncio
import time


# ============================================================================
# Streaming Protocol
# ============================================================================

@dataclass
class StreamChunk:
    """Single chunk in a streaming response."""

    event: str  # "token", "complete", "error"
    data: str   # Token text or final JSON
    metadata: Dict[str, Any] = None  # Optional metadata (timestamp, etc.)

    def to_sse_format(self) -> str:
        """
        Convert to Server-Sent Events format.

        Returns:
            SSE-formatted string (event: ...\ndata: ...\n\n)
        """
        lines = []
        lines.append(f"event: {self.event}")

        # For multi-line data, use multiple "data:" lines
        for line in self.data.split('\n'):
            lines.append(f"data: {line}")

        # Add metadata if present
        if self.metadata:
            lines.append(f"data: __metadata__:{json.dumps(self.metadata)}")

        lines.append("")  # Empty line terminates event
        return "\n".join(lines) + "\n"


class StreamingLLMClient(Protocol):
    """
    Protocol for streaming LLM clients.

    All streaming implementations must provide this interface.
    """

    async def stream_complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream completion tokens.

        Args:
            prompt: User prompt
            system_prompt: System/instruction prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            StreamChunk objects with event="token" (during) and event="complete" (end)
        """
        ...


# ============================================================================
# Streaming Client Implementations
# ============================================================================

class StreamingDummyClient:
    """
    Dummy streaming client for testing.

    Simulates realistic streaming behavior with configurable delays.
    """

    def __init__(
        self,
        *,
        response_mode: str = "npc_dialogue",
        tokens_per_second: int = 20,  # Simulate ~20 tokens/sec (realistic for GPT-4)
    ):
        """
        Initialize dummy streaming client.

        Args:
            response_mode: Response mode ("npc_dialogue", "hint", etc.)
            tokens_per_second: Simulated token generation speed
        """
        self.response_mode = response_mode
        self.token_delay = 1.0 / tokens_per_second
        self.call_count = 0

    async def stream_complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamChunk]:
        """Stream dummy tokens."""
        self.call_count += 1

        # Generate appropriate response based on prompt
        if "talk_to_npc" in prompt.lower():
            response_text = self._npc_dialogue_json()
        elif "enter_scene" in prompt.lower():
            response_text = self._world_reaction_json()
        elif "request_hint" in prompt.lower():
            response_text = self._hint_json()
        else:
            response_text = self._npc_dialogue_json()

        # Stream character by character (simulating token generation)
        buffer = ""
        start_time = time.time()

        for i, char in enumerate(response_text):
            buffer += char

            # Yield token every ~5 characters (rough token approximation)
            if len(buffer) >= 5 or i == len(response_text) - 1:
                yield StreamChunk(
                    event="token",
                    data=buffer,
                    metadata={
                        "timestamp": time.time(),
                        "position": i,
                        "total": len(response_text),
                    }
                )
                buffer = ""

                # Simulate realistic delay
                await asyncio.sleep(self.token_delay)

        # Final complete event with full response
        duration_ms = (time.time() - start_time) * 1000
        yield StreamChunk(
            event="complete",
            data=response_text,
            metadata={
                "timestamp": time.time(),
                "duration_ms": duration_ms,
                "total_tokens": len(response_text) // 5,  # Rough estimate
            }
        )

    def _npc_dialogue_json(self) -> str:
        """Generate NPC dialogue JSON."""
        return json.dumps({
            "mode": "npc_dialogue",
            "priority": "medium",
            "dialogue": [
                {
                    "npc_id": "innkeeper",
                    "text": "Greetings, traveler! Welcome to the Silver Stag Inn.",
                    "tone": "warm"
                }
            ],
            "hint_text": None,
            "world_reaction": None,
            "debug_notes": "Streamed from DummyClient"
        }, indent=2)

    def _world_reaction_json(self) -> str:
        """Generate world reaction JSON."""
        return json.dumps({
            "mode": "world_reaction",
            "priority": "low",
            "dialogue": [],
            "hint_text": None,
            "world_reaction": {
                "description": "The evening air is calm. Lanterns flicker along the cobblestone streets.",
                "flag_changes": {"scene_entered": True}
            },
            "debug_notes": "Streamed from DummyClient"
        }, indent=2)

    def _hint_json(self) -> str:
        """Generate hint JSON."""
        return json.dumps({
            "mode": "hint",
            "priority": "medium",
            "dialogue": [],
            "hint_text": "Try speaking with the innkeeper. They often have valuable information about local rumors.",
            "world_reaction": None,
            "debug_notes": "Streamed from DummyClient"
        }, indent=2)


class StreamingAnthropicClient:
    """
    Streaming Anthropic Claude client.

    Uses native streaming API for token-by-token generation.
    """

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize Anthropic streaming client.

        Args:
            api_key: Anthropic API key
            model: Model to use
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    async def stream_complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion using Claude."""
        messages = [{"role": "user", "content": prompt}]

        start_time = time.time()
        accumulated_text = ""
        token_count = 0

        # Use Anthropic's streaming API
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                accumulated_text += text
                token_count += 1

                yield StreamChunk(
                    event="token",
                    data=text,
                    metadata={
                        "timestamp": time.time(),
                        "token_count": token_count,
                    }
                )

        # Final complete event
        duration_ms = (time.time() - start_time) * 1000
        yield StreamChunk(
            event="complete",
            data=accumulated_text,
            metadata={
                "timestamp": time.time(),
                "duration_ms": duration_ms,
                "total_tokens": token_count,
            }
        )


class StreamingOpenAIClient:
    """
    Streaming OpenAI client.

    Uses native streaming API for token-by-token generation.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI streaming client.

        Args:
            api_key: OpenAI API key
            model: Model to use
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def stream_complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion using OpenAI."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        accumulated_text = ""
        token_count = 0

        # Use OpenAI's streaming API
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                accumulated_text += text
                token_count += 1

                yield StreamChunk(
                    event="token",
                    data=text,
                    metadata={
                        "timestamp": time.time(),
                        "token_count": token_count,
                    }
                )

        # Final complete event
        duration_ms = (time.time() - start_time) * 1000
        yield StreamChunk(
            event="complete",
            data=accumulated_text,
            metadata={
                "timestamp": time.time(),
                "duration_ms": duration_ms,
                "total_tokens": token_count,
            }
        )


class StreamingLocalClient:
    """
    Streaming local LLM client using Ollama.

    Uses Ollama's streaming API for token-by-token generation.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        """
        Initialize local streaming client.

        Args:
            base_url: Ollama API base URL
            model: Model name
        """
        self.base_url = base_url.rstrip('/')
        self.model = model

    async def stream_complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion using local Ollama model."""
        import httpx

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        start_time = time.time()
        accumulated_text = ""
        token_count = 0

        # Use Ollama's streaming API
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                },
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                text = data["response"]
                                accumulated_text += text
                                token_count += 1

                                yield StreamChunk(
                                    event="token",
                                    data=text,
                                    metadata={
                                        "timestamp": time.time(),
                                        "token_count": token_count,
                                    }
                                )
                        except json.JSONDecodeError:
                            continue

        # Final complete event
        duration_ms = (time.time() - start_time) * 1000
        yield StreamChunk(
            event="complete",
            data=accumulated_text,
            metadata={
                "timestamp": time.time(),
                "duration_ms": duration_ms,
                "total_tokens": token_count,
            }
        )


# ============================================================================
# Factory
# ============================================================================

def create_streaming_client(provider: str = "dummy", **kwargs) -> StreamingLLMClient:
    """
    Factory function to create streaming LLM clients.

    Args:
        provider: "dummy", "anthropic", "openai", "local"
        **kwargs: Provider-specific arguments

    Returns:
        Streaming LLM client instance

    Examples:
        >>> # Dummy client for testing
        >>> client = create_streaming_client("dummy", tokens_per_second=20)

        >>> # Real streaming clients
        >>> client = create_streaming_client("anthropic", api_key="sk-...")
        >>> client = create_streaming_client("openai", api_key="sk-...")
        >>> client = create_streaming_client("local", model="llama3.2:3b")
    """
    if provider == "dummy":
        return StreamingDummyClient(**kwargs)
    elif provider == "anthropic":
        return StreamingAnthropicClient(**kwargs)
    elif provider == "openai":
        return StreamingOpenAIClient(**kwargs)
    elif provider == "local":
        return StreamingLocalClient(**kwargs)
    else:
        raise ValueError(f"Unknown streaming provider: {provider}")
