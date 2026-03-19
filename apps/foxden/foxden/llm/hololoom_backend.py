"""HoloLoom backend — use HoloLoom's UnifiedLLMClient if available."""

from __future__ import annotations

from .backend import LLMResponse, Message


class HoloLoomBackend:
    """Bridge to HoloLoom's multi-provider LLM client."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen3:30b",
        **kwargs,
    ):
        try:
            from hololoom.llm.unified_client import UnifiedLLMClient, LLMConfig
        except ImportError:
            raise ImportError(
                "HoloLoom not found. Install it or use a different backend.\n"
                "  pip install -e /path/to/hololoom"
            )

        self._config = LLMConfig(provider=provider, model=model, **kwargs)
        self._client = UnifiedLLMClient(self._config)
        self.model = model

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        # Convert our Message format to what HoloLoom expects.
        system = None
        prompt_parts = []

        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                prompt_parts.append(f"[{m.role.upper()}]\n{m.content}")

        prompt = "\n\n".join(prompt_parts)

        resp = await self._client.generate(
            prompt=prompt,
            system_prompt=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return LLMResponse(
            content=resp.content,
            model=resp.model,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        )
