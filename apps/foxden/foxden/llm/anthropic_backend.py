"""Anthropic backend — Claude API."""

from __future__ import annotations

from .backend import LLMResponse, Message


class AnthropicBackend:
    """Talk to the Claude API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_retries: int = 2,
    ):
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("pip install foxden[anthropic]")
            self._client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                max_retries=self.max_retries,
            )
        return self._client

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        client = self._get_client()

        # Extract system message if present.
        system = None
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        kwargs = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        resp = await client.messages.create(**kwargs)

        return LLMResponse(
            content=resp.content[0].text,
            model=self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
