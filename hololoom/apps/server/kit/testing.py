"""
Test harness for agent pipelines.

    MockBackend  — deterministic LLM backend for testing without infrastructure
    mock         — factory shorthand

Usage:
    from hololoom.apps.server.kit.testing import mock, MockBackend

    async def test_single_pass():
        backend = mock(["Hello from mock."])
        agent = Agent(soul="Test.", memory=NullMemory(), backend=backend, name="test")
        resp = await agent(ChatRequest(text="hi", room_id="t"))
        assert resp.response == "Hello from mock."
        assert backend.calls == 1

    async def test_multi_pass_pipeline():
        fast = mock(["Quick draft."])
        quality = mock(["Refined response."])
        agent = Agent(
            soul="Test.",
            memory=NullMemory(),
            backend=fast,
            pipeline=[think(fast), think(quality)],
            name="test",
        )
        resp = await agent(ChatRequest(text="hi", room_id="t"))
        assert resp.response == "Refined response."
        assert len(resp.metadata["trace"]) == 2
"""
from __future__ import annotations


class MockBackend:
    """Deterministic LLM backend for testing pipelines without infrastructure.

    Satisfies LLMBackend protocol. Returns canned responses in order.
    Tracks every call for assertions.

    Args:
        responses: List of response strings. Popped in order; cycles last one.
        model: Model name reported in responses.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        model: str = "mock",
    ) -> None:
        self.model = model
        self._responses = list(responses or ["Mock response."])
        self._history: list[list[dict[str, str]]] = []

    async def call(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self._history.append(messages)
        # Pop next response, or repeat the last one
        text = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return {
            "message": {"content": text},
            "model": self.model,
            "eval_count": len(text.split()),
            "prompt_eval_count": sum(len(m.get("content", "").split()) for m in messages),
        }

    @property
    def calls(self) -> int:
        """Number of times call() was invoked."""
        return len(self._history)

    @property
    def last_messages(self) -> list[dict[str, str]]:
        """Messages from the most recent call."""
        return self._history[-1] if self._history else []


def mock(
    responses: list[str] | None = None,
    model: str = "mock",
) -> MockBackend:
    """Factory: create a MockBackend."""
    return MockBackend(responses=responses, model=model)
