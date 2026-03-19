"""Tests for weaverlet.shuttle.http_client — DaemonInferClient."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from hololoom.weaverlet.shuttle.http_client import (
    DaemonInferClient,
    DaemonUnavailableError,
    InferResponse,
)


# ── InferResponse ────────────────────────────────────────────────────────

class TestInferResponse:
    def test_fields(self):
        r = InferResponse(content="hello", tokens_used=42, model="qwen", latency_ms=100.5)
        assert r.content == "hello"
        assert r.tokens_used == 42
        assert r.model == "qwen"
        assert r.latency_ms == 100.5


# ── Mock helpers ─────────────────────────────────────────────────────────

class MockResponse:
    """Minimal mock for aiohttp response context manager."""

    def __init__(self, json_data, status=200, text=""):
        self._json = json_data
        self.status = status
        self._text = text

    async def json(self):
        return self._json

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockSession:
    """Minimal mock for aiohttp.ClientSession."""

    def __init__(self):
        self.closed = False
        self._post_response = None
        self._get_response = None

    def post(self, url, json=None):
        return self._post_response

    def get(self, url):
        return self._get_response

    async def close(self):
        self.closed = True


# ── DaemonInferClient ────────────────────────────────────────────────────

class TestDaemonInferClient:

    @pytest.mark.asyncio
    async def test_infer_success(self):
        client = DaemonInferClient("http://localhost:4243")
        session = MockSession()
        session._post_response = MockResponse({
            "response": "Hello world",
            "tokens_used": 50,
            "model": "qwen3.5:9b",
            "latency_ms": 200.0,
        })
        client._session = session

        resp = await client.infer("Say hello")
        assert resp.content == "Hello world"
        assert resp.tokens_used == 50
        assert resp.model == "qwen3.5:9b"

    @pytest.mark.asyncio
    async def test_infer_content_key(self):
        """Daemon may return 'content' instead of 'response'."""
        client = DaemonInferClient()
        session = MockSession()
        session._post_response = MockResponse({
            "content": "Result text",
            "tokens_used": 30,
        })
        client._session = session

        resp = await client.infer("test")
        assert resp.content == "Result text"

    @pytest.mark.asyncio
    async def test_infer_error_status(self):
        client = DaemonInferClient()
        session = MockSession()
        session._post_response = MockResponse(
            {}, status=500, text="Internal Server Error"
        )
        client._session = session

        with pytest.raises(DaemonUnavailableError, match="500"):
            await client.infer("test")

    @pytest.mark.asyncio
    async def test_infer_with_system_prompt(self):
        """System prompt gets prepended to prompt."""
        client = DaemonInferClient()
        session = MockSession()
        captured_payloads = []

        original_post = session.post

        def capturing_post(url, json=None):
            captured_payloads.append(json)
            return MockResponse({
                "response": "ok",
                "tokens_used": 10,
            })

        session.post = capturing_post
        client._session = session

        await client.infer("user prompt", system_prompt="system instructions")
        assert len(captured_payloads) == 1
        assert "system instructions" in captured_payloads[0]["prompt"]
        assert "user prompt" in captured_payloads[0]["prompt"]

    @pytest.mark.asyncio
    async def test_health_success(self):
        client = DaemonInferClient()
        session = MockSession()
        session._get_response = MockResponse({"running": True})
        client._session = session

        assert await client.health() is True

    @pytest.mark.asyncio
    async def test_health_failure(self):
        client = DaemonInferClient()
        session = MockSession()
        session._get_response = MockResponse({}, status=503)
        client._session = session

        assert await client.health() is False

    @pytest.mark.asyncio
    async def test_store_memory(self):
        client = DaemonInferClient()
        session = MockSession()
        session._post_response = MockResponse({"id": "mem_123"})
        client._session = session

        result = await client.store_memory("test content", tags=["weaverlet"])
        assert result["id"] == "mem_123"

    @pytest.mark.asyncio
    async def test_recall_memory(self):
        client = DaemonInferClient()
        session = MockSession()
        session._post_response = MockResponse({
            "memories": [{"content": "recalled item", "score": 0.9}]
        })
        client._session = session

        results = await client.recall_memory("search query", limit=5)
        assert len(results) == 1
        assert results[0]["content"] == "recalled item"

    @pytest.mark.asyncio
    async def test_recall_memory_results_key(self):
        """Daemon may use 'results' instead of 'memories'."""
        client = DaemonInferClient()
        session = MockSession()
        session._post_response = MockResponse({
            "results": [{"content": "item"}]
        })
        client._session = session

        results = await client.recall_memory("query")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_close(self):
        client = DaemonInferClient()
        session = MockSession()
        client._session = session

        await client.close()
        assert session.closed

    @pytest.mark.asyncio
    async def test_context_manager(self):
        client = DaemonInferClient()
        session = MockSession()
        client._session = session

        async with client:
            pass
        assert session.closed

    def test_url_trailing_slash_stripped(self):
        client = DaemonInferClient("http://localhost:4243/")
        assert client.base_url == "http://localhost:4243"
