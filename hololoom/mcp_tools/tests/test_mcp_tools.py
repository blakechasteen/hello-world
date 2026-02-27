"""
Test HoloLoom MCP Tools
=======================

**Created**: November 21, 2025
**Status**: Basic tests (will expand with more tools)
"""

import pytest
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import server module
try:
    from HoloLoom.mcp_tools import server
    MCP_TOOLS_AVAILABLE = True
except ImportError:
    MCP_TOOLS_AVAILABLE = False


@pytest.mark.skipif(not MCP_TOOLS_AVAILABLE, reason="MCP tools not available")
class TestMCPServer:
    """Test MCP server functionality."""

    def test_server_import(self):
        """Test that server module can be imported."""
        assert server is not None
        assert hasattr(server, 'app')
        assert hasattr(server, 'create_hololoom_mcp_server')

    def test_get_hololoom(self):
        """Test HoloLoom instance creation."""
        if not server.HOLOLOOM_AVAILABLE:
            pytest.skip("HoloLoom not available")

        loom = server.get_hololoom()
        assert loom is not None

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test listing available tools."""
        if not server.MCP_AVAILABLE:
            pytest.skip("MCP SDK not installed")

        tools = await server.list_tools()
        assert len(tools) >= 8  # At minimum: experience, recall, metrics, weave, summary, reflect, etc.

        # Check for core tools
        tool_names = [t.name for t in tools]
        assert "hololoom_experience" in tool_names
        assert "hololoom_recall" in tool_names
        assert "hololoom_metrics" in tool_names
        assert "hololoom_weave" in tool_names

    @pytest.mark.asyncio
    async def test_tool_experience(self):
        """Test memory storage tool."""
        if not server.HOLOLOOM_AVAILABLE or not server.MCP_AVAILABLE:
            pytest.skip("HoloLoom or MCP not available")

        args = {
            "content": "Test memory: HoloLoom uses Thompson Sampling",
            "metadata": {"source": "test", "tags": ["testing"]}
        }

        result = await server.tool_experience(args)
        assert len(result) == 1
        assert result[0].type == "text"

        # Parse JSON result
        import json
        data = json.loads(result[0].text)
        assert data["status"] == "success"
        assert data["content_length"] > 0

    @pytest.mark.asyncio
    async def test_tool_recall(self):
        """Test memory retrieval tool."""
        if not server.HOLOLOOM_AVAILABLE or not server.MCP_AVAILABLE:
            pytest.skip("HoloLoom or MCP not available")

        # First store a memory
        await server.tool_experience({
            "content": "Thompson Sampling is a Bayesian exploration algorithm"
        })

        # Then recall it
        args = {
            "query": "What is Thompson Sampling?",
            "limit": 5
        }

        result = await server.tool_recall(args)
        assert len(result) == 1

        import json
        data = json.loads(result[0].text)
        assert "memories_found" in data
        assert data["memories_found"] >= 0

    @pytest.mark.asyncio
    async def test_tool_metrics(self):
        """Test metrics retrieval."""
        if not server.HOLOLOOM_AVAILABLE or not server.MCP_AVAILABLE:
            pytest.skip("HoloLoom or MCP not available")

        args = {
            "include_learning": True,
            "include_performance": True
        }

        result = await server.tool_metrics(args)
        assert len(result) == 1

        import json
        data = json.loads(result[0].text)
        assert "metrics" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_tool_summary(self):
        """Test system summary."""
        if not server.HOLOLOOM_AVAILABLE or not server.MCP_AVAILABLE:
            pytest.skip("HoloLoom or MCP not available")

        result = await server.tool_summary({})
        assert len(result) == 1
        assert len(result[0].text) > 0  # Summary should have content

    @pytest.mark.asyncio
    async def test_tool_weave(self):
        """Test weaving cycle."""
        if not server.HOLOLOOM_AVAILABLE or not server.MCP_AVAILABLE:
            pytest.skip("HoloLoom or MCP not available")

        args = {
            "query": "What is 2+2?",
            "mode": "FAST",
            "enable_reflection": False
        }

        result = await server.tool_weave(args)
        assert len(result) == 1

        import json
        data = json.loads(result[0].text)
        assert "query" in data
        assert "response" in data
        assert "confidence" in data

    @pytest.mark.asyncio
    async def test_tool_reason(self):
        """Test agentic reasoning (if available)."""
        if not server.AGENTIC_AVAILABLE or not server.MCP_AVAILABLE:
            pytest.skip("Agentic reasoning not available")

        args = {
            "query": "What is Thompson Sampling?",
            "mode": "direct",
            "max_steps": 3
        }

        result = await server.tool_reason(args)
        assert len(result) == 1

        import json
        data = json.loads(result[0].text)
        assert "response" in data
        assert "confidence" in data
        assert "mode" in data


@pytest.mark.skipif(not MCP_TOOLS_AVAILABLE, reason="MCP tools not available")
class TestToolRegistry:
    """Test tool registration and discovery."""

    @pytest.mark.asyncio
    async def test_all_tools_have_names(self):
        """Verify all tools have names."""
        if not server.MCP_AVAILABLE:
            pytest.skip("MCP SDK not installed")

        tools = await server.list_tools()
        for tool in tools:
            assert hasattr(tool, 'name')
            assert len(tool.name) > 0
            assert tool.name.startswith('hololoom_')

    @pytest.mark.asyncio
    async def test_all_tools_have_descriptions(self):
        """Verify all tools have descriptions."""
        if not server.MCP_AVAILABLE:
            pytest.skip("MCP SDK not installed")

        tools = await server.list_tools()
        for tool in tools:
            assert hasattr(tool, 'description')
            assert len(tool.description) > 0

    @pytest.mark.asyncio
    async def test_all_tools_have_schemas(self):
        """Verify all tools have input schemas."""
        if not server.MCP_AVAILABLE:
            pytest.skip("MCP SDK not installed")

        tools = await server.list_tools()
        for tool in tools:
            assert hasattr(tool, 'inputSchema')
            assert tool.inputSchema is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
