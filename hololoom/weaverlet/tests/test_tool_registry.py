"""Tests for weaverlet.tools.registry — WeaverletToolRegistry."""

import pytest

from hololoom.weaverlet.schemas import Budget, BudgetExhaustedError, BudgetLedger
from hololoom.weaverlet.tools.registry import WeaverletToolRegistry


class TestWeaverletToolRegistry:

    def test_list_builtin_tools(self):
        reg = WeaverletToolRegistry()
        tools = reg.list_tools()
        assert "code_execute" in tools
        assert "memory_query_yarn" in tools
        assert "memory_query_warp" in tools

    def test_register_custom_tool(self):
        reg = WeaverletToolRegistry()

        async def my_tool(**kwargs):
            return {"result": "ok", "success": True}

        reg.register("my_tool", my_tool)
        assert "my_tool" in reg.list_tools()

    @pytest.mark.asyncio
    async def test_execute_custom_tool(self):
        reg = WeaverletToolRegistry()

        async def echo_tool(**kwargs):
            return {"echo": kwargs.get("msg", ""), "success": True}

        reg.register("echo", echo_tool)
        result = await reg.execute("echo", msg="hello")
        assert result["echo"] == "hello"
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        reg = WeaverletToolRegistry()
        result = await reg.execute("nonexistent_tool")
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    @pytest.mark.asyncio
    async def test_charges_ledger(self):
        ledger = BudgetLedger(budget=Budget(max_tool_calls=10))
        reg = WeaverletToolRegistry(ledger=ledger)

        async def noop(**kwargs):
            return {"success": True}

        reg.register("noop", noop)
        await reg.execute("noop")
        assert ledger.tool_calls_used == 1
        await reg.execute("noop")
        assert ledger.tool_calls_used == 2

    @pytest.mark.asyncio
    async def test_budget_exhausted_before_call(self):
        ledger = BudgetLedger(budget=Budget(max_tool_calls=1))
        reg = WeaverletToolRegistry(ledger=ledger)

        async def noop(**kwargs):
            return {"success": True}

        reg.register("noop", noop)
        # First call works (charges to 1, which exceeds max of 1 via charge_tool_call)
        # Actually charge_tool_call increments THEN checks > max
        await reg.execute("noop")  # tool_calls_used = 1
        # Ledger is now exhausted (1 >= 1)
        with pytest.raises(BudgetExhaustedError):
            await reg.execute("noop")

    @pytest.mark.asyncio
    async def test_tool_error_returns_error_dict(self):
        reg = WeaverletToolRegistry()

        async def bad_tool(**kwargs):
            raise ValueError("something went wrong")

        reg.register("bad", bad_tool)
        result = await reg.execute("bad")
        assert result["success"] is False
        assert "something went wrong" in result["error"]

    @pytest.mark.asyncio
    async def test_no_ledger_still_works(self):
        reg = WeaverletToolRegistry(ledger=None)

        async def noop(**kwargs):
            return {"success": True}

        reg.register("noop", noop)
        result = await reg.execute("noop")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_yarn_query_no_backend(self):
        """Memory tools with no backend return empty results."""
        reg = WeaverletToolRegistry(yarn=None, warp=None)
        result = await reg.execute("memory_query_yarn", cypher="MATCH (n) RETURN n")
        assert result["results"] == []
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_warp_query_no_backend(self):
        reg = WeaverletToolRegistry(yarn=None, warp=None)
        result = await reg.execute("memory_query_warp", query="test")
        assert result["results"] == []
        assert result["success"] is True
