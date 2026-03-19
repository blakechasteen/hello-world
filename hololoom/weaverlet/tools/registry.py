"""
Weaverlet Tool Registry
========================

Extends ToolExecutor pattern with budget-aware execution.
Every tool call charges the BudgetLedger.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from hololoom.weaverlet.schemas import BudgetExhaustedError, BudgetLedger
from hololoom.weaverlet.tools.code_executor import SandboxedCodeExecutor
from hololoom.weaverlet.tools.memory_tools import MemoryTools

logger = logging.getLogger(__name__)

ToolHandler = Callable[..., Awaitable[dict[str, Any]]]


class WeaverletToolRegistry:

    def __init__(
        self,
        ledger: BudgetLedger | None = None,
        yarn=None,
        warp=None,
    ):
        self.ledger = ledger
        self._handlers: dict[str, ToolHandler] = {}

        # Register built-in tools
        self._code_executor = SandboxedCodeExecutor()
        self._memory_tools = MemoryTools(yarn=yarn, warp=warp)

        self.register("code_execute", self._handle_code_execute)
        self.register("memory_query_yarn", self._handle_yarn_query)
        self.register("memory_query_warp", self._handle_warp_query)

    def register(self, name: str, handler: ToolHandler) -> None:
        self._handlers[name] = handler
        logger.debug(f"Registered tool: {name}")

    async def execute(self, tool_name: str, **kwargs) -> dict[str, Any]:
        """Execute a tool by name. Charges the budget ledger."""
        if tool_name not in self._handlers:
            return {"error": f"Unknown tool: {tool_name}", "success": False}

        if self.ledger and self.ledger.is_exhausted():
            raise BudgetExhaustedError(
                f"Budget exhausted before tool call: {self.ledger.exhaustion_reason()}"
            )

        if self.ledger:
            self.ledger.charge_tool_call()

        try:
            result = await self._handlers[tool_name](**kwargs)
            return result
        except BudgetExhaustedError:
            raise
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"error": str(e), "success": False}

    def list_tools(self) -> list[str]:
        return list(self._handlers.keys())

    # ── Built-in handlers ─────────────────────────────────────────────

    async def _handle_code_execute(self, **kwargs) -> dict[str, Any]:
        return await self._code_executor.execute(
            code=kwargs.get("code", ""),
            language=kwargs.get("language", "python"),
            timeout=kwargs.get("timeout", 30.0),
        )

    async def _handle_yarn_query(self, **kwargs) -> dict[str, Any]:
        results = await self._memory_tools.query_yarn(
            cypher=kwargs.get("cypher", ""),
            params=kwargs.get("params"),
        )
        return {"results": results, "success": True}

    async def _handle_warp_query(self, **kwargs) -> dict[str, Any]:
        results = await self._memory_tools.query_warp(
            query=kwargs.get("query", ""),
            limit=kwargs.get("limit", 5),
        )
        return {"results": results, "success": True}
