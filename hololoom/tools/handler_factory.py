"""
Tool Handler Factory - Eliminates duplication in tool execution.

This module provides a registry-based factory pattern for tool handlers,
reducing ~140 LOC of duplicated code in weaving_orchestrator.py.

Design Pattern:
- Registry pattern for tool handler registration
- Type-safe handler signatures
- Default fallback for unknown tools
- Easy extensibility (just register new handlers)

Usage:
    from hololoom.tools.handler_factory import ToolHandlerFactory

    # Create factory
    factory = ToolHandlerFactory(logger=logger, llm=llm_client)

    # Execute tool
    result = await factory.execute(
        tool_name="answer",
        query=query,
        context=context
    )
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from hololoom.protocols.types import Context, Query


@dataclass
class ToolHandler:
    """
    Tool handler registration.

    Encapsulates handler function and metadata.
    """
    name: str
    handler: Callable[[Query, Context, Any], Awaitable[dict]]
    description: str
    requires_llm: bool = False


class ToolHandlerFactory:
    """
    Factory for tool execution with registry pattern.

    Eliminates code duplication by centralizing tool handler logic.
    Handlers are registered once and executed by name.

    Example:
        factory = ToolHandlerFactory(logger=my_logger, llm=my_llm)

        # Register custom handler
        async def custom_handler(query, context, deps):
            return {"tool": "custom", "result": "success"}

        factory.register("custom_tool", custom_handler, "My custom tool")

        # Execute
        result = await factory.execute("custom_tool", query, context)
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        llm: Any | None = None
    ):
        """
        Initialize tool handler factory.

        Args:
            logger: Logger for debugging
            llm: LLM client for answer generation
        """
        self.logger = logger or logging.getLogger(__name__)
        self.llm = llm
        self._handlers: dict[str, ToolHandler] = {}

        # Register default handlers
        self._register_defaults()

    def register(
        self,
        name: str,
        handler: Callable[[Query, Context, Any], Awaitable[dict]],
        description: str,
        requires_llm: bool = False
    ):
        """
        Register a tool handler.

        Args:
            name: Tool name
            handler: Async handler function
            description: Handler description
            requires_llm: Whether handler requires LLM

        Example:
            async def my_handler(query, context, deps):
                return {"tool": "my_tool", "result": "success"}

            factory.register("my_tool", my_handler, "My custom tool")
        """
        self._handlers[name] = ToolHandler(
            name=name,
            handler=handler,
            description=description,
            requires_llm=requires_llm
        )
        self.logger.debug(f"Registered tool handler: {name}")

    async def execute(
        self,
        tool_name: str,
        query: Query,
        context: Context,
        **kwargs
    ) -> dict:
        """
        Execute a tool handler by name.

        Args:
            tool_name: Name of tool to execute
            query: Query object
            context: Context object
            **kwargs: Additional dependencies passed to handler

        Returns:
            Dict with tool execution results

        Example:
            result = await factory.execute(
                "answer",
                query=Query(text="What is Thompson Sampling?"),
                context=context
            )
        """
        # Get handler (fallback to unknown handler)
        handler_obj = self._handlers.get(tool_name, self._handlers["unknown"])

        # Check LLM requirement
        if handler_obj.requires_llm and not self._llm_available():
            self.logger.warning(
                f"Tool '{tool_name}' requires LLM but LLM unavailable. "
                f"Using fallback."
            )

        # Build dependencies
        deps = {
            "logger": self.logger,
            "llm": self.llm,
            **kwargs
        }

        # Execute handler
        try:
            result = await handler_obj.handler(query, context, deps)
            return result
        except Exception as e:
            self.logger.error(f"Tool handler '{tool_name}' failed: {e}")
            return {
                "tool": tool_name,
                "result": f"Tool execution failed: {e}",
                "error": str(e),
                "status": "error"
            }

    def list_handlers(self) -> dict[str, str]:
        """
        List all registered handlers.

        Returns:
            Dict mapping handler name → description
        """
        return {
            name: handler.description
            for name, handler in self._handlers.items()
        }

    def _llm_available(self) -> bool:
        """Check if LLM is available."""
        return (
            self.llm is not None and
            hasattr(self.llm, 'is_available') and
            self.llm.is_available()
        )

    # ========================================================================
    # Default Handler Implementations
    # ========================================================================

    def _register_defaults(self):
        """Register default tool handlers."""
        self.register("answer", self._handle_answer, "Generate answer from context", requires_llm=True)
        self.register("search", self._handle_search, "Perform search")
        self.register("notion_write", self._handle_notion_write, "Write to Notion")
        self.register("calc", self._handle_calc, "Perform calculation")
        self.register("unknown", self._handle_unknown, "Fallback for unknown tools")

    async def _handle_answer(self, query: Query, context: Context, deps: dict) -> dict:
        """
        Generate an answer based on context.

        Uses LLM if available, otherwise returns fallback.
        """
        logger = deps["logger"]
        llm = deps.get("llm")

        # Extract packed context (beta wave optimization)
        packed_ctx = context.metadata.get('packed_context') if context and hasattr(context, 'metadata') else None

        if packed_ctx:
            # Use optimized physics-based packed context
            llm_context = packed_ctx.format_for_llm(include_metadata=False)
            context_tokens = packed_ctx.total_tokens
            packing_stats = {
                'using_packed_context': True,
                'elements_included': packed_ctx.elements_included,
                'elements_compressed': packed_ctx.elements_compressed,
                'elements_excluded': packed_ctx.elements_excluded,
                'avg_activation': packed_ctx.avg_activation,
                'token_budget_used': f"{packed_ctx.total_tokens}/{context.metadata.get('packing_stats', {}).get('budget_available', 'N/A')}"
            }
            logger.debug(
                f"Using packed context: {packed_ctx.elements_included} elements, "
                f"{context_tokens} tokens (avg_activation={packed_ctx.avg_activation:.3f})"
            )
        else:
            # Use raw shard texts (legacy behavior)
            shard_texts = context.shard_texts[:5] if context and hasattr(context, 'shard_texts') else []
            llm_context = "\n\n".join(shard_texts)
            context_tokens = len(llm_context) // 4  # Rough token estimate
            packing_stats = {
                'using_packed_context': False,
                'raw_shards_count': len(shard_texts)
            }
            logger.debug(f"Using raw context: {len(shard_texts)} shards")

        # Build LLM prompt
        system_prompt = (
            "You are a helpful AI assistant. "
            "Answer based on the provided context. "
            "Be concise and accurate."
        )

        user_prompt = f"""Context:
{llm_context}

Question: {query.text}

Answer:"""

        # Call LLM if available
        if llm and hasattr(llm, 'is_available') and llm.is_available():
            try:
                response = await llm.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=500,
                    temperature=0.7
                )

                return {
                    "tool": "answer",
                    "result": response.content,
                    "confidence": 0.85,
                    "sources": len(context.shards) if context and hasattr(context, 'shards') else 0,
                    "llm_provider": response.provider.value if hasattr(response, 'provider') else "ollama",
                    "llm_model": response.model if hasattr(response, 'model') else "unknown",
                    "usage": response.usage if hasattr(response, 'usage') else {},
                    "context_tokens": context_tokens,
                    "packing_stats": packing_stats
                }
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                # Fall through to fallback

        # Fallback (LLM unavailable)
        logger.warning("Using fallback response (LLM unavailable)")
        return {
            "tool": "answer",
            "result": f"[Fallback] Generated answer for: {query.text}\n\nContext: {llm_context[:300]}...",
            "confidence": 0.5,
            "sources": len(context.shards) if context and hasattr(context, 'shards') else 0,
            "context_preview": llm_context[:200] + "..." if len(llm_context) > 200 else llm_context,
            "context_tokens": context_tokens,
            "packing_stats": packing_stats
        }

    async def _handle_search(self, query: Query, context: Context, deps: dict) -> dict:
        """Perform a search."""
        return {
            "tool": "search",
            "result": "Search results based on query",
            "sources": ["source1", "source2", "source3"],
            "count": 3
        }

    async def _handle_notion_write(self, query: Query, context: Context, deps: dict) -> dict:
        """Write to Notion database."""
        return {
            "tool": "notion_write",
            "result": "Successfully wrote to Notion database",
            "status": "success",
            "page_id": "mock_page_123"
        }

    async def _handle_calc(self, query: Query, context: Context, deps: dict) -> dict:
        """Perform calculation."""
        return {
            "tool": "calc",
            "result": "Calculation completed",
            "value": 42,
            "expression": "mock_calculation"
        }

    async def _handle_unknown(self, query: Query, context: Context, deps: dict) -> dict:
        """Handle unknown tool."""
        return {
            "tool": "unknown",
            "result": "Unknown tool",
            "error": "Tool not implemented",
            "status": "error"
        }


# Convenience factory function
def create_tool_handler_factory(
    logger: logging.Logger | None = None,
    llm: Any | None = None
) -> ToolHandlerFactory:
    """
    Create a ToolHandlerFactory with default handlers.

    Args:
        logger: Logger instance
        llm: LLM client

    Returns:
        Configured ToolHandlerFactory

    Example:
        factory = create_tool_handler_factory(logger=my_logger, llm=my_llm)
        result = await factory.execute("answer", query, context)
    """
    return ToolHandlerFactory(logger=logger, llm=llm)
