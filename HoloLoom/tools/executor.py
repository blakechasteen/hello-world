#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool Execution Module
=====================

Executes tools based on convergence engine decisions.

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: lines 164-329

This module provides the ToolExecutor class which handles the execution of
various tools (answer, search, notion_write, calc) based on the convergence
engine's decisions.

Key Features:
- LLM-powered answer generation (with fallback)
- Beta wave context packing integration
- Graceful degradation when LLM unavailable
- Extensible tool handler system

Author: Claude Code (Elegance Pass Refactoring)
Date: 2025-11-22
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, TYPE_CHECKING

from HoloLoom.protocols.types import Query, Context

if TYPE_CHECKING:
    from HoloLoom.awareness.llm_integration import OllamaLLM


class ToolExecutor:
    """
    Executes tools based on convergence engine decisions.

    Supports multiple tools:
    - answer: LLM-powered answer generation
    - search: Search results retrieval
    - notion_write: Notion database writes
    - calc: Calculation execution

    The executor integrates with:
    - Beta wave context packing for optimized LLM prompts
    - Ollama LLM for local generation
    - Fallback responses when LLM unavailable

    Example:
        executor = ToolExecutor()
        result = await executor.execute("answer", query, context)

    Attributes:
        tools (List[str]): Available tool names
        llm (Optional[OllamaLLM]): LLM instance (lazy loaded)
        logger (logging.Logger): Logger instance
    """

    def __init__(self, llm: Optional['OllamaLLM'] = None):
        """
        Initialize the tool executor.

        Args:
            llm: Optional pre-initialized LLM instance.
                If None, will attempt to lazy-load Ollama LLM.

        Note:
            LLM initialization gracefully fails if dependencies unavailable.
            Executor falls back to mock responses.
        """
        self.tools = ["answer", "search", "notion_write", "calc"]
        self.logger = logging.getLogger(__name__)

        # Initialize LLM (lazy loading)
        self.llm = llm
        if self.llm is None:
            try:
                from HoloLoom.awareness.llm_integration import OllamaLLM
                self.llm = OllamaLLM(model="llama3.2:3b")
                self.logger.info("Initialized Ollama LLM (llama3.2:3b)")
            except Exception as e:
                self.logger.warning(f"LLM unavailable, using fallback: {e}")
                self.llm = None

    async def execute(self, tool: str, query: Query, context: Context) -> Dict:
        """
        Execute a tool based on the convergence decision.

        Routes the tool execution to the appropriate handler method.
        Falls back to _handle_unknown for unrecognized tools.

        Args:
            tool: Tool name from CollapseResult (e.g., "answer", "search")
            query: Original user query
            context: Retrieved context (shards + metadata)

        Returns:
            Dict containing execution results with keys:
                - tool: Tool name
                - result: Tool output
                - confidence: Confidence score (0.0-1.0)
                - Additional tool-specific metadata

        Example:
            result = await executor.execute("answer", query, context)
            print(result['result'])  # LLM-generated answer
        """
        self.logger.info(f"Executing tool: {tool}")

        # Tool implementations (stubs - replace with real implementations)
        tool_handlers = {
            "answer": self._handle_answer,
            "search": self._handle_search,
            "notion_write": self._handle_notion_write,
            "calc": self._handle_calc
        }

        handler = tool_handlers.get(tool, self._handle_unknown)
        return await handler(query, context)

    async def _handle_answer(self, query: Query, context: Context) -> Dict:
        """
        Generate an answer based on context.

        Uses beta wave packed context when available for optimized token usage.
        Falls back to raw shard texts for legacy compatibility.

        LLM Generation Flow:
        1. Extract context (packed or raw)
        2. Build system + user prompts
        3. Call LLM.generate() if available
        4. Return result or fallback

        Args:
            query: User query
            context: Retrieved context with metadata

        Returns:
            Dict with keys:
                - tool: "answer"
                - result: Generated answer text
                - confidence: 0.85 (LLM) or 0.5 (fallback)
                - sources: Number of source shards
                - llm_provider: "ollama" or provider name
                - llm_model: Model name
                - usage: Token usage stats
                - context_tokens: Context token count
                - packing_stats: Beta wave packing statistics
        """
        # Use packed context if available (beta wave optimization)
        packed_ctx = context.metadata.get('packed_context') if context and hasattr(context, 'metadata') else None

        if packed_ctx:
            # Use optimized physics-based packed context
            # Format: query section + awareness section + memory section
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
            self.logger.debug(
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
            self.logger.debug(f"Using raw context: {len(shard_texts)} shards")

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
        if self.llm and hasattr(self.llm, 'is_available') and self.llm.is_available():
            try:
                response = await self.llm.generate(
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
                self.logger.error(f"LLM generation failed: {e}")
                # Fall through to fallback

        # Fallback (LLM unavailable)
        self.logger.warning("Using fallback response (LLM unavailable)")
        return {
            "tool": "answer",
            "result": f"[Fallback] Generated answer for: {query.text}\n\nContext: {llm_context[:300]}...",
            "confidence": 0.5,
            "sources": len(context.shards) if context and hasattr(context, 'shards') else 0,
            "context_preview": llm_context[:200] + "..." if len(llm_context) > 200 else llm_context,
            "context_tokens": context_tokens,
            "packing_stats": packing_stats
        }

    async def _handle_search(self, query: Query, context: Context) -> Dict:
        """
        Perform a search operation.

        Stub implementation - replace with actual search logic.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Dict with search results
        """
        return {
            "tool": "search",
            "result": "Search results based on query",
            "sources": ["source1", "source2", "source3"],
            "count": 3
        }

    async def _handle_notion_write(self, query: Query, context: Context) -> Dict:
        """
        Write to Notion database.

        Stub implementation - replace with actual Notion API integration.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Dict with write status
        """
        return {
            "tool": "notion_write",
            "result": "Successfully wrote to Notion database",
            "status": "success",
            "page_id": "mock_page_123"
        }

    async def _handle_calc(self, query: Query, context: Context) -> Dict:
        """
        Perform calculation.

        Stub implementation - replace with actual calculation logic.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Dict with calculation result
        """
        return {
            "tool": "calc",
            "result": "Calculation completed",
            "value": 42,
            "expression": "mock_calculation"
        }

    async def _handle_unknown(self, query: Query, context: Context) -> Dict:
        """
        Handle unknown or unimplemented tool.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Dict with error status
        """
        return {
            "tool": "unknown",
            "result": "Unknown tool",
            "error": "Tool not implemented",
            "status": "error"
        }
