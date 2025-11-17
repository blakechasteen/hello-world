#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool Executor - Executes tools based on convergence decisions
==============================================================

Handles the execution of tools selected by the convergence engine.
Supports multiple tool types: answer, search, notion_write, calc.

**Extracted: November 2025** - Refactored from weaving_orchestrator.py
**Lines: 167** (reduced from 3,476-line monolith)

Author: Claude Code (refactoring pass)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, TYPE_CHECKING

from HoloLoom.Documentation.types import Query, Context

if TYPE_CHECKING:
    from HoloLoom.awareness.llm_integration import OllamaLLM


class ToolExecutor:
    """
    Executes tools based on convergence engine decisions.

    This class handles the execution of tools selected by the convergence
    engine. In production, this would call actual APIs, databases, etc.

    **Supported Tools:**
    - answer: Generate an answer using LLM with context
    - search: Perform a search operation
    - notion_write: Write to Notion database
    - calc: Perform calculations

    **LLM Integration:**
    The executor uses Ollama LLM for answer generation, with graceful
    fallback to mock responses if LLM is unavailable.

    **Context Packing:**
    Supports physics-based packed context (beta wave optimization) for
    efficient token usage.

    Example:
        >>> executor = ToolExecutor()
        >>> result = await executor.execute("answer", query, context)
        >>> print(result['result'])

    Args:
        llm: Optional pre-initialized LLM instance. If None, will attempt
             to lazy-load Ollama LLM.
    """

    def __init__(self, llm: Optional['OllamaLLM'] = None):
        """
        Initialize the tool executor.

        Args:
            llm: Optional LLM instance. If None, attempts lazy loading.
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

        Routes the execution to the appropriate tool handler based on
        the tool name from the convergence result.

        Args:
            tool: Tool name from CollapseResult (e.g., "answer", "search")
            query: Original query object
            context: Retrieved context with shards and metadata

        Returns:
            Dict with execution results containing:
                - tool: Tool name
                - result: Tool-specific result
                - Additional tool-specific fields (confidence, sources, etc.)

        Example:
            >>> result = await executor.execute("answer", query, context)
            >>> print(f"Confidence: {result['confidence']}")
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
        Generate an answer based on context using LLM.

        Supports two context formats:
        1. **Packed context** (physics-based beta wave optimization)
        2. **Raw shards** (legacy fallback)

        The LLM is prompted with context and generates a response.
        Falls back to mock response if LLM unavailable.

        Args:
            query: User query
            context: Retrieved context with shards or packed_context

        Returns:
            Dict with answer results:
                - result: Generated answer text
                - confidence: Answer confidence (0.5-0.85)
                - sources: Number of context shards used
                - llm_provider: LLM provider (e.g., "ollama")
                - llm_model: Model name
                - usage: Token usage statistics
                - context_tokens: Tokens in context
                - packing_stats: Context packing statistics
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
                    "result": response.content,  # Actual LLM response
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

        Args:
            query: Search query
            context: Retrieved context

        Returns:
            Dict with search results:
                - result: Search result description
                - sources: List of source URLs/IDs
                - count: Number of results
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

        Args:
            query: Write request
            context: Retrieved context

        Returns:
            Dict with write results:
                - result: Success message
                - status: Operation status
                - page_id: Created/updated page ID
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

        Args:
            query: Calculation request
            context: Retrieved context

        Returns:
            Dict with calculation results:
                - result: Result description
                - value: Calculated value
                - expression: Evaluated expression
        """
        return {
            "tool": "calc",
            "result": "Calculation completed",
            "value": 42,
            "expression": "mock_calculation"
        }

    async def _handle_unknown(self, query: Query, context: Context) -> Dict:
        """
        Handle unknown tool requests.

        Args:
            query: Original query
            context: Retrieved context

        Returns:
            Dict with error information:
                - tool: "unknown"
                - result: Error message
                - error: Error description
                - status: "error"
        """
        return {
            "tool": "unknown",
            "result": "Unknown tool",
            "error": "Tool not implemented",
            "status": "error"
        }
