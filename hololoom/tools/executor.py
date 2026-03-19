#!/usr/bin/env python3
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
import os
import re
from typing import TYPE_CHECKING

from hololoom.fabric.spacetime import Artifact, ArtifactType
from hololoom.protocols.types import Context, Query

if TYPE_CHECKING:
    from hololoom.awareness.llm_integration import OllamaLLM

# Skills integration (November 2025)
try:
    from hololoom.tools.skills_bridge import get_skills_bridge
    SKILLS_AVAILABLE = True
except ImportError:
    SKILLS_AVAILABLE = False


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

    def __init__(self, llm: OllamaLLM | None = None, enable_skills: bool = True):
        """
        Initialize the tool executor.

        Args:
            llm: Optional pre-initialized LLM instance.
                If None, will attempt to lazy-load Ollama LLM.
            enable_skills: Enable skills system integration (default: True)

        Note:
            LLM initialization gracefully fails if dependencies unavailable.
            Executor falls back to mock responses.
            Skills integration gracefully degrades if unavailable.
        """
        self.logger = logging.getLogger(__name__)
        disable_skills = os.getenv("HOLOLOOM_DISABLE_SKILLS", "").lower() in {"1", "true", "yes"}
        if disable_skills:
            enable_skills = False
        self.require_llm = os.getenv("HOLOLOOM_REQUIRE_LLM", "").lower() in {"1", "true", "yes"}
        model_name = os.getenv("HOLOLOOM_OLLAMA_MODEL", "llama3.2:3b")

        # Core tools
        self.core_tools = ["answer", "search", "notion_write", "calc"]

        # Initialize skills bridge
        self.skills_bridge = None
        if enable_skills and SKILLS_AVAILABLE:
            try:
                self.skills_bridge = get_skills_bridge()
                skill_tools = self.skills_bridge.get_skill_tools()
                self.logger.info(
                    f"Skills integration enabled: {len(skill_tools)} skills available"
                )
            except Exception as e:
                self.logger.warning(f"Skills integration failed: {e}")
                self.skills_bridge = None

        # Combined tools list (core + skills)
        self.tools = self.core_tools.copy()
        if self.skills_bridge:
            self.tools.extend(self.skills_bridge.get_skill_tools())

        # Initialize LLM (lazy loading)
        self.llm = llm
        if self.llm is None:
            try:
                from hololoom.awareness.llm_integration import OllamaLLM
                self.llm = OllamaLLM(model=model_name)
                self.logger.info(f"Initialized Ollama LLM ({model_name})")
            except Exception as e:
                self.logger.warning(f"LLM unavailable, using fallback: {e}")
                self.llm = None

    async def execute(self, tool: str, query: Query, context: Context) -> dict:
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

        # Check if this is a skill tool
        if self.skills_bridge and tool not in self.core_tools:
            # Route to skills system
            return await self._handle_skill(tool, query, context)

        # Core tool implementations
        tool_handlers = {
            "answer": self._handle_answer,
            "search": self._handle_search,
            "notion_write": self._handle_notion_write,
            "calc": self._handle_calc
        }

        handler = tool_handlers.get(tool, self._handle_unknown)
        return await handler(query, context)

    async def _handle_answer(self, query: Query, context: Context) -> dict:
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
        if self.require_llm:
            if not self.llm or not hasattr(self.llm, 'is_available') or not self.llm.is_available():
                raise RuntimeError(
                    "LLM required but unavailable. Start Ollama and pull the model "
                    f"'{os.getenv('HOLOLOOM_OLLAMA_MODEL', 'llama3.2:3b')}'."
                )

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
                    "packing_stats": packing_stats,
                    "artifacts": self._extract_artifacts(response.content)
                }

            except Exception as e:
                self.logger.error(f"LLM generation failed: {e}")
                if self.require_llm:
                    raise RuntimeError("LLM required but generation failed.") from e
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
            "packing_stats": packing_stats,
            "artifacts": self._extract_artifacts(f"[Fallback] Generated answer for: {query.text}\n\nContext: {llm_context[:300]}...") # unlikely to find artifacts here but consistent
        }


    async def _handle_search(self, query: Query, context: Context) -> dict:
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

    async def _handle_notion_write(self, query: Query, context: Context) -> dict:
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

    async def _handle_calc(self, query: Query, context: Context) -> dict:
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

    async def _handle_unknown(self, query: Query, context: Context) -> dict:
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

    async def _handle_skill(self, skill_name: str, query: Query, context: Context) -> dict:
        """
        Handle skill execution.

        Routes skill calls to the skills bridge for execution.
        Automatically determines operation and parameters from query/context.

        Args:
            skill_name: Name of the skill to execute
            query: User query (may contain operation/parameters)
            context: Retrieved context

        Returns:
            Dict with skill execution results

        Note:
            This is a simplified handler. For more control, skills can be
            called directly via the skills bridge with explicit operation
            and parameters.
        """
        # Extract operation and parameters from query
        # For now, use default operation and minimal parameters
        # In production, this could be enhanced with NLU to extract structured params

        # Default operation (most skills have a primary operation)
        operation = "execute"  # Generic operation name

        # Build parameters from query text
        parameters = {
            "input": query.text,
            "query": query.text
        }

        # Add context if available
        if context and hasattr(context, 'shard_texts'):
            parameters["context"] = context.shard_texts[:3]  # First 3 shards

        try:
            result = await self.skills_bridge.execute_skill(
                skill_name=skill_name,
                operation=operation,
                parameters=parameters,
                query=query,
                context=context,

                use_cache=True
            )

            # Extract artifacts if result is text-based
            if isinstance(result, dict) and 'result' in result and isinstance(result['result'], str):
                artifacts = self._extract_artifacts(result['result'])
                if artifacts:
                    result['artifacts'] = artifacts

            return result

        except ValueError as e:
            # Skill not found or invalid operation
            self.logger.error(f"Skill execution failed: {e}")
            return {
                "tool": skill_name,
                "result": None,
                "status": "error",
                "error": str(e),
                "confidence": 0.0
            }

        except Exception as e:
            # Other execution errors
            self.logger.error(f"Skill execution error: {e}", exc_info=True)
            return {
                "tool": skill_name,
                "result": None,
                "status": "error",
                "error": f"Execution failed: {e}",
                "confidence": 0.0
            }

    def _extract_artifacts(self, text: str) -> list[dict]:
        """
        Extract artifacts from text response.
        
        Looks for code blocks with filename headers.
        """
        artifacts = []
        if not text:
            return artifacts

        # Pattern 1: Code block with filename comment inside
        code_block_pattern = re.compile(r"```(?:\w+)?\n(?:#\s*filename:\s*([^\n]+)\n)(.*?)```", re.DOTALL)

        for match in code_block_pattern.finditer(text):
            filename = match.group(1).strip()
            content = match.group(2)

            # Simple heuristic to determine type
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.gif']:
                 atype = ArtifactType.IMAGE
            elif ext in ['.zip', '.tar', '.gz']:
                 atype = ArtifactType.ARCHIVE
            else:
                 atype = ArtifactType.CODE

            artifacts.append(Artifact(
                name=os.path.basename(filename),
                type=atype,
                content=content,
                destination_path=filename,
                metadata={'extracted_from': 'code_block'}
            ).to_dict())

        # Pattern 2: Explicit XML-like tags (optional)
        xml_pattern = re.compile(r"<artifact\s+name=\"([^\"]+)\"\s+type=\"([^\"]+)\">(.*?)</artifact>", re.DOTALL)
        for match in xml_pattern.finditer(text):
            filename = match.group(1).strip()
            art_type = match.group(2).strip()
            content = match.group(3).strip()

            # Map string type to enum if possible
            try:
                type_enum = ArtifactType(art_type)
            except ValueError:
                type_enum = ArtifactType.CODE # Default

            artifacts.append(Artifact(
                name=os.path.basename(filename),
                type=type_enum,
                content=content,
                destination_path=filename,
                metadata={'extracted_from': 'xml_tag'}
            ).to_dict())

        return artifacts
