"""
Promptly RAG Integration

Dedicated RAG (Retrieval-Augmented Generation) integration for context-aware prompts.
Provides retrieval, enhancement, and execution with RAG-injected context.

Usage:
    from promptly.integrations.rag_integration import PromptRAGIntegration

    rag = PromptRAGIntegration(db)

    # Retrieve relevant context
    contexts = await rag.retrieve_context("prompt content", "summarization", k=5)

    # Enhance prompt with context
    enhanced = await rag.enhance_prompt("my_prompt", {"text": "..."})

    # Execute with RAG
    result = await rag.execute_with_rag("my_prompt", {"text": "..."}, "summarization")
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import asyncio
import re


@dataclass
class RAGContext:
    """Context item retrieved for RAG enhancement."""
    content: str
    source: str  # e.g., "execution:123", "pattern:factual", "knowledge:thompson_sampling"
    relevance: float  # 0.0-1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "relevance": self.relevance,
            "metadata": self.metadata
        }


@dataclass
class RAGEnhancedPrompt:
    """Result of prompt enhancement with RAG context."""
    original_prompt: str
    enhanced_prompt: str
    context_items: List[RAGContext]
    injection_method: str  # "prefix", "suffix", "inline"
    total_context_tokens: int
    avg_relevance: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_prompt": self.original_prompt,
            "enhanced_prompt": self.enhanced_prompt,
            "context_items": [c.to_dict() for c in self.context_items],
            "injection_method": self.injection_method,
            "total_context_tokens": self.total_context_tokens,
            "avg_relevance": self.avg_relevance
        }


@dataclass
class RAGExecutionResult:
    """Result of RAG-enhanced prompt execution."""
    prompt_name: str
    response: str
    confidence: float
    context_used: List[RAGContext]
    execution_time_ms: float
    rag_overhead_ms: float
    quality_boost: float  # Estimated quality improvement from RAG
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_name": self.prompt_name,
            "response": self.response,
            "confidence": self.confidence,
            "context_used": [c.to_dict() for c in self.context_used],
            "execution_time_ms": self.execution_time_ms,
            "rag_overhead_ms": self.rag_overhead_ms,
            "quality_boost": self.quality_boost,
            "metadata": self.metadata
        }


class PromptRAGIntegration:
    """
    RAG integration for context-aware prompt execution.

    Retrieves relevant context from multiple sources:
    - Past prompt executions with high quality scores
    - Task-type specific patterns and best practices
    - Domain knowledge from hololoom memory

    Context is injected using configurable strategies:
    - prefix: Context appears before the main prompt
    - suffix: Context appears after the main prompt
    - inline: Context is woven into the prompt structure
    """

    def __init__(
        self,
        db=None,
        hololoom_bridge=None,
        default_context_k: int = 5,
        min_relevance_threshold: float = 0.3,
        max_context_tokens: int = 2000,
        injection_method: str = "prefix"
    ):
        """
        Initialize RAG integration.

        Args:
            db: Promptly database connection
            hololoom_bridge: HoloLoomBridge instance for memory access
            default_context_k: Default number of context items to retrieve
            min_relevance_threshold: Minimum relevance score for context inclusion
            max_context_tokens: Maximum tokens to use for context
            injection_method: How to inject context ("prefix", "suffix", "inline")
        """
        self.db = db
        self._hololoom_bridge = hololoom_bridge
        self.default_context_k = default_context_k
        self.min_relevance_threshold = min_relevance_threshold
        self.max_context_tokens = max_context_tokens
        self.injection_method = injection_method

        # Cache for recently retrieved contexts
        self._context_cache: Dict[str, List[RAGContext]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_timestamps: Dict[str, float] = {}

    async def retrieve_context(
        self,
        prompt_content: str,
        task_type: Optional[str] = None,
        k: int = None
    ) -> List[RAGContext]:
        """
        Retrieve relevant context for a prompt.

        Searches multiple sources:
        1. High-quality past executions for similar prompts
        2. Task-type specific patterns and examples
        3. Domain knowledge from hololoom memory (if available)

        Args:
            prompt_content: The prompt content to find context for
            task_type: Optional task type for targeted retrieval
            k: Number of context items to retrieve

        Returns:
            List of RAGContext items sorted by relevance
        """
        k = k or self.default_context_k
        contexts: List[RAGContext] = []

        # Check cache first
        cache_key = f"{hash(prompt_content)}:{task_type}:{k}"
        if self._is_cache_valid(cache_key):
            return self._context_cache[cache_key]

        # Source 1: Past executions from database
        if self.db:
            execution_contexts = await self._retrieve_from_executions(
                prompt_content, task_type, k
            )
            contexts.extend(execution_contexts)

        # Source 2: Task-type patterns
        if task_type:
            pattern_contexts = await self._retrieve_task_patterns(task_type, k)
            contexts.extend(pattern_contexts)

        # Source 3: HoloLoom memory
        if self._hololoom_bridge:
            try:
                memory_contexts = await self._retrieve_from_hololoom(
                    prompt_content, task_type, k
                )
                contexts.extend(memory_contexts)
            except Exception:
                pass  # Continue without HoloLoom context

        # Filter by relevance threshold
        contexts = [c for c in contexts if c.relevance >= self.min_relevance_threshold]

        # Sort by relevance and limit
        contexts.sort(key=lambda c: c.relevance, reverse=True)
        contexts = contexts[:k]

        # Update cache
        self._context_cache[cache_key] = contexts
        self._cache_timestamps[cache_key] = datetime.now().timestamp()

        return contexts

    async def enhance_prompt(
        self,
        prompt_name: str,
        variables: Optional[Dict[str, Any]] = None,
        inject_context: bool = True,
        context_k: int = None
    ) -> RAGEnhancedPrompt:
        """
        Enhance a prompt with RAG-retrieved context.

        Args:
            prompt_name: Name of the prompt to enhance
            variables: Variables to substitute in the prompt
            inject_context: Whether to inject retrieved context
            context_k: Number of context items to retrieve

        Returns:
            RAGEnhancedPrompt with original, enhanced prompt, and context
        """
        # Get prompt content
        prompt_content = await self._get_prompt_content(prompt_name, variables)

        if not inject_context:
            return RAGEnhancedPrompt(
                original_prompt=prompt_content,
                enhanced_prompt=prompt_content,
                context_items=[],
                injection_method=self.injection_method,
                total_context_tokens=0,
                avg_relevance=0.0
            )

        # Get task type from prompt metadata
        task_type = await self._get_prompt_task_type(prompt_name)

        # Retrieve context
        context_items = await self.retrieve_context(
            prompt_content, task_type, context_k or self.default_context_k
        )

        if not context_items:
            return RAGEnhancedPrompt(
                original_prompt=prompt_content,
                enhanced_prompt=prompt_content,
                context_items=[],
                injection_method=self.injection_method,
                total_context_tokens=0,
                avg_relevance=0.0
            )

        # Build context string respecting token limit
        context_string, used_items = self._build_context_string(context_items)

        # Inject context into prompt
        enhanced_prompt = self._inject_context(prompt_content, context_string)

        # Calculate metrics
        total_tokens = self._estimate_tokens(context_string)
        avg_relevance = sum(c.relevance for c in used_items) / len(used_items)

        return RAGEnhancedPrompt(
            original_prompt=prompt_content,
            enhanced_prompt=enhanced_prompt,
            context_items=used_items,
            injection_method=self.injection_method,
            total_context_tokens=total_tokens,
            avg_relevance=avg_relevance
        )

    async def execute_with_rag(
        self,
        prompt_name: str,
        variables: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None,
        llm_executor: Optional[Callable] = None
    ) -> RAGExecutionResult:
        """
        Execute a prompt with RAG-enhanced context.

        Args:
            prompt_name: Name of the prompt to execute
            variables: Variables to substitute in the prompt
            task_type: Task type for targeted context retrieval
            llm_executor: Optional LLM execution function

        Returns:
            RAGExecutionResult with response and metrics
        """
        start_time = datetime.now()

        # Enhance prompt with RAG
        enhanced = await self.enhance_prompt(prompt_name, variables, inject_context=True)

        rag_end_time = datetime.now()
        rag_overhead_ms = (rag_end_time - start_time).total_seconds() * 1000

        # Execute the enhanced prompt
        if llm_executor:
            response = await llm_executor(enhanced.enhanced_prompt)
        else:
            # Placeholder response for when no LLM is configured
            response = f"[RAG-Enhanced Response for {prompt_name}]"

        end_time = datetime.now()
        execution_time_ms = (end_time - start_time).total_seconds() * 1000

        # Estimate quality boost from context
        quality_boost = self._estimate_quality_boost(enhanced.context_items)

        # Calculate confidence based on context quality
        confidence = min(0.95, 0.6 + (enhanced.avg_relevance * 0.35) + (quality_boost * 0.1))

        return RAGExecutionResult(
            prompt_name=prompt_name,
            response=response,
            confidence=confidence,
            context_used=enhanced.context_items,
            execution_time_ms=execution_time_ms,
            rag_overhead_ms=rag_overhead_ms,
            quality_boost=quality_boost,
            metadata={
                "injection_method": enhanced.injection_method,
                "total_context_tokens": enhanced.total_context_tokens,
                "task_type": task_type
            }
        )

    # ==================== Private Methods ====================

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached context is still valid."""
        if cache_key not in self._context_cache:
            return False

        timestamp = self._cache_timestamps.get(cache_key, 0)
        return (datetime.now().timestamp() - timestamp) < self._cache_ttl

    async def _retrieve_from_executions(
        self,
        prompt_content: str,
        task_type: Optional[str],
        k: int
    ) -> List[RAGContext]:
        """Retrieve context from past prompt executions."""
        contexts = []

        if not self.db:
            return contexts

        try:
            # Query high-quality executions
            query = """
                SELECT pe.id, pe.prompt_id, pe.output, pe.quality_score,
                       pe.task_type, pe.metadata, p.name as prompt_name
                FROM prompt_executions pe
                JOIN prompts p ON pe.prompt_id = p.id
                WHERE pe.quality_score >= 0.7
            """
            params = []

            if task_type:
                query += " AND pe.task_type = ?"
                params.append(task_type)

            query += " ORDER BY pe.quality_score DESC LIMIT ?"
            params.append(k * 2)  # Get more to filter by relevance

            cursor = self.db.execute(query, params)
            rows = cursor.fetchall()

            # Calculate relevance based on content similarity
            prompt_words = set(prompt_content.lower().split())

            for row in rows:
                output = row[2] or ""
                output_words = set(output.lower().split())

                # Simple word overlap relevance
                overlap = len(prompt_words & output_words)
                relevance = min(1.0, overlap / max(len(prompt_words), 1) * 2)

                if relevance >= self.min_relevance_threshold:
                    contexts.append(RAGContext(
                        content=output[:500],  # Truncate long outputs
                        source=f"execution:{row[0]}",
                        relevance=relevance * row[3],  # Weight by quality score
                        metadata={
                            "prompt_name": row[6],
                            "quality_score": row[3],
                            "task_type": row[4]
                        }
                    ))
        except Exception:
            pass  # Continue without execution context

        return contexts

    async def _retrieve_task_patterns(
        self,
        task_type: str,
        k: int
    ) -> List[RAGContext]:
        """Retrieve task-type specific patterns and best practices."""
        # Built-in patterns for common task types
        patterns = {
            "summarization": [
                "Focus on key points and main arguments",
                "Maintain original meaning while reducing length",
                "Use clear, concise language"
            ],
            "code_review": [
                "Check for security vulnerabilities",
                "Verify error handling coverage",
                "Assess code maintainability and readability"
            ],
            "explanation": [
                "Start with high-level overview",
                "Use analogies for complex concepts",
                "Provide concrete examples"
            ],
            "creative": [
                "Balance originality with coherence",
                "Develop consistent voice and tone",
                "Build narrative tension appropriately"
            ],
            "analysis": [
                "Consider multiple perspectives",
                "Support claims with evidence",
                "Acknowledge limitations and uncertainties"
            ]
        }

        contexts = []
        task_patterns = patterns.get(task_type, [])

        for i, pattern in enumerate(task_patterns[:k]):
            contexts.append(RAGContext(
                content=pattern,
                source=f"pattern:{task_type}:{i}",
                relevance=0.7,  # Fixed relevance for patterns
                metadata={"task_type": task_type, "pattern_index": i}
            ))

        return contexts

    async def _retrieve_from_hololoom(
        self,
        prompt_content: str,
        task_type: Optional[str],
        k: int
    ) -> List[RAGContext]:
        """Retrieve context from hololoom memory."""
        contexts = []

        if not self._hololoom_bridge:
            return contexts

        try:
            # Use HoloLoom bridge to get context
            hololoom_contexts = await self._hololoom_bridge.get_context_for_prompt(
                prompt_content, task_type or "general", k
            )

            for ctx in hololoom_contexts:
                contexts.append(RAGContext(
                    content=ctx.get("content", ""),
                    source=f"hololoom:{ctx.get('id', 'unknown')}",
                    relevance=ctx.get("relevance", 0.5),
                    metadata=ctx.get("metadata", {})
                ))
        except Exception:
            pass  # Continue without HoloLoom context

        return contexts

    async def _get_prompt_content(
        self,
        prompt_name: str,
        variables: Optional[Dict[str, Any]]
    ) -> str:
        """Get prompt content with variable substitution."""
        content = ""

        if self.db:
            try:
                cursor = self.db.execute(
                    "SELECT content FROM prompts WHERE name = ?",
                    (prompt_name,)
                )
                row = cursor.fetchone()
                if row:
                    content = row[0]
            except Exception:
                pass

        if not content:
            content = f"[Prompt: {prompt_name}]"

        # Substitute variables
        if variables:
            for key, value in variables.items():
                content = content.replace(f"{{{key}}}", str(value))
                content = content.replace(f"{{{{ {key} }}}}", str(value))

        return content

    async def _get_prompt_task_type(self, prompt_name: str) -> Optional[str]:
        """Get task type from prompt metadata."""
        if not self.db:
            return None

        try:
            cursor = self.db.execute(
                "SELECT metadata FROM prompts WHERE name = ?",
                (prompt_name,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                import json
                metadata = json.loads(row[0])
                return metadata.get("task_type")
        except Exception:
            pass

        return None

    def _build_context_string(
        self,
        context_items: List[RAGContext]
    ) -> tuple[str, List[RAGContext]]:
        """Build context string respecting token limit."""
        used_items = []
        context_parts = []
        total_tokens = 0

        for item in context_items:
            item_tokens = self._estimate_tokens(item.content)

            if total_tokens + item_tokens > self.max_context_tokens:
                break

            context_parts.append(f"- {item.content}")
            used_items.append(item)
            total_tokens += item_tokens

        if context_parts:
            context_string = "Relevant context:\n" + "\n".join(context_parts)
        else:
            context_string = ""

        return context_string, used_items

    def _inject_context(self, prompt_content: str, context_string: str) -> str:
        """Inject context into prompt using configured method."""
        if not context_string:
            return prompt_content

        if self.injection_method == "prefix":
            return f"{context_string}\n\n{prompt_content}"
        elif self.injection_method == "suffix":
            return f"{prompt_content}\n\n{context_string}"
        elif self.injection_method == "inline":
            # Try to insert after first paragraph or sentence
            paragraphs = prompt_content.split("\n\n")
            if len(paragraphs) > 1:
                return f"{paragraphs[0]}\n\n{context_string}\n\n" + "\n\n".join(paragraphs[1:])
            else:
                return f"{prompt_content}\n\n{context_string}"
        else:
            return f"{context_string}\n\n{prompt_content}"

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple approximation: ~4 characters per token
        return len(text) // 4

    def _estimate_quality_boost(self, context_items: List[RAGContext]) -> float:
        """Estimate quality improvement from context."""
        if not context_items:
            return 0.0

        # Quality boost based on number and relevance of context items
        count_factor = min(1.0, len(context_items) / 5)  # Max boost at 5 items
        relevance_factor = sum(c.relevance for c in context_items) / len(context_items)

        return count_factor * relevance_factor * 0.3  # Max 30% boost


# ==================== Factory Functions ====================

def create_rag_integration(
    db=None,
    hololoom_bridge=None,
    **kwargs
) -> PromptRAGIntegration:
    """
    Create a PromptRAGIntegration instance.

    Args:
        db: Promptly database connection
        hololoom_bridge: HoloLoomBridge instance
        **kwargs: Additional configuration options

    Returns:
        Configured PromptRAGIntegration instance
    """
    return PromptRAGIntegration(
        db=db,
        hololoom_bridge=hololoom_bridge,
        **kwargs
    )


async def retrieve_prompt_context(
    prompt_content: str,
    task_type: Optional[str] = None,
    k: int = 5,
    db=None,
    hololoom_bridge=None
) -> List[RAGContext]:
    """
    Convenience function to retrieve context for a prompt.

    Args:
        prompt_content: The prompt content
        task_type: Optional task type
        k: Number of context items
        db: Database connection
        hololoom_bridge: HoloLoom bridge instance

    Returns:
        List of RAGContext items
    """
    rag = create_rag_integration(db=db, hololoom_bridge=hololoom_bridge)
    return await rag.retrieve_context(prompt_content, task_type, k)


async def execute_prompt_with_rag(
    prompt_name: str,
    variables: Optional[Dict[str, Any]] = None,
    task_type: Optional[str] = None,
    db=None,
    hololoom_bridge=None,
    llm_executor: Optional[Callable] = None
) -> RAGExecutionResult:
    """
    Convenience function to execute a prompt with RAG.

    Args:
        prompt_name: Name of the prompt
        variables: Variables to substitute
        task_type: Task type for context retrieval
        db: Database connection
        hololoom_bridge: HoloLoom bridge instance
        llm_executor: LLM execution function

    Returns:
        RAGExecutionResult
    """
    rag = create_rag_integration(db=db, hololoom_bridge=hololoom_bridge)
    return await rag.execute_with_rag(
        prompt_name, variables, task_type, llm_executor
    )
