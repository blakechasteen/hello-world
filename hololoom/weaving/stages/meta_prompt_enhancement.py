"""
Meta-Prompt Enhancement Stage
==============================
Step 0 (optional): Enhance the query via LLM before pipeline processing.

Rewrites, clarifies, or expands the user query. Stores both original
and enhanced queries on the context.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class MetaPromptEnhancementStage:
    """
    Meta-Prompt Enhancement Stage (Step 0: Query Rewriting).

    Optionally enhances the query through an LLM call before the
    main pipeline processes it.
    """

    reads: ClassVar[frozenset[str]] = frozenset({'query', 'auto_enhance'})
    writes: ClassVar[frozenset[str]] = frozenset({'enhanced_query', 'enhancement_applied'})
    cacheable: ClassVar[bool] = True

    def __init__(
        self,
        proto_llm_call: Callable | None = None,
        enable_enhancement: bool = True,
        logger: logging.Logger | None = None,
    ):
        self.proto_llm_call = proto_llm_call
        self.enable_enhancement = enable_enhancement
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: 'WeavingContext') -> None:
        """Enhance query via LLM, writing enhanced_query to ctx."""
        should_enhance = ctx.auto_enhance if ctx.auto_enhance is not None else self.enable_enhancement

        if not should_enhance or self.proto_llm_call is None:
            ctx.enhancement_applied = False
            return

        try:
            from hololoom.protocols.types import Query

            original_text = ctx.query.text
            enhanced_text = await self.proto_llm_call(original_text)

            enhanced_metadata = dict(ctx.query.metadata) if ctx.query.metadata else {}
            enhanced_metadata['original_query'] = original_text
            enhanced_metadata['enhanced'] = True

            ctx.enhanced_query = Query(text=enhanced_text, metadata=enhanced_metadata)
            ctx.enhancement_applied = True

            self.logger.info(
                f"  [0] Enhanced query ({len(original_text)} -> {len(enhanced_text)} chars)"
            )

        except (TypeError, AttributeError, ValueError, RuntimeError) as e:
            self.logger.warning(f"  [0] Auto-enhancement failed: {e}")
            ctx.enhancement_applied = False

    def get_stage_name(self) -> str:
        return "meta_prompt_enhancement"

    def get_required_context_keys(self) -> list[str]:
        return []

    def get_output_context_keys(self) -> list[str]:
        return ["enhanced_query", "enhancement_applied"]
