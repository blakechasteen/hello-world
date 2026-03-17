"""
Jenny Compilation Stage
=======================
Post-pipeline visualization panel compilation.

Compiles Spacetime results into Jenny panel specifications for
multi-target rendering (HTML, React, Terminal, AR/VR).
Side-effect stage — no context writes.
"""

import logging
from typing import Any, ClassVar, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class JennyCompilationStage:
    """
    Jenny Compilation Stage (Visualization).

    Compiles pipeline results into Jenny panel specifications.
    Side-effect stage — writes to external Jenny runtime, not context.
    """

    reads: ClassVar[frozenset[str]] = frozenset({'spacetime', 'features'})
    writes: ClassVar[frozenset[str]] = frozenset()  # Side-effect only
    cacheable: ClassVar[bool] = False  # Side-effect stage

    def __init__(
        self,
        jenny_runtime: Optional[Any] = None,
        panel_count: int = 4,
        logger: Optional[logging.Logger] = None,
    ):
        self.jenny_runtime = jenny_runtime
        self.panel_count = panel_count
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: 'WeavingContext') -> None:
        """Compile spacetime into Jenny panels."""
        if self.jenny_runtime is None:
            return

        spacetime = getattr(ctx, 'spacetime', None)
        if spacetime is None:
            return

        features = getattr(ctx, 'features', None)

        try:
            panels = await self.jenny_runtime.compile_spacetime(
                spacetime=spacetime,
                features=features,
                panel_count=self.panel_count,
            )
            self.logger.info(f"  [J] Jenny compiled {len(panels)} panels")

        except Exception as e:
            self.logger.warning(f"  [J] Jenny compilation failed: {e}")

    def get_stage_name(self) -> str:
        return "jenny_compilation"

    def get_required_context_keys(self) -> list[str]:
        return ["spacetime"]

    def get_output_context_keys(self) -> list[str]:
        return []
