"""
GovernanceMiddleware Protocol — the contract every middleware implements.
"""

from __future__ import annotations

from typing import Callable, Awaitable, Protocol, runtime_checkable

from .context import GovernanceContext

# Type alias for the next-middleware callable
NextFn = Callable[[GovernanceContext], Awaitable[GovernanceContext]]


@runtime_checkable
class GovernanceMiddleware(Protocol):
    """Async middleware in the governance chain.

    Each middleware receives the context and a `next` callable.
    Call `await next(ctx)` to continue the chain, or raise
    an HTTPException to short-circuit.
    """

    name: str

    async def __call__(
        self, ctx: GovernanceContext, next: NextFn
    ) -> GovernanceContext: ...
