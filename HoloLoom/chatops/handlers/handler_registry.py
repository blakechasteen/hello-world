"""
Handler Registry - Decorator-based ChatOps handler registration.

Provides clean architecture for ChatOps command handlers with:
- Decorator-based registration
- Auto-generated help from metadata
- Category grouping (query, system, learning, admin)
- Command aliases (!w -> !weave)
- Auth requirements per handler

Usage:
    from HoloLoom.chatops.handlers.handler_registry import (
        HandlerRegistry, HandlerCategory, chatops_handler
    )

    class MyHandlers:
        @chatops_handler(
            command="weave",
            description="Query HoloLoom",
            usage="!weave <question>",
            category=HandlerCategory.QUERY,
            aliases=["w"]
        )
        async def handle_weave(self, room, event, args: str):
            ...

    # Auto-dispatch
    registry = HandlerRegistry()
    registry.register_instance(MyHandlers())
    await registry.dispatch(command, room, event, args)

Created: 2025-12-05
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum
import asyncio
import functools


class HandlerCategory(Enum):
    """Categories for grouping handlers in help output."""
    QUERY = "query"         # !weave, !memory, !analyze
    SYSTEM = "system"       # !stats, !ping, !help
    LEARNING = "learning"   # !learn, !trace
    ADMIN = "admin"         # !audit
    PORTAL = "portal"       # Portal-specific commands


@dataclass
class HandlerSpec:
    """Specification for a registered handler."""
    command: str                  # Primary command name (without !)
    handler: Callable             # Async function reference
    description: str              # Short description for help
    usage: str = ""               # Full usage string
    category: HandlerCategory = HandlerCategory.QUERY
    aliases: List[str] = field(default_factory=list)
    requires_auth: bool = False   # Require authenticated user
    requires_room_admin: bool = False  # Require room admin
    hidden: bool = False          # Hide from help output
    cooldown_seconds: float = 0.0  # Per-user cooldown

    def __post_init__(self):
        if not self.usage:
            self.usage = f"!{self.command}"


class HandlerRegistry:
    """
    Central registry for ChatOps command handlers.

    Supports both class-based (decorator on methods) and function-based handlers.
    """

    # Class-level storage for decorated handlers
    _pending_handlers: Dict[str, HandlerSpec] = {}
    _instances: List[Any] = []

    def __init__(self):
        self._handlers: Dict[str, HandlerSpec] = {}
        self._command_to_spec: Dict[str, str] = {}  # Maps aliases to primary command
        self._cooldowns: Dict[str, Dict[str, float]] = {}  # command -> user_id -> last_used

    @classmethod
    def register(
        cls,
        command: str,
        description: str,
        usage: str = "",
        category: HandlerCategory = HandlerCategory.QUERY,
        aliases: Optional[List[str]] = None,
        requires_auth: bool = False,
        requires_room_admin: bool = False,
        hidden: bool = False,
        cooldown_seconds: float = 0.0
    ) -> Callable:
        """
        Decorator to register a handler method.

        Example:
            @HandlerRegistry.register(
                command="weave",
                description="Query HoloLoom with full weaving cycle",
                usage="!weave <your question>",
                category=HandlerCategory.QUERY,
                aliases=["w"]
            )
            async def handle_weave(self, room, event, args: str):
                ...
        """
        def decorator(func: Callable) -> Callable:
            spec = HandlerSpec(
                command=command,
                handler=func,
                description=description,
                usage=usage or f"!{command}",
                category=category,
                aliases=aliases or [],
                requires_auth=requires_auth,
                requires_room_admin=requires_room_admin,
                hidden=hidden,
                cooldown_seconds=cooldown_seconds
            )

            # Store in pending handlers (class-level)
            cls._pending_handlers[command] = spec

            # Mark function with metadata for instance binding
            func._handler_spec = spec

            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            wrapper._handler_spec = spec
            return wrapper

        return decorator

    def register_instance(self, instance: Any) -> int:
        """
        Register all decorated handlers from a class instance.

        Returns count of handlers registered.
        """
        count = 0

        for attr_name in dir(instance):
            if attr_name.startswith('_'):
                continue

            attr = getattr(instance, attr_name, None)
            if attr and hasattr(attr, '_handler_spec'):
                spec: HandlerSpec = attr._handler_spec

                # Create bound spec
                bound_spec = HandlerSpec(
                    command=spec.command,
                    handler=attr,  # Bound method
                    description=spec.description,
                    usage=spec.usage,
                    category=spec.category,
                    aliases=list(spec.aliases),
                    requires_auth=spec.requires_auth,
                    requires_room_admin=spec.requires_room_admin,
                    hidden=spec.hidden,
                    cooldown_seconds=spec.cooldown_seconds
                )

                # Register primary command
                self._handlers[spec.command] = bound_spec
                self._command_to_spec[spec.command] = spec.command

                # Register aliases
                for alias in spec.aliases:
                    self._command_to_spec[alias] = spec.command

                count += 1

        self._instances.append(instance)
        return count

    def register_function(
        self,
        command: str,
        handler: Callable,
        description: str,
        usage: str = "",
        category: HandlerCategory = HandlerCategory.QUERY,
        aliases: Optional[List[str]] = None,
        requires_auth: bool = False,
        hidden: bool = False
    ) -> None:
        """Register a standalone function as a handler."""
        spec = HandlerSpec(
            command=command,
            handler=handler,
            description=description,
            usage=usage or f"!{command}",
            category=category,
            aliases=aliases or [],
            requires_auth=requires_auth,
            hidden=hidden
        )

        self._handlers[command] = spec
        self._command_to_spec[command] = command

        for alias in (aliases or []):
            self._command_to_spec[alias] = command

    def get_handler(self, command: str) -> Optional[HandlerSpec]:
        """Get handler spec for a command (resolves aliases)."""
        primary = self._command_to_spec.get(command)
        if primary:
            return self._handlers.get(primary)
        return None

    def get_all_commands(self) -> List[str]:
        """Get all registered commands (primary only, no aliases)."""
        return list(self._handlers.keys())

    def get_commands_by_category(self) -> Dict[HandlerCategory, List[HandlerSpec]]:
        """Get handlers grouped by category."""
        result: Dict[HandlerCategory, List[HandlerSpec]] = {
            cat: [] for cat in HandlerCategory
        }

        for spec in self._handlers.values():
            if not spec.hidden:
                result[spec.category].append(spec)

        return result

    async def dispatch(
        self,
        command: str,
        room: Any,
        event: Any,
        args: str,
        user_id: Optional[str] = None
    ) -> Optional[Any]:
        """
        Dispatch a command to its registered handler.

        Returns:
            Handler result, or None if command not found

        Raises:
            PermissionError: If auth required but user not authorized
            RuntimeError: If cooldown not expired
        """
        spec = self.get_handler(command)
        if not spec:
            return None

        # Check auth requirements
        if spec.requires_auth and not user_id:
            raise PermissionError(f"Command !{command} requires authentication")

        # Check cooldown
        if spec.cooldown_seconds > 0 and user_id:
            last_used = self._cooldowns.get(command, {}).get(user_id, 0)
            now = asyncio.get_event_loop().time()

            if now - last_used < spec.cooldown_seconds:
                remaining = spec.cooldown_seconds - (now - last_used)
                raise RuntimeError(f"Cooldown active. Try again in {remaining:.1f}s")

            # Update cooldown
            if command not in self._cooldowns:
                self._cooldowns[command] = {}
            self._cooldowns[command][user_id] = now

        # Execute handler
        return await spec.handler(room, event, args)

    def generate_help(
        self,
        prefix: str = "!",
        show_aliases: bool = True,
        show_usage: bool = True,
        filter_category: Optional[HandlerCategory] = None
    ) -> str:
        """
        Generate help text from registered handlers.

        Returns formatted help string grouped by category.
        """
        lines = ["**Available Commands**\n"]

        by_category = self.get_commands_by_category()

        for category in HandlerCategory:
            if filter_category and category != filter_category:
                continue

            handlers = by_category.get(category, [])
            if not handlers:
                continue

            # Category header
            lines.append(f"\n**{category.value.title()}**")

            for spec in sorted(handlers, key=lambda s: s.command):
                # Command line
                cmd_str = f"`{prefix}{spec.command}`"

                if show_aliases and spec.aliases:
                    alias_str = ", ".join(f"`{prefix}{a}`" for a in spec.aliases)
                    cmd_str += f" (aliases: {alias_str})"

                lines.append(f"  {cmd_str} - {spec.description}")

                if show_usage and spec.usage != f"{prefix}{spec.command}":
                    lines.append(f"    Usage: `{spec.usage}`")

        return "\n".join(lines)

    def generate_help_for_command(self, command: str, prefix: str = "!") -> Optional[str]:
        """Generate detailed help for a specific command."""
        spec = self.get_handler(command)
        if not spec:
            return None

        lines = [
            f"**{prefix}{spec.command}**",
            f"",
            f"{spec.description}",
            f"",
            f"**Usage:** `{spec.usage}`"
        ]

        if spec.aliases:
            lines.append(f"**Aliases:** {', '.join(f'`{prefix}{a}`' for a in spec.aliases)}")

        lines.append(f"**Category:** {spec.category.value}")

        if spec.requires_auth:
            lines.append("**Requires:** Authentication")

        if spec.requires_room_admin:
            lines.append("**Requires:** Room admin")

        if spec.cooldown_seconds > 0:
            lines.append(f"**Cooldown:** {spec.cooldown_seconds}s")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()
        self._command_to_spec.clear()
        self._cooldowns.clear()
        self._instances.clear()


# Convenience alias for the decorator
chatops_handler = HandlerRegistry.register


# Global registry instance (optional, for simple use cases)
_global_registry: Optional[HandlerRegistry] = None


def get_global_registry() -> HandlerRegistry:
    """Get or create the global handler registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = HandlerRegistry()
    return _global_registry


def register_handlers_from(instance: Any) -> int:
    """Register all handlers from an instance to the global registry."""
    return get_global_registry().register_instance(instance)
