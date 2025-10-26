#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom Commands Framework
==========================
Extensible command system for user-defined chatops commands.

Features:
- Dynamic command registration
- Parameter validation
- Access control (roles/permissions)
- Command aliases
- Help generation
- Hot-reload support

Usage:
    manager = CustomCommandManager()

    # Register command
    @manager.command(
        name="deploy",
        description="Deploy application",
        params=["environment"],
        admin_only=True
    )
    async def deploy_handler(ctx, environment):
        return f"Deploying to {environment}..."

    # Execute
    result = await manager.execute("deploy", ctx, "production")
"""

import logging
import inspect
import re
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import importlib.util
import sys

logger = logging.getLogger(__name__)


# ============================================================================
# Command Data Structures
# ============================================================================

@dataclass
class CommandParameter:
    """Command parameter definition."""
    name: str
    type: str = "str"  # str, int, float, bool, list
    required: bool = True
    default: Any = None
    description: str = ""
    choices: Optional[List[Any]] = None

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate parameter value.

        Returns:
            (is_valid, error_message)
        """
        # Check required
        if self.required and value is None:
            return False, f"Parameter '{self.name}' is required"

        if value is None:
            return True, None

        # Type conversion and validation
        try:
            if self.type == "int":
                value = int(value)
            elif self.type == "float":
                value = float(value)
            elif self.type == "bool":
                value = str(value).lower() in ("true", "1", "yes")
            elif self.type == "list":
                if isinstance(value, str):
                    value = [v.strip() for v in value.split(",")]

            # Check choices
            if self.choices and value not in self.choices:
                return False, f"Invalid choice for '{self.name}'. Must be one of: {self.choices}"

            return True, None

        except (ValueError, TypeError) as e:
            return False, f"Invalid type for '{self.name}': {str(e)}"


@dataclass
class CommandContext:
    """Context provided to command handlers."""
    user_id: str
    conversation_id: str
    message_id: str
    is_admin: bool = False
    room_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandDefinition:
    """Custom command definition."""
    name: str
    handler: Callable
    description: str = ""
    params: List[CommandParameter] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    admin_only: bool = False
    enabled: bool = True
    category: str = "general"
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def generate_help(self) -> str:
        """Generate help text for command."""
        lines = [f"**!{self.name}**"]

        if self.description:
            lines.append(f"{self.description}")

        if self.params:
            lines.append("\n**Parameters:**")
            for param in self.params:
                param_str = f"• `{param.name}`"
                if not param.required:
                    param_str += " (optional)"
                if param.description:
                    param_str += f" - {param.description}"
                if param.default is not None:
                    param_str += f" (default: {param.default})"
                lines.append(param_str)

        if self.aliases:
            lines.append(f"\n**Aliases:** {', '.join(f'!{a}' for a in self.aliases)}")

        if self.examples:
            lines.append("\n**Examples:**")
            for example in self.examples:
                lines.append(f"• `{example}`")

        if self.admin_only:
            lines.append("\n🔒 *Admin only*")

        return "\n".join(lines)


# ============================================================================
# Custom Command Manager
# ============================================================================

class CustomCommandManager:
    """
    Manages custom user-defined commands.

    Features:
    - Dynamic command registration
    - Parameter validation
    - Access control
    - Command aliases
    - Help generation
    - Hot-reload from files

    Usage:
        manager = CustomCommandManager()

        # Register via decorator
        @manager.command(name="hello", description="Say hello")
        async def hello_handler(ctx):
            return f"Hello, {ctx.user_id}!"

        # Register programmatically
        manager.register(
            name="status",
            handler=status_handler,
            description="System status"
        )

        # Execute command
        result = await manager.execute("hello", context)
    """

    def __init__(self):
        """Initialize command manager."""
        self.commands: Dict[str, CommandDefinition] = {}
        self.aliases: Dict[str, str] = {}  # alias -> command_name
        self.categories: Set[str] = set()

        logger.info("CustomCommandManager initialized")

    # ========================================================================
    # Command Registration
    # ========================================================================

    def command(
        self,
        name: str,
        description: str = "",
        params: Optional[List[Dict]] = None,
        aliases: Optional[List[str]] = None,
        admin_only: bool = False,
        category: str = "general",
        examples: Optional[List[str]] = None
    ):
        """
        Decorator for registering commands.

        Args:
            name: Command name
            description: Command description
            params: List of parameter definitions
            aliases: Command aliases
            admin_only: Require admin privileges
            category: Command category
            examples: Usage examples

        Usage:
            @manager.command(
                name="deploy",
                description="Deploy application",
                params=[{"name": "env", "type": "str", "required": True}],
                admin_only=True
            )
            async def deploy_handler(ctx, env):
                return f"Deploying to {env}"
        """
        def decorator(func: Callable):
            # Parse parameters from params list
            param_objs = []
            if params:
                for p in params:
                    param_objs.append(CommandParameter(
                        name=p.get("name"),
                        type=p.get("type", "str"),
                        required=p.get("required", True),
                        default=p.get("default"),
                        description=p.get("description", ""),
                        choices=p.get("choices")
                    ))

            self.register(
                name=name,
                handler=func,
                description=description,
                params=param_objs,
                aliases=aliases or [],
                admin_only=admin_only,
                category=category,
                examples=examples or []
            )

            return func

        return decorator

    def register(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        params: Optional[List[CommandParameter]] = None,
        aliases: Optional[List[str]] = None,
        admin_only: bool = False,
        category: str = "general",
        examples: Optional[List[str]] = None
    ) -> None:
        """
        Register a command.

        Args:
            name: Command name
            handler: Async handler function
            description: Command description
            params: Parameter definitions
            aliases: Command aliases
            admin_only: Require admin
            category: Category
            examples: Examples
        """
        # Validate handler
        if not callable(handler):
            raise ValueError(f"Handler must be callable: {handler}")

        if not inspect.iscoroutinefunction(handler):
            raise ValueError(f"Handler must be async: {handler}")

        # Create definition
        cmd_def = CommandDefinition(
            name=name,
            handler=handler,
            description=description,
            params=params or [],
            aliases=aliases or [],
            admin_only=admin_only,
            category=category,
            examples=examples or []
        )

        # Register command
        self.commands[name] = cmd_def
        self.categories.add(category)

        # Register aliases
        for alias in cmd_def.aliases:
            self.aliases[alias] = name

        logger.info(f"Registered command: {name} (category: {category})")

    def unregister(self, name: str) -> bool:
        """
        Unregister a command.

        Args:
            name: Command name

        Returns:
            True if unregistered
        """
        if name not in self.commands:
            return False

        cmd_def = self.commands[name]

        # Remove aliases
        for alias in cmd_def.aliases:
            if alias in self.aliases:
                del self.aliases[alias]

        # Remove command
        del self.commands[name]

        logger.info(f"Unregistered command: {name}")
        return True

    # ========================================================================
    # Command Execution
    # ========================================================================

    async def execute(
        self,
        command_name: str,
        context: CommandContext,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a command.

        Args:
            command_name: Command name or alias
            context: Command context
            *args, **kwargs: Command arguments

        Returns:
            Command result

        Raises:
            ValueError: If command not found or validation fails
            PermissionError: If user lacks permissions
        """
        # Resolve alias
        if command_name in self.aliases:
            command_name = self.aliases[command_name]

        # Get command
        if command_name not in self.commands:
            raise ValueError(f"Unknown command: {command_name}")

        cmd_def = self.commands[command_name]

        # Check if enabled
        if not cmd_def.enabled:
            raise ValueError(f"Command is disabled: {command_name}")

        # Check permissions
        if cmd_def.admin_only and not context.is_admin:
            raise PermissionError(f"Command requires admin privileges: {command_name}")

        # Validate and convert parameters
        validated_args, validated_kwargs = self._validate_params(cmd_def, args, kwargs)

        # Execute handler
        try:
            result = await cmd_def.handler(context, *validated_args, **validated_kwargs)
            return result

        except Exception as e:
            logger.error(f"Error executing command {command_name}: {e}", exc_info=True)
            raise

    def _validate_params(
        self,
        cmd_def: CommandDefinition,
        args: tuple,
        kwargs: dict
    ) -> tuple[tuple, dict]:
        """Validate and convert parameters."""
        validated_args = []
        validated_kwargs = {}

        # Process positional parameters
        for i, param in enumerate(cmd_def.params):
            value = None

            # Get value from args or kwargs
            if i < len(args):
                value = args[i]
            elif param.name in kwargs:
                value = kwargs[param.name]
            elif param.default is not None:
                value = param.default
            elif not param.required:
                continue

            # Validate
            is_valid, error = param.validate(value)
            if not is_valid:
                raise ValueError(error)

            validated_args.append(value)

        return tuple(validated_args), validated_kwargs

    # ========================================================================
    # Help & Discovery
    # ========================================================================

    def get_help(self, command_name: Optional[str] = None) -> str:
        """
        Get help text.

        Args:
            command_name: Specific command or None for all

        Returns:
            Help text
        """
        if command_name:
            # Resolve alias
            if command_name in self.aliases:
                command_name = self.aliases[command_name]

            if command_name not in self.commands:
                return f"Unknown command: {command_name}"

            return self.commands[command_name].generate_help()

        # Generate help for all commands
        lines = ["**Custom Commands:**\n"]

        # Group by category
        for category in sorted(self.categories):
            category_commands = [
                cmd for cmd in self.commands.values()
                if cmd.category == category and cmd.enabled
            ]

            if not category_commands:
                continue

            lines.append(f"**{category.title()}:**")
            for cmd in sorted(category_commands, key=lambda c: c.name):
                admin_marker = "🔒 " if cmd.admin_only else ""
                lines.append(f"• `!{cmd.name}` - {admin_marker}{cmd.description}")

        lines.append(f"\nUse `!help <command>` for detailed help")

        return "\n".join(lines)

    def list_commands(
        self,
        category: Optional[str] = None,
        include_disabled: bool = False
    ) -> List[str]:
        """
        List command names.

        Args:
            category: Filter by category
            include_disabled: Include disabled commands

        Returns:
            List of command names
        """
        commands = []

        for name, cmd_def in self.commands.items():
            if category and cmd_def.category != category:
                continue

            if not include_disabled and not cmd_def.enabled:
                continue

            commands.append(name)

        return sorted(commands)

    # ========================================================================
    # Hot Reload from Files
    # ========================================================================

    def load_from_file(self, file_path: str) -> int:
        """
        Load commands from Python file.

        The file should define commands using decorators:

            from chatops_custom_commands import manager

            @manager.command(name="example")
            async def example_handler(ctx):
                return "Example result"

        Args:
            file_path: Path to Python file

        Returns:
            Number of commands loaded
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load module
        spec = importlib.util.spec_from_file_location("custom_commands", path)
        if not spec or not spec.loader:
            raise ImportError(f"Failed to load {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["custom_commands"] = module

        # Track initial count
        initial_count = len(self.commands)

        # Execute module (will register commands via decorators)
        spec.loader.exec_module(module)

        # Return number of new commands
        new_count = len(self.commands) - initial_count
        logger.info(f"Loaded {new_count} commands from {file_path}")

        return new_count

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get command statistics."""
        return {
            "total_commands": len(self.commands),
            "enabled_commands": sum(1 for cmd in self.commands.values() if cmd.enabled),
            "admin_commands": sum(1 for cmd in self.commands.values() if cmd.admin_only),
            "categories": len(self.categories),
            "aliases": len(self.aliases),
            "by_category": {
                category: len([
                    cmd for cmd in self.commands.values()
                    if cmd.category == category
                ])
                for category in self.categories
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("Custom Commands Framework Demo")
    print("="*80)
    print()

    # Create manager
    manager = CustomCommandManager()

    # Register commands via decorator
    @manager.command(
        name="hello",
        description="Say hello to user",
        category="fun"
    )
    async def hello_handler(ctx):
        return f"Hello, {ctx.user_id}! 👋"

    @manager.command(
        name="echo",
        description="Echo back message",
        params=[{"name": "message", "type": "str", "required": True}],
        examples=["!echo Hello world"],
        category="utils"
    )
    async def echo_handler(ctx, message):
        return f"You said: {message}"

    @manager.command(
        name="deploy",
        description="Deploy application",
        params=[
            {"name": "environment", "type": "str", "required": True, "choices": ["dev", "staging", "prod"]},
            {"name": "version", "type": "str", "required": False, "default": "latest"}
        ],
        admin_only=True,
        examples=["!deploy prod v2.0"],
        category="admin"
    )
    async def deploy_handler(ctx, environment, version="latest"):
        return f"🚀 Deploying {version} to {environment}"

    # Demo execution
    async def demo():
        print("1. Registered Commands:")
        for cmd in manager.list_commands():
            print(f"  • !{cmd}")
        print()

        print("2. General Help:")
        print(manager.get_help())
        print()

        print("3. Command-Specific Help:")
        print(manager.get_help("deploy"))
        print()

        # Create context
        ctx = CommandContext(
            user_id="@alice:matrix.org",
            conversation_id="room_123",
            message_id="msg_456",
            is_admin=True
        )

        print("4. Executing Commands:")

        # Execute hello
        result = await manager.execute("hello", ctx)
        print(f"  !hello → {result}")

        # Execute echo
        result = await manager.execute("echo", ctx, "Hello, World!")
        print(f"  !echo Hello, World! → {result}")

        # Execute deploy
        result = await manager.execute("deploy", ctx, "staging", "v2.0")
        print(f"  !deploy staging v2.0 → {result}")
        print()

        # Statistics
        print("5. Statistics:")
        stats = manager.get_statistics()
        for key, value in stats.items():
            if key != "by_category":
                print(f"  {key}: {value}")
        print()

    asyncio.run(demo())

    print("✓ Demo complete!")
