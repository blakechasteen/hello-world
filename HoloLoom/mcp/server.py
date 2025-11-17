"""
MCP Server - Tool Registry and Execution
=========================================
Central server for tool discovery, registration, and execution.

Philosophy:
The MCP server is the "tool shed" where all external capabilities live.
It provides discovery (what tools exist?), validation (are inputs correct?),
and execution (run the tool safely) with timeout and error handling.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from collections import defaultdict

from .protocol import ToolDefinition, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


@dataclass
class MCPConfig:
    """Configuration for MCP server."""
    max_concurrent_tools: int = 10
    default_timeout: float = 30.0
    enable_analytics: bool = True
    rate_limit_per_minute: int = 100


class MCPServer:
    """
    MCP server for tool management.

    Manages tool lifecycle:
    1. Registration: Tools register their capabilities
    2. Discovery: Clients query available tools
    3. Validation: Parameters are checked before execution
    4. Execution: Tools run with timeout and error handling
    5. Analytics: Track usage and performance

    Usage:
        server = MCPServer(MCPConfig())

        # Register tools
        @server.tool("calculator", "Perform math operations")
        async def calculator(expression: str) -> float:
            return eval(expression)

        # Execute
        result = await server.execute("calculator", {"expression": "2 + 2"})
        print(result.output)  # 4

        # Get available tools
        tools = server.list_tools()
    """

    def __init__(self, config: MCPConfig):
        """
        Initialize MCP server.

        Args:
            config: Server configuration
        """
        self.config = config

        # Tool registry
        self.tools: Dict[str, ToolDefinition] = {}

        # Execution tracking
        self.execution_count = defaultdict(int)
        self.execution_time = defaultdict(float)
        self.success_count = defaultdict(int)
        self.failure_count = defaultdict(int)

        # Rate limiting
        self.rate_limit_window = defaultdict(list)

        # Concurrency control
        self.semaphore = asyncio.Semaphore(config.max_concurrent_tools)

        logger.info(f"MCPServer initialized (max_concurrent={config.max_concurrent_tools})")

    def register_tool(self, tool: ToolDefinition) -> bool:
        """
        Register a tool.

        Args:
            tool: Tool definition

        Returns:
            True if successful
        """
        if tool.name in self.tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")

        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} ({tool.category})")

        return True

    def tool(
        self,
        name: str,
        description: str,
        category: str = "general",
        timeout: Optional[float] = None,
        **param_specs
    ):
        """
        Decorator for registering tool functions.

        Args:
            name: Tool name
            description: Tool description
            category: Tool category
            timeout: Optional timeout override
            **param_specs: Parameter specifications

        Usage:
            @server.tool("search", "Search the web", timeout=10.0)
            async def search_web(query: str, max_results: int = 5):
                return await search_api(query, limit=max_results)
        """
        def decorator(func: Callable[..., Awaitable[Any]]):
            # Build parameters from function signature
            import inspect
            sig = inspect.signature(func)
            parameters = []

            for param_name, param in sig.parameters.items():
                # Infer type from annotation
                param_type = "string"
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int or param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == dict:
                        param_type = "object"
                    elif param.annotation == list:
                        param_type = "array"

                # Get default value
                default = None
                required = True
                if param.default != inspect.Parameter.empty:
                    default = param.default
                    required = False

                from .protocol import ToolParameter
                parameters.append(ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=param_specs.get(param_name, f"Parameter: {param_name}"),
                    required=required,
                    default=default
                ))

            # Create tool definition
            tool_def = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                category=category,
                timeout=timeout or self.config.default_timeout,
                handler=func
            )

            self.register_tool(tool_def)

            return func

        return decorator

    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> ToolResult:
        """
        Execute a tool.

        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            timeout: Optional timeout override

        Returns:
            ToolResult with output or error
        """
        # Check tool exists
        if tool_name not in self.tools:
            return ToolResult.failure(f"Tool '{tool_name}' not found")

        tool = self.tools[tool_name]

        # Validate parameters
        valid, error = tool.validate_params(parameters)
        if not valid:
            return ToolResult.failure(f"Parameter validation failed: {error}")

        # Check rate limit
        if not self._check_rate_limit(tool_name):
            return ToolResult.failure("Rate limit exceeded")

        # Execute with timeout and concurrency control
        start_time = time.time()

        try:
            async with self.semaphore:
                exec_timeout = timeout or tool.timeout

                if tool.handler:
                    result = await asyncio.wait_for(
                        tool.handler(**parameters),
                        timeout=exec_timeout
                    )

                    execution_time = time.time() - start_time

                    # Track success
                    if self.config.enable_analytics:
                        self._record_execution(tool_name, execution_time, success=True)

                    return ToolResult.success(
                        output=result,
                        execution_time=execution_time,
                        tool=tool_name
                    )
                else:
                    return ToolResult.failure("Tool has no handler function")

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            if self.config.enable_analytics:
                self._record_execution(tool_name, execution_time, success=False)

            return ToolResult.timeout_result(
                execution_time=execution_time,
                tool=tool_name
            )

        except Exception as e:
            execution_time = time.time() - start_time
            if self.config.enable_analytics:
                self._record_execution(tool_name, execution_time, success=False)

            logger.error(f"Tool execution error ({tool_name}): {e}")
            return ToolResult.failure(
                error=str(e),
                execution_time=execution_time,
                tool=tool_name
            )

    def _check_rate_limit(self, tool_name: str) -> bool:
        """Check if tool is within rate limit."""
        if not self.config.enable_analytics:
            return True

        now = time.time()
        window_start = now - 60  # 1 minute window

        # Clean old entries
        self.rate_limit_window[tool_name] = [
            t for t in self.rate_limit_window[tool_name]
            if t > window_start
        ]

        # Check limit
        if len(self.rate_limit_window[tool_name]) >= self.config.rate_limit_per_minute:
            return False

        # Record this request
        self.rate_limit_window[tool_name].append(now)
        return True

    def _record_execution(self, tool_name: str, execution_time: float, success: bool):
        """Record execution analytics."""
        self.execution_count[tool_name] += 1
        self.execution_time[tool_name] += execution_time

        if success:
            self.success_count[tool_name] += 1
        else:
            self.failure_count[tool_name] += 1

    def list_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available tools.

        Args:
            category: Optional category filter

        Returns:
            List of tool schemas
        """
        tools = []

        for tool in self.tools.values():
            if category is None or tool.category == category:
                tools.append(tool.to_schema())

        return tools

    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Get tool definition.

        Args:
            tool_name: Tool name

        Returns:
            ToolDefinition or None
        """
        return self.tools.get(tool_name)

    def get_analytics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get execution analytics.

        Args:
            tool_name: Optional specific tool

        Returns:
            Analytics dictionary
        """
        if tool_name:
            if tool_name not in self.tools:
                return {}

            count = self.execution_count[tool_name]
            avg_time = (
                self.execution_time[tool_name] / count
                if count > 0 else 0
            )

            return {
                "tool": tool_name,
                "executions": count,
                "success": self.success_count[tool_name],
                "failure": self.failure_count[tool_name],
                "success_rate": (
                    self.success_count[tool_name] / count
                    if count > 0 else 0
                ),
                "avg_execution_time": avg_time
            }

        # Global analytics
        total_executions = sum(self.execution_count.values())
        total_success = sum(self.success_count.values())
        total_failure = sum(self.failure_count.values())

        # Per-tool breakdown
        tools_analytics = {}
        for tool_name in self.tools:
            tools_analytics[tool_name] = self.get_analytics(tool_name)

        return {
            "total_executions": total_executions,
            "total_success": total_success,
            "total_failure": total_failure,
            "success_rate": (
                total_success / total_executions
                if total_executions > 0 else 0
            ),
            "tools": tools_analytics
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("="*80)
        print("MCP Server Demo")
        print("="*80 + "\n")

        # Create server
        config = MCPConfig(
            max_concurrent_tools=5,
            default_timeout=10.0,
            enable_analytics=True
        )
        server = MCPServer(config)

        # Register tools via decorator
        @server.tool("calculator", "Perform mathematical calculations")
        async def calculator(expression: str) -> float:
            """Calculate math expression."""
            await asyncio.sleep(0.1)  # Simulate work
            return eval(expression)

        @server.tool("greet", "Generate greeting", category="text")
        async def greet(name: str, language: str = "en") -> str:
            """Generate personalized greeting."""
            greetings = {
                "en": f"Hello, {name}!",
                "es": f"¡Hola, {name}!",
                "fr": f"Bonjour, {name}!"
            }
            return greetings.get(language, greetings["en"])

        print("1. Registered Tools:")
        tools = server.list_tools()
        for tool in tools:
            print(f"   - {tool['name']}: {tool['description']}")
        print()

        # Execute tools
        print("2. Executing Tools:\n")

        result1 = await server.execute("calculator", {"expression": "2 + 2 * 3"})
        print(f"   calculator(2 + 2 * 3) → {result1.output} ({result1.status.value})")

        result2 = await server.execute("greet", {"name": "Alice", "language": "es"})
        print(f"   greet(Alice, es) → {result2.output} ({result2.status.value})")

        # Test validation error
        result3 = await server.execute("calculator", {})  # Missing required param
        print(f"   calculator() → Error: {result3.error}")
        print()

        # Analytics
        print("3. Analytics:")
        analytics = server.get_analytics()
        print(f"   Total executions: {analytics['total_executions']}")
        print(f"   Success rate: {analytics['success_rate']:.1%}")
        print()

        for tool_name, stats in analytics['tools'].items():
            if stats['executions'] > 0:
                print(f"   {tool_name}:")
                print(f"     Executions: {stats['executions']}")
                print(f"     Avg time: {stats['avg_execution_time']*1000:.1f}ms")

        print("\n✓ Demo complete!")

    asyncio.run(demo())
