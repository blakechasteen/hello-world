"""
Tool Executor

Safely executes custom tools with sandboxing, resource limits, and error handling.
"""

from typing import Dict, Any, Optional, Set
import asyncio
import sys
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import resource
import signal
from .definition import ToolDefinition
from .validator import ToolValidator
from .generator import CodeGenerator
from .registry import ToolRegistry


class ExecutionError(Exception):
    """Tool execution error"""
    pass


class ExecutionTimeout(Exception):
    """Tool execution timeout"""
    pass


class ToolExecutor:
    """Executes custom tools in a sandboxed environment"""

    # Safe built-ins that are allowed
    SAFE_BUILTINS = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
        'chr', 'dict', 'divmod', 'enumerate', 'filter', 'float', 'format',
        'frozenset', 'hex', 'int', 'isinstance', 'issubclass', 'iter',
        'len', 'list', 'map', 'max', 'min', 'next', 'oct', 'ord', 'pow',
        'print', 'range', 'repr', 'reversed', 'round', 'set', 'slice',
        'sorted', 'str', 'sum', 'tuple', 'type', 'zip',
        # Type hints
        'Dict', 'List', 'Any', 'Optional', 'Union', 'Tuple', 'Set',
    }

    # Safe modules that can be imported
    SAFE_MODULES = {
        'json', 'math', 'datetime', 'typing', 'functools', 'itertools',
        'collections', 're', 'string', 'uuid', 'hashlib', 'base64',
        'aiohttp', 'aiofiles',
    }

    # Default resource limits
    DEFAULT_TIMEOUT = 30  # seconds
    DEFAULT_MEMORY_LIMIT = 512 * 1024 * 1024  # 512 MB
    MAX_OUTPUT_SIZE = 10 * 1024 * 1024  # 10 MB

    def __init__(
        self,
        registry: ToolRegistry,
        timeout: int = DEFAULT_TIMEOUT,
        memory_limit: int = DEFAULT_MEMORY_LIMIT,
        enable_sandbox: bool = True,
    ):
        """
        Initialize tool executor

        Args:
            registry: Tool registry
            timeout: Execution timeout in seconds
            memory_limit: Memory limit in bytes
            enable_sandbox: Whether to enable sandboxing
        """
        self.registry = registry
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.enable_sandbox = enable_sandbox
        self.validator = ToolValidator()
        self.generator = CodeGenerator()

        # Execution tracking
        self.execution_count = 0
        self.execution_history: list = []

    async def execute(
        self,
        tool_id: str,
        parameters: Dict[str, Any],
        timeout_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a custom tool

        Args:
            tool_id: Tool ID to execute
            parameters: Input parameters
            timeout_override: Override default timeout

        Returns:
            Execution result

        Raises:
            ExecutionError: If execution fails
            ExecutionTimeout: If execution times out
        """
        # Get tool definition
        tool_def = self.registry.get(tool_id)
        if not tool_def:
            raise ExecutionError(f"Tool {tool_id} not found")

        if not tool_def.enabled:
            raise ExecutionError(f"Tool {tool_id} is disabled")

        # Validate tool definition
        try:
            self.validator.validate(tool_def)
        except Exception as e:
            raise ExecutionError(f"Tool validation failed: {e}")

        # Get or generate code
        code = self.registry.get_generated_code(tool_id)
        if not code:
            code = self.generator.generate(tool_def)
            self.registry.save_generated_code(tool_id, code)

        # Validate parameters
        self._validate_parameters(tool_def, parameters)

        # Execute with timeout
        timeout = timeout_override or self.timeout

        try:
            result = await asyncio.wait_for(
                self._execute_in_sandbox(tool_def, code, parameters),
                timeout=timeout
            )

            # Track execution
            self._track_execution(tool_id, parameters, result, success=True)

            return result

        except asyncio.TimeoutError:
            error = f"Execution timed out after {timeout} seconds"
            self._track_execution(tool_id, parameters, {"error": error}, success=False)
            raise ExecutionTimeout(error)

        except Exception as e:
            error = f"Execution failed: {str(e)}"
            self._track_execution(tool_id, parameters, {"error": error}, success=False)
            raise ExecutionError(error)

    async def _execute_in_sandbox(
        self,
        tool_def: ToolDefinition,
        code: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute code in sandboxed environment"""
        if self.enable_sandbox:
            return await self._execute_sandboxed(tool_def, code, parameters)
        else:
            return await self._execute_direct(tool_def, code, parameters)

    async def _execute_sandboxed(
        self,
        tool_def: ToolDefinition,
        code: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute with full sandboxing"""
        # Create restricted namespace
        namespace = self._create_sandbox_namespace()

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Set resource limits (Unix only)
            if hasattr(resource, 'RLIMIT_AS'):
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (self.memory_limit, hard)
                )

            # Execute code in namespace
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, namespace, namespace)

                # Call the generated function
                func = namespace.get(tool_def.name)
                if not func:
                    raise ExecutionError(f"Function {tool_def.name} not found in generated code")

                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(**parameters)
                else:
                    result = func(**parameters)

            # Get captured output
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            # Check output size
            if len(stdout_output) > self.MAX_OUTPUT_SIZE:
                stdout_output = stdout_output[:self.MAX_OUTPUT_SIZE] + "\n[OUTPUT TRUNCATED]"

            # Return result with metadata
            return {
                "result": result,
                "stdout": stdout_output,
                "stderr": stderr_output,
                "status": "success",
            }

        except Exception as e:
            return {
                "result": None,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue() + f"\n{traceback.format_exc()}",
                "status": "error",
                "error": str(e),
            }

        finally:
            # Reset resource limits
            if hasattr(resource, 'RLIMIT_AS'):
                resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

    async def _execute_direct(
        self,
        tool_def: ToolDefinition,
        code: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute without sandboxing (faster but less safe)"""
        namespace = {}

        try:
            exec(code, namespace, namespace)

            func = namespace.get(tool_def.name)
            if not func:
                raise ExecutionError(f"Function {tool_def.name} not found")

            if asyncio.iscoroutinefunction(func):
                result = await func(**parameters)
            else:
                result = func(**parameters)

            return {
                "result": result,
                "status": "success",
            }

        except Exception as e:
            return {
                "result": None,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def _create_sandbox_namespace(self) -> Dict[str, Any]:
        """Create a restricted namespace for execution"""
        # Start with safe builtins
        safe_builtins = {}
        for name in self.SAFE_BUILTINS:
            if hasattr(__builtins__, name):
                safe_builtins[name] = getattr(__builtins__, name)

        namespace = {
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',
            '__doc__': None,
        }

        # Add safe module imports
        import json
        import math
        import datetime
        import typing
        import functools
        import itertools
        import collections
        import re
        import string
        import uuid
        import hashlib
        import base64

        namespace.update({
            'json': json,
            'math': math,
            'datetime': datetime,
            'typing': typing,
            'functools': functools,
            'itertools': itertools,
            'collections': collections,
            're': re,
            'string': string,
            'uuid': uuid,
            'hashlib': hashlib,
            'base64': base64,
        })

        # Add async support
        namespace['asyncio'] = asyncio

        # Try to add optional modules
        try:
            import aiohttp
            namespace['aiohttp'] = aiohttp
        except ImportError:
            pass

        try:
            import aiofiles
            namespace['aiofiles'] = aiofiles
        except ImportError:
            pass

        return namespace

    def _validate_parameters(
        self,
        tool_def: ToolDefinition,
        parameters: Dict[str, Any],
    ):
        """Validate input parameters"""
        # Check required parameters
        for param in tool_def.parameters:
            if param.required and param.name not in parameters:
                raise ExecutionError(f"Missing required parameter: {param.name}")

        # Type validation
        for param in tool_def.parameters:
            if param.name not in parameters:
                continue

            value = parameters[param.name]

            # Skip None values for optional parameters
            if value is None and not param.required:
                continue

            # Validate type
            if param.validation:
                self._validate_parameter_value(param.name, value, param.validation)

    def _validate_parameter_value(
        self,
        param_name: str,
        value: Any,
        validation: Dict[str, Any],
    ):
        """Validate parameter value against validation rules"""
        # Min/max for numbers
        if 'min' in validation and isinstance(value, (int, float)):
            if value < validation['min']:
                raise ExecutionError(
                    f"Parameter {param_name} value {value} is less than minimum {validation['min']}"
                )

        if 'max' in validation and isinstance(value, (int, float)):
            if value > validation['max']:
                raise ExecutionError(
                    f"Parameter {param_name} value {value} exceeds maximum {validation['max']}"
                )

        # Pattern matching for strings
        if 'pattern' in validation and isinstance(value, str):
            import re
            if not re.match(validation['pattern'], value):
                raise ExecutionError(
                    f"Parameter {param_name} value does not match pattern {validation['pattern']}"
                )

        # Length validation
        if 'min_length' in validation and hasattr(value, '__len__'):
            if len(value) < validation['min_length']:
                raise ExecutionError(
                    f"Parameter {param_name} length {len(value)} is less than minimum {validation['min_length']}"
                )

        if 'max_length' in validation and hasattr(value, '__len__'):
            if len(value) > validation['max_length']:
                raise ExecutionError(
                    f"Parameter {param_name} length {len(value)} exceeds maximum {validation['max_length']}"
                )

    def _track_execution(
        self,
        tool_id: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
        success: bool,
    ):
        """Track execution for monitoring"""
        self.execution_count += 1

        execution_record = {
            "execution_id": self.execution_count,
            "tool_id": tool_id,
            "timestamp": datetime.utcnow().isoformat(),
            "parameters": parameters,
            "result": result,
            "success": success,
        }

        self.execution_history.append(execution_record)

        # Keep only last 1000 executions
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "success_rate": 0.0,
            }

        total = len(self.execution_history)
        successful = sum(1 for e in self.execution_history if e["success"])
        failed = total - successful

        # Tool usage statistics
        tool_usage = {}
        for execution in self.execution_history:
            tool_id = execution["tool_id"]
            tool_usage[tool_id] = tool_usage.get(tool_id, 0) + 1

        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "tool_usage": tool_usage,
            "most_used_tool": max(tool_usage.items(), key=lambda x: x[1])[0] if tool_usage else None,
        }

    def get_recent_executions(self, limit: int = 10) -> list:
        """Get recent execution records"""
        return self.execution_history[-limit:]

    def clear_history(self):
        """Clear execution history"""
        self.execution_history = []
        self.execution_count = 0
