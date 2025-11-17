"""
MCP Protocol - Tool Definition and Execution Protocol
======================================================
Implements Model Context Protocol for tool integration.

Philosophy:
Tools extend the intelligence beyond neural computation. The MCP protocol
provides a standardized way to discover, validate, and execute external
capabilities that the neural core can orchestrate.
"""

import logging
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ToolStatus(Enum):
    """Tool execution status."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ToolParameter:
    """Tool parameter specification."""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None

    def validate(self, value: Any) -> bool:
        """Validate parameter value."""
        # Type checking
        type_map = {
            "string": str,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list
        }

        expected_type = type_map.get(self.type)
        if expected_type and not isinstance(value, expected_type):
            return False

        # Enum validation
        if self.enum and value not in self.enum:
            return False

        return True


@dataclass
class ToolDefinition:
    """
    MCP tool definition.

    Defines a tool's interface, parameters, and metadata for discovery.
    """
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    category: str = "general"
    version: str = "1.0.0"
    timeout: float = 30.0  # Max execution time
    handler: Optional[Callable[..., Awaitable[Any]]] = None

    def to_schema(self) -> Dict[str, Any]:
        """
        Convert to JSON schema for MCP discovery.

        Returns:
            OpenAPI-compatible tool schema
        """
        properties = {}
        required = []

        for param in self.parameters:
            param_schema = {
                "type": param.type,
                "description": param.description
            }

            if param.enum:
                param_schema["enum"] = param.enum

            if param.default is not None:
                param_schema["default"] = param.default

            properties[param.name] = param_schema

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            },
            "timeout": self.timeout
        }

    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate input parameters.

        Args:
            params: Parameter values to validate

        Returns:
            (valid, error_message)
        """
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                return False, f"Missing required parameter: {param.name}"

            # Validate type if present
            if param.name in params:
                if not param.validate(params[param.name]):
                    return False, f"Invalid value for parameter: {param.name}"

        return True, None


@dataclass
class ToolResult:
    """Result from tool execution."""
    status: ToolStatus
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def success(cls, output: Any, execution_time: float = 0.0, **metadata) -> 'ToolResult':
        """Create success result."""
        return cls(
            status=ToolStatus.SUCCESS,
            output=output,
            execution_time=execution_time,
            metadata=metadata
        )

    @classmethod
    def failure(cls, error: str, execution_time: float = 0.0, **metadata) -> 'ToolResult':
        """Create failure result."""
        return cls(
            status=ToolStatus.FAILURE,
            output=None,
            error=error,
            execution_time=execution_time,
            metadata=metadata
        )

    @classmethod
    def timeout_result(cls, execution_time: float, **metadata) -> 'ToolResult':
        """Create timeout result."""
        return cls(
            status=ToolStatus.TIMEOUT,
            output=None,
            error="Tool execution timed out",
            execution_time=execution_time,
            metadata=metadata
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("MCP Protocol Demo")
    print("="*80 + "\n")

    # Define a tool
    calc_tool = ToolDefinition(
        name="calculator",
        description="Perform mathematical calculations",
        parameters=[
            ToolParameter(
                name="expression",
                type="string",
                description="Mathematical expression to evaluate",
                required=True
            ),
            ToolParameter(
                name="precision",
                type="number",
                description="Decimal precision",
                required=False,
                default=2
            )
        ],
        category="math",
        timeout=5.0
    )

    # Show schema
    print("Tool Schema:")
    schema = calc_tool.to_schema()
    print(json.dumps(schema, indent=2))
    print()

    # Validate parameters
    print("Parameter Validation:")
    valid_params = {"expression": "2 + 2", "precision": 2}
    valid, error = calc_tool.validate_params(valid_params)
    print(f"  Valid params: {valid_params} → {valid}")

    invalid_params = {"precision": 2}  # Missing required 'expression'
    valid, error = calc_tool.validate_params(invalid_params)
    print(f"  Invalid params: {invalid_params} → {valid} (Error: {error})")
    print()

    # Create results
    print("Tool Results:")
    success = ToolResult.success(output=4, execution_time=0.01, input="2+2")
    print(f"  Success: {success.to_dict()}")

    failure = ToolResult.failure(error="Division by zero", execution_time=0.005)
    print(f"  Failure: {failure.to_dict()}")

    print("\n✓ Demo complete!")
