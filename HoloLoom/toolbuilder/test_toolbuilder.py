"""
Integration Tests for Tool Builder

Tests all components of the custom tool builder system.
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path

from .definition import (
    ToolDefinition,
    Parameter,
    LogicBlock,
    Condition,
    ParameterType,
    LogicBlockType,
    ActionType,
    OutputFormat,
    HTTPRequestAction,
    ComputeAction,
)
from .validator import ToolValidator, ValidationError
from .generator import CodeGenerator
from .registry import ToolRegistry, ToolRegistryError
from .executor import ToolExecutor, ExecutionError


class TestToolDefinition:
    """Test tool definition schema"""

    def test_create_simple_tool(self):
        """Test creating a simple tool definition"""
        tool = ToolDefinition(
            id="test_calculator",
            name="calculator",
            description="A simple calculator",
            version="1.0.0",
            parameters=[
                Parameter(
                    name="x",
                    type=ParameterType.NUMBER,
                    description="First number",
                    required=True
                ),
                Parameter(
                    name="y",
                    type=ParameterType.NUMBER,
                    description="Second number",
                    required=True
                ),
            ],
            logic_blocks=[
                LogicBlock(
                    id="block_1",
                    type=LogicBlockType.VARIABLE,
                    variable_name="result",
                    variable_value="${x} + ${y}"
                ),
                LogicBlock(
                    id="block_2",
                    type=LogicBlockType.RETURN,
                    return_value="${result}"
                ),
            ],
            output_format=OutputFormat.JSON
        )

        assert tool.name == "calculator"
        assert len(tool.parameters) == 2
        assert len(tool.logic_blocks) == 2

    def test_tool_serialization(self):
        """Test tool definition serialization"""
        tool = ToolDefinition(
            id="test_tool",
            name="test",
            description="Test tool",
            parameters=[
                Parameter(
                    name="input",
                    type=ParameterType.TEXT,
                    description="Input text",
                    required=True
                )
            ],
            logic_blocks=[]
        )

        # Convert to dict
        tool_dict = tool.to_dict()
        assert tool_dict["id"] == "test_tool"
        assert tool_dict["name"] == "test"

        # Convert to JSON
        tool_json = tool.to_json()
        assert isinstance(tool_json, str)
        assert "test_tool" in tool_json

        # Convert back from dict
        tool_restored = ToolDefinition.from_dict(tool_dict)
        assert tool_restored.id == tool.id
        assert tool_restored.name == tool.name


class TestValidator:
    """Test tool validator"""

    def test_validate_valid_tool(self):
        """Test validating a valid tool"""
        tool = ToolDefinition(
            id="valid_tool",
            name="valid_tool",
            description="A valid tool for testing",
            version="1.0.0",
            parameters=[
                Parameter(
                    name="input",
                    type=ParameterType.TEXT,
                    description="Input parameter",
                    required=True
                )
            ],
            logic_blocks=[
                LogicBlock(
                    id="block_1",
                    type=LogicBlockType.RETURN,
                    return_value="${input}"
                )
            ]
        )

        validator = ToolValidator()
        assert validator.validate(tool) == True
        assert len(validator.get_errors()) == 0

    def test_validate_missing_name(self):
        """Test validation fails for missing name"""
        tool = ToolDefinition(
            id="test",
            name="",  # Empty name
            description="Test",
            logic_blocks=[]
        )

        validator = ToolValidator()
        with pytest.raises(ValidationError):
            validator.validate(tool)

    def test_validate_invalid_name(self):
        """Test validation fails for invalid name"""
        tool = ToolDefinition(
            id="test",
            name="123invalid",  # Starts with number
            description="Test",
            logic_blocks=[]
        )

        validator = ToolValidator()
        with pytest.raises(ValidationError):
            validator.validate(tool)

    def test_validate_duplicate_parameters(self):
        """Test validation fails for duplicate parameter names"""
        tool = ToolDefinition(
            id="test",
            name="test",
            description="Test",
            parameters=[
                Parameter(name="param1", type=ParameterType.TEXT, description="Param 1"),
                Parameter(name="param1", type=ParameterType.TEXT, description="Param 1 duplicate"),
            ],
            logic_blocks=[]
        )

        validator = ToolValidator()
        with pytest.raises(ValidationError):
            validator.validate(tool)

    def test_validate_dangerous_expression(self):
        """Test validation fails for dangerous expressions"""
        tool = ToolDefinition(
            id="test",
            name="test",
            description="Test",
            logic_blocks=[
                LogicBlock(
                    id="block_1",
                    type=LogicBlockType.VARIABLE,
                    variable_name="result",
                    variable_value="eval('1+1')"  # Dangerous eval
                )
            ]
        )

        validator = ToolValidator()
        with pytest.raises(ValidationError) as excinfo:
            validator.validate(tool)
        assert "dangerous" in str(excinfo.value).lower()


class TestCodeGenerator:
    """Test code generator"""

    def test_generate_simple_tool(self):
        """Test generating code for a simple tool"""
        tool = ToolDefinition(
            id="simple_tool",
            name="simple_tool",
            description="A simple tool",
            version="1.0.0",
            parameters=[
                Parameter(
                    name="message",
                    type=ParameterType.TEXT,
                    description="Message to return",
                    required=True
                )
            ],
            logic_blocks=[
                LogicBlock(
                    id="block_1",
                    type=LogicBlockType.RETURN,
                    return_value="${message}"
                )
            ]
        )

        generator = CodeGenerator()
        code = generator.generate(tool)

        # Check generated code
        assert "async def simple_tool" in code
        assert "message: str" in code
        assert "return" in code
        assert "Dict[str, Any]" in code

    def test_generate_with_logic(self):
        """Test generating code with logic blocks"""
        tool = ToolDefinition(
            id="calculator",
            name="calculator",
            description="Calculator tool",
            parameters=[
                Parameter(name="x", type=ParameterType.NUMBER, description="First number"),
                Parameter(name="y", type=ParameterType.NUMBER, description="Second number"),
            ],
            logic_blocks=[
                LogicBlock(
                    id="block_1",
                    type=LogicBlockType.VARIABLE,
                    variable_name="sum",
                    variable_value="${x} + ${y}"
                ),
                LogicBlock(
                    id="block_2",
                    type=LogicBlockType.RETURN,
                    return_value="${sum}"
                ),
            ]
        )

        generator = CodeGenerator()
        code = generator.generate(tool)

        assert "async def calculator" in code
        assert "context['sum']" in code
        assert "return" in code


class TestRegistry:
    """Test tool registry"""

    @pytest.fixture
    def temp_registry(self):
        """Create temporary registry"""
        temp_dir = tempfile.mkdtemp()
        registry = ToolRegistry(storage_path=temp_dir)
        yield registry
        shutil.rmtree(temp_dir)

    def test_register_tool(self, temp_registry):
        """Test registering a tool"""
        tool = ToolDefinition(
            id="test_tool",
            name="test",
            description="Test tool",
            logic_blocks=[]
        )

        tool_id = temp_registry.register(tool)
        assert tool_id == "test_tool"

        # Verify tool can be retrieved
        retrieved = temp_registry.get(tool_id)
        assert retrieved is not None
        assert retrieved.name == "test"

    def test_register_duplicate_fails(self, temp_registry):
        """Test registering duplicate tool fails"""
        tool = ToolDefinition(
            id="test_tool",
            name="test",
            description="Test tool",
            logic_blocks=[]
        )

        temp_registry.register(tool)

        # Try to register again without overwrite
        with pytest.raises(ToolRegistryError):
            temp_registry.register(tool, overwrite=False)

    def test_list_tools(self, temp_registry):
        """Test listing tools"""
        tool1 = ToolDefinition(id="tool1", name="tool1", description="Tool 1", category="utility", logic_blocks=[])
        tool2 = ToolDefinition(id="tool2", name="tool2", description="Tool 2", category="integration", logic_blocks=[])

        temp_registry.register(tool1)
        temp_registry.register(tool2)

        all_tools = temp_registry.list_all()
        assert len(all_tools) == 2

        # Filter by category
        utility_tools = temp_registry.list_all(category="utility")
        assert len(utility_tools) == 1
        assert utility_tools[0].name == "tool1"

    def test_search_tools(self, temp_registry):
        """Test searching tools"""
        tool1 = ToolDefinition(id="tool1", name="calculator", description="Math calculator", logic_blocks=[])
        tool2 = ToolDefinition(id="tool2", name="scraper", description="Web scraper", logic_blocks=[])

        temp_registry.register(tool1)
        temp_registry.register(tool2)

        results = temp_registry.search("calculator")
        assert len(results) == 1
        assert results[0].name == "calculator"

    def test_delete_tool(self, temp_registry):
        """Test deleting a tool"""
        tool = ToolDefinition(id="test_tool", name="test", description="Test", logic_blocks=[])

        temp_registry.register(tool)
        assert temp_registry.get("test_tool") is not None

        temp_registry.delete("test_tool")
        assert temp_registry.get("test_tool") is None

    def test_version_management(self, temp_registry):
        """Test version management"""
        tool = ToolDefinition(id="test_tool", name="test", description="Test v1", version="1.0.0", logic_blocks=[])

        temp_registry.register(tool)

        # Save version
        version_id = temp_registry.save_version("test_tool")
        assert version_id is not None

        # Update tool
        tool.description = "Test v2"
        tool.version = "2.0.0"
        temp_registry.register(tool, overwrite=True)

        # Get versions
        versions = temp_registry.get_versions("test_tool")
        assert len(versions) >= 1


class TestExecutor:
    """Test tool executor"""

    @pytest.fixture
    def executor_setup(self):
        """Setup executor with registry"""
        temp_dir = tempfile.mkdtemp()
        registry = ToolRegistry(storage_path=temp_dir)
        executor = ToolExecutor(registry=registry, enable_sandbox=False)  # Disable sandbox for testing
        yield registry, executor
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_execute_simple_tool(self, executor_setup):
        """Test executing a simple tool"""
        registry, executor = executor_setup

        tool = ToolDefinition(
            id="echo_tool",
            name="echo_tool",
            description="Echo tool",
            parameters=[
                Parameter(name="message", type=ParameterType.TEXT, description="Message", required=True)
            ],
            logic_blocks=[
                LogicBlock(
                    id="block_1",
                    type=LogicBlockType.RETURN,
                    return_value="${message}"
                )
            ]
        )

        registry.register(tool)

        # Execute tool
        result = await executor.execute("echo_tool", {"message": "Hello"})

        assert result["status"] == "success"
        assert "result" in result

    @pytest.mark.asyncio
    async def test_execute_missing_parameter(self, executor_setup):
        """Test executing with missing required parameter"""
        registry, executor = executor_setup

        tool = ToolDefinition(
            id="test_tool",
            name="test_tool",
            description="Test",
            parameters=[
                Parameter(name="required_param", type=ParameterType.TEXT, description="Required", required=True)
            ],
            logic_blocks=[]
        )

        registry.register(tool)

        # Execute without required parameter
        with pytest.raises(ExecutionError):
            await executor.execute("test_tool", {})

    @pytest.mark.asyncio
    async def test_execution_timeout(self, executor_setup):
        """Test execution timeout"""
        from .executor import ExecutionTimeout

        registry, executor = executor_setup

        # Create a tool that would run forever (while loop without exit)
        tool = ToolDefinition(
            id="infinite_tool",
            name="infinite_tool",
            description="Infinite loop",
            logic_blocks=[
                LogicBlock(
                    id="block_1",
                    type=LogicBlockType.WHILE_LOOP,
                    condition=Condition(left="True", operator="==", right=True),
                    loop_blocks=[]
                )
            ]
        )

        registry.register(tool)

        # Execute with very short timeout
        with pytest.raises(ExecutionTimeout):
            await executor.execute("infinite_tool", {}, timeout_override=1)

    def test_execution_stats(self, executor_setup):
        """Test execution statistics"""
        registry, executor = executor_setup

        stats = executor.get_execution_stats()
        assert "total_executions" in stats
        assert "successful_executions" in stats
        assert "failed_executions" in stats


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
