"""Tests for Code Execution Ability.

Test coverage:
- Parameter validation
- Successful code execution
- Error handling
- Timeout enforcement
- Output truncation
- Sandbox vs direct execution
- Permission checking
- Lifecycle management

Status: Production ready (2025-12-03)
"""

import tempfile
from datetime import datetime

import pytest

from ..protocol import (
    AbilityContext,
    AbilityResult,
    AbilityTrustLevel,
)
from .code_execution import (
    CodeExecutionAbility,
    CodeExecutionConfig,
    create_code_execution_ability,
)

# Test fixtures

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def basic_context(temp_dir):
    """Create basic execution context."""
    return AbilityContext(
        session_id="test_session",
        working_directory=temp_dir,
        user_confirmed=True,
        trust_level=AbilityTrustLevel.VERIFIED,
        timeout_seconds=30.0
    )


@pytest.fixture
def ability():
    """Create code execution ability."""
    return CodeExecutionAbility()


@pytest.fixture
def ability_with_config():
    """Create ability with custom config."""
    config = CodeExecutionConfig(
        max_timeout=10.0,
        max_output_length=1000,
        enable_sandbox=True
    )
    return CodeExecutionAbility(config)


# Test manifest and metadata

class TestManifest:
    """Test ability manifest."""

    def test_manifest_name(self, ability):
        """Test manifest has correct name."""
        assert ability.manifest.name == "code_execution"

    def test_manifest_version(self, ability):
        """Test manifest has version."""
        assert ability.manifest.version == "1.0.0"

    def test_manifest_tier(self, ability):
        """Test manifest specifies PLUGIN tier."""
        from ..protocol import AbilityTier
        assert ability.manifest.tier == AbilityTier.PLUGIN

    def test_manifest_trust_level(self, ability):
        """Test manifest specifies VERIFIED trust."""
        assert ability.manifest.trust_level == AbilityTrustLevel.VERIFIED

    def test_manifest_permissions(self, ability):
        """Test manifest lists required permissions."""
        assert "execute_code" in ability.manifest.permissions
        assert "read_file" in ability.manifest.permissions
        assert "write_file" in ability.manifest.permissions

    def test_manifest_requires_confirmation(self, ability):
        """Test manifest requires user confirmation."""
        assert ability.manifest.requires_confirmation is True

    def test_manifest_tags(self, ability):
        """Test manifest has appropriate tags."""
        assert "code" in ability.manifest.tags
        assert "execution" in ability.manifest.tags
        assert "sandbox" in ability.manifest.tags
        assert "dangerous" in ability.manifest.tags

    def test_manifest_dependencies(self, ability):
        """Test manifest lists dependencies."""
        assert "subprocess" in ability.manifest.requires
        assert "asyncio" in ability.manifest.requires


# Test preflight checks

class TestPreflight:
    """Test preflight validation."""

    @pytest.mark.asyncio
    async def test_preflight_requires_confirmation(self, ability):
        """Test preflight fails without user confirmation."""
        context = AbilityContext(user_confirmed=False)
        result = await ability.preflight(context)
        assert result.can_execute is False
        assert "confirmation" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_preflight_success_with_confirmation(self, ability, basic_context):
        """Test preflight succeeds with confirmation."""
        result = await ability.preflight(basic_context)
        assert result.can_execute is True

    @pytest.mark.asyncio
    async def test_preflight_warns_invalid_directory(self, ability):
        """Test preflight warns about invalid working directory."""
        context = AbilityContext(
            user_confirmed=True,
            working_directory="/nonexistent/path"
        )
        result = await ability.preflight(context)
        # Should succeed but with warning
        assert result.can_execute is True
        assert any("directory" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_preflight_warns_excessive_timeout(self, ability_with_config):
        """Test preflight warns about excessive timeout."""
        context = AbilityContext(
            user_confirmed=True,
            timeout_seconds=500.0
        )
        result = await ability_with_config.preflight(context)
        assert any("timeout" in w.lower() for w in result.warnings)


# Test parameter validation

class TestParameterValidation:
    """Test parameter validation."""

    def test_validate_empty_code(self, ability):
        """Test validation rejects empty code."""
        result = ability._validate_params("", "python", 10.0)
        assert result.success is False
        assert "empty" in result.error.lower()

    def test_validate_invalid_language(self, ability):
        """Test validation rejects invalid language."""
        result = ability._validate_params("print('hi')", "javascript", 10.0)
        assert result.success is False
        assert "unsupported" in result.error.lower()

    def test_validate_python_language(self, ability):
        """Test validation accepts 'python' and 'py'."""
        result1 = ability._validate_params("print('hi')", "python", 10.0)
        result2 = ability._validate_params("print('hi')", "py", 10.0)
        assert result1.success is True
        assert result2.success is True

    def test_validate_negative_timeout(self, ability):
        """Test validation rejects negative timeout."""
        result = ability._validate_params("print('hi')", "python", -1.0)
        assert result.success is False
        assert "positive" in result.error.lower()

    def test_validate_zero_timeout(self, ability):
        """Test validation rejects zero timeout."""
        result = ability._validate_params("print('hi')", "python", 0.0)
        assert result.success is False

    def test_validate_excessive_timeout(self, ability_with_config):
        """Test validation rejects timeout exceeding max."""
        result = ability_with_config._validate_params("print('hi')", "python", 100.0)
        assert result.success is False
        assert "exceeds maximum" in result.error.lower()

    def test_validate_valid_params(self, ability):
        """Test validation accepts valid parameters."""
        result = ability._validate_params("print('hello')", "python", 10.0)
        assert result.success is True


# Test code execution

class TestCodeExecution:
    """Test code execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_print(self, ability, basic_context):
        """Test executing simple print statement."""
        result = await ability.execute(
            {
                "code": "print('Hello, World!')",
                "language": "python",
                "timeout": 5.0,
                "sandbox": True
            },
            basic_context
        )
        assert result.success is True
        assert "Hello, World!" in result.output
        assert result.error is None or result.error == ""

    @pytest.mark.asyncio
    async def test_execute_arithmetic(self, ability, basic_context):
        """Test executing arithmetic code."""
        code = """
x = 5
y = 10
print(x + y)
"""
        result = await ability.execute(
            {"code": code, "timeout": 5.0},
            basic_context
        )
        assert result.success is True
        assert "15" in result.output

    @pytest.mark.asyncio
    async def test_execute_with_error(self, ability, basic_context):
        """Test execution with runtime error."""
        code = """
try:
    x = 1 / 0
except Exception as e:
    print(f"Error: {e}")
"""
        result = await ability.execute(
            {"code": code, "timeout": 5.0},
            basic_context
        )
        assert result.success is True
        assert "division" in result.output.lower()

    @pytest.mark.asyncio
    async def test_execute_syntax_error(self, ability, basic_context):
        """Test execution with syntax error."""
        result = await ability.execute(
            {"code": "this is not valid python!", "timeout": 5.0},
            basic_context
        )
        assert result.success is False
        assert "SyntaxError" in result.error or "error" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_timeout(self, ability, basic_context):
        """Test execution timeout."""
        code = """
import time
time.sleep(10)
"""
        result = await ability.execute(
            {"code": code, "timeout": 0.5, "sandbox": True},
            basic_context
        )
        assert result.success is False
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_large_output(self, ability_with_config, basic_context):
        """Test output truncation for large output."""
        code = """
for i in range(1000):
    print(f"Line {i}: " + "x" * 50)
"""
        result = await ability_with_config.execute(
            {"code": code, "timeout": 5.0},
            basic_context
        )
        assert result.success is True
        assert result.metadata.get("truncated") is True
        assert len(result.output) <= ability_with_config.config.max_output_length + 100


# Test statistics tracking

class TestStatistics:
    """Test execution statistics tracking."""

    @pytest.mark.asyncio
    async def test_execution_count_increments(self, ability, basic_context):
        """Test execution count increments."""
        initial_count = ability.execution_count
        await ability.execute(
            {"code": "print('test')", "timeout": 5.0},
            basic_context
        )
        assert ability.execution_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_last_execution_timestamp(self, ability, basic_context):
        """Test last execution timestamp is updated."""
        before = datetime.now()
        await ability.execute(
            {"code": "print('test')", "timeout": 5.0},
            basic_context
        )
        after = datetime.now()
        assert ability.last_execution is not None
        assert before <= ability.last_execution <= after

    @pytest.mark.asyncio
    async def test_total_duration_accumulates(self, ability, basic_context):
        """Test total duration accumulates."""
        initial_duration = ability.total_duration_ms
        await ability.execute(
            {"code": "print('test')", "timeout": 5.0},
            basic_context
        )
        assert ability.total_duration_ms > initial_duration


# Test verification

class TestVerification:
    """Test output verification."""

    @pytest.mark.asyncio
    async def test_verify_successful_execution(self, ability):
        """Test verification of successful execution."""
        result = AbilityResult(
            success=True,
            output="Expected output",
            confidence=0.95,
            metadata={}
        )
        verification = await ability.verify(result)
        assert verification.verified is True
        assert len(verification.issues) == 0

    @pytest.mark.asyncio
    async def test_verify_failed_execution(self, ability):
        """Test verification of failed execution."""
        result = AbilityResult(
            success=False,
            error="Execution failed",
            confidence=0.0,
            metadata={}
        )
        verification = await ability.verify(result)
        assert verification.verified is False
        assert "Execution failed" in verification.issues[0]

    @pytest.mark.asyncio
    async def test_verify_truncated_output(self, ability):
        """Test verification detects truncated output."""
        result = AbilityResult(
            success=True,
            output="x" * 100_000,
            confidence=0.8,
            metadata={"truncated": True}
        )
        verification = await ability.verify(result)
        assert any("truncated" in issue.lower() for issue in verification.issues)

    @pytest.mark.asyncio
    async def test_verify_low_confidence(self, ability):
        """Test verification detects low confidence."""
        result = AbilityResult(
            success=True,
            output="Output",
            confidence=0.3,
            metadata={}
        )
        verification = await ability.verify(result)
        assert any("confidence" in issue.lower() for issue in verification.issues)


# Test direct execution

class TestDirectExecution:
    """Test direct (non-sandboxed) execution."""

    @pytest.mark.asyncio
    async def test_execute_direct_mode(self, ability, basic_context):
        """Test direct execution without sandbox."""
        result = await ability.execute(
            {
                "code": "x = 42\nprint(x)",
                "timeout": 5.0,
                "sandbox": False
            },
            basic_context
        )
        assert result.success is True
        assert "42" in result.output


# Test convenience function

class TestConvenienceFunction:
    """Test convenience function."""

    def test_create_ability(self):
        """Test creating ability via convenience function."""
        ability = create_code_execution_ability()
        assert isinstance(ability, CodeExecutionAbility)
        assert ability.manifest.name == "code_execution"

    def test_create_ability_with_config(self):
        """Test creating ability with custom config."""
        config = CodeExecutionConfig(max_timeout=5.0)
        ability = create_code_execution_ability(config)
        assert ability.config.max_timeout == 5.0


# Integration tests

class TestIntegration:
    """Integration tests with full workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, ability, basic_context):
        """Test full preflight -> execute -> verify workflow."""
        # Preflight
        preflight = await ability.preflight(basic_context)
        assert preflight.can_execute is True

        # Execute
        result = await ability.execute(
            {
                "code": """
result = 0
for i in range(1, 5):
    result += i
print(f"Sum: {result}")
""",
                "timeout": 5.0
            },
            basic_context
        )
        assert result.success is True
        assert "Sum: 10" in result.output

        # Verify
        verification = await ability.verify(result)
        assert verification.verified is True

    @pytest.mark.asyncio
    async def test_execution_id_generation(self, ability, basic_context):
        """Test execution ID is generated and included in metadata."""
        result = await ability.execute(
            {"code": "print('test')", "timeout": 5.0},
            basic_context
        )
        assert "execution_id" in result.metadata
        assert basic_context.session_id in result.metadata["execution_id"]

    @pytest.mark.asyncio
    async def test_multiple_executions(self, ability, basic_context):
        """Test multiple executions accumulate statistics."""
        for i in range(3):
            await ability.execute(
                {"code": f"print({i})", "timeout": 5.0},
                basic_context
            )
        assert ability.execution_count == 3
        assert ability.last_execution is not None
        assert ability.total_duration_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
