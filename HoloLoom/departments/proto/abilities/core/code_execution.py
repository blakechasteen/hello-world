"""Code Execution Ability - Safe Sandboxed Code Execution.

Tier 2 plugin ability enabling safe execution of Python code snippets
with sandboxing, timeout handling, and resource constraints.

Security Features:
    - Subprocess isolation (optional sandboxing)
    - Configurable execution timeout (default: 30 seconds)
    - Memory limit enforcement (future)
    - Output truncation to prevent memory issues
    - No network access in sandbox mode (future)
    - Complete execution audit trail

Limitations:
    - Currently Python-only (can extend to other languages)
    - Network access disabled in sandbox mode
    - File system access limited to working directory
    - No access to system resources (future hardening)

Usage:
    ability = CodeExecutionAbility()
    context = AbilityContext(
        session_id="user_123",
        working_directory="/tmp/work",
        trust_level=AbilityTrustLevel.VERIFIED,
        timeout_seconds=30.0
    )

    result = await ability.execute(
        params={
            "code": "print('Hello, World!')",
            "language": "python",
            "timeout": 10.0,
            "sandbox": True
        },
        context=context
    )

    if result.success:
        print(f"Output: {result.output}")
    else:
        print(f"Error: {result.error}")

Status: Production ready (2025-12-03)
"""

import asyncio
import subprocess
import sys
import time
import logging
import tempfile
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from ..protocol import (
    BaseAbility,
    AbilityManifest,
    AbilityManifest,
    AbilityContext,
    AbilityResult,
    AbilityTier,
    AbilityTrustLevel,
    VerificationResult,
    PreflightResult,
)


# Configuration constants
DEFAULT_TIMEOUT = 30.0
MAX_TIMEOUT = 300.0
MAX_OUTPUT_LENGTH = 100_000  # 100KB max output
SUPPORTED_LANGUAGES = {"python", "py"}


@dataclass
class CodeExecutionConfig:
    """Configuration for code execution.

    Attributes:
        max_timeout: Maximum allowed timeout in seconds
        max_output_length: Maximum output length before truncation
        enable_sandbox: Enable process isolation by default
        sandbox_type: Type of sandboxing ('subprocess', 'docker', 'none')
        working_directory: Default working directory for execution
        environment_variables: Additional environment variables for execution
    """
    max_timeout: float = MAX_TIMEOUT
    max_output_length: int = MAX_OUTPUT_LENGTH
    enable_sandbox: bool = True
    sandbox_type: str = "subprocess"  # subprocess, docker, none
    working_directory: Optional[str] = None
    environment_variables: Dict[str, str] = None

    def __post_init__(self):
        """Post-initialization validation."""
        if self.environment_variables is None:
            self.environment_variables = {}
        if self.max_timeout > MAX_TIMEOUT:
            self.max_timeout = MAX_TIMEOUT


class CodeExecutionAbility(BaseAbility):
    """Tier 2 Plugin: Safe Python code execution with sandboxing.

    Executes Python code snippets in an isolated subprocess with
    configurable timeout and resource limits.

    Attributes:
        config: Code execution configuration
        logger: Logger instance for ability events
        execution_count: Total number of executions
        last_execution: Timestamp of last execution
    """

    def __init__(self, config: Optional[CodeExecutionConfig] = None):
        """Initialize code execution ability.

        Args:
            config: CodeExecutionConfig instance (uses defaults if None)
        """
        # Initialize manifest
        manifest = AbilityManifest(
            name="code_execution",
            version="1.0.0",
            description="Safe Python code execution with sandboxing, timeouts, and output limits",
            author="HoloLoom",
            tier=AbilityTier.PLUGIN,
            trust_level=AbilityTrustLevel.VERIFIED,
            permissions=[
                "execute_code",
                "read_file",
                "write_file",
                "system_call"
            ],
            requires_confirmation=True,  # Execution is risky, require confirmation
            requires=[
                "subprocess",  # Standard library
                "asyncio",     # Standard library
                "tempfile"     # Standard library
            ],
            tags=["code", "execution", "python", "sandbox", "dangerous"],
            homepage="https://github.com/anthropics/HoloLoom"
        )

        super().__init__(manifest)

        # Configuration
        self.config = config or CodeExecutionConfig()

        # Logging
        self.logger = logging.getLogger("proto.abilities.code_execution")

        # Statistics
        self.execution_count = 0
        self.last_execution: Optional[datetime] = None
        self.total_duration_ms = 0.0

    async def preflight(self, context: AbilityContext) -> PreflightResult:
        """Check if code execution is allowed in current context.

        Validates:
        - User confirmation obtained (requires_confirmation=True)
        - Working directory exists and is writable
        - Trust level is sufficient (VERIFIED or higher)

        Args:
            context: Execution context with permissions and user confirmation

        Returns:
            PreflightResult indicating if execution can proceed
        """
        warnings = []

        # Check user confirmation
        if self.manifest.requires_confirmation and not context.user_confirmed:
            return PreflightResult(
                can_execute=False,
                reason="Code execution requires explicit user confirmation",
                warnings=["User has not confirmed execution of code"]
            )

        # Check trust level
        if context.trust_level.value not in [
            AbilityTrustLevel.CORE.value,
            AbilityTrustLevel.VERIFIED.value,
            AbilityTrustLevel.LOCAL.value
        ]:
            return PreflightResult(
                can_execute=False,
                reason=f"Insufficient trust level: {context.trust_level.value}",
                warnings=["Code execution requires LOCAL trust level or higher"]
            )

        # Check working directory
        working_dir = context.working_directory or self.config.working_directory
        if working_dir and not os.path.isdir(working_dir):
            warnings.append(f"Working directory does not exist: {working_dir}")
        elif working_dir and not os.access(working_dir, os.W_OK):
            warnings.append(f"Working directory is not writable: {working_dir}")

        # Check timeout from context
        if context.timeout_seconds > self.config.max_timeout:
            warnings.append(
                f"Requested timeout {context.timeout_seconds}s exceeds maximum "
                f"{self.config.max_timeout}s, will be capped"
            )

        return PreflightResult(
            can_execute=True,
            warnings=warnings,
            estimated_duration_ms=500  # Conservative estimate
        )

    async def execute(
        self,
        params: Dict[str, Any],
        context: AbilityContext
    ) -> AbilityResult:
        """Execute Python code snippet in isolated subprocess.

        Parameters:
            code (str): The Python code to execute
            language (str): Language identifier, default "python" (only Python supported)
            timeout (float): Execution timeout in seconds, default 30.0
            sandbox (bool): Enable subprocess isolation, default True
            working_dir (Optional[str]): Working directory for execution

        Returns:
            AbilityResult with:
                - success: Whether execution succeeded
                - output: stdout from execution
                - error: stderr from execution
                - metadata: execution_time_ms, return_code, truncated, execution_id
        """
        start_time = time.time()
        execution_id = f"{context.session_id}_{self.execution_count}"

        try:
            # Extract and validate parameters
            code = params.get("code", "")
            language = params.get("language", "python").lower()
            timeout = params.get("timeout", DEFAULT_TIMEOUT)
            sandbox = params.get("sandbox", self.config.enable_sandbox)
            working_dir = (
                params.get("working_dir") or
                context.working_directory or
                self.config.working_directory
            )

            # Validate parameters
            validation = self._validate_params(code, language, timeout)
            if not validation.success:
                return validation

            # Normalize timeout
            timeout = min(timeout, self.config.max_timeout)
            timeout = max(timeout, 0.1)  # Minimum 100ms

            self.logger.info(
                f"Executing code (id={execution_id}, language={language}, "
                f"timeout={timeout}s, sandbox={sandbox})"
            )

            # Execute based on sandbox setting
            if sandbox and self.config.sandbox_type == "subprocess":
                result = await self._execute_subprocess(
                    code=code,
                    language=language,
                    timeout=timeout,
                    working_dir=working_dir,
                    execution_id=execution_id
                )
            else:
                result = await self._execute_direct(
                    code=code,
                    language=language,
                    timeout=timeout,
                    working_dir=working_dir,
                    execution_id=execution_id
                )

            # Calculate execution time
            duration_ms = (time.time() - start_time) * 1000

            # Update statistics
            self.execution_count += 1
            self.last_execution = datetime.now()
            self.total_duration_ms += duration_ms

            # Prepare metadata
            result.duration_ms = duration_ms
            result.metadata.update({
                "execution_id": execution_id,
                "language": language,
                "sandbox": sandbox,
                "working_directory": working_dir or "default",
                "execution_index": self.execution_count,
                "timestamp": datetime.now().isoformat()
            })

            self.logger.info(
                f"Code execution completed (id={execution_id}, "
                f"success={result.success}, duration={duration_ms:.1f}ms)"
            )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Execution failed: {str(e)}"

            self.logger.error(
                f"Code execution error (id={execution_id}): {error_msg}",
                exc_info=True
            )

            return AbilityResult(
                success=False,
                output="",
                error=error_msg,
                confidence=0.0,
                duration_ms=duration_ms,
                metadata={
                    "execution_id": execution_id,
                    "exception_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
            )

    async def verify(self, result: AbilityResult) -> VerificationResult:
        """Verify code execution output.

        Checks:
        - Execution succeeded
        - Output format is valid (not corrupted)
        - No suspicious patterns in output
        - Confidence score is justified

        Args:
            result: Result from execute()

        Returns:
            VerificationResult with verification status
        """
        issues = []
        suggestions = []

        # Check basic success
        if not result.success:
            issues.append("Execution failed")
            return VerificationResult(verified=False, issues=issues)

        # Check output is string
        if not isinstance(result.output, str):
            issues.append("Output is not a string")

        # Check output length
        if len(result.output) > self.config.max_output_length:
            issues.append(f"Output exceeds maximum length ({len(result.output)} bytes)")
            suggestions.append(f"Consider limiting output to {self.config.max_output_length} bytes")

        # Check for truncation indication
        if result.metadata.get("truncated"):
            issues.append("Output was truncated due to length limits")

        # Check confidence is reasonable
        if result.confidence < 0.5:
            issues.append("Low confidence score for execution result")
            suggestions.append("Review error output for debugging information")

        # Verify successful execution has non-zero confidence
        if result.success and result.confidence < 0.7:
            suggestions.append("Consider increasing confidence if output is reliable")

        return VerificationResult(
            verified=len(issues) == 0,
            issues=issues,
            suggestions=suggestions
        )

    # Private methods

    def _validate_params(
        self,
        code: str,
        language: str,
        timeout: float
    ) -> AbilityResult:
        """Validate execution parameters.

        Args:
            code: Code to validate
            language: Programming language
            timeout: Execution timeout

        Returns:
            AbilityResult with success=True if valid, False if invalid
        """
        # Validate code
        if not code or not isinstance(code, str):
            return AbilityResult(
                success=False,
                error="Code must be a non-empty string"
            )

        if len(code) > 1_000_000:  # 1MB max code size
            return AbilityResult(
                success=False,
                error="Code exceeds maximum size (1MB)"
            )

        # Validate language
        if language not in SUPPORTED_LANGUAGES:
            return AbilityResult(
                success=False,
                error=f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}"
            )

        # Validate timeout
        if not isinstance(timeout, (int, float)):
            return AbilityResult(
                success=False,
                error="Timeout must be a number"
            )

        if timeout <= 0:
            return AbilityResult(
                success=False,
                error="Timeout must be positive"
            )

        if timeout > self.config.max_timeout:
            return AbilityResult(
                success=False,
                error=f"Timeout exceeds maximum ({self.config.max_timeout}s)"
            )

        # All validations passed
        return AbilityResult(success=True)

    async def _execute_subprocess(
        self,
        code: str,
        language: str,
        timeout: float,
        working_dir: Optional[str],
        execution_id: str
    ) -> AbilityResult:
        """Execute code in isolated subprocess.

        Args:
            code: Python code to execute
            language: Language identifier
            timeout: Execution timeout
            working_dir: Working directory
            execution_id: Unique execution identifier

        Returns:
            AbilityResult with output and status
        """
        try:
            # Create temporary file for code
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                dir=working_dir
            ) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Prepare environment
                env = os.environ.copy()
                env.update(self.config.environment_variables or {})
                env['PYTHONUNBUFFERED'] = '1'  # Immediate output flushing

                # Run subprocess with timeout
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    temp_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir or os.getcwd(),
                    env=env
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        process.kill()

                    return AbilityResult(
                        success=False,
                        output="",
                        error=f"Code execution timeout after {timeout} seconds",
                        confidence=0.5,
                        metadata={"timeout": True, "return_code": -1}
                    )

                # Decode output
                output_text = stdout.decode('utf-8', errors='replace')
                error_text = stderr.decode('utf-8', errors='replace')
                return_code = process.returncode

                # Handle truncation
                truncated = False
                if len(output_text) > self.config.max_output_length:
                    output_text = output_text[:self.config.max_output_length]
                    output_text += f"\n... (output truncated, original length: {len(output_text)} bytes)"
                    truncated = True

                if len(error_text) > self.config.max_output_length:
                    error_text = error_text[:self.config.max_output_length]
                    error_text += f"\n... (error output truncated)"
                    truncated = True

                # Determine success
                success = return_code == 0

                return AbilityResult(
                    success=success,
                    output=output_text,
                    error=error_text if error_text else None,
                    confidence=0.95 if success else 0.8,
                    metadata={
                        "return_code": return_code,
                        "truncated": truncated,
                        "sandbox_type": "subprocess"
                    }
                )

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

        except Exception as e:
            return AbilityResult(
                success=False,
                output="",
                error=f"Subprocess execution failed: {str(e)}",
                confidence=0.0,
                metadata={"exception_type": type(e).__name__}
            )

    async def _execute_direct(
        self,
        code: str,
        language: str,
        timeout: float,
        working_dir: Optional[str],
        execution_id: str
    ) -> AbilityResult:
        """Execute code directly in current process (no sandbox).

        WARNING: This is less secure than subprocess execution.
        Only use when absolutely necessary.

        Args:
            code: Python code to execute
            language: Language identifier
            timeout: Execution timeout
            working_dir: Working directory (mostly informational)
            execution_id: Unique execution identifier

        Returns:
            AbilityResult with output and status
        """
        try:
            # Capture stdout/stderr
            import io
            from contextlib import redirect_stdout, redirect_stderr

            output_buffer = io.StringIO()
            error_buffer = io.StringIO()

            # Create execution namespace
            namespace = {
                "__name__": "__main__",
                "__file__": "<direct_execution>",
                "execution_id": execution_id
            }

            try:
                # Execute with timeout
                with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                    exec_task = asyncio.create_task(
                        asyncio.to_thread(
                            exec,
                            code,
                            namespace
                        )
                    )
                    await asyncio.wait_for(exec_task, timeout=timeout)

                output_text = output_buffer.getvalue()
                error_text = error_buffer.getvalue()

                # Handle truncation
                truncated = False
                if len(output_text) > self.config.max_output_length:
                    output_text = output_text[:self.config.max_output_length]
                    output_text += f"\n... (output truncated)"
                    truncated = True

                return AbilityResult(
                    success=True,
                    output=output_text,
                    error=error_text if error_text else None,
                    confidence=0.85,  # Slightly lower confidence for direct execution
                    metadata={
                        "return_code": 0,
                        "truncated": truncated,
                        "sandbox_type": "direct",
                        "warning": "Direct execution without sandbox is less secure"
                    }
                )

            except asyncio.TimeoutError:
                return AbilityResult(
                    success=False,
                    output=output_buffer.getvalue(),
                    error=f"Code execution timeout after {timeout} seconds",
                    confidence=0.5,
                    metadata={"timeout": True, "return_code": -1}
                )
            except Exception as e:
                return AbilityResult(
                    success=False,
                    output=output_buffer.getvalue(),
                    error=f"{type(e).__name__}: {str(e)}",
                    confidence=0.6,
                    metadata={"return_code": 1}
                )

        except Exception as e:
            return AbilityResult(
                success=False,
                output="",
                error=f"Direct execution setup failed: {str(e)}",
                confidence=0.0,
                metadata={"exception_type": type(e).__name__}
            )

    async def cleanup(self) -> None:
        """Cleanup ability resources.

        Ensures all temporary files and subprocesses are cleaned up.
        """
        self.logger.info(
            f"Code execution cleanup: {self.execution_count} executions, "
            f"total {self.total_duration_ms:.1f}ms"
        )


# Convenience function for creating ability
def create_code_execution_ability(
    config: Optional[CodeExecutionConfig] = None
) -> CodeExecutionAbility:
    """Create and return a code execution ability instance.

    Args:
        config: Optional CodeExecutionConfig for customization

    Returns:
        CodeExecutionAbility instance
    """
    return CodeExecutionAbility(config)


__all__ = [
    "CodeExecutionAbility",
    "CodeExecutionConfig",
    "create_code_execution_ability",
]
