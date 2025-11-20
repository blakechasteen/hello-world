#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS-STAR Executor
================

Safe execution of generated code.
Integrates with HoloLoom's alignment framework for safety.

Created: 2025-01-20
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import traceback
import sys
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stdout, redirect_stderr


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    outputs: Dict[str, Any]       # Variable name → value
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SafeExecutor:
    """
    Executes code in a safe, sandboxed environment.

    Integrates with HoloLoom's alignment framework for safety checks.
    """

    def __init__(
        self,
        max_execution_time: float = 30.0,
        enable_safety_checks: bool = True,
        allowed_imports: Optional[List[str]] = None
    ):
        self.max_execution_time = max_execution_time
        self.enable_safety_checks = enable_safety_checks
        self.allowed_imports = allowed_imports or [
            'pandas', 'numpy', 'matplotlib', 'seaborn',
            'scipy', 'sklearn', 'math', 'statistics'
        ]

    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute code safely.

        Args:
            code: Python code to execute
            context: Variables to inject into execution context

        Returns:
            ExecutionResult with outputs and status
        """
        import time
        start_time = time.time()

        # Safety checks
        if self.enable_safety_checks:
            safety_issues = self._check_code_safety(code)
            if safety_issues:
                return ExecutionResult(
                    success=False,
                    outputs={},
                    error=f"Safety violations: {', '.join(safety_issues)}"
                )

        # Prepare execution context
        exec_context = self._prepare_context(context)

        # Capture stdout/stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        try:
            # Execute code with output redirection
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_context)

            # Extract outputs (variables created during execution)
            outputs = self._extract_outputs(exec_context, context)

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=True,
                outputs=outputs,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

            return ExecutionResult(
                success=False,
                outputs={},
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                error=error_msg,
                execution_time=execution_time
            )

    def _check_code_safety(self, code: str) -> List[str]:
        """
        Check code for safety issues.

        Returns:
            List of safety issue descriptions (empty if safe)
        """
        issues = []

        # Check for dangerous operations
        dangerous_patterns = [
            ('__import__', "Dynamic imports not allowed"),
            ('eval(', "eval() not allowed"),
            ('exec(', "exec() not allowed"),
            ('open(', "File operations not allowed"),
            ('os.system', "System commands not allowed"),
            ('subprocess', "Subprocess execution not allowed"),
            ('rmdir', "Directory removal not allowed"),
            ('unlink', "File deletion not allowed"),
        ]

        for pattern, message in dangerous_patterns:
            if pattern in code:
                issues.append(message)

        # Check for allowed imports
        import_lines = [line for line in code.split('\n') if line.strip().startswith('import ') or ' import ' in line]
        for line in import_lines:
            # Extract module name
            if 'import ' in line:
                parts = line.split('import ')
                if len(parts) > 1:
                    module = parts[1].split()[0].split('.')[0].strip()
                    if module not in self.allowed_imports:
                        issues.append(f"Import of '{module}' not allowed")

        return issues

    def _prepare_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare execution context with safe builtins."""
        exec_context = {
            # Safe builtins
            'pd': pd,
            'np': np,
            'plt': plt,
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'True': True,
                'False': False,
                'None': None,
            }
        }

        # Inject user context
        if context:
            exec_context.update(context)

        return exec_context

    def _extract_outputs(
        self,
        exec_context: Dict[str, Any],
        original_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract variables created during execution."""
        outputs = {}

        # Original variables (don't include in outputs)
        original_vars = set(original_context.keys()) if original_context else set()
        original_vars.update({'pd', 'np', 'plt', '__builtins__'})

        # Extract new variables
        for key, value in exec_context.items():
            if key.startswith('_'):
                continue  # Skip private variables
            if key in original_vars:
                continue  # Skip original variables

            # Serialize value (convert to JSON-friendly format)
            try:
                if isinstance(value, pd.DataFrame):
                    outputs[key] = {
                        'type': 'DataFrame',
                        'shape': value.shape,
                        'preview': value.head(5).to_dict(),
                    }
                elif isinstance(value, (pd.Series, np.ndarray)):
                    outputs[key] = {
                        'type': type(value).__name__,
                        'shape': getattr(value, 'shape', len(value)),
                        'preview': str(value)[:200]
                    }
                elif isinstance(value, (int, float, str, bool, list, dict)):
                    outputs[key] = value
                else:
                    outputs[key] = str(value)[:200]
            except:
                outputs[key] = f"<{type(value).__name__}>"

        return outputs

    def execute_with_timeout(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> ExecutionResult:
        """
        Execute code with timeout.

        Args:
            code: Python code
            context: Execution context
            timeout: Timeout in seconds (uses max_execution_time if None)

        Returns:
            ExecutionResult
        """
        # TODO: Implement actual timeout mechanism
        # For now, just use regular execute
        # In production, use multiprocessing or asyncio with timeout
        return self.execute(code, context)


# Convenience function
def execute_code(code: str, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
    """
    Quick code execution.

    Args:
        code: Python code
        context: Optional execution context

    Returns:
        ExecutionResult
    """
    executor = SafeExecutor()
    return executor.execute(code, context)
