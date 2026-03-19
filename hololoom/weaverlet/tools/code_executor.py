"""
Sandboxed Code Executor
========================

Runs code in a subprocess with timeout and output capture.
Same pattern as TestSignal in tapestry/signals/.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SandboxedCodeExecutor:

    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout: float = 30.0,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """
        Run code in subprocess, capture stdout/stderr.

        Returns dict with: success, stdout, stderr, exit_code, timed_out.
        """
        if language == "python":
            cmd = ["python3", "-c", code]
        elif language == "shell":
            cmd = ["bash", "-c", code]
        else:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Unsupported language: {language}",
                "exit_code": -1,
                "timed_out": False,
            }

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            return {
                "success": proc.returncode == 0,
                "stdout": stdout.decode(errors="replace")[:10_000],
                "stderr": stderr.decode(errors="replace")[:10_000],
                "exit_code": proc.returncode,
                "timed_out": False,
            }
        except asyncio.TimeoutError:
            proc.kill()
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Timed out after {timeout}s",
                "exit_code": -1,
                "timed_out": True,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "timed_out": False,
            }
