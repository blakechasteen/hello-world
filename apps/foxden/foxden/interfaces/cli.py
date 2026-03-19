"""CLI interface — test command-line applications via subprocess."""

from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass, field

from .base import Action, Observation


@dataclass
class CLIInterface:
    """Interact with a CLI application."""

    command: str  # base command to run (e.g. "python app.py", "npm start")
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    shell_history: list[str] = field(default_factory=list, init=False)

    async def start(self) -> Observation:
        welcome = f"CLI session ready. Base command: {self.command}\nYou can run commands by sending actions with kind='command'."
        return Observation(content=welcome)

    async def act(self, action: Action) -> Observation:
        if action.kind != "command":
            return Observation(content=f"Unknown action kind: {action.kind}. Use kind='command'.")

        cmd = action.value
        self.shell_history.append(cmd)

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env={**dict(__import__("os").environ), **self.env} if self.env else None,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            return Observation(
                content=f"TIMEOUT after {self.timeout}s running: {cmd}",
                raw={"exit_code": -1, "timeout": True},
            )

        stdout_text = stdout.decode(errors="replace").strip()
        stderr_text = stderr.decode(errors="replace").strip()

        parts = []
        if stdout_text:
            parts.append(f"STDOUT:\n{stdout_text}")
        if stderr_text:
            parts.append(f"STDERR:\n{stderr_text}")
        parts.append(f"EXIT CODE: {proc.returncode}")

        return Observation(
            content="\n\n".join(parts),
            raw={
                "exit_code": proc.returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
            },
        )

    async def close(self) -> None:
        pass
