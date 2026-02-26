"""
Proto Matrix ChatOps Handlers
==============================

Matrix/Element bot integration for Proto code agent department.

Provides chat commands for code assistance, review, testing, and security scanning
in Matrix rooms. Built on the HoloLoom ChatOps architecture.

Commands:
    !proto <query>          - Ask Proto any code-related question
    !proto review <file>    - Run code review on a file
    !proto explain <code>   - Explain code snippet
    !proto test <file>      - Run tests on a file/directory
    !proto security <file>  - Run security scan on a file
    !proto git <cmd>        - Execute git operation (status, diff, log, etc.)
    !proto run <code>       - Execute Python code in sandbox
    !proto help             - Show available commands

Architecture:
    - Follows HoloLoomMatrixHandlers pattern
    - Routes through ProtoEngine.process() for unified handling
    - Supports room-based session persistence
    - Async message streaming for long operations

Integration:
    - Uses Tier 2 abilities (code_execution, git_operations, test_runner, security_scan)
    - Safety gating via alignment framework
    - Audit trail for all operations

Status: Production ready (2025-12-04)
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Awaitable

from HoloLoom.apps.departments.proto.core import ProtoEngine, ProtoConfig
from HoloLoom.apps.departments.proto.domain import CodeContext

logger = logging.getLogger("proto.matrix")


@dataclass
class MatrixMessage:
    """Represents a Matrix message for processing."""
    room_id: str
    sender: str
    content: str
    event_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    reply_to: Optional[str] = None
    attachments: List[str] = field(default_factory=list)


@dataclass
class ProtoSession:
    """Room-based Proto session state."""
    room_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    context_file: Optional[str] = None
    history: List[Dict[str, str]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)

    def add_exchange(self, user_msg: str, proto_response: str) -> None:
        """Add a message exchange to history."""
        self.history.append({
            "user": user_msg,
            "proto": proto_response,
            "timestamp": datetime.now().isoformat()
        })
        self.last_activity = datetime.now()
        # Keep last 20 exchanges
        if len(self.history) > 20:
            self.history = self.history[-20:]


class ProtoMatrixHandlers:
    """
    Matrix ChatOps handlers for Proto code agent.

    Provides command handling for Proto interactions in Matrix rooms.
    Follows the HoloLoom ChatOps handler pattern with safety gating
    and audit trail integration.

    Usage:
        handlers = ProtoMatrixHandlers(config)
        await handlers.startup()

        # Register with Matrix client
        handlers.register_all(matrix_client)

        # Handle incoming message
        response = await handlers.handle_message(message)

        # Cleanup
        await handlers.shutdown()

    Attributes:
        config: ProtoConfig instance
        engine: ProtoEngine for processing requests
        sessions: Room-based session storage
        handlers: Registered command handlers
    """

    # Command prefix for Proto
    PREFIX = "!proto"

    # Help text
    HELP_TEXT = """
**Proto Code Assistant - Commands**

`!proto <question>` - Ask any code-related question
`!proto review <file>` - Review code for issues and improvements
`!proto explain` - Explain selected/pasted code
`!proto test <file>` - Run tests on file or directory
`!proto security <file>` - Scan for security vulnerabilities
`!proto git <cmd>` - Run git command (status, diff, log, show, blame)
`!proto run <code>` - Execute Python code in sandbox
`!proto context <file>` - Set context file for session
`!proto history` - Show recent conversation history
`!proto help` - Show this help message

**Examples:**
- `!proto What is the difference between async and threading?`
- `!proto review src/main.py`
- `!proto test tests/ -v`
- `!proto security app.py`
- `!proto git diff HEAD~3`
- `!proto run print("Hello, World!")`
"""

    def __init__(
        self,
        config: Optional[ProtoConfig] = None,
        engine: Optional[ProtoEngine] = None,
        enable_audit: bool = True,
    ):
        """Initialize Proto Matrix handlers.

        Args:
            config: ProtoConfig instance (creates default if None)
            engine: ProtoEngine instance (creates from config if None)
            enable_audit: Enable audit trail logging
        """
        self.config = config or ProtoConfig.default()
        self.engine = engine
        self.enable_audit = enable_audit

        # Session storage (room_id -> ProtoSession)
        self.sessions: Dict[str, ProtoSession] = {}

        # Command handlers (command -> handler function)
        self.handlers: Dict[str, Callable[[MatrixMessage, str], Awaitable[str]]] = {}

        # Message send callback (set by Matrix client)
        self._send_message: Optional[Callable[[str, str], Awaitable[None]]] = None

        # Audit trail
        self._audit_log: List[Dict[str, Any]] = []

        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register all command handlers."""
        self.handlers = {
            "help": self._handle_help,
            "review": self._handle_review,
            "explain": self._handle_explain,
            "test": self._handle_test,
            "security": self._handle_security,
            "git": self._handle_git,
            "run": self._handle_run,
            "context": self._handle_context,
            "history": self._handle_history,
        }

    async def startup(self) -> None:
        """Initialize handler resources."""
        logger.info("Proto Matrix handlers starting...")

        # Create engine if not provided
        if self.engine is None:
            self.engine = ProtoEngine(self.config)
            await self.engine.startup()

        logger.info("Proto Matrix handlers started")

    async def shutdown(self) -> None:
        """Clean up handler resources."""
        logger.info("Proto Matrix handlers shutting down...")

        # Shutdown engine if we created it
        if self.engine:
            await self.engine.shutdown()

        # Clear sessions
        self.sessions.clear()

        logger.info("Proto Matrix handlers shut down")

    def register_all(self, client: Any) -> None:
        """Register handlers with Matrix client.

        Args:
            client: Matrix client instance
        """
        # Store send callback
        if hasattr(client, "send_message"):
            self._send_message = client.send_message

        # Register message handler
        if hasattr(client, "on_message"):
            client.on_message(self.handle_message)

        logger.info(f"Registered Proto handlers with prefix '{self.PREFIX}'")

    async def handle_message(self, message: MatrixMessage) -> Optional[str]:
        """Handle incoming Matrix message.

        Args:
            message: Matrix message to process

        Returns:
            Response text if message was handled, None otherwise
        """
        content = message.content.strip()

        # Check for Proto prefix
        if not content.lower().startswith(self.PREFIX):
            return None

        # Extract command and args
        parts = content[len(self.PREFIX):].strip().split(maxsplit=1)
        command = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        # Get or create session
        session = self._get_session(message.room_id)

        # Log to audit trail
        if self.enable_audit:
            self._log_audit(message, command, args)

        # Route to handler
        if command in self.handlers:
            try:
                response = await self.handlers[command](message, args)
            except Exception as e:
                logger.error(f"Handler error for '{command}': {e}", exc_info=True)
                response = f"Error: {str(e)}"
        elif command == "" or command == "ask":
            # Default: treat as general question
            response = await self._handle_ask(message, args if args else command)
        else:
            # Unknown command with args - treat as question
            full_query = f"{command} {args}".strip()
            response = await self._handle_ask(message, full_query)

        # Update session history
        session.add_exchange(content, response)

        return response

    def _get_session(self, room_id: str) -> ProtoSession:
        """Get or create session for room."""
        if room_id not in self.sessions:
            self.sessions[room_id] = ProtoSession(room_id=room_id)
        return self.sessions[room_id]

    def _log_audit(self, message: MatrixMessage, command: str, args: str) -> None:
        """Log command to audit trail."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "room_id": message.room_id,
            "sender": message.sender,
            "command": command,
            "args": args[:100] + "..." if len(args) > 100 else args,
            "event_id": message.event_id,
        }
        self._audit_log.append(entry)

        # Keep last 1000 entries
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]

        logger.debug(f"Audit: {message.sender} -> {command} in {message.room_id}")

    # =========================================================================
    # Command Handlers
    # =========================================================================

    async def _handle_help(self, message: MatrixMessage, args: str) -> str:
        """Handle !proto help command."""
        return self.HELP_TEXT

    async def _handle_ask(self, message: MatrixMessage, query: str) -> str:
        """Handle general Proto query.

        This is the default handler for any question without a specific command.
        """
        if not query:
            return "Please provide a question. Example: `!proto What is dependency injection?`"

        session = self._get_session(message.room_id)
        context = None

        # Use context file if set
        if session.context_file:
            try:
                from pathlib import Path
                path = Path(session.context_file)
                if path.exists():
                    context = CodeContext(
                        file_path=str(path),
                        content=path.read_text(),
                        language=path.suffix.lstrip('.') or 'text'
                    )
            except Exception as e:
                logger.warning(f"Could not load context file: {e}")

        # Process through engine
        response = await self.engine.process(query, context)

        return response.content

    async def _handle_review(self, message: MatrixMessage, args: str) -> str:
        """Handle !proto review <file> command."""
        if not args:
            return "Please specify a file to review. Example: `!proto review src/main.py`"

        file_path = args.split()[0]

        try:
            from pathlib import Path
            path = Path(file_path)
            if not path.exists():
                return f"File not found: {file_path}"

            content = path.read_text()
            language = path.suffix.lstrip('.') or 'text'

            context = CodeContext(
                file_path=str(path),
                content=content,
                language=language
            )

            response = await self.engine.process(
                f"Review this {language} code for issues, improvements, and best practices",
                context
            )

            return f"**Code Review: {file_path}**\n\n{response.content}"

        except Exception as e:
            return f"Error reviewing file: {e}"

    async def _handle_explain(self, message: MatrixMessage, args: str) -> str:
        """Handle !proto explain command."""
        if not args:
            session = self._get_session(message.room_id)
            if session.context_file:
                args = f"the code in {session.context_file}"
            else:
                return "Please provide code to explain or set a context file with `!proto context <file>`"

        # Check if args is a file path
        try:
            from pathlib import Path
            path = Path(args.split()[0])
            if path.exists() and path.is_file():
                content = path.read_text()
                language = path.suffix.lstrip('.') or 'text'
                context = CodeContext(
                    file_path=str(path),
                    content=content,
                    language=language
                )
                response = await self.engine.process(
                    f"Explain this {language} code in detail",
                    context
                )
                return f"**Explanation: {path.name}**\n\n{response.content}"
        except Exception:
            pass

        # Treat as inline code
        response = await self.engine.process(f"Explain this code:\n```\n{args}\n```")
        return response.content

    async def _handle_test(self, message: MatrixMessage, args: str) -> str:
        """Handle !proto test <file/directory> command."""
        if not args:
            return "Please specify a file or directory to test. Example: `!proto test tests/`"

        from HoloLoom.apps.departments.proto.abilities.core import TestRunnerAbility

        runner = TestRunnerAbility()
        target = args.split()[0]
        extra_args = args[len(target):].strip()

        try:
            # Check if TestRunnerAbility has execute method
            if hasattr(runner, 'execute'):
                result = await runner.execute({
                    "operation": "run_tests",
                    "target": target,
                    "args": extra_args.split() if extra_args else [],
                })
            else:
                # Fallback: use engine
                response = await self.engine.process(
                    f"Run tests for {target} {extra_args}".strip()
                )
                return f"**Test Results: {target}**\n\n{response.content}"

            # Format test results
            if result.get("success"):
                status = "[PASS]"
            else:
                status = "[FAIL]"

            output = result.get("output", "No output")
            summary = result.get("summary", {})

            lines = [
                f"**Test Results: {target}** {status}",
                "",
                f"**Summary:**",
                f"- Passed: {summary.get('passed', 'N/A')}",
                f"- Failed: {summary.get('failed', 'N/A')}",
                f"- Skipped: {summary.get('skipped', 'N/A')}",
                f"- Duration: {summary.get('duration_s', 'N/A')}s",
                "",
                "**Output:**",
                f"```\n{output[:2000]}{'...' if len(output) > 2000 else ''}\n```"
            ]

            return "\n".join(lines)

        except Exception as e:
            return f"Error running tests: {e}"

    async def _handle_security(self, message: MatrixMessage, args: str) -> str:
        """Handle !proto security <file> command."""
        if not args:
            return "Please specify a file to scan. Example: `!proto security app.py`"

        from HoloLoom.apps.departments.proto.abilities.core import SecurityScanAbility

        scanner = SecurityScanAbility()
        file_path = args.split()[0]

        try:
            from pathlib import Path
            path = Path(file_path)
            if not path.exists():
                return f"File not found: {file_path}"

            # Check if SecurityScanAbility has execute method
            if hasattr(scanner, 'execute'):
                result = await scanner.execute({
                    "file_path": str(path),
                })
            else:
                # Fallback: use engine
                response = await self.engine.process(
                    f"Scan {file_path} for security vulnerabilities"
                )
                return f"**Security Scan: {file_path}**\n\n{response.content}"

            # Format security results
            findings = result.get("findings", [])
            severity_counts = result.get("severity_counts", {})

            if not findings:
                return f"**Security Scan: {file_path}**\n\n[OK] No security issues found."

            lines = [
                f"**Security Scan: {file_path}**",
                "",
                f"**Summary:**",
                f"- Critical: {severity_counts.get('CRITICAL', 0)}",
                f"- High: {severity_counts.get('HIGH', 0)}",
                f"- Medium: {severity_counts.get('MEDIUM', 0)}",
                f"- Low: {severity_counts.get('LOW', 0)}",
                "",
                "**Findings:**",
            ]

            for finding in findings[:10]:  # Limit to 10 findings
                severity = finding.get("severity", "UNKNOWN")
                issue = finding.get("issue_type", "Unknown")
                line = finding.get("line_number", "?")
                desc = finding.get("description", "No description")

                emoji = {
                    "CRITICAL": "!!!",
                    "HIGH": "!!",
                    "MEDIUM": "!",
                    "LOW": ".",
                }.get(severity, "?")

                lines.append(f"- {emoji} **{severity}** (line {line}): {issue}")
                lines.append(f"  {desc[:100]}{'...' if len(desc) > 100 else ''}")

            if len(findings) > 10:
                lines.append(f"\n... and {len(findings) - 10} more findings")

            return "\n".join(lines)

        except Exception as e:
            return f"Error scanning file: {e}"

    async def _handle_git(self, message: MatrixMessage, args: str) -> str:
        """Handle !proto git <command> command."""
        if not args:
            return """Git commands available:
- `!proto git status` - Show working tree status
- `!proto git diff [file]` - Show changes
- `!proto git log [-n N]` - Show commit history
- `!proto git show <commit>` - Show commit details
- `!proto git blame <file>` - Show line-by-line authorship
- `!proto git branch` - List branches"""

        from HoloLoom.apps.departments.proto.abilities.core import GitOperationsAbility

        git = GitOperationsAbility()
        parts = args.split(maxsplit=1)
        operation = parts[0]
        op_args = parts[1] if len(parts) > 1 else ""

        # Map commands to operations
        op_map = {
            "status": "status",
            "diff": "diff",
            "log": "log",
            "show": "show",
            "blame": "blame",
            "branch": "branch",
            "branches": "branch",
            "stash": "stash_list",
        }

        if operation not in op_map:
            return f"Unknown git command: {operation}. Use `!proto git` for available commands."

        try:
            if hasattr(git, 'execute'):
                result = await git.execute({
                    "operation": op_map[operation],
                    "args": op_args.split() if op_args else [],
                })
            else:
                # Fallback
                response = await self.engine.process(f"git {args}")
                return f"```\n{response.content}\n```"

            output = result.get("output", "No output")
            if result.get("success"):
                return f"```\n{output[:3000]}{'...' if len(output) > 3000 else ''}\n```"
            else:
                return f"Git error:\n```\n{output}\n```"

        except Exception as e:
            return f"Error running git command: {e}"

    async def _handle_run(self, message: MatrixMessage, args: str) -> str:
        """Handle !proto run <code> command."""
        if not args:
            return "Please provide Python code to run. Example: `!proto run print('Hello!')`"

        from HoloLoom.apps.departments.proto.abilities.core import CodeExecutionAbility

        executor = CodeExecutionAbility()

        # Extract code from markdown code block if present
        code = args
        code_match = re.search(r'```(?:python)?\s*(.*?)```', args, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()

        try:
            if hasattr(executor, 'execute'):
                result = await executor.execute({
                    "code": code,
                    "timeout": 10.0,  # 10 second timeout for chat
                })
            else:
                # Fallback
                response = await self.engine.process(f"Run this Python code:\n```python\n{code}\n```")
                return response.content

            output = result.get("output", "")
            error = result.get("error", "")
            success = result.get("success", False)

            if success:
                return f"**Output:**\n```\n{output[:2000]}{'...' if len(output) > 2000 else ''}\n```"
            else:
                return f"**Error:**\n```\n{error[:2000]}{'...' if len(error) > 2000 else ''}\n```"

        except Exception as e:
            return f"Error executing code: {e}"

    async def _handle_context(self, message: MatrixMessage, args: str) -> str:
        """Handle !proto context <file> command."""
        session = self._get_session(message.room_id)

        if not args:
            if session.context_file:
                return f"Current context file: `{session.context_file}`\n\nUse `!proto context <file>` to change or `!proto context clear` to remove."
            else:
                return "No context file set. Use `!proto context <file>` to set one."

        if args.lower() == "clear":
            session.context_file = None
            return "Context file cleared."

        try:
            from pathlib import Path
            path = Path(args)
            if not path.exists():
                return f"File not found: {args}"
            if not path.is_file():
                return f"Not a file: {args}"

            session.context_file = str(path.absolute())
            return f"Context set to: `{session.context_file}`\n\nAll queries will now include this file as context."

        except Exception as e:
            return f"Error setting context: {e}"

    async def _handle_history(self, message: MatrixMessage, args: str) -> str:
        """Handle !proto history command."""
        session = self._get_session(message.room_id)

        if not session.history:
            return "No conversation history in this room."

        lines = ["**Recent Conversation History:**", ""]

        # Show last 5 exchanges
        for i, exchange in enumerate(session.history[-5:], 1):
            user_msg = exchange["user"][:80] + "..." if len(exchange["user"]) > 80 else exchange["user"]
            proto_msg = exchange["proto"][:80] + "..." if len(exchange["proto"]) > 80 else exchange["proto"]

            lines.append(f"**{i}.** You: {user_msg}")
            lines.append(f"   Proto: {proto_msg}")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries.

        Args:
            limit: Maximum entries to return

        Returns:
            List of audit log entries
        """
        return self._audit_log[-limit:]

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions.

        Returns:
            Session statistics dict
        """
        return {
            "active_sessions": len(self.sessions),
            "total_exchanges": sum(len(s.history) for s in self.sessions.values()),
            "rooms": list(self.sessions.keys()),
        }

    async def send_typing(self, room_id: str, timeout: int = 5000) -> None:
        """Send typing indicator to room.

        Args:
            room_id: Matrix room ID
            timeout: Typing timeout in milliseconds
        """
        # Would be implemented by Matrix client
        pass


# Factory function
def create_proto_handlers(
    config: Optional[ProtoConfig] = None,
    enable_audit: bool = True,
) -> ProtoMatrixHandlers:
    """Create Proto Matrix handlers instance.

    Args:
        config: ProtoConfig (creates default if None)
        enable_audit: Enable audit trail logging

    Returns:
        Configured ProtoMatrixHandlers instance
    """
    return ProtoMatrixHandlers(config=config, enable_audit=enable_audit)


__all__ = [
    "ProtoMatrixHandlers",
    "MatrixMessage",
    "ProtoSession",
    "create_proto_handlers",
]
