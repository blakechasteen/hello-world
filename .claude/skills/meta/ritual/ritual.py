"""
/ritual - Coding Ritual Command Router
======================================

Single entry point for all ritual commands.

Usage:
    from ritual import ritual

    # Open a session
    await ritual("open", task="Build feature X")

    # Check status
    await ritual("status")

    # Build wave 1
    await ritual("build", wave=1)
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from state.session_manager import SessionManager


# Shared session manager (singleton pattern)
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create the shared session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


async def ritual(command: str, **kwargs) -> Dict[str, Any]:
    """
    Execute a ritual command.

    Commands:
        open    - Start a new coding session
        design  - Protocol-first interface design
        build   - Wave validation (1=Core, 2=Edge, 3=Integration)
        check   - Safety checklist review
        close   - End session and archive
        status  - Show current progress
        help    - Show command reference

    Examples:
        await ritual("open", task="Build REST API", complexity="fast")
        await ritual("design", component="UserService")
        await ritual("build", wave=1, wave_status="pass")
        await ritual("check", check_results={"input_validation": True})
        await ritual("close", lessons=["Learned about caching"])
        await ritual("status")

    Returns:
        Dict with: phase, session_id, status, message, next_steps, data, metadata
    """
    command = command.lower().strip()
    sm = get_session_manager()

    if command == "open":
        from phases.open import OpenPhaseHandler
        handler = OpenPhaseHandler(session_manager=sm)
        return await handler.execute(
            task=kwargs.get("task", "Unnamed task"),
            complexity=kwargs.get("complexity", "fast"),
            success_criteria=kwargs.get("criteria", kwargs.get("success_criteria", [])),
        )

    elif command == "design":
        from phases.design import DesignPhaseHandler
        handler = DesignPhaseHandler(session_manager=sm)
        return await handler.execute(
            component=kwargs.get("component"),
            design_type=kwargs.get("design_type"),
            requirements=kwargs.get("requirements", []),
            constraints=kwargs.get("constraints", []),
        )

    elif command == "build":
        from phases.build import BuildPhaseHandler
        handler = BuildPhaseHandler(session_manager=sm)
        return await handler.execute(
            wave=kwargs.get("wave"),
            wave_status=kwargs.get("wave_status"),
            wave_notes=kwargs.get("wave_notes"),
            checklist_results=kwargs.get("checklist_results"),
        )

    elif command == "check":
        from phases.check import CheckPhaseHandler
        handler = CheckPhaseHandler(session_manager=sm)
        return await handler.execute(
            check_results=kwargs.get("check_results"),
            notes=kwargs.get("notes"),
            run_automated=kwargs.get("run_automated", False),
        )

    elif command == "close":
        from phases.close import ClosePhaseHandler
        handler = ClosePhaseHandler(session_manager=sm)
        return await handler.execute(
            lessons=kwargs.get("lessons", []),
            notes=kwargs.get("notes"),
        )

    elif command == "status":
        from phases.status import StatusPhaseHandler
        handler = StatusPhaseHandler(session_manager=sm)
        return await handler.execute()

    elif command == "help":
        return _help_response()

    else:
        return {
            "phase": "error",
            "status": "error",
            "message": f"Unknown command: {command}",
            "next_steps": ["Use /ritual help for available commands"],
            "data": {"available_commands": ["open", "design", "build", "check", "close", "status", "help"]},
        }


def _help_response() -> Dict[str, Any]:
    """Generate help response."""
    return {
        "phase": "help",
        "status": "success",
        "message": "Coding Ritual - Structured Development Sessions",
        "next_steps": ["Use /ritual open <task> to start a session"],
        "data": {
            "commands": {
                "open": {
                    "description": "Start a new coding session",
                    "args": {
                        "task": "What you're building (required)",
                        "complexity": "lite|fast|full|research (default: fast)",
                        "criteria": "List of success criteria",
                    },
                    "example": 'ritual("open", task="Build auth system", complexity="fast")',
                },
                "design": {
                    "description": "Protocol-first interface design",
                    "args": {
                        "component": "Name of component to design",
                        "design_type": "protocol|interface|schema|api",
                        "requirements": "List of requirements",
                    },
                    "example": 'ritual("design", component="AuthService")',
                },
                "build": {
                    "description": "Wave validation build process",
                    "args": {
                        "wave": "1 (Core), 2 (Edge), or 3 (Integration)",
                        "wave_status": "pass|fail|skip",
                        "wave_notes": "Notes about this wave",
                    },
                    "example": 'ritual("build", wave=1, wave_status="pass")',
                },
                "check": {
                    "description": "Safety-first checklist review",
                    "args": {
                        "check_results": "Dict of check_id -> bool",
                        "notes": "Dict of check_id -> note",
                        "run_automated": "Run automated checks (bool)",
                    },
                    "example": 'ritual("check", check_results={"input_validation": True})',
                },
                "close": {
                    "description": "End session and archive",
                    "args": {
                        "lessons": "List of lessons learned",
                        "notes": "Additional notes",
                    },
                    "example": 'ritual("close", lessons=["Learned about caching"])',
                },
                "status": {
                    "description": "Show current session status",
                    "args": {},
                    "example": 'ritual("status")',
                },
            },
            "workflow": [
                "1. open   - Start with a clear task",
                "2. design - Define interfaces first",
                "3. build  - Wave 1 (Core) -> Wave 2 (Edge) -> Wave 3 (Integration)",
                "4. check  - Review safety checklist",
                "5. close  - Capture lessons learned",
            ],
            "complexity_levels": {
                "lite": "Quick tasks (<30 min), minimal ceremony",
                "fast": "Standard tasks (30-120 min), balanced (default)",
                "full": "Complex tasks (2-8 hrs), full validation",
                "research": "Exploration, no time limit",
            },
        },
    }


# Convenience shortcuts
async def open_session(task: str, complexity: str = "fast", criteria: List[str] = None) -> Dict[str, Any]:
    """Shortcut: Start a new ritual session."""
    return await ritual("open", task=task, complexity=complexity, criteria=criteria or [])


async def design(component: str = None, **kwargs) -> Dict[str, Any]:
    """Shortcut: Start design phase."""
    return await ritual("design", component=component, **kwargs)


async def build(wave: int = None, status: str = None, **kwargs) -> Dict[str, Any]:
    """Shortcut: Run a build wave."""
    return await ritual("build", wave=wave, wave_status=status, **kwargs)


async def check(results: Dict[str, bool] = None, **kwargs) -> Dict[str, Any]:
    """Shortcut: Run safety check."""
    return await ritual("check", check_results=results, **kwargs)


async def close_session(lessons: List[str] = None, **kwargs) -> Dict[str, Any]:
    """Shortcut: Close the session."""
    return await ritual("close", lessons=lessons or [], **kwargs)


async def status() -> Dict[str, Any]:
    """Shortcut: Get current status."""
    return await ritual("status")


# Pretty print helper
def print_result(result: Dict[str, Any]) -> None:
    """Pretty print a ritual result."""
    status_icons = {
        "success": "✓",
        "in_progress": "→",
        "blocked": "⚠",
        "error": "✗",
        "no_session": "○",
    }

    icon = status_icons.get(result.get("status", ""), "•")
    phase = result.get("phase", "unknown")
    message = result.get("message", "")

    print(f"\n{icon} [{phase.upper()}] {message}")

    if result.get("next_steps"):
        print("\nNext steps:")
        for step in result["next_steps"]:
            print(f"  → {step}")

    if result.get("data", {}).get("progress_percent"):
        progress = result["data"]["progress_percent"]
        filled = int(progress / 10)
        bar = "█" * filled + "░" * (10 - filled)
        print(f"\nProgress: [{bar}] {progress}%")


# Demo / CLI
if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=" * 50)
        print("Coding Ritual Demo")
        print("=" * 50)

        # Show help
        result = await ritual("help")
        print_result(result)

        # Open a session
        print("\n" + "-" * 50)
        result = await open_session(
            task="Demo: Build a greeting service",
            complexity="fast",
            criteria=["Returns greeting", "Handles empty name"]
        )
        print_result(result)

        # Check status
        print("\n" + "-" * 50)
        result = await status()
        print_result(result)

        # Design
        print("\n" + "-" * 50)
        result = await design(component="GreetingService")
        print_result(result)

        # Build wave 1
        print("\n" + "-" * 50)
        result = await build(wave=1, status="pass")
        print_result(result)

        # Check
        print("\n" + "-" * 50)
        result = await check(results={
            "input_validation": True,
            "error_messages": True,
        })
        print_result(result)

        # Close
        print("\n" + "-" * 50)
        result = await close_session(lessons=["Ritual system works!"])
        print_result(result)

        print("\n" + "=" * 50)
        print("Demo complete!")

    asyncio.run(demo())
