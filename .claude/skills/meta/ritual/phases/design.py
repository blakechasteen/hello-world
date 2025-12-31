"""
/ritual design - Protocol-First Design Phase
=============================================

Guides through interface-first design:
- Clarifying questions about the component
- Interface/protocol definition
- Dependencies and fallbacks
- Decision documentation with rationale

Maps to: LIFT_OFF lifecycle state
Emits: ritual.design.started, ritual.design.completed events
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from state.session_manager import (
    SessionManager,
    SessionState,
    RitualPhase,
)


class DesignPhaseHandler:
    """
    Handles /ritual design command.

    Guides through protocol-first design:
    - Ask clarifying questions
    - Generate interface definitions
    - Identify dependencies
    - Document decisions
    """

    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        event_bus: Optional[Any] = None,
    ):
        """
        Initialize design phase handler.

        Args:
            session_manager: SessionManager instance (created if not provided)
            event_bus: Optional EventBus for emitting events
        """
        self.session_manager = session_manager or SessionManager()
        self.event_bus = event_bus

    async def execute(
        self,
        component: Optional[str] = None,
        clarifications: Optional[Dict[str, str]] = None,
        interface_spec: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        fallbacks: Optional[List[str]] = None,
        rationale: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the design phase.

        Args:
            component: Name of component being designed
            clarifications: Answers to clarifying questions
            interface_spec: Interface/protocol definition
            dependencies: List of identified dependencies
            fallbacks: List of fallback strategies
            rationale: Design decision rationale

        Returns:
            Structured result with design guidance or completion
        """
        start_time = datetime.now()

        # Check for active session
        if not self.session_manager.has_active_session():
            return self._error_response(
                "No active session. Use /ritual open to start a session first."
            )

        # Get current session
        session = self.session_manager.current_session

        # Validate we're in a valid phase for design
        valid_phases = [RitualPhase.OPEN, RitualPhase.OPENING, RitualPhase.DESIGNING]
        if session.phase not in valid_phases:
            return self._error_response(
                f"Cannot design in phase '{session.phase.value}'. "
                f"Design is available after opening and before building."
            )

        # Emit design started event
        events_emitted = []
        if self.event_bus and session.phase != RitualPhase.DESIGNING:
            event_id = await self._emit_design_started(session, component)
            if event_id:
                events_emitted.append(event_id)

        # Update session phase to designing
        if session.phase != RitualPhase.DESIGNING:
            self.session_manager.update_phase(RitualPhase.DESIGNING)

        # Determine what stage of design we're in
        if not component:
            # Stage 1: Need to know what we're designing
            return self._needs_component_response(
                session=session,
                start_time=start_time,
                events_emitted=events_emitted,
            )

        if not clarifications:
            # Stage 2: Need clarifying questions answered
            questions = self._generate_clarifying_questions(component, session.task)
            return self._needs_clarification_response(
                session=session,
                component=component,
                questions=questions,
                start_time=start_time,
                events_emitted=events_emitted,
            )

        if not interface_spec:
            # Stage 3: Generate interface suggestion
            suggested_interface = self._suggest_interface(
                component=component,
                clarifications=clarifications,
                task=session.task,
            )
            suggested_deps = self._suggest_dependencies(component, clarifications)
            suggested_fallbacks = self._suggest_fallbacks(component, clarifications)

            return self._suggest_interface_response(
                session=session,
                component=component,
                interface=suggested_interface,
                dependencies=suggested_deps,
                fallbacks=suggested_fallbacks,
                start_time=start_time,
                events_emitted=events_emitted,
            )

        # Stage 4: Record the design decision
        design_record = {
            "component": component,
            "interface_spec": interface_spec,
            "dependencies": dependencies or [],
            "fallbacks": fallbacks or [],
            "rationale": rationale or "No rationale provided",
            "clarifications": clarifications,
            "designed_at": datetime.now().isoformat(),
        }

        self.session_manager.add_design(design_record)

        # Emit design completed event
        if self.event_bus:
            event_id = await self._emit_design_completed(session, design_record)
            if event_id:
                events_emitted.append(event_id)

        # Calculate execution time
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return self._design_completed_response(
            session=session,
            design_record=design_record,
            execution_time_ms=execution_time_ms,
            events_emitted=events_emitted,
        )

    def _generate_clarifying_questions(
        self, component: str, task: str
    ) -> List[Dict[str, str]]:
        """Generate clarifying questions for the component."""
        # Generic questions that apply to most components
        questions = [
            {
                "id": "inputs",
                "question": f"What inputs will {component} receive?",
                "hint": "Consider data types, formats, and sources",
            },
            {
                "id": "outputs",
                "question": f"What outputs will {component} produce?",
                "hint": "Consider return types, side effects, and consumers",
            },
            {
                "id": "errors",
                "question": f"What errors can {component} encounter?",
                "hint": "Consider validation errors, external failures, edge cases",
            },
            {
                "id": "scale",
                "question": f"What scale does {component} need to handle?",
                "hint": "Consider data volume, concurrent users, frequency",
            },
        ]

        # Add context-specific questions based on task keywords
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["auth", "login", "user", "session"]):
            questions.append({
                "id": "security",
                "question": "What security requirements apply?",
                "hint": "Consider authentication, authorization, encryption",
            })

        if any(kw in task_lower for kw in ["api", "endpoint", "service"]):
            questions.append({
                "id": "api_style",
                "question": "What API style is preferred?",
                "hint": "REST, GraphQL, RPC, WebSocket, etc.",
            })

        if any(kw in task_lower for kw in ["data", "store", "save", "persist"]):
            questions.append({
                "id": "storage",
                "question": "What storage approach is needed?",
                "hint": "Database type, caching strategy, persistence requirements",
            })

        return questions

    def _suggest_interface(
        self,
        component: str,
        clarifications: Dict[str, str],
        task: str,
    ) -> str:
        """Suggest an interface definition based on clarifications."""
        # Build a protocol-style interface suggestion
        inputs = clarifications.get("inputs", "data")
        outputs = clarifications.get("outputs", "result")

        interface = f"""class {component}Protocol(Protocol):
    \"\"\"
    Protocol for {component}.

    Part of: {task}
    \"\"\"

    async def process(self, {self._to_param_name(inputs)}: Any) -> {self._to_type_name(outputs)}:
        \"\"\"
        Main processing method.

        Args:
            {self._to_param_name(inputs)}: Input data

        Returns:
            Processed result

        Raises:
            ValidationError: If input is invalid
            ProcessingError: If processing fails
        \"\"\"
        ...

    async def validate(self, {self._to_param_name(inputs)}: Any) -> ValidationResult:
        \"\"\"Validate input before processing.\"\"\"
        ...

    def get_status(self) -> ComponentStatus:
        \"\"\"Get current component status.\"\"\"
        ...
"""
        return interface

    def _suggest_dependencies(
        self, component: str, clarifications: Dict[str, str]
    ) -> List[str]:
        """Suggest likely dependencies."""
        deps = []

        # Analyze clarifications for dependency hints
        all_text = " ".join(str(v) for v in clarifications.values()).lower()

        if any(kw in all_text for kw in ["http", "api", "request", "fetch"]):
            deps.append("httpx or requests for HTTP client")

        if any(kw in all_text for kw in ["database", "sql", "query", "persist"]):
            deps.append("Database adapter (SQLAlchemy, asyncpg, etc.)")

        if any(kw in all_text for kw in ["json", "serialize", "parse"]):
            deps.append("pydantic for data validation")

        if any(kw in all_text for kw in ["async", "concurrent", "parallel"]):
            deps.append("asyncio for async operations")

        if any(kw in all_text for kw in ["log", "monitor", "trace"]):
            deps.append("Logging/monitoring framework")

        # Always suggest typing
        deps.append("typing for type hints")

        return deps

    def _suggest_fallbacks(
        self, component: str, clarifications: Dict[str, str]
    ) -> List[str]:
        """Suggest fallback strategies."""
        fallbacks = []

        # Default fallbacks for common scenarios
        errors = clarifications.get("errors", "").lower()

        if any(kw in errors for kw in ["network", "timeout", "connection"]):
            fallbacks.append("Retry with exponential backoff on network failures")
            fallbacks.append("Circuit breaker for persistent failures")

        if any(kw in errors for kw in ["validation", "invalid", "bad"]):
            fallbacks.append("Return clear error messages for validation failures")
            fallbacks.append("Log invalid inputs for debugging")

        # Generic fallbacks
        fallbacks.append("Graceful degradation if optional dependencies unavailable")
        fallbacks.append("Clear error messages with recovery suggestions")

        return fallbacks

    def _to_param_name(self, description: str) -> str:
        """Convert description to parameter name."""
        # Simple conversion: take first word, lowercase
        words = description.split()
        if words:
            return words[0].lower().replace("-", "_")
        return "data"

    def _to_type_name(self, description: str) -> str:
        """Convert description to type name."""
        # Simple conversion: capitalize first word, add Result suffix
        words = description.split()
        if words:
            base = words[0].capitalize()
            if not base.endswith("Result"):
                return f"{base}Result"
            return base
        return "Result"

    def _needs_component_response(
        self,
        session: SessionState,
        start_time: datetime,
        events_emitted: List[str],
    ) -> Dict[str, Any]:
        """Build response when we need to know what component to design."""
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "phase": "design",
            "session_id": session.session_id,
            "status": "needs_input",
            "message": f"Protocol-first design for: {session.task}",
            "next_steps": [
                "Identify the component or interface to design",
                "Consider the key abstractions needed",
                "Use /ritual design <component> to begin",
            ],
            "data": {
                "task": session.task,
                "stage": "component_selection",
                "prompt": "What component or interface should we design first?",
                "suggestions": self._suggest_components(session.task),
            },
            "metadata": {
                "execution_time_ms": round(execution_time_ms, 2),
                "events_emitted": events_emitted,
                "correlation_id": session.correlation_id,
            },
        }

    def _needs_clarification_response(
        self,
        session: SessionState,
        component: str,
        questions: List[Dict[str, str]],
        start_time: datetime,
        events_emitted: List[str],
    ) -> Dict[str, Any]:
        """Build response when we need clarifying questions answered."""
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "phase": "design",
            "session_id": session.session_id,
            "status": "needs_input",
            "message": f"Designing: {component}",
            "next_steps": [
                "Answer the clarifying questions below",
                "Consider edge cases and error scenarios",
                "Think about dependencies and integrations",
            ],
            "data": {
                "task": session.task,
                "component": component,
                "stage": "clarification",
                "questions": questions,
            },
            "metadata": {
                "execution_time_ms": round(execution_time_ms, 2),
                "events_emitted": events_emitted,
                "correlation_id": session.correlation_id,
            },
        }

    def _suggest_interface_response(
        self,
        session: SessionState,
        component: str,
        interface: str,
        dependencies: List[str],
        fallbacks: List[str],
        start_time: datetime,
        events_emitted: List[str],
    ) -> Dict[str, Any]:
        """Build response with suggested interface."""
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "phase": "design",
            "session_id": session.session_id,
            "status": "needs_input",
            "message": f"Suggested interface for: {component}",
            "next_steps": [
                "Review the proposed interface",
                "Modify as needed for your requirements",
                "Confirm to record the design decision",
            ],
            "data": {
                "task": session.task,
                "component": component,
                "stage": "interface_review",
                "proposed_interface": interface,
                "proposed_dependencies": dependencies,
                "proposed_fallbacks": fallbacks,
                "prompt": "Review and confirm (or modify) the proposed design:",
            },
            "metadata": {
                "execution_time_ms": round(execution_time_ms, 2),
                "events_emitted": events_emitted,
                "correlation_id": session.correlation_id,
            },
        }

    def _design_completed_response(
        self,
        session: SessionState,
        design_record: Dict[str, Any],
        execution_time_ms: float,
        events_emitted: List[str],
    ) -> Dict[str, Any]:
        """Build response when design is complete."""
        return {
            "phase": "design",
            "session_id": session.session_id,
            "status": "success",
            "message": f"Design recorded for: {design_record['component']}",
            "next_steps": [
                "Use /ritual design to design another component",
                "Use /ritual build to start implementation",
                "Use /ritual status to check progress",
            ],
            "data": {
                "task": session.task,
                "component": design_record["component"],
                "designs_count": len(session.designs),
                "design_record": design_record,
            },
            "metadata": {
                "execution_time_ms": round(execution_time_ms, 2),
                "events_emitted": events_emitted,
                "correlation_id": session.correlation_id,
            },
        }

    def _suggest_components(self, task: str) -> List[str]:
        """Suggest potential components based on task description."""
        suggestions = []
        task_lower = task.lower()

        # Common component patterns
        if any(kw in task_lower for kw in ["api", "endpoint", "service"]):
            suggestions.extend(["APIHandler", "RequestValidator", "ResponseBuilder"])

        if any(kw in task_lower for kw in ["auth", "login", "user"]):
            suggestions.extend(["AuthProvider", "SessionManager", "UserValidator"])

        if any(kw in task_lower for kw in ["data", "store", "save"]):
            suggestions.extend(["DataRepository", "CacheManager", "StorageAdapter"])

        if any(kw in task_lower for kw in ["process", "transform", "convert"]):
            suggestions.extend(["Processor", "Transformer", "Converter"])

        if any(kw in task_lower for kw in ["ui", "display", "view", "render"]):
            suggestions.extend(["ViewRenderer", "DisplayManager", "UIComponent"])

        # Default suggestions if nothing specific
        if not suggestions:
            suggestions = ["CoreProcessor", "DataHandler", "ServiceInterface"]

        return suggestions[:5]  # Limit to 5 suggestions

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Build error response."""
        return {
            "phase": "design",
            "session_id": None,
            "status": "error",
            "message": message,
            "next_steps": [],
            "data": {},
            "metadata": {},
        }

    async def _emit_design_started(
        self,
        session: SessionState,
        component: Optional[str],
    ) -> Optional[str]:
        """Emit ritual.design.started event."""
        if not self.event_bus:
            return None

        try:
            from HoloLoom.skills.protocol import SkillEvent, EventType

            event = SkillEvent(
                event_type=EventType.SKILL_STARTED,
                skill_name="ritual",
                timestamp=datetime.now().isoformat(),
                payload={
                    "session_id": session.session_id,
                    "task": session.task,
                    "component": component,
                    "phase": "design",
                },
                event_id=f"ritual.design.started-{uuid.uuid4().hex[:8]}",
                topic="ritual.design.started",
                correlation_id=session.correlation_id,
                causation_id=f"ritual.session.opened-{session.correlation_id}",
                sequence=session.event_sequence + 1,
            )

            await self.event_bus.emit(event)
            return event.event_id

        except ImportError:
            return None
        except Exception:
            return None

    async def _emit_design_completed(
        self,
        session: SessionState,
        design_record: Dict[str, Any],
    ) -> Optional[str]:
        """Emit ritual.design.completed event."""
        if not self.event_bus:
            return None

        try:
            from HoloLoom.skills.protocol import SkillEvent, EventType

            event = SkillEvent(
                event_type=EventType.SKILL_COMPLETED,
                skill_name="ritual",
                timestamp=datetime.now().isoformat(),
                payload={
                    "session_id": session.session_id,
                    "component": design_record["component"],
                    "dependencies_count": len(design_record.get("dependencies", [])),
                    "fallbacks_count": len(design_record.get("fallbacks", [])),
                },
                event_id=f"ritual.design.completed-{uuid.uuid4().hex[:8]}",
                topic="ritual.design.completed",
                correlation_id=session.correlation_id,
                sequence=session.event_sequence + 2,
            )

            await self.event_bus.emit(event)
            return event.event_id

        except ImportError:
            return None
        except Exception:
            return None


# Convenience function for direct invocation
async def start_ritual_design(
    component: Optional[str] = None,
    clarifications: Optional[Dict[str, str]] = None,
    interface_spec: Optional[str] = None,
    dependencies: Optional[List[str]] = None,
    fallbacks: Optional[List[str]] = None,
    rationale: Optional[str] = None,
    event_bus: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Start or continue the design phase.

    Args:
        component: Component name to design
        clarifications: Answers to clarifying questions
        interface_spec: Interface definition
        dependencies: List of dependencies
        fallbacks: Fallback strategies
        rationale: Design rationale
        event_bus: Optional EventBus for events

    Returns:
        Design phase result
    """
    handler = DesignPhaseHandler(event_bus=event_bus)
    return await handler.execute(
        component=component,
        clarifications=clarifications,
        interface_spec=interface_spec,
        dependencies=dependencies,
        fallbacks=fallbacks,
        rationale=rationale,
    )


def format_design_response(result: Dict[str, Any]) -> str:
    """
    Format design response for display.

    Returns formatted string suitable for terminal output.
    """
    if result["status"] == "error":
        return f"Error: {result['message']}"

    lines = [
        "",
        "Protocol-First Design",
        "",
    ]

    data = result["data"]
    stage = data.get("stage", "unknown")

    if stage == "component_selection":
        lines.extend([
            f"**Task**: {data['task']}",
            "",
            "What component should we design?",
            "",
            "**Suggestions**:",
        ])
        for suggestion in data.get("suggestions", []):
            lines.append(f"  - {suggestion}")
        lines.append("")
        lines.append("Enter component name to begin design:")

    elif stage == "clarification":
        lines.extend([
            f"**Designing**: {data['component']}",
            "",
            "**Clarifying Questions**:",
            "",
        ])
        for q in data.get("questions", []):
            lines.append(f"  **{q['id']}**: {q['question']}")
            if q.get("hint"):
                lines.append(f"    _{q['hint']}_")
            lines.append("")

    elif stage == "interface_review":
        lines.extend([
            f"**Proposed Interface for**: {data['component']}",
            "",
            "```python",
            data.get("proposed_interface", ""),
            "```",
            "",
            "**Proposed Dependencies**:",
        ])
        for dep in data.get("proposed_dependencies", []):
            lines.append(f"  - {dep}")
        lines.extend([
            "",
            "**Proposed Fallbacks**:",
        ])
        for fb in data.get("proposed_fallbacks", []):
            lines.append(f"  - {fb}")
        lines.append("")
        lines.append("Review and confirm to record this design.")

    else:  # success
        lines.extend([
            f"**Design Recorded**: {data['component']}",
            "",
            f"Total designs: {data['designs_count']}",
            "",
            "**Next Steps**:",
        ])
        for step in result["next_steps"]:
            lines.append(f"  - {step}")

    return "\n".join(lines)


if __name__ == "__main__":
    import asyncio

    async def demo():
        # First open a session
        from open import open_ritual_session

        await open_ritual_session(
            task="Build a file upload handler",
            complexity="fast",
            success_criteria=["Validates file types", "Handles errors gracefully"],
        )

        # Start design - stage 1: need component
        result = await start_ritual_design()
        print(format_design_response(result))
        print("\n---\n")

        # Stage 2: provide component, get questions
        result = await start_ritual_design(component="FileValidator")
        print(format_design_response(result))
        print("\n---\n")

        # Stage 3: provide clarifications, get interface
        result = await start_ritual_design(
            component="FileValidator",
            clarifications={
                "inputs": "File object with name, size, content",
                "outputs": "Validation result with status and errors",
                "errors": "Invalid type, size exceeded, corrupted content",
                "scale": "Up to 100 files per minute",
            },
        )
        print(format_design_response(result))
        print("\n---\n")

        # Stage 4: confirm interface
        result = await start_ritual_design(
            component="FileValidator",
            clarifications={
                "inputs": "File object with name, size, content",
                "outputs": "Validation result with status and errors",
                "errors": "Invalid type, size exceeded, corrupted content",
                "scale": "Up to 100 files per minute",
            },
            interface_spec="class FileValidatorProtocol(Protocol): ...",
            dependencies=["pydantic", "python-magic"],
            fallbacks=["Return generic error on unknown file types"],
            rationale="Protocol-first design enables testing with mocks",
        )
        print(format_design_response(result))

    asyncio.run(demo())