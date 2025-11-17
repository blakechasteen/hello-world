#!/usr/bin/env python3
"""
Chain Execution Tracing

Provides detailed execution tracing, performance monitoring, and debugging capabilities.
"""

import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class TraceLevel(Enum):
    """Trace detail levels"""
    MINIMAL = "minimal"     # Only step names and status
    STANDARD = "standard"   # + timing and basic results
    DETAILED = "detailed"   # + full inputs/outputs
    DEBUG = "debug"         # + internal processor state


@dataclass
class TraceEvent:
    """Single trace event"""
    timestamp: float
    event_type: str
    step_name: str
    data: Dict[str, Any]
    level: TraceLevel = TraceLevel.STANDARD


@dataclass
class StepTrace:
    """Trace for a single step execution"""
    step_name: str
    processor_type: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "pending"
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    events: List[TraceEvent] = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        """Calculate execution duration in seconds"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "step_name": self.step_name,
            "processor_type": self.processor_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "metadata": self.metadata,
            "events": [
                {
                    "timestamp": e.timestamp,
                    "type": e.event_type,
                    "step": e.step_name,
                    "data": e.data
                }
                for e in self.events
            ]
        }


class ExecutionTracer:
    """Trace chain execution with detailed logging"""

    def __init__(self, trace_level: TraceLevel = TraceLevel.STANDARD):
        self.trace_level = trace_level
        self.traces: List[StepTrace] = []
        self.current_step: Optional[StepTrace] = None
        self.global_start_time: Optional[float] = None
        self.global_end_time: Optional[float] = None
        self._event_handlers: List[Callable[[TraceEvent], None]] = []

    def start_execution(self) -> None:
        """Mark start of chain execution"""
        self.global_start_time = time.time()
        self.traces = []

    def end_execution(self) -> None:
        """Mark end of chain execution"""
        self.global_end_time = time.time()

    def start_step(
        self,
        step_name: str,
        processor_type: str,
        input_data: Optional[Dict[str, Any]] = None
    ) -> StepTrace:
        """Start tracing a step"""
        trace = StepTrace(
            step_name=step_name,
            processor_type=processor_type,
            start_time=time.time(),
            status="running"
        )

        # Store input based on trace level
        if self.trace_level in [TraceLevel.DETAILED, TraceLevel.DEBUG]:
            trace.input_data = input_data
        elif self.trace_level == TraceLevel.STANDARD:
            # Store only summary
            trace.metadata["input_summary"] = self._summarize_data(input_data)

        self.current_step = trace
        self.traces.append(trace)

        # Emit event
        self._emit_event(TraceEvent(
            timestamp=time.time(),
            event_type="step_start",
            step_name=step_name,
            data={"processor_type": processor_type}
        ))

        return trace

    def end_step(
        self,
        step_name: str,
        status: str = "completed",
        output_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """End tracing a step"""
        trace = next((t for t in self.traces if t.step_name == step_name), None)
        if not trace:
            return

        trace.end_time = time.time()
        trace.status = status
        trace.error = error

        # Store output based on trace level
        if self.trace_level in [TraceLevel.DETAILED, TraceLevel.DEBUG]:
            trace.output_data = output_data
        elif self.trace_level == TraceLevel.STANDARD:
            trace.metadata["output_summary"] = self._summarize_data(output_data)

        # Emit event
        self._emit_event(TraceEvent(
            timestamp=time.time(),
            event_type="step_end",
            step_name=step_name,
            data={
                "status": status,
                "duration": trace.duration,
                "error": error
            }
        ))

        self.current_step = None

    def add_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        step_name: Optional[str] = None
    ) -> None:
        """Add a custom event to the trace"""
        if step_name is None and self.current_step:
            step_name = self.current_step.step_name

        event = TraceEvent(
            timestamp=time.time(),
            event_type=event_type,
            step_name=step_name or "global",
            data=data,
            level=self.trace_level
        )

        if self.current_step:
            self.current_step.events.append(event)

        self._emit_event(event)

    def register_event_handler(self, handler: Callable[[TraceEvent], None]) -> None:
        """Register a handler for trace events"""
        self._event_handlers.append(handler)

    def get_trace(self) -> List[StepTrace]:
        """Get all step traces"""
        return self.traces

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        total_duration = 0.0
        if self.global_start_time and self.global_end_time:
            total_duration = self.global_end_time - self.global_start_time

        return {
            "total_duration": total_duration,
            "total_steps": len(self.traces),
            "completed": sum(1 for t in self.traces if t.status == "completed"),
            "failed": sum(1 for t in self.traces if t.status == "failed"),
            "skipped": sum(1 for t in self.traces if t.status == "skipped"),
            "steps": [
                {
                    "name": t.step_name,
                    "type": t.processor_type,
                    "duration": t.duration,
                    "status": t.status
                }
                for t in self.traces
            ]
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        metrics = {
            "total_duration": 0.0,
            "step_durations": {},
            "processor_stats": {},
            "slowest_steps": [],
            "failed_steps": []
        }

        if self.global_start_time and self.global_end_time:
            metrics["total_duration"] = self.global_end_time - self.global_start_time

        # Collect step durations
        for trace in self.traces:
            if trace.duration is not None:
                metrics["step_durations"][trace.step_name] = trace.duration

                # Group by processor type
                if trace.processor_type not in metrics["processor_stats"]:
                    metrics["processor_stats"][trace.processor_type] = {
                        "count": 0,
                        "total_duration": 0.0,
                        "avg_duration": 0.0
                    }

                stats = metrics["processor_stats"][trace.processor_type]
                stats["count"] += 1
                stats["total_duration"] += trace.duration

            # Track failures
            if trace.status == "failed":
                metrics["failed_steps"].append({
                    "name": trace.step_name,
                    "error": trace.error
                })

        # Calculate averages
        for processor_type, stats in metrics["processor_stats"].items():
            if stats["count"] > 0:
                stats["avg_duration"] = stats["total_duration"] / stats["count"]

        # Find slowest steps
        sorted_traces = sorted(
            [t for t in self.traces if t.duration is not None],
            key=lambda t: t.duration,
            reverse=True
        )
        metrics["slowest_steps"] = [
            {
                "name": t.step_name,
                "type": t.processor_type,
                "duration": t.duration
            }
            for t in sorted_traces[:5]
        ]

        return metrics

    def export_trace(self, format: str = "json") -> str:
        """Export trace in specified format"""
        if format == "json":
            return json.dumps({
                "summary": self.get_summary(),
                "traces": [t.to_dict() for t in self.traces]
            }, indent=2)

        elif format == "csv":
            import csv
            from io import StringIO

            output = StringIO()
            writer = csv.writer(output)

            # Header
            writer.writerow(["Step", "Type", "Status", "Duration", "Error"])

            # Rows
            for trace in self.traces:
                writer.writerow([
                    trace.step_name,
                    trace.processor_type,
                    trace.status,
                    trace.duration or "",
                    trace.error or ""
                ])

            return output.getvalue()

        elif format == "markdown":
            lines = ["# Execution Trace", ""]
            lines.append("## Summary")
            summary = self.get_summary()
            lines.append(f"- Total Duration: {summary['total_duration']:.3f}s")
            lines.append(f"- Total Steps: {summary['total_steps']}")
            lines.append(f"- Completed: {summary['completed']}")
            lines.append(f"- Failed: {summary['failed']}")
            lines.append("")
            lines.append("## Steps")
            lines.append("")
            lines.append("| Step | Type | Status | Duration |")
            lines.append("|------|------|--------|----------|")

            for trace in self.traces:
                duration_str = f"{trace.duration:.3f}s" if trace.duration else "N/A"
                lines.append(f"| {trace.step_name} | {trace.processor_type} | {trace.status} | {duration_str} |")

            return "\n".join(lines)

        else:
            raise ValueError(f"Unknown format: {format}")

    def _emit_event(self, event: TraceEvent) -> None:
        """Emit event to registered handlers"""
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception:
                # Don't let handler errors break tracing
                pass

    def _summarize_data(self, data: Optional[Dict[str, Any]]) -> str:
        """Create a summary of data for compact logging"""
        if data is None:
            return "None"

        summary_parts = []
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)):
                summary_parts.append(f"{key}={value}")
            elif isinstance(value, (list, dict)):
                summary_parts.append(f"{key}=<{type(value).__name__}>")

        return ", ".join(summary_parts[:5])  # Limit to first 5 fields


class PerformanceMonitor:
    """Monitor performance and detect bottlenecks"""

    def __init__(self):
        self.measurements: Dict[str, List[float]] = {}

    def measure(self, name: str, duration: float) -> None:
        """Record a performance measurement"""
        if name not in self.measurements:
            self.measurements[name] = []
        self.measurements[name].append(duration)

    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a measurement"""
        if name not in self.measurements:
            return {}

        values = self.measurements[name]
        return {
            "count": len(values),
            "total": sum(values),
            "average": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "median": sorted(values)[len(values) // 2]
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all measurements"""
        return {
            name: self.get_stats(name)
            for name in self.measurements.keys()
        }

    def detect_bottlenecks(self, threshold_multiplier: float = 2.0) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks"""
        bottlenecks = []

        all_stats = self.get_all_stats()
        if not all_stats:
            return bottlenecks

        # Calculate average duration across all measurements
        total_avg = sum(stats["average"] for stats in all_stats.values()) / len(all_stats)

        # Find measurements significantly slower than average
        for name, stats in all_stats.items():
            if stats["average"] > total_avg * threshold_multiplier:
                bottlenecks.append({
                    "name": name,
                    "average_duration": stats["average"],
                    "baseline_average": total_avg,
                    "slowdown_factor": stats["average"] / total_avg,
                    "max_duration": stats["max"]
                })

        return sorted(bottlenecks, key=lambda x: x["slowdown_factor"], reverse=True)


# Convenience functions
def create_tracer(trace_level: str = "standard") -> ExecutionTracer:
    """Create an execution tracer"""
    level_map = {
        "minimal": TraceLevel.MINIMAL,
        "standard": TraceLevel.STANDARD,
        "detailed": TraceLevel.DETAILED,
        "debug": TraceLevel.DEBUG
    }
    level = level_map.get(trace_level.lower(), TraceLevel.STANDARD)
    return ExecutionTracer(trace_level=level)


__all__ = [
    "ExecutionTracer",
    "PerformanceMonitor",
    "TraceLevel",
    "TraceEvent",
    "StepTrace",
    "create_tracer"
]
