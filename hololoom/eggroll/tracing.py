"""
Eggroll Observability - Distributed Tracing.

Provides OpenTelemetry-compatible tracing for distributed evolutionary training.

Span Decorators:
- @trace_epoch: Trace evolutionary epochs
- @trace_worker: Trace individual worker steps
- @trace_selection: Trace selection operations
- @trace_mutation: Trace mutation operations
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, TypeVar, Union

from HoloLoom.telemetry.tracing import (
    get_tracer,
    span,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TYPE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

F = TypeVar("F", bound=Callable[..., Any])


# ═══════════════════════════════════════════════════════════════════════════════
#  EGGROLL TRACER
# ═══════════════════════════════════════════════════════════════════════════════


def get_eggroll_tracer():
    """Get the Eggroll-specific tracer."""
    return get_tracer(service_name="eggroll")


# ═══════════════════════════════════════════════════════════════════════════════
#  SPAN DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════


def trace_epoch(
    population_id: Optional[str] = None,
    complexity: str = "FULL",
) -> Callable[[F], F]:
    """
    Decorator to trace evolutionary epochs.

    Usage:
        @trace_epoch(population_id="pop_001", complexity="FULL")
        async def run_epoch(self, ...):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_eggroll_tracer()
            span_name = f"eggroll.epoch.{func.__name__}"

            with tracer.trace(span_name) as s:
                s.set_attribute("eggroll.population_id", population_id or "unknown")
                s.set_attribute("eggroll.complexity", complexity)
                s.set_attribute("eggroll.operation", "epoch")

                result = await func(*args, **kwargs)

                # Extract epoch metadata if available
                if isinstance(result, dict):
                    if "epoch" in result:
                        s.set_attribute("eggroll.epoch_number", result["epoch"])
                    if "best_fitness" in result:
                        s.set_attribute("eggroll.best_fitness", result["best_fitness"])
                    if "workers" in result:
                        s.set_attribute("eggroll.worker_count", len(result["workers"]))

                return result

        return wrapper  # type: ignore
    return decorator


def trace_worker(worker_id: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator to trace individual worker steps.

    Usage:
        @trace_worker(worker_id="worker_0")
        def step(self, task_input, target):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_eggroll_tracer()
            span_name = f"eggroll.worker.{func.__name__}"

            # Try to get worker_id from self if not provided
            effective_worker_id = worker_id
            if effective_worker_id is None and len(args) > 0:
                self_obj = args[0]
                if hasattr(self_obj, "worker_id"):
                    effective_worker_id = str(self_obj.worker_id)

            with tracer.trace(span_name) as s:
                s.set_attribute("eggroll.worker_id", effective_worker_id or "unknown")
                s.set_attribute("eggroll.operation", "worker_step")

                result = func(*args, **kwargs)

                # Extract metrics if tuple (fitness, metrics, output)
                if isinstance(result, tuple) and len(result) >= 2:
                    fitness = result[0]
                    metrics = result[1] if isinstance(result[1], dict) else {}
                    s.set_attribute("eggroll.fitness", float(fitness))
                    if "loss" in metrics:
                        s.set_attribute("eggroll.loss", float(metrics["loss"]))

                return result

        return wrapper  # type: ignore
    return decorator


def trace_selection(method: str = "tournament") -> Callable[[F], F]:
    """
    Decorator to trace selection operations.

    Usage:
        @trace_selection(method="tournament")
        def select_parents(self, population, fitness_scores):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_eggroll_tracer()
            span_name = f"eggroll.selection.{func.__name__}"

            with tracer.trace(span_name) as s:
                s.set_attribute("eggroll.selection_method", method)
                s.set_attribute("eggroll.operation", "selection")

                result = func(*args, **kwargs)

                # Extract selection metadata
                if isinstance(result, (list, tuple)):
                    s.set_attribute("eggroll.selected_count", len(result))

                return result

        return wrapper  # type: ignore
    return decorator


def trace_mutation(mutation_rate: Optional[float] = None) -> Callable[[F], F]:
    """
    Decorator to trace mutation operations.

    Usage:
        @trace_mutation(mutation_rate=0.1)
        def mutate(self, individual):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_eggroll_tracer()
            span_name = f"eggroll.mutation.{func.__name__}"

            with tracer.trace(span_name) as s:
                s.set_attribute("eggroll.operation", "mutation")
                if mutation_rate is not None:
                    s.set_attribute("eggroll.mutation_rate", mutation_rate)

                result = func(*args, **kwargs)
                return result

        return wrapper  # type: ignore
    return decorator


def trace_crossover(crossover_type: str = "uniform") -> Callable[[F], F]:
    """
    Decorator to trace crossover operations.

    Usage:
        @trace_crossover(crossover_type="uniform")
        def crossover(self, parent1, parent2):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_eggroll_tracer()
            span_name = f"eggroll.crossover.{func.__name__}"

            with tracer.trace(span_name) as s:
                s.set_attribute("eggroll.crossover_type", crossover_type)
                s.set_attribute("eggroll.operation", "crossover")

                result = func(*args, **kwargs)
                return result

        return wrapper  # type: ignore
    return decorator


def trace_rust_operation(operation: str) -> Callable[[F], F]:
    """
    Decorator to trace Rust-accelerated operations.

    Usage:
        @trace_rust_operation("kmeans_clustering")
        def cluster_population(self, embeddings):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_eggroll_tracer()
            span_name = f"eggroll.rust.{operation}"

            with tracer.trace(span_name) as s:
                s.set_attribute("eggroll.operation", operation)
                s.set_attribute("eggroll.accelerated", True)
                s.set_attribute("eggroll.backend", "rust")

                result = func(*args, **kwargs)

                # Extract result metadata
                if isinstance(result, dict):
                    if "n_clusters" in result:
                        s.set_attribute("eggroll.n_clusters", result["n_clusters"])
                    if "silhouette" in result:
                        s.set_attribute("eggroll.silhouette", result["silhouette"])

                return result

        return wrapper  # type: ignore
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: INLINE SPAN CREATION
# ═══════════════════════════════════════════════════════════════════════════════


def eggroll_span(
    name: str,
    population_id: Optional[str] = None,
    worker_id: Optional[str] = None,
    **attributes,
):
    """
    Create an Eggroll span with standard attributes.

    Usage:
        with eggroll_span("fitness_evaluation", population_id="pop_001") as span:
            # Do work
            span.set_attribute("custom_metric", 0.95)
    """
    tracer = get_eggroll_tracer()

    # Build attributes
    span_attributes = {"eggroll.span_name": name}
    if population_id:
        span_attributes["eggroll.population_id"] = population_id
    if worker_id:
        span_attributes["eggroll.worker_id"] = worker_id
    span_attributes.update(attributes)

    return tracer.trace(f"eggroll.{name}", attributes=span_attributes)
