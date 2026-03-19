"""
Pipeline Resolver — Level-aware topological sort.

Produces parallel execution levels from stage dependency declarations.
Stages at the same level have all dependencies satisfied and can run
concurrently via asyncio.gather.

This module is generic — it knows nothing about WeavingContext or any
specific pipeline. The caller provides pre_satisfied fields.
"""

import dataclasses
from difflib import get_close_matches
from typing import Any

from hololoom.pipeline.exceptions import CyclicDependencyError, WriteConflictError


def resolve_levels(
    stages: dict[str, Any],
    pre_satisfied: frozenset[str] = frozenset(),
) -> list[list[str]]:
    """
    Topological sort into parallel execution levels.

    Args:
        stages: name -> stage instance mapping. Each stage must have
                .reads and .writes frozenset attributes.
        pre_satisfied: fields available before any stage runs.
            Caller provides these — resolver stays generic.

    Returns:
        List of levels, where each level is a list of stage names.
        Stages within a level can run concurrently.

    Raises:
        CyclicDependencyError: if stages have circular dependencies.

    Example:
        >>> levels = resolve_levels(
        ...     {'a': StageA(), 'b': StageB(), 'c': StageC()},
        ...     pre_satisfied=frozenset({'query'}),
        ... )
        >>> # [['a'], ['b', 'c']]  — b and c can run in parallel
    """
    satisfied = set(pre_satisfied)
    remaining = dict(stages)
    levels: list[list[str]] = []

    while remaining:
        ready = sorted(
            name for name, stage in remaining.items()
            if stage.reads <= satisfied
        )
        if not ready:
            unsatisfied = ', '.join(
                f"'{n}': needs {s.reads - satisfied}"
                for n, s in remaining.items()
            )
            raise CyclicDependencyError(
                f"Cannot resolve remaining stages — unsatisfied reads: {unsatisfied}"
            )
        levels.append(ready)
        for name in ready:
            satisfied |= remaining.pop(name).writes

    validate_disjoint_writes(levels, stages)
    return levels


def validate_disjoint_writes(
    levels: list[list[str]],
    stages: dict[str, Any],
) -> None:
    """
    Assert stages at the same level have disjoint writes.

    Two concurrent stages writing the same field is a race condition.
    Called automatically by resolve_levels after level assignment.

    Raises:
        WriteConflictError: if concurrent stages write the same field.
    """
    for level in levels:
        if len(level) < 2:
            continue
        seen: dict[str, str] = {}  # field -> first stage that writes it
        for name in level:
            for field in stages[name].writes:
                if field in seen:
                    raise WriteConflictError(
                        f"Stages '{seen[field]}' and '{name}' both write '{field}' "
                        f"and would run concurrently at the same level.\n"
                        f"  Fix: one stage should read the other's output to create a dependency."
                    )
                seen[field] = name


def validate_completeness(
    stages: dict[str, Any],
    pre_satisfied: frozenset[str] = frozenset(),
) -> list[str]:
    """
    Warn if any stage reads a field no included stage writes
    and that field isn't in pre_satisfied.

    Returns list of warning messages (empty if all reads are satisfied).
    Does not raise — missing reads may be intentional (optional fields).
    """
    all_writes = pre_satisfied | frozenset().union(
        *(s.writes for s in stages.values())
    )
    warnings = []
    for name, stage in stages.items():
        missing = stage.reads - all_writes
        if missing:
            warnings.append(
                f"Stage '{name}' reads {missing} but no included stage writes these fields "
                f"and they're not in pre_satisfied."
            )
    return warnings


def validate_stage_fields(
    stages: dict[str, Any],
    context_class: type,
    pre_satisfied: frozenset[str] = frozenset(),
) -> None:
    """
    Assert every declared read/write is an actual field on the context dataclass.

    Catches typos and forgotten fields at init time, not runtime.

    Args:
        stages: name -> stage instance mapping
        context_class: the dataclass type (e.g., WeavingContext)
        pre_satisfied: fields that may not be on the dataclass (e.g., computed)

    Raises:
        ValueError: if a declared field doesn't exist on the context.
    """
    if not dataclasses.is_dataclass(context_class):
        raise TypeError(f"{context_class} is not a dataclass")

    valid_fields = {f.name for f in dataclasses.fields(context_class)}
    all_allowed = valid_fields | pre_satisfied

    for name, stage in stages.items():
        for field_name in stage.reads | stage.writes:
            if field_name not in all_allowed:
                suggestion = ""
                close = get_close_matches(field_name, valid_fields, n=1, cutoff=0.6)
                if close:
                    suggestion = f"\n  Did you mean: '{close[0]}'?"
                raise ValueError(
                    f"Stage '{name}' declares '{field_name}' but "
                    f"{context_class.__name__} has no field '{field_name}'.{suggestion}"
                )
