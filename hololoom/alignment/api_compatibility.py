"""
API Compatibility Layer for Alignment Framework

Provides wrapper functions to match the specification API from
ALIGNMENT_FRAMEWORK_INTEGRATION.md while maintaining backward compatibility
with the existing implementation.

This module adds specification-compliant methods to the alignment classes
without breaking existing code.

API Differences Addressed:
1. SafetyGuardrails: evaluate_action() wrapper for evaluate()
2. DeceptionDetector: register_goal() and evaluate_probe() wrappers
3. InstrumentalConvergenceGuard: Unified ResourceBounds constructor

Usage:
    from hololoom.alignment.api_compatibility import patch_alignment_api
    patch_alignment_api()  # Apply compatibility patches

    # Now both APIs work:
    guardrails.evaluate_action("query", "QUERY", context={})  # Spec API
    guardrails.evaluate(ActionRequest(...))  # Implementation API
"""

from typing import Optional, Dict, Any
import logging

from .safety_guardrails import (
    SafetyGuardrails,
    ActionRequest,
    ActionCategory,
    SafetyDecision,
)
from .deception_detection import (
    DeceptionDetector,
    GoalStatement,
    BehavioralProbe,
)
from .instrumental_convergence import (
    InstrumentalConvergenceGuard,
    ResourceBounds,
    ResourceType,
)


logger = logging.getLogger(__name__)


# =============================================================================
# SAFETY GUARDRAILS COMPATIBILITY
# =============================================================================


def evaluate_action_wrapper(
    self: SafetyGuardrails,
    action: str,
    category: str,
    context: Optional[Dict[str, Any]] = None,
) -> SafetyDecision:
    """
    Specification-compliant API for SafetyGuardrails.evaluate().

    Converts spec API to implementation API.

    Args:
        action: Action description
        category: Action category (string)
        context: Additional context (optional)

    Returns:
        SafetyDecision
    """
    # Convert category string to enum
    try:
        category_enum = ActionCategory[category.upper()]
    except KeyError:
        logger.warning(f"Unknown category '{category}', defaulting to QUERY")
        category_enum = ActionCategory.QUERY

    # Create ActionRequest
    request = ActionRequest(
        action_id=f"action_{hash(action)}",
        category=category_enum,
        description=action,
        context=context or {},
    )

    # Call implementation method
    return self.evaluate(request, text_input=action)


# =============================================================================
# DECEPTION DETECTION COMPATIBILITY
# =============================================================================


def register_goal_wrapper(
    self: DeceptionDetector,
    description: str,
    priority: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Specification-compliant API for goal registration.

    Converts spec API to implementation API.

    Args:
        description: Goal description
        priority: Goal priority (1-10)
        metadata: Additional metadata

    Returns:
        goal_id: Unique identifier for the goal
    """
    goal_id = f"goal_{hash(description)}"

    goal = GoalStatement(
        goal_id=goal_id,
        description=description,
        priority=priority,
        metadata=metadata or {},
    )

    self.goal_tracker.declare_goal(goal)
    return goal_id


def evaluate_probe_wrapper(
    self: DeceptionDetector,
    probe: BehavioralProbe,
    response: str,
) -> tuple[bool, float]:
    """
    Specification-compliant API for probe evaluation.

    Converts spec API to implementation API.

    Args:
        probe: Behavioral probe to run
        response: Actual behavior/response

    Returns:
        (passed, deception_score): Probe result
    """
    return self.run_probe(probe, actual_behavior=response)


# =============================================================================
# INSTRUMENTAL CONVERGENCE COMPATIBILITY
# =============================================================================


def create_unified_resource_bounds(
    max_compute_seconds: Optional[float] = None,
    max_memory_mb: Optional[float] = None,
    max_storage_mb: Optional[float] = None,
    max_network_mb: Optional[float] = None,
    max_api_calls: Optional[int] = None,
    max_data_access_mb: Optional[float] = None,
) -> Dict[ResourceType, ResourceBounds]:
    """
    Specification-compliant unified ResourceBounds constructor.

    Creates per-resource bounds from unified parameters.

    Args:
        max_compute_seconds: Max compute time (soft=80%, hard=100%)
        max_memory_mb: Max memory usage
        max_storage_mb: Max storage usage
        max_network_mb: Max network bandwidth
        max_api_calls: Max API calls per minute
        max_data_access_mb: Max data access per minute

    Returns:
        Dictionary mapping ResourceType to ResourceBounds
    """
    bounds_dict = {}

    if max_compute_seconds is not None:
        bounds_dict[ResourceType.COMPUTE] = ResourceBounds(
            resource_type=ResourceType.COMPUTE,
            soft_limit=max_compute_seconds * 0.8,
            hard_limit=max_compute_seconds,
            time_window_seconds=60.0,
            rate_limit=max_compute_seconds / 60.0,
        )

    if max_memory_mb is not None:
        bounds_dict[ResourceType.MEMORY] = ResourceBounds(
            resource_type=ResourceType.MEMORY,
            soft_limit=max_memory_mb * 0.8,
            hard_limit=max_memory_mb,
            time_window_seconds=60.0,
        )

    if max_storage_mb is not None:
        bounds_dict[ResourceType.STORAGE] = ResourceBounds(
            resource_type=ResourceType.STORAGE,
            soft_limit=max_storage_mb * 0.8,
            hard_limit=max_storage_mb,
            time_window_seconds=3600.0,  # 1 hour window
        )

    if max_network_mb is not None:
        bounds_dict[ResourceType.NETWORK] = ResourceBounds(
            resource_type=ResourceType.NETWORK,
            soft_limit=max_network_mb * 0.8,
            hard_limit=max_network_mb,
            time_window_seconds=60.0,
            rate_limit=max_network_mb / 60.0,
        )

    if max_api_calls is not None:
        bounds_dict[ResourceType.API_CALLS] = ResourceBounds(
            resource_type=ResourceType.API_CALLS,
            soft_limit=max_api_calls * 0.8,
            hard_limit=float(max_api_calls),
            time_window_seconds=60.0,
            rate_limit=max_api_calls / 60.0,
        )

    if max_data_access_mb is not None:
        bounds_dict[ResourceType.DATA_ACCESS] = ResourceBounds(
            resource_type=ResourceType.DATA_ACCESS,
            soft_limit=max_data_access_mb * 0.8,
            hard_limit=max_data_access_mb,
            time_window_seconds=60.0,
            rate_limit=max_data_access_mb / 60.0,
        )

    return bounds_dict


def configure_from_unified_bounds_wrapper(
    self: InstrumentalConvergenceGuard,
    max_compute_seconds: Optional[float] = None,
    max_memory_mb: Optional[float] = None,
    max_storage_mb: Optional[float] = None,
    max_network_mb: Optional[float] = None,
    max_api_calls: Optional[int] = None,
    max_data_access_mb: Optional[float] = None,
):
    """
    Specification-compliant unified bounds configuration.

    Configures guard using unified parameters instead of per-resource.

    Args:
        max_compute_seconds: Max compute time
        max_memory_mb: Max memory usage
        max_storage_mb: Max storage usage
        max_network_mb: Max network bandwidth
        max_api_calls: Max API calls per minute
        max_data_access_mb: Max data access per minute
    """
    bounds_dict = create_unified_resource_bounds(
        max_compute_seconds=max_compute_seconds,
        max_memory_mb=max_memory_mb,
        max_storage_mb=max_storage_mb,
        max_network_mb=max_network_mb,
        max_api_calls=max_api_calls,
        max_data_access_mb=max_data_access_mb,
    )

    for resource_type, bounds in bounds_dict.items():
        self.set_resource_bounds(resource_type, bounds)

    logger.info(f"Configured {len(bounds_dict)} resource bounds from unified parameters")


# =============================================================================
# PATCHING SYSTEM
# =============================================================================


_PATCHED = False


def patch_alignment_api():
    """
    Apply API compatibility patches to alignment classes.

    This adds specification-compliant methods to the existing classes
    without breaking backward compatibility.

    Safe to call multiple times (idempotent).
    """
    global _PATCHED

    if _PATCHED:
        logger.debug("API compatibility already patched")
        return

    # Patch SafetyGuardrails
    SafetyGuardrails.evaluate_action = evaluate_action_wrapper

    # Patch DeceptionDetector
    DeceptionDetector.register_goal = register_goal_wrapper
    DeceptionDetector.evaluate_probe = evaluate_probe_wrapper

    # Patch InstrumentalConvergenceGuard
    InstrumentalConvergenceGuard.configure_from_unified_bounds = (
        configure_from_unified_bounds_wrapper
    )

    _PATCHED = True
    logger.info("Applied API compatibility patches to alignment framework")


def unpatch_alignment_api():
    """
    Remove API compatibility patches.

    Useful for testing or reverting to implementation-only API.
    """
    global _PATCHED

    if not _PATCHED:
        return

    # Remove patched methods
    if hasattr(SafetyGuardrails, "evaluate_action"):
        delattr(SafetyGuardrails, "evaluate_action")

    if hasattr(DeceptionDetector, "register_goal"):
        delattr(DeceptionDetector, "register_goal")

    if hasattr(DeceptionDetector, "evaluate_probe"):
        delattr(DeceptionDetector, "evaluate_probe")

    if hasattr(InstrumentalConvergenceGuard, "configure_from_unified_bounds"):
        delattr(InstrumentalConvergenceGuard, "configure_from_unified_bounds")

    _PATCHED = False
    logger.info("Removed API compatibility patches")


# =============================================================================
# AUTO-PATCH ON IMPORT (OPTIONAL)
# =============================================================================

# Uncomment to auto-patch when module is imported:
# patch_alignment_api()
