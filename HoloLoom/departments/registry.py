#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Department Registry - Central registry for department discovery and management.

This module provides the infrastructure for a marketplace-ready department
ecosystem, enabling:
- Dynamic department registration and discovery
- Version management and compatibility checking
- Dependency resolution
- Load balancing across department instances
- Health monitoring and failover

Design Philosophy:
"Departments are plugins. The registry is the plugin system that makes them
composable, discoverable, and marketplace-ready."
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
import logging
import asyncio

from .protocol import Department, DepartmentManifest, DepartmentRequest, DepartmentResponse
from .base import BaseDepartment


logger = logging.getLogger(__name__)


@dataclass
class DepartmentInstance:
    """
    Registered department instance with metadata.

    Tracks both the department itself and runtime metadata for
    load balancing, health monitoring, and failover.
    """
    department: Department  # The actual department instance
    manifest: DepartmentManifest  # Metadata for discovery
    health_status: str = "healthy"  # "healthy" | "degraded" | "unhealthy"
    request_count: int = 0  # Total requests served
    active_requests: int = 0  # Currently executing requests
    last_health_check: Optional[float] = None  # Timestamp of last health check


class DepartmentRegistry:
    """
    Central registry for department discovery and management.

    Responsibilities:
    - Register departments with metadata
    - Discover departments by name/domain/task_type
    - Manage department versions
    - Resolve dependencies
    - Load balance requests across instances
    - Monitor health and handle failover

    Example:

        # Create registry
        registry = DepartmentRegistry()

        # Register departments
        context_dept = ContextDepartment(orchestrator)
        await registry.register(context_dept)

        master_weaver = MasterWeaverDepartment()
        await registry.register(master_weaver)

        # Discover departments
        dept = registry.get_department("context")
        assert dept is not None

        # Find departments by capability
        depts = registry.find_by_task("retrieve_context")
        assert "context" in [d.name for d in depts]

        # Route request
        request = DepartmentRequest(
            task_id="req_001",
            task_type="retrieve_context",
            parameters={"query": "What is Thompson Sampling?"}
        )
        response = await registry.route_request(request)
    """

    def __init__(self):
        """Initialize empty registry."""
        # Core registry: name → [instances]
        self._departments: Dict[str, List[DepartmentInstance]] = {}

        # Indexes for fast lookup
        self._by_domain: Dict[str, Set[str]] = {}  # domain → {department_names}
        self._by_task: Dict[str, Set[str]] = {}    # task_type → {department_names}

        # Version tracking: name → {version → instance}
        self._versions: Dict[str, Dict[str, DepartmentInstance]] = {}

        # Dependency graph: name → {required_names}
        self._dependencies: Dict[str, Set[str]] = {}

        # Health monitoring
        self._health_check_interval = 60.0  # seconds
        self._health_check_task: Optional[asyncio.Task] = None

        # Lifecycle
        self._closed = False

        logger.info("Department registry initialized")

    # ========================================================================
    # Registration
    # ========================================================================

    async def register(
        self,
        department: Department,
        manifest: Optional[DepartmentManifest] = None
    ) -> None:
        """
        Register a department in the registry.

        Args:
            department: Department instance to register
            manifest: Optional manifest (auto-generated if not provided)

        Raises:
            ValueError: If department already registered with same version
        """
        # Generate manifest if not provided
        if manifest is None:
            if isinstance(department, BaseDepartment):
                manifest = department.get_manifest()
            else:
                # Create minimal manifest for non-BaseDepartment
                manifest = DepartmentManifest(
                    name=department.name,
                    domain=department.domain,
                    version=department.version,
                    supported_tasks=department.supported_tasks,
                    confidence_range=department.confidence_range
                )

        # Check if already registered
        if department.name in self._versions:
            if department.version in self._versions[department.name]:
                raise ValueError(
                    f"Department {department.name} version {department.version} "
                    f"already registered"
                )

        # Create instance wrapper
        instance = DepartmentInstance(
            department=department,
            manifest=manifest
        )

        # Register in core registry
        if department.name not in self._departments:
            self._departments[department.name] = []
        self._departments[department.name].append(instance)

        # Register in version tracking
        if department.name not in self._versions:
            self._versions[department.name] = {}
        self._versions[department.name][department.version] = instance

        # Index by domain
        if manifest.domain not in self._by_domain:
            self._by_domain[manifest.domain] = set()
        self._by_domain[manifest.domain].add(department.name)

        # Index by task types
        for task_type in manifest.supported_tasks:
            if task_type not in self._by_task:
                self._by_task[task_type] = set()
            self._by_task[task_type].add(department.name)

        # Register dependencies
        if manifest.requires:
            self._dependencies[department.name] = set(manifest.requires)

        logger.info(
            f"Registered department: {department.name} v{department.version} "
            f"(domain: {manifest.domain})"
        )

        # Start health monitoring if this is first department
        if len(self._departments) == 1 and self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_monitor_loop())

    async def unregister(self, name: str, version: Optional[str] = None) -> None:
        """
        Unregister a department from the registry.

        Args:
            name: Department name
            version: Specific version to unregister (None = all versions)
        """
        if name not in self._departments:
            logger.warning(f"Department {name} not registered")
            return

        if version is None:
            # Unregister all versions
            del self._departments[name]
            del self._versions[name]

            # Remove from domain index
            for domain, dept_names in self._by_domain.items():
                dept_names.discard(name)

            # Remove from task index
            for task_type, dept_names in self._by_task.items():
                dept_names.discard(name)

            # Remove dependencies
            if name in self._dependencies:
                del self._dependencies[name]

            logger.info(f"Unregistered all versions of department: {name}")

        else:
            # Unregister specific version
            if version not in self._versions.get(name, {}):
                logger.warning(f"Department {name} version {version} not registered")
                return

            # Remove from versions
            instance = self._versions[name][version]
            del self._versions[name][version]

            # Remove from core registry
            self._departments[name] = [
                inst for inst in self._departments[name]
                if inst.department.version != version
            ]

            logger.info(f"Unregistered department: {name} v{version}")

    # ========================================================================
    # Discovery
    # ========================================================================

    def get_department(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[Department]:
        """
        Get a department by name and optional version.

        Args:
            name: Department name
            version: Specific version (None = latest)

        Returns:
            Department instance, or None if not found
        """
        if name not in self._departments:
            return None

        if version is None:
            # Return latest version (first instance)
            instances = self._departments[name]
            if not instances:
                return None
            return instances[0].department

        # Return specific version
        if name not in self._versions or version not in self._versions[name]:
            return None

        return self._versions[name][version].department

    def find_by_domain(self, domain: str) -> List[Department]:
        """
        Find all departments in a specific domain.

        Args:
            domain: Domain name (e.g., "beekeeping", "healthcare")

        Returns:
            List of department instances
        """
        if domain not in self._by_domain:
            return []

        dept_names = self._by_domain[domain]
        return [
            self.get_department(name)
            for name in dept_names
            if self.get_department(name) is not None
        ]

    def find_by_task(self, task_type: str) -> List[Department]:
        """
        Find all departments that support a specific task type.

        Args:
            task_type: Task type (e.g., "retrieve_context", "extract_entities")

        Returns:
            List of department instances that support this task
        """
        if task_type not in self._by_task:
            return []

        dept_names = self._by_task[task_type]
        return [
            self.get_department(name)
            for name in dept_names
            if self.get_department(name) is not None
        ]

    def list_departments(self) -> List[DepartmentManifest]:
        """
        List all registered departments with their manifests.

        Returns:
            List of department manifests
        """
        manifests = []
        for instances in self._departments.values():
            for instance in instances:
                manifests.append(instance.manifest)
        return manifests

    # ========================================================================
    # Dependency Resolution
    # ========================================================================

    def resolve_dependencies(self, name: str) -> List[str]:
        """
        Resolve all dependencies for a department (recursive).

        Args:
            name: Department name

        Returns:
            List of department names in dependency order (dependencies first)

        Raises:
            ValueError: If circular dependency detected
        """
        visited = set()
        result = []

        def visit(dept_name: str, path: Set[str]):
            if dept_name in path:
                # Circular dependency
                cycle = " -> ".join(list(path) + [dept_name])
                raise ValueError(f"Circular dependency detected: {cycle}")

            if dept_name in visited:
                return

            visited.add(dept_name)
            path.add(dept_name)

            # Visit dependencies first
            if dept_name in self._dependencies:
                for dep in self._dependencies[dept_name]:
                    visit(dep, path.copy())

            result.append(dept_name)

        visit(name, set())
        return result

    def check_dependencies(self, name: str) -> bool:
        """
        Check if all dependencies for a department are registered.

        Args:
            name: Department name

        Returns:
            True if all dependencies available, False otherwise
        """
        if name not in self._dependencies:
            return True  # No dependencies

        for dep in self._dependencies[name]:
            if dep not in self._departments:
                logger.warning(f"Missing dependency for {name}: {dep}")
                return False

        return True

    # ========================================================================
    # Request Routing
    # ========================================================================

    async def route_request(
        self,
        request: DepartmentRequest,
        department_name: Optional[str] = None
    ) -> DepartmentResponse:
        """
        Route a request to the appropriate department.

        If department_name is specified, routes to that department.
        Otherwise, discovers department by task_type and load balances.

        Args:
            request: Standardized request
            department_name: Optional specific department to use

        Returns:
            DepartmentResponse from the selected department

        Raises:
            ValueError: If no suitable department found
        """
        # Select department
        if department_name is not None:
            # Route to specific department
            instance = await self._select_instance(department_name)
        else:
            # Discover by task_type
            candidates = self.find_by_task(request.task_type)
            if not candidates:
                raise ValueError(
                    f"No department found for task type: {request.task_type}"
                )

            # Load balance across candidates
            instance = await self._select_best_instance(
                [dept.name for dept in candidates]
            )

        if instance is None:
            raise ValueError(f"No healthy instance available")

        # Execute request
        instance.active_requests += 1
        try:
            response = await instance.department.execute(request)
            instance.request_count += 1
            return response
        finally:
            instance.active_requests -= 1

    async def _select_instance(
        self,
        name: str
    ) -> Optional[DepartmentInstance]:
        """
        Select a specific department instance.

        If multiple versions exist, selects the healthiest one.

        Args:
            name: Department name

        Returns:
            Selected instance, or None if none available
        """
        if name not in self._departments:
            return None

        instances = self._departments[name]

        # Filter to healthy instances
        healthy = [inst for inst in instances if inst.health_status == "healthy"]
        if not healthy:
            # Fall back to degraded instances
            healthy = [inst for inst in instances if inst.health_status == "degraded"]

        if not healthy:
            return None

        # Select instance with fewest active requests
        return min(healthy, key=lambda inst: inst.active_requests)

    async def _select_best_instance(
        self,
        candidate_names: List[str]
    ) -> Optional[DepartmentInstance]:
        """
        Select the best department instance from multiple candidates.

        Selection criteria (in order):
        1. Health status (healthy > degraded > unhealthy)
        2. Active request count (lower is better)
        3. Historical success rate (if available)

        Args:
            candidate_names: List of department names to choose from

        Returns:
            Selected instance, or None if none available
        """
        instances = []
        for name in candidate_names:
            inst = await self._select_instance(name)
            if inst is not None:
                instances.append(inst)

        if not instances:
            return None

        # Sort by health status, then active requests
        def sort_key(inst: DepartmentInstance):
            health_priority = {"healthy": 0, "degraded": 1, "unhealthy": 2}
            return (
                health_priority.get(inst.health_status, 3),
                inst.active_requests
            )

        return min(instances, key=sort_key)

    # ========================================================================
    # Health Monitoring
    # ========================================================================

    async def _health_monitor_loop(self) -> None:
        """Background task that periodically checks department health."""
        while not self._closed:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._check_all_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check failed: {e}")

    async def _check_all_health(self) -> None:
        """Check health of all registered departments."""
        import time

        for name, instances in self._departments.items():
            for instance in instances:
                try:
                    health = await instance.department.health_check()
                    instance.health_status = health.get('status', 'unknown')
                    instance.last_health_check = time.time()

                    if instance.health_status != "healthy":
                        logger.warning(
                            f"Department {name} v{instance.department.version} "
                            f"status: {instance.health_status}"
                        )
                except Exception as e:
                    logger.error(f"Health check failed for {name}: {e}")
                    instance.health_status = "unhealthy"

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def close(self) -> None:
        """Cleanup registry resources."""
        if self._closed:
            return

        # Stop health monitoring
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close all departments
        for instances in self._departments.values():
            for instance in instances:
                if isinstance(instance.department, BaseDepartment):
                    await instance.department.close()

        self._closed = True
        logger.info("Department registry closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
