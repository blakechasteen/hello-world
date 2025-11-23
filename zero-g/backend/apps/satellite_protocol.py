"""
Satellite App Protocol for Zero-G

Defines the interface for satellite apps that dock to Zero-G's orbital Loom.

Apps that implement this protocol can:
- Share Zero-G's Loom infrastructure (WarpSpace, YarnGraph, Rift, etc.)
- Register domain-specific tools for execution
- Communicate with other docked apps via shared semantics
- Participate in complete provenance tracking

Lifecycle:
1. App created with manifest
2. App docks to Loom via on_dock()
3. App registers tools via Rift
4. App executes queries/tools using shared Loom
5. App undocks gracefully via on_undock()
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass
from enum import Enum

# Import Loom core protocols
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from loom_core.protocols import LoomCore, RiftAction


class GLevelRequirement(Enum):
    """G-Series data access level required by app"""
    G1_CONTACT = "G1"        # Zero-move access (data in place)
    G2_LIFTOFF = "G2"        # Schema discovery, indexing
    G3_ORBITAL = "G3"        # Real-time streaming
    G4_SPACEWALK = "G4"      # Manual heddle tensioning
    G5_LEGACY_REPAIR = "G5"  # Database modernization


@dataclass
class DataSource:
    """Data source declaration for G-Series onboarding"""
    source_id: str
    source_type: str  # "database_sql", "json_files", "api", etc.
    access_mode: str  # "read", "read_write"
    sensitivity: str  # "public", "confidential", "secret"
    connection_params: Dict[str, Any]


@dataclass
class AppManifest:
    """
    Satellite app manifest - declares capabilities and requirements

    Used by Zero-G to validate docking compatibility and provision resources.
    """
    app_id: str
    app_name: str
    version: str
    description: str

    # G-Series requirement
    required_g_level: GLevelRequirement

    # Loom subsystem requirements
    requires_warp_space: bool = True
    requires_yarn_graph: bool = True
    requires_resonance_shed: bool = False
    requires_convergence_engine: bool = False
    requires_rift: bool = True
    requires_spacetime_fabric: bool = True
    requires_reflection_buffer: bool = False
    requires_thread_spinner: bool = False

    # Data sources (for G-Series onboarding)
    data_sources: List[DataSource] = None

    # Capabilities (tools this app provides)
    capabilities: List[str] = None

    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = []
        if self.capabilities is None:
            self.capabilities = []


class SatelliteApp(ABC):
    """
    Base class for Zero-G satellite applications

    Satellite apps dock to Zero-G's orbital Loom and share:
    - Semantic layer (WarpSpace + YarnGraph)
    - Tool execution (Rift)
    - Provenance tracking (SpacetimeFabric)
    - Learning (ReflectionBuffer)

    Apps provide:
    - Domain-specific UI/UX
    - Business logic and workflows
    - Specialized tools registered via Rift

    Example:
        class COZSatellite(SatelliteApp):
            def get_manifest(self) -> AppManifest:
                return AppManifest(
                    app_id="coz",
                    app_name="Cozbi's Organix Zentrum",
                    version="1.0.0",
                    required_g_level=GLevelRequirement.G2_LIFTOFF,
                    requires_warp_space=True,
                    requires_yarn_graph=True,
                    requires_rift=True
                )

            async def on_dock(self, loom: LoomCore):
                # Access shared Loom infrastructure
                self.warp = loom.warp_space
                self.yarn = loom.yarn_graph
                self.rift = loom.rift

                # Register COZ-specific tools
                await self.rift.register_tool(
                    "coz.create_batch",
                    executor=self.create_batch,
                    schema=BATCH_SCHEMA
                )
    """

    def __init__(self):
        self.loom: Optional[LoomCore] = None
        self.is_docked: bool = False

    @abstractmethod
    def get_manifest(self) -> AppManifest:
        """
        Return app manifest declaring capabilities and requirements

        Called during docking to validate compatibility.
        """
        ...

    @abstractmethod
    async def on_dock(self, loom: LoomCore) -> None:
        """
        Called when app docks to Zero-G's Loom

        Use this to:
        - Store references to Loom subsystems
        - Register tools via loom.rift
        - Initialize app-specific state
        - Set up event listeners

        Args:
            loom: The shared Loom Core instance
        """
        ...

    @abstractmethod
    async def on_undock(self) -> None:
        """
        Called when app undocks from Loom

        Use this to:
        - Clean up resources
        - Unregister tools
        - Save state
        - Close connections
        """
        ...

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint for monitoring

        Returns:
            Dict with health status and diagnostics
        """
        return {
            "app_id": self.get_manifest().app_id,
            "is_docked": self.is_docked,
            "loom_available": self.loom is not None,
            "status": "healthy" if self.is_docked else "not_docked"
        }

    async def get_capabilities(self) -> List[str]:
        """
        Get list of capabilities (tools) this app provides

        Returns:
            List of tool names (e.g., ["coz.create_batch", "coz.update_inventory"])
        """
        return self.get_manifest().capabilities


# ==================== Orbit Manager ====================

class OrbitManager:
    """
    Manages multiple satellite apps docked to a single Loom instance

    Responsibilities:
    - Validate app manifests before docking
    - Coordinate inter-app communication
    - Monitor app health
    - Handle graceful shutdown

    Example:
        orbit = OrbitManager(loom_core)
        await orbit.dock_app(coz_satellite)
        await orbit.dock_app(elle_satellite)

        # Apps now share same Loom instance
        # Can communicate via shared WarpSpace, YarnGraph, Rift
    """

    def __init__(self, loom_core: LoomCore):
        self.loom = loom_core
        self.docked_apps: Dict[str, SatelliteApp] = {}

    async def dock_app(self, app: SatelliteApp) -> bool:
        """
        Dock a satellite app to the orbital Loom

        Returns:
            True if docking successful, False otherwise
        """
        manifest = app.get_manifest()

        # Validate not already docked
        if manifest.app_id in self.docked_apps:
            print(f"⚠️  App {manifest.app_id} already docked")
            return False

        # Validate G-Level compatibility
        # (In full implementation, check if Loom has required G-Level support)

        # Execute docking sequence
        await app.on_dock(self.loom)
        app.is_docked = True
        app.loom = self.loom

        # Register in orbit
        self.docked_apps[manifest.app_id] = app

        print(f"✅ App {manifest.app_name} ({manifest.app_id}) docked successfully")
        return True

    async def undock_app(self, app_id: str) -> bool:
        """
        Undock a satellite app from the orbital Loom

        Returns:
            True if undocking successful, False if app not found
        """
        if app_id not in self.docked_apps:
            print(f"⚠️  App {app_id} not docked")
            return False

        app = self.docked_apps[app_id]

        # Execute undocking sequence
        await app.on_undock()
        app.is_docked = False
        app.loom = None

        # Remove from orbit
        del self.docked_apps[app_id]

        print(f"✅ App {app_id} undocked successfully")
        return True

    async def get_orbit_status(self) -> Dict[str, Any]:
        """Get status of all docked apps"""
        return {
            "total_apps": len(self.docked_apps),
            "apps": {
                app_id: await app.health_check()
                for app_id, app in self.docked_apps.items()
            }
        }

    async def shutdown_all(self) -> None:
        """Gracefully undock all apps"""
        for app_id in list(self.docked_apps.keys()):
            await self.undock_app(app_id)
