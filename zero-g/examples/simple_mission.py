"""
Simple Mission Example - Complete Zero-G MVP Walkthrough

This script demonstrates a complete Zero-G mission from Preflight through Orbit.

Flow:
1. Initialize Loom Core
2. Create Launch Orchestrator
3. Execute Preflight checks
4. Run Launch Countdown (T-10 to T-0)
5. Execute Lift-Off (metadata scanning)
6. Execute Boost (schema discovery, indexing)
7. Achieve Orbit (stable operation)
8. Execute queries using Loom Core
9. (Optional) Enter EVA mode for manual intervention
10. Land gracefully

Requirements:
- Python 3.9+
- No external dependencies (uses built-in libraries only for MVP)

Usage:
    python examples/simple_mission.py
"""

import asyncio
import sys
from pathlib import Path
import logging
import json
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Import Zero-G components
from loom_core.simple_loom import create_simple_loom_core
from launch_system.simple_orchestrator import create_simple_launch_orchestrator
from g_series.g1_contact.simple_json_connector import create_json_connector
from g_series.protocols import DataSensitivity
from launch_system.protocols import LaunchStage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Mission Configuration ====================

MISSION_CONFIG = {
    "mission_id": "zero_g_mvp_demo",
    "astronaut_id": "astronaut_1",
    "mission_name": "First Zero-G MVP Mission",
    "data_sources": [
        "data/example.json"
    ],
    "permissions": {
        "read": True,
        "write": False,
        "delete": False
    },
    "environment": {
        "platform": "local",
        "region": "localhost"
    }
}


# ==================== Helper Functions ====================

def print_header(title: str, symbol: str = "="):
    """Print a formatted section header"""
    width = 60
    print(f"\n{symbol * width}")
    print(f"{title:^{width}}")
    print(f"{symbol * width}\n")


def print_status(stage: str, status: str, message: str):
    """Print a status message with formatting"""
    status_symbols = {
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "info": "ℹ️",
        "rocket": "🚀",
        "orbit": "🌍"
    }
    symbol = status_symbols.get(status, "•")
    print(f"{symbol} [{stage}] {message}")


async def create_sample_data():
    """Create sample JSON data file for the mission"""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    sample_data = {
        "users": [
            {"id": 1, "name": "Alice", "role": "engineer"},
            {"id": 2, "name": "Bob", "role": "scientist"},
            {"id": 3, "name": "Charlie", "role": "astronaut"}
        ],
        "projects": [
            {"id": 1, "name": "Zero-G MVP", "status": "in_progress"},
            {"id": 2, "name": "HoloLoom Integration", "status": "planned"}
        ],
        "metadata": {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "description": "Sample data for Zero-G MVP demonstration"
        }
    }

    sample_file = data_dir / "example.json"
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)

    print_status("Setup", "success", f"Sample data created: {sample_file}")
    return sample_file


# ==================== Main Mission Flow ====================

async def execute_mission():
    """
    Execute a complete Zero-G mission

    This demonstrates the full vertical slice of functionality.
    """
    print_header("🚀 ZERO-G MVP MISSION CONTROL 🚀", "=")
    print(f"Mission: {MISSION_CONFIG['mission_name']}")
    print(f"Astronaut: {MISSION_CONFIG['astronaut_id']}")
    print(f"Mission ID: {MISSION_CONFIG['mission_id']}\n")

    try:
        # ==================== Phase 0: Setup ====================
        print_header("Phase 0: Mission Setup")

        # Create sample data
        sample_file = await create_sample_data()

        # ==================== Phase 1: Initialize Loom Core ====================
        print_header("Phase 1: Loom Core Initialization")

        print_status("LoomCore", "info", "Initializing Loom Core...")
        loom_core = await create_simple_loom_core()
        print_status("LoomCore", "success", "Loom Core initialized successfully")

        # ==================== Phase 2: Create Launch Orchestrator ====================
        print_header("Phase 2: Launch System Initialization")

        print_status("LaunchSystem", "info", "Creating Launch Orchestrator...")
        orchestrator = await create_simple_launch_orchestrator(loom_core)
        print_status("LaunchSystem", "success", "Launch Orchestrator created")

        # ==================== Phase 3: Execute Preflight ====================
        print_header("Phase 3: Preflight Checks")

        print_status("Preflight", "info", "Beginning preflight checklist...")
        preflight_report = await orchestrator.execute_preflight(MISSION_CONFIG)

        # Display preflight results
        print(f"\n{'-' * 60}")
        print("PREFLIGHT REPORT")
        print(f"{'-' * 60}")
        print(f"Suit Fitting:       {preflight_report.suit_fitting.status.value.upper()}")
        print(f"                    {preflight_report.suit_fitting.message}")
        print(f"\nTraining:           {preflight_report.training_complete.status.value.upper()}")
        print(f"                    {preflight_report.training_complete.message}")
        print(f"\nPhysical Exam:      {preflight_report.physical_exam.status.value.upper()}")
        print(f"                    {preflight_report.physical_exam.message}")
        print(f"\nMission Briefing:   {preflight_report.briefing_acknowledged.status.value.upper()}")
        print(f"                    {preflight_report.briefing_acknowledged.message}")
        print(f"\nOVERALL STATUS:     {preflight_report.overall_status.value.upper()}")
        print(f"CAN PROCEED:        {'YES' if preflight_report.can_proceed else 'NO'}")
        print(f"{'-' * 60}\n")

        if not preflight_report.can_proceed:
            print_status("Preflight", "error", "Preflight checks failed. Aborting mission.")
            return

        print_status("Preflight", "success", "Preflight complete - GO for launch!")

        # ==================== Phase 4: Launch Countdown ====================
        print_header("Phase 4: Launch Countdown")

        print_status("Countdown", "rocket", "Initiating launch sequence...")
        print("\n🎯 LAUNCH SEQUENCE INITIATED\n")

        countdown_status = await orchestrator.execute_countdown()

        print_status("Countdown", "rocket", "T-0 REACHED - IGNITION!")

        # ==================== Phase 5: Lift-Off ====================
        print_header("Phase 5: Lift-Off")

        print_status("LiftOff", "rocket", "Beginning lift-off operations...")
        liftoff_check = await orchestrator.execute_liftoff()
        print_status("LiftOff", "success", liftoff_check.message)

        # ==================== Phase 6: Boost ====================
        print_header("Phase 6: Boosters")

        print_status("Boost", "rocket", "Engaging boosters...")
        boost_check = await orchestrator.execute_boost()
        print_status("Boost", "success", boost_check.message)

        # ==================== Phase 7: Connect Data Source (G1) ====================
        print_header("Phase 7: G1 Data Connection")

        print_status("G1", "info", "Creating JSON connector...")
        json_connector = await create_json_connector(str(sample_file.parent))

        print_status("G1", "info", f"Connecting to {sample_file.name} (zero-move)...")
        connector_health = await json_connector.connect(
            source_id="sample_data",
            file_path=sample_file.name,
            sensitivity=DataSensitivity.INTERNAL
        )

        if connector_health.status == "healthy":
            print_status("G1", "success", "Data source connected successfully")

            # Get metadata (no read)
            print_status("G1", "info", "Retrieving metadata (zero-move)...")
            metadata = await json_connector.get_metadata("sample_data")

            print(f"\n{'-' * 60}")
            print("DATA SOURCE METADATA")
            print(f"{'-' * 60}")
            print(f"File Name:          {metadata['file_name']}")
            print(f"File Size:          {metadata['file_size_human']}")
            print(f"Modified:           {metadata['modified_time']}")
            print(f"Access Mode:        {metadata['access_mode']}")
            print(f"Sensitivity:        {metadata['sensitivity']}")
            print(f"{'-' * 60}\n")

            # Discover schema
            print_status("G1", "info", "Discovering schema...")
            schema = await json_connector.discover_schema("sample_data", sample_size=100)

            print(f"\n{'-' * 60}")
            print("SCHEMA DISCOVERY RESULT")
            print(f"{'-' * 60}")
            for element in schema.elements:
                print(f"Element: {element.name}")
                print(f"Type:    {element.element_type}")
                print(f"Properties:")
                print(json.dumps(element.properties, indent=2))
            print(f"{'-' * 60}\n")

        else:
            print_status("G1", "error", f"Connection failed: {connector_health.metadata.get('error')}")

        # ==================== Phase 8: Achieve Orbit ====================
        print_header("Phase 8: Orbital Insertion")

        print_status("Orbit", "orbit", "Achieving orbit...")
        orbit_status = await orchestrator.achieve_orbit()

        print(f"\n{'-' * 60}")
        print("🌍 ORBIT ACHIEVED 🌍")
        print(f"{'-' * 60}")
        print(f"Orbital Altitude:   {orbit_status.orbital_altitude}")
        print(f"Apps Docked:        {len(orbit_status.apps_docked)}")
        print(f"Yarn Stable:        {'YES' if orbit_status.yarn_stable else 'NO'}")
        print(f"Warp Aligned:       {'YES' if orbit_status.warp_aligned else 'NO'}")
        print(f"Uptime:             {orbit_status.uptime_seconds:.1f}s")
        print(f"{'-' * 60}\n")

        print_status("Orbit", "success", "Zero-G is now operational!")

        # ==================== Phase 9: Query Execution ====================
        print_header("Phase 9: Query Execution")

        # Execute sample queries
        queries = [
            "What is Thompson Sampling?",
            "Search for information about zero-move access",
            "Analyze the mission status"
        ]

        for i, query in enumerate(queries, 1):
            print(f"\n📝 Query {i}: {query}")
            print("-" * 60)

            result = await loom_core.weave(query)

            print(f"Action:         {result['action']}")
            print(f"Duration:       {result['duration_ms']:.1f}ms")
            print(f"Threads:        {result['threads_consulted']}")
            print(f"\nResult:")
            print(result['result'])
            print()

        # ==================== Phase 10: Optional EVA ====================
        print_header("Phase 10: EVA (Space Walk) - Optional")

        eva_choice = input("Would you like to enter EVA mode? (y/n): ").lower().strip()

        if eva_choice == 'y':
            print_status("EVA", "info", "Entering EVA mode...")

            eva_session = await orchestrator.enter_eva_mode(
                astronaut_id=MISSION_CONFIG['astronaut_id'],
                purpose="Manual graph inspection"
            )

            print(f"\n{'-' * 60}")
            print("🚶 EVA SESSION ACTIVE 🚶")
            print(f"{'-' * 60}")
            print(f"Session ID:     {eva_session.eva_id}")
            print(f"Astronaut:      {eva_session.astronaut_id}")
            print(f"Purpose:        {eva_session.purpose}")
            print(f"{'-' * 60}\n")

            print_status("EVA", "info", "In a production system, you would see:")
            print("  • Heddle visualization (graph structures)")
            print("  • Manual tension adjustments")
            print("  • Semantic pathway tuning")
            print("  • Real-time validation")

            print_status("EVA", "success", "EVA complete - returning to cabin")

        # ==================== Phase 11: Mission Status ====================
        print_header("Phase 11: Mission Status")

        current_stage = await orchestrator.get_current_stage()
        telemetry = await orchestrator.mission_control.get_telemetry()

        print(f"\n{'-' * 60}")
        print("MISSION STATUS")
        print(f"{'-' * 60}")
        print(f"Current Stage:      {current_stage.value.upper()}")
        print(f"Loom Health:        {telemetry.loom_health['status']}")
        print(f"G-Series Status:    {telemetry.g_series_status['status']}")
        print(f"Safety Status:      {telemetry.safety_status.value.upper()}")
        print(f"Active Missions:    {telemetry.active_missions}")
        print(f"Checkpoints:        {len(orchestrator.checkpoints)}")
        print(f"{'-' * 60}\n")

        # ==================== Phase 12: Landing ====================
        print_header("Phase 12: Landing Sequence")

        print_status("Landing", "info", "Initiating landing sequence...")
        landing_check = await orchestrator.land()
        print_status("Landing", "success", landing_check.message)

        # Shutdown Loom Core
        print_status("Shutdown", "info", "Shutting down Loom Core...")
        await loom_core.shutdown()
        print_status("Shutdown", "success", "Loom Core shutdown complete")

        # ==================== Mission Complete ====================
        print_header("🎉 MISSION COMPLETE 🎉")

        print("Mission Summary:")
        print(f"  • Mission ID:        {MISSION_CONFIG['mission_id']}")
        print(f"  • Astronaut:         {MISSION_CONFIG['astronaut_id']}")
        print(f"  • Data Sources:      {len(MISSION_CONFIG['data_sources'])}")
        print(f"  • Queries Executed:  {len(queries)}")
        print(f"  • Final Stage:       {current_stage.value.upper()}")
        print(f"  • Status:            SUCCESS")
        print()
        print("Zero-G MVP demonstration complete!")
        print("Next steps:")
        print("  1. Review the architecture docs in /docs")
        print("  2. Explore other examples in /examples")
        print("  3. Try the frontend UI: cd frontend/cabin_ui && npm start")
        print("  4. Build your own connectors and apps!")
        print()

    except Exception as e:
        logger.error(f"Mission failed: {e}", exc_info=True)
        print_status("Mission", "error", f"Mission failed: {e}")
        print("\nAttempting emergency abort...")

        try:
            abort_check = await orchestrator.abort_mission(
                reason=str(e),
                rollback_to=LaunchStage.GROUNDED
            )
            print_status("Abort", "warning", abort_check.message)
        except Exception as abort_error:
            logger.error(f"Abort failed: {abort_error}")
            print_status("Abort", "error", "Emergency abort failed")


# ==================== Entry Point ====================

def main():
    """Main entry point"""
    print("\n")
    print("=" * 70)
    print(" " * 15 + "ZERO-G: A GRAVITY-FREE OPERATING ENVIRONMENT")
    print(" " * 20 + "FOR AI-NATIVE APPLICATIONS")
    print("=" * 70)
    print()
    print("This demonstration shows a complete Zero-G mission:")
    print("  • Loom-based reasoning (WarpSpace, YarnGraph, etc.)")
    print("  • Spaceflight lifecycle (Preflight → Orbit)")
    print("  • Zero-move data access (G-Series connectors)")
    print("  • Safe, reversible operations")
    print()
    print("Press Ctrl+C at any time to abort the mission.")
    print()
    print("-" * 70)
    print()

    try:
        asyncio.run(execute_mission())
    except KeyboardInterrupt:
        print("\n\n⚠️  Mission aborted by user (Ctrl+C)")
        print("Emergency shutdown initiated...")
        print("✅ Graceful shutdown complete")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        logger.error("Fatal error", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
