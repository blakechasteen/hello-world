"""
Domain Harness Demo
===================

Demonstrates the Initializer/Worker pattern without Neo4j dependency.
Uses in-memory storage for the demo.

Run:
    python HoloLoom/domain_harness/demo.py
"""

import asyncio
from datetime import datetime

from hololoom.domain_harness.initializer import DomainInitializer
from hololoom.domain_harness.protocol import DomainProtocol
from hololoom.domain_harness.schema import (
    DomainContext,
    DomainState,
    Feature,
    FeatureStatus,
    ProgressAction,
    ProgressEntry,
    TestResult,
)
from hololoom.domain_harness.worker import DomainWorker, ImplementationResult, WorkerConfig

# ============================================================================
# In-Memory Storage (Demo Only)
# ============================================================================

class InMemoryYarn:
    """Simple in-memory graph storage for demo purposes."""

    def __init__(self):
        self.features: dict[str, Feature] = {}
        self.progress: list[ProgressEntry] = []
        self.state: DomainState | None = None
        self.test_results: list[TestResult] = []

    def add_feature(self, feature: Feature):
        self.features[feature.id] = feature

    def get_feature(self, feature_id: str) -> Feature | None:
        return self.features.get(feature_id)

    def get_actionable_features(self) -> list[Feature]:
        return [f for f in self.features.values() if f.is_actionable()]

    def update_feature_status(self, feature_id: str, status: FeatureStatus):
        if feature_id in self.features:
            self.features[feature_id].status = status
            self.features[feature_id].updated_at = datetime.now()

    def add_progress(self, entry: ProgressEntry):
        self.progress.append(entry)

    def get_recent_progress(self, limit: int = 10) -> list[ProgressEntry]:
        return sorted(self.progress, key=lambda p: p.timestamp, reverse=True)[:limit]


# ============================================================================
# Demo Protocol (In-Memory)
# ============================================================================

class DemoProtocol(DomainProtocol):
    """Protocol implementation using in-memory storage."""

    def __init__(self, yarn: InMemoryYarn, project_name: str = "demo"):
        super().__init__(yarn=yarn, project_name=project_name)
        self.yarn: InMemoryYarn = yarn

    async def _get_or_create_state(self) -> DomainState:
        if not self.yarn.state:
            self.yarn.state = DomainState(
                project_name=self.project_name,
                total_features=len(self.yarn.features)
            )
        return self.yarn.state

    async def _get_actionable_features(self) -> list[Feature]:
        return self.yarn.get_actionable_features()

    async def _get_recent_progress(self, limit: int = 10) -> list[ProgressEntry]:
        return self.yarn.get_recent_progress(limit)

    async def _update_feature_status(self, feature_id: str, status: FeatureStatus):
        self.yarn.update_feature_status(feature_id, status)

        # Update state counts
        if self.yarn.state:
            passing = len([f for f in self.yarn.features.values() if f.status == FeatureStatus.PASSING])
            failing = len([f for f in self.yarn.features.values() if f.status == FeatureStatus.FAILING])
            self.yarn.state.passing_features = passing
            self.yarn.state.failing_features = failing

    async def _save_progress_entry(self, entry: ProgressEntry):
        self.yarn.add_progress(entry)

    async def _save_test_result(self, result: TestResult):
        self.yarn.test_results.append(result)

    async def _update_state(self, **kwargs):
        if self.yarn.state:
            for k, v in kwargs.items():
                if hasattr(self.yarn.state, k):
                    setattr(self.yarn.state, k, v)


# ============================================================================
# Demo Implementation Functions
# ============================================================================

async def demo_implementation(feature: Feature, context: DomainContext) -> ImplementationResult:
    """Simulate implementing a feature."""
    print(f"    📝 Implementing: {feature.name}")
    await asyncio.sleep(0.1)  # Simulate work

    return ImplementationResult(
        success=True,
        files_changed=[f"src/{feature.name.lower().replace(' ', '_')}.py"],
        output=f"Created implementation for {feature.name}"
    )


async def demo_test(feature: Feature) -> TestResult:
    """Simulate running tests - 80% pass rate."""
    import random

    print(f"    🧪 Testing: {feature.name}")
    await asyncio.sleep(0.05)  # Simulate test run

    # 80% pass rate
    passed = random.random() < 0.8

    return TestResult(
        feature_id=feature.id,
        passed=passed,
        output="All tests passed!" if passed else "AssertionError: Expected behavior not met",
        duration_ms=random.randint(50, 500)
    )


# ============================================================================
# Demo Worker
# ============================================================================

class DemoWorker(DomainWorker):
    """Worker using in-memory storage."""

    def __init__(self, yarn: InMemoryYarn, project_name: str = "demo"):
        # Don't call super().__init__ - we'll set up manually
        self.yarn = yarn
        self.config = WorkerConfig(verbose=True)
        self.implementation_fn = demo_implementation
        self.test_fn = demo_test
        self.working_memory = None
        self.warp = None

        # Use demo protocol
        self.protocol = DemoProtocol(yarn=yarn, project_name=project_name)

        self._current_context = None
        self._run_start_time = None


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    print("\n" + "="*70)
    print("🧵 HoloLoom Domain Harness Demo")
    print("="*70)

    # 1. Initialize domain from prompt
    print("\n📋 STEP 1: Initialize Domain Memory")
    print("-" * 40)

    prompt = """
    Build a user management system:
    - Database schema design [HIGH]
    - User registration (depends on database)
    - User login with JWT (depends on registration)
    - Password reset functionality (depends on login)
    - User profile management (depends on login)
    - Admin dashboard (depends on user profile)
    """

    initializer = DomainInitializer()
    result = await initializer.initialize(prompt, project_name="UserManagement")

    print(f"\n✨ Created {len(result.features)} features:")
    for f in result.features:
        print(f"   [{f.id}] {f.name} (priority: {f.priority})")

    print(f"\n🔗 Dependencies: {len(result.dependencies)}")
    for d in result.dependencies:
        print(f"   {d.from_feature_id} → {d.to_feature_id}")

    # 2. Set up in-memory storage
    print("\n💾 STEP 2: Create In-Memory Storage")
    print("-" * 40)

    yarn = InMemoryYarn()
    for feature in result.features:
        yarn.add_feature(feature)

    yarn.state = result.state
    print(f"   Loaded {len(yarn.features)} features into memory")

    # 3. Run worker cycles
    print("\n🔄 STEP 3: Run Worker Cycles")
    print("-" * 40)

    worker = DemoWorker(yarn=yarn, project_name="UserManagement")

    max_cycles = 20
    cycle = 0

    while cycle < max_cycles:
        cycle += 1

        # Check completion
        actionable = yarn.get_actionable_features()
        if not actionable:
            print("\n✅ All features complete!")
            break

        print(f"\n--- Cycle {cycle} ---")
        result = await worker.run()

        if result.feature_id == "none":
            print("   No work available (dependencies not met)")
            break

    # 4. Show final status
    print("\n📊 STEP 4: Final Status")
    print("-" * 40)

    for f in yarn.features.values():
        status_icon = {
            FeatureStatus.PASSING: "✅",
            FeatureStatus.FAILING: "❌",
            FeatureStatus.PENDING: "⏳",
            FeatureStatus.IN_PROGRESS: "🔄",
            FeatureStatus.BLOCKED: "🚫",
            FeatureStatus.SKIPPED: "⏭️"
        }.get(f.status, "?")
        print(f"   {status_icon} [{f.id}] {f.name} - {f.status.value}")

    print(f"\n📈 Progress Log ({len(yarn.progress)} entries):")
    for entry in yarn.progress[:10]:
        icon = "✓" if entry.action == ProgressAction.COMPLETED else "✗"
        print(f"   {icon} {entry.timestamp.strftime('%H:%M:%S')} - {entry.summary[:50]}")

    # Summary
    passing = len([f for f in yarn.features.values() if f.status == FeatureStatus.PASSING])
    total = len(yarn.features)
    print(f"\n🎯 Completion: {passing}/{total} ({100*passing/total:.0f}%)")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
