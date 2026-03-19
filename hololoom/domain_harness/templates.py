"""
Domain Harness Templates - Concrete JSON/File Formats

These templates match the GPT handoff exactly.
Used by the initializer to generate flat-file domain memory.

Two modes:
1. Flat files (features.json, state.json, progress.log) - works anywhere
2. Neo4j graph (Feature nodes, relationships) - for HoloLoom integration

The flat-file format is the "source of truth" schema.
Neo4j is an optimization layer on top.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hololoom.domain_harness.schema import DomainState, Feature, FeatureStatus

# ============================================================================
# Template Schemas
# ============================================================================

FEATURES_SCHEMA_VERSION = "features-1.0"
STATE_SCHEMA_VERSION = "state-1.0"


def get_utc_now() -> str:
    """ISO timestamp in UTC."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================================
# features.json Template
# ============================================================================

def generate_features_json(
    features: list[Feature],
    description: str = "Feature backlog for the agent-driven project."
) -> dict[str, Any]:
    """
    Generate features.json in the canonical format.
    
    All features start with status: "failing" until tests pass.
    """
    return {
        "metadata": {
            "version": 1,
            "created_at": get_utc_now(),
            "description": description,
            "schema": FEATURES_SCHEMA_VERSION
        },
        "features": [
            {
                "id": f.id,
                "title": f.name,
                "description": f.description,
                "status": "failing",  # Always start failing
                "priority": f.priority,
                "depends_on": f.depends_on,
                "acceptance_criteria": f.acceptance_criteria,
                "last_updated": None,
                "notes": []
            }
            for f in features
        ]
    }


def parse_features_json(data: dict[str, Any]) -> list[Feature]:
    """Parse features.json back into Feature objects."""
    features = []
    for f in data.get("features", []):
        status_map = {
            "failing": FeatureStatus.FAILING,
            "passing": FeatureStatus.PASSING,
            "pending": FeatureStatus.PENDING,
            "in_progress": FeatureStatus.IN_PROGRESS,
            "blocked": FeatureStatus.BLOCKED,
            "skipped": FeatureStatus.SKIPPED
        }
        features.append(Feature(
            id=f["id"],
            name=f["title"],
            description=f["description"],
            status=status_map.get(f["status"], FeatureStatus.FAILING),
            priority=f.get("priority", 0),
            depends_on=f.get("depends_on", []),
            acceptance_criteria=f.get("acceptance_criteria", [])
        ))
    return features


# ============================================================================
# state.json Template
# ============================================================================

def generate_state_json(
    project_name: str,
    language: str = "python",
    runtime: str = ">=3.10",
    constraints: list[str] | None = None,
    environment: dict[str, Any] | None = None,
    test_command: str = "pytest -q"
) -> dict[str, Any]:
    """
    Generate state.json with project config and constraints.
    
    This encodes the "rules of engagement" for worker agents.
    """
    return {
        "project": {
            "name": project_name,
            "version": "0.1.0",
            "language": language,
            "runtime": runtime
        },
        "constraints": {
            "must_pass_all_tests": True,
            "atomic_work_unit": True,
            "worker_must_update_feature_state": True,
            "worker_must_append_progress_log": True,
            "custom": constraints or []
        },
        "rules_of_engagement": {
            "allowed_changes": ["source_code", "tests", "features.json", "progress.log"],
            "forbidden_changes": ["memory_schema", "initializer_artifacts"],
            "test_command": test_command
        },
        "environment": environment or {
            "dependencies": ["pytest"],
            "install_instructions": "pip install -r requirements.txt"
        }
    }


def parse_state_json(data: dict[str, Any]) -> DomainState:
    """Parse state.json back into DomainState."""
    project = data.get("project", {})
    constraints = data.get("constraints", {})

    return DomainState(
        project_name=project.get("name", "unknown"),
        constraints=constraints.get("custom", []),
        environment=data.get("environment", {})
    )


# ============================================================================
# progress.log Template
# ============================================================================

def generate_progress_log_header(project_name: str) -> str:
    """Generate initial progress.log content."""
    return f"""# progress.log
# Created: {get_utc_now()}
# Project: {project_name}
# Purpose: Linear chronological record of all worker agent runs.

[INIT] Project initialized by initializer agent.
       features.json created with all features marked 'failing'.
       Test scaffolding generated.
       Worker agents will make atomic progress on one feature at a time.

"""


def append_progress_entry(
    feature_id: str,
    action: str,
    summary: str,
    worker_id: str | None = None
) -> str:
    """Generate a progress log entry."""
    timestamp = get_utc_now()
    worker_tag = f" [{worker_id}]" if worker_id else ""
    return f"[{timestamp}]{worker_tag} {action.upper()} {feature_id}: {summary}\n"


# ============================================================================
# Test Scaffolding Templates
# ============================================================================

def generate_test_file(feature: Feature) -> str:
    """Generate a pytest file for a feature."""
    safe_id = feature.id.replace("-", "_").lower()

    # Build acceptance criteria as comments
    criteria_comments = ""
    if feature.acceptance_criteria:
        criteria_comments = "\n".join(
            f"    # - {c}" for c in feature.acceptance_criteria
        )
    else:
        criteria_comments = f"    # - {feature.description}"

    return f'''"""
Test: {feature.name}
Feature ID: {feature.id}

Acceptance Criteria:
{criteria_comments}

This test intentionally fails until implementation is done.
The worker agent must implement real functionality to make it pass.
"""

import pytest


def test_{safe_id}_implementation():
    """
    Worker agent must replace this with real tests.
    
    Expected behavior:
{criteria_comments}
    """
    # TODO: Implement actual test logic
    assert False, "{feature.name} not yet implemented."


def test_{safe_id}_edge_cases():
    """Edge case handling for {feature.name}."""
    # TODO: Add edge case tests
    pytest.skip("Edge case tests not yet defined.")
'''


def generate_conftest(project_name: str) -> str:
    """Generate pytest conftest.py."""
    return f'''"""
Pytest configuration for {project_name}
Generated by Domain Harness Initializer
"""

import pytest
import json
from pathlib import Path


@pytest.fixture
def domain_memory_path():
    """Path to domain memory directory."""
    return Path("domain_memory")


@pytest.fixture
def features(domain_memory_path):
    """Load current features.json."""
    features_path = domain_memory_path / "features.json"
    if features_path.exists():
        with open(features_path) as f:
            return json.load(f)
    return {{"features": []}}


@pytest.fixture
def state(domain_memory_path):
    """Load current state.json."""
    state_path = domain_memory_path / "state.json"
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {{}}


@pytest.fixture
def project_context(features, state):
    """Combined project context for tests."""
    return {{
        "project": state.get("project", {{}}),
        "features": features.get("features", []),
        "constraints": state.get("constraints", {{}})
    }}
'''


# ============================================================================
# README Template
# ============================================================================

def generate_worker_readme(project_name: str) -> str:
    """Generate README.md for the domain_memory directory."""
    return f'''# {project_name} - Agent Harness Domain Memory

## Overview

This directory contains the persistent state for agent-driven development.
Workers are stateless - all knowledge comes from these files.

## Files

| File | Purpose | Who Writes |
|------|---------|------------|
| `features.json` | Feature backlog with pass/fail status | Initializer creates, Worker updates |
| `state.json` | Project config, constraints, rules | Initializer creates, rarely changes |
| `progress.log` | Chronological worker run history | Worker appends |
| `tests/` | Pytest files for each feature | Initializer scaffolds, Worker implements |

## Worker Boot Protocol

1. Read `features.json`
2. Read `progress.log` (last N entries)
3. Read last git commit(s)
4. Select **ONE** failing feature
5. Implement only that feature
6. Run tests: `pytest -q`
7. Update `features.json` with pass/fail result
8. Append entry to `progress.log`
9. Output a git-ready diff

## Rules

### Do:
- Work on exactly ONE feature per run
- Run tests before updating status
- Log all actions to progress.log
- Leave codebase in working state

### Do Not:
- Modify multiple features in one run
- Assume anything not in these files
- Skip the test step
- Modify state.json or schema

## Status Values

| Status | Meaning |
|--------|---------|
| `failing` | Not implemented or tests fail |
| `passing` | Tests pass |
| `in_progress` | Worker currently active |
| `blocked` | Waiting on dependency |
| `skipped` | Explicitly skipped |

## Test Command

```bash
pytest -q
```

Or run specific feature tests:

```bash
pytest tests/test_feature_F001.py -v
```
'''


# ============================================================================
# Full Directory Generator
# ============================================================================

class FlatFileGenerator:
    """
    Generate complete flat-file domain memory structure.
    
    Creates:
        domain_memory/
            features.json
            state.json
            progress.log
            README.md
            tests/
                conftest.py
                test_feature_*.py
    """

    def __init__(self, output_dir: Path = Path("domain_memory")):
        self.output_dir = Path(output_dir)
        self.tests_dir = self.output_dir / "tests"

    def generate(
        self,
        features: list[Feature],
        project_name: str,
        language: str = "python",
        constraints: list[str] | None = None,
        environment: dict[str, Any] | None = None
    ) -> dict[str, Path]:
        """
        Generate all domain memory files.
        
        Returns dict of {name: path} for created files.
        """
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tests_dir.mkdir(parents=True, exist_ok=True)

        created_files = {}

        # features.json
        features_path = self.output_dir / "features.json"
        features_data = generate_features_json(features, f"Feature backlog for {project_name}")
        with open(features_path, 'w') as f:
            json.dump(features_data, f, indent=2)
        created_files["features.json"] = features_path

        # state.json
        state_path = self.output_dir / "state.json"
        state_data = generate_state_json(
            project_name=project_name,
            language=language,
            constraints=constraints,
            environment=environment
        )
        with open(state_path, 'w') as f:
            json.dump(state_data, f, indent=2)
        created_files["state.json"] = state_path

        # progress.log
        progress_path = self.output_dir / "progress.log"
        with open(progress_path, 'w') as f:
            f.write(generate_progress_log_header(project_name))
        created_files["progress.log"] = progress_path

        # README.md
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(generate_worker_readme(project_name))
        created_files["README.md"] = readme_path

        # conftest.py
        conftest_path = self.tests_dir / "conftest.py"
        with open(conftest_path, 'w') as f:
            f.write(generate_conftest(project_name))
        created_files["conftest.py"] = conftest_path

        # Test files for each feature
        for feature in features:
            safe_id = feature.id.replace("-", "_").lower()
            test_path = self.tests_dir / f"test_feature_{safe_id}.py"
            with open(test_path, 'w') as f:
                f.write(generate_test_file(feature))
            created_files[f"test_{safe_id}.py"] = test_path

        return created_files


# ============================================================================
# Flat File Backend (alternative to Neo4j)
# ============================================================================

class FlatFileBackend:
    """
    File-based domain memory backend.
    
    Drop-in alternative to Neo4j for simpler deployments.
    """

    def __init__(self, domain_memory_dir: Path = Path("domain_memory")):
        self.dir = Path(domain_memory_dir)
        self.features_path = self.dir / "features.json"
        self.state_path = self.dir / "state.json"
        self.progress_path = self.dir / "progress.log"

    def load_features(self) -> list[Feature]:
        """Load features from features.json."""
        if not self.features_path.exists():
            return []
        with open(self.features_path) as f:
            data = json.load(f)
        return parse_features_json(data)

    def save_features(self, features: list[Feature]):
        """Save features to features.json."""
        # Load existing to preserve metadata
        if self.features_path.exists():
            with open(self.features_path) as f:
                data = json.load(f)
        else:
            data = {"metadata": {}, "features": []}

        # Update features
        data["features"] = [
            {
                "id": f.id,
                "title": f.name,
                "description": f.description,
                "status": f.status.value,
                "priority": f.priority,
                "depends_on": f.depends_on,
                "acceptance_criteria": f.acceptance_criteria,
                "last_updated": get_utc_now(),
                "notes": []
            }
            for f in features
        ]

        with open(self.features_path, 'w') as f:
            json.dump(data, f, indent=2)

    def update_feature_status(self, feature_id: str, status: FeatureStatus):
        """Update a single feature's status."""
        features = self.load_features()
        for f in features:
            if f.id == feature_id:
                f.status = status
                f.updated_at = datetime.now()
                break
        self.save_features(features)

    def load_state(self) -> DomainState:
        """Load state from state.json."""
        if not self.state_path.exists():
            return DomainState(project_name="unknown")
        with open(self.state_path) as f:
            data = json.load(f)
        return parse_state_json(data)

    def append_progress(
        self,
        feature_id: str,
        action: str,
        summary: str,
        worker_id: str | None = None
    ):
        """Append entry to progress.log."""
        entry = append_progress_entry(feature_id, action, summary, worker_id)
        with open(self.progress_path, 'a') as f:
            f.write(entry)

    def get_recent_progress(self, limit: int = 10) -> list[str]:
        """Get last N lines from progress.log."""
        if not self.progress_path.exists():
            return []
        with open(self.progress_path) as f:
            lines = f.readlines()
        # Filter to actual entries (not comments/headers)
        entries = [l.strip() for l in lines if l.startswith("[") and not l.startswith("# ")]
        return entries[-limit:]

    def get_actionable_features(self) -> list[Feature]:
        """Get features that can be worked on."""
        features = self.load_features()

        # Build dependency map
        feature_map = {f.id: f for f in features}

        actionable = []
        for f in features:
            if f.status not in (FeatureStatus.FAILING, FeatureStatus.PENDING):
                continue

            # Check dependencies
            blocked = False
            for dep_id in f.depends_on:
                dep = feature_map.get(dep_id)
                if dep and dep.status != FeatureStatus.PASSING:
                    blocked = True
                    break

            if not blocked:
                actionable.append(f)

        return sorted(actionable, key=lambda x: (-x.priority, x.created_at))
