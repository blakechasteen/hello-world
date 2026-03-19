#!/usr/bin/env python3
"""
Domain Harness Bootstrap
========================

Drop this in any project directory and run:
    python bootstrap_harness.py "Build a REST API with user auth and blog posts"

Creates:
    domain_memory/
    ├── features.json
    ├── state.json  
    ├── progress.log
    ├── strands/
    │   └── loom_state.json
    └── tests/
        ├── conftest.py
        └── test_feature_FXXX.py (one per feature)
    
    CLAUDE.md (worker instructions)
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path


def parse_prompt_to_features(prompt: str) -> list:
    """
    Parse natural language prompt into features.
    
    Simple heuristic: split on conjunctions and list patterns.
    """
    # Common patterns
    patterns = [
        r'with\s+(.+?)(?:,|\s+and\s+|$)',
        r'including\s+(.+?)(?:,|\s+and\s+|$)',
        r'that\s+(?:can|has|supports)\s+(.+?)(?:,|\s+and\s+|$)',
    ]

    features = []
    feature_id = 1

    # Extract main subject
    main_match = re.match(r'^(?:Build|Create|Make|Implement)\s+(?:a\s+)?(.+?)(?:\s+with|\s+that|\s+including|$)', prompt, re.I)
    if main_match:
        main_subject = main_match.group(1).strip()
        features.append({
            "id": f"F{feature_id:03d}",
            "title": f"Core {main_subject}",
            "description": f"Set up the basic {main_subject} structure",
            "priority": 10,
            "depends_on": []
        })
        feature_id += 1

    # Extract sub-features
    for pattern in patterns:
        matches = re.findall(pattern, prompt, re.I)
        for match in matches:
            # Split on commas and "and"
            parts = re.split(r',\s*|\s+and\s+', match)
            for part in parts:
                part = part.strip()
                if part and len(part) > 2:
                    features.append({
                        "id": f"F{feature_id:03d}",
                        "title": part.title(),
                        "description": f"Implement {part}",
                        "priority": max(1, 10 - feature_id),
                        "depends_on": ["F001"] if feature_id > 1 else []
                    })
                    feature_id += 1

    # Fallback if no features extracted
    if len(features) < 2:
        features = [
            {"id": "F001", "title": "Core Setup", "description": "Basic project structure", "priority": 10, "depends_on": []},
            {"id": "F002", "title": "Main Feature", "description": prompt, "priority": 9, "depends_on": ["F001"]},
        ]

    return features


def generate_features_json(features: list, project_name: str) -> dict:
    """Generate features.json content."""
    return {
        "metadata": {
            "version": 1,
            "schema": "features-1.0",
            "project": project_name,
            "created": datetime.utcnow().isoformat() + "Z"
        },
        "features": [
            {
                "id": f["id"],
                "title": f["title"],
                "description": f["description"],
                "status": "failing",
                "priority": f["priority"],
                "depends_on": f["depends_on"],
                "acceptance_criteria": [f"Implement {f['title'].lower()}"],
                "test_file": f"tests/test_feature_{f['id']}.py",
                "notes": []
            }
            for f in features
        ]
    }


def generate_state_json(project_name: str) -> dict:
    """Generate state.json content."""
    return {
        "project": {
            "name": project_name,
            "version": "0.1.0",
            "language": "python"
        },
        "constraints": {
            "must_pass_all_tests": True,
            "atomic_work_unit": True,
            "max_features_per_session": 1
        },
        "rules_of_engagement": {
            "allowed_paths": ["domain_memory/", "src/", "tests/"],
            "test_command": "pytest -v",
            "forbidden": ["Delete features", "Skip tests"]
        },
        "environment": {
            "python_version": "3.11",
            "dependencies": ["pytest"]
        }
    }


def generate_loom_state(features: list) -> dict:
    """Generate initial loom_state.json."""
    strands = {}
    for f in features:
        strands[f["id"]] = {
            "feature_id": f["id"],
            "tension": 5.0,
            "weft_position": 0.0,
            "is_regression": False,
            "consecutive_failures": 0,
            "last_updated": datetime.utcnow().isoformat() + "Z"
        }

    return {
        "strands": strands,
        "total_tension": 5.0 * len(features),
        "average_tension": 5.0,
        "is_stable": True,
        "is_critical": False,
        "is_complete": False
    }


def generate_test_file(feature: dict) -> str:
    """Generate test file content."""
    return f'''"""
Feature: {feature["id"]} - {feature["title"]}
Status: failing

Description: {feature["description"]}
"""
import pytest


class Test{feature["id"]}{feature["title"].replace(" ", "")}:
    """Tests for {feature["title"]}."""
    
    def test_{feature["id"].lower()}_implementation(self):
        """Test that {feature["title"].lower()} works correctly."""
        # TODO: Implement this test
        assert False, "Not yet implemented - {feature['title']}"
'''


def generate_conftest() -> str:
    """Generate conftest.py content."""
    return '''"""
Pytest configuration for domain harness tests.
"""
import pytest
from pathlib import Path


@pytest.fixture
def domain_memory_path():
    """Path to domain memory directory."""
    return Path(__file__).parent.parent / "domain_memory"


@pytest.fixture
def features(domain_memory_path):
    """Load features from features.json."""
    import json
    with open(domain_memory_path / "features.json") as f:
        return json.load(f)["features"]
'''


def generate_progress_log(project_name: str, feature_count: int) -> str:
    """Generate initial progress.log."""
    timestamp = datetime.utcnow().isoformat() + "Z"
    return f"""# progress.log
# Project: {project_name}
# Created: {timestamp}

[INIT] Domain memory initialized. {feature_count} features defined, all failing.
"""


CLAUDE_MD = '''# CLAUDE.md - Domain Harness Worker

You are a stateless worker. No memory between sessions. Everything must be in files.

## The Ritual

### BOOT
```bash
cat domain_memory/features.json   # What to build
cat domain_memory/progress.log    # History
```

Select ONE feature: status=failing, dependencies satisfied, highest priority.

### ACTION
1. Set status to `in_progress`
2. Implement in `src/`
3. Update tests
4. Run: `pytest tests/test_feature_FXXX.py -v`
5. Set status to `passing` or `failing`

### EXIT
1. Append to progress.log: `[TIMESTAMP] PASS/FAIL FXXX: Summary`
2. **STOP.** Do not continue.

## Rules
- ONE feature per session
- Tests are truth
- Dependencies must be passing first
- Record everything in files

## Status Values
`pending` | `failing` | `in_progress` | `passing` | `blocked`
'''


def bootstrap(prompt: str, base_dir: Path = Path(".")):
    """Bootstrap domain harness from prompt."""

    # Parse prompt
    project_name = "domain-project"
    name_match = re.search(r'(?:called|named)\s+"?([^"]+)"?', prompt, re.I)
    if name_match:
        project_name = name_match.group(1).strip().lower().replace(" ", "-")

    features = parse_prompt_to_features(prompt)
    print(f"Parsed {len(features)} features from prompt")

    # Create directories
    domain_dir = base_dir / "domain_memory"
    tests_dir = domain_dir / "tests"
    strands_dir = domain_dir / "strands"
    src_dir = base_dir / "src"

    for d in [domain_dir, tests_dir, strands_dir, src_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Generate files
    files_created = []

    # features.json
    features_data = generate_features_json(features, project_name)
    with open(domain_dir / "features.json", "w") as f:
        json.dump(features_data, f, indent=2)
    files_created.append("domain_memory/features.json")

    # state.json
    state_data = generate_state_json(project_name)
    with open(domain_dir / "state.json", "w") as f:
        json.dump(state_data, f, indent=2)
    files_created.append("domain_memory/state.json")

    # loom_state.json
    loom_data = generate_loom_state(features)
    with open(strands_dir / "loom_state.json", "w") as f:
        json.dump(loom_data, f, indent=2)
    files_created.append("domain_memory/strands/loom_state.json")

    # progress.log
    progress = generate_progress_log(project_name, len(features))
    with open(domain_dir / "progress.log", "w") as f:
        f.write(progress)
    files_created.append("domain_memory/progress.log")

    # Test files
    with open(tests_dir / "conftest.py", "w") as f:
        f.write(generate_conftest())
    files_created.append("domain_memory/tests/conftest.py")

    for feature in features_data["features"]:
        test_file = tests_dir / f"test_feature_{feature['id']}.py"
        with open(test_file, "w") as f:
            f.write(generate_test_file(feature))
        files_created.append(f"domain_memory/tests/test_feature_{feature['id']}.py")

    # CLAUDE.md
    with open(base_dir / "CLAUDE.md", "w") as f:
        f.write(CLAUDE_MD)
    files_created.append("CLAUDE.md")

    # src/__init__.py
    with open(src_dir / "__init__.py", "w") as f:
        f.write(f'"""{project_name}"""\n')
    files_created.append("src/__init__.py")

    print("\n✓ Domain harness initialized")
    print(f"  Project: {project_name}")
    print(f"  Features: {len(features)}")
    print(f"  Files created: {len(files_created)}")
    print("\nFeatures:")
    for f in features:
        print(f"  {f['id']}: {f['title']} (priority={f['priority']})")
    print("\nNext: Open Claude Code and begin the ritual.")

    return files_created


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bootstrap_harness.py \"Your project description\"")
        print("\nExample:")
        print('  python bootstrap_harness.py "Build a REST API with user auth and blog posts"')
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    bootstrap(prompt)
