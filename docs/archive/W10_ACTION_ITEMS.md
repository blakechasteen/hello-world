# W10: Setup Complexity - Action Items

**Analysis Date**: December 31, 2025
**Priority**: Critical path analysis
**Time Estimate**: Phase 1 = 2-3 hours, Phase 2 = 4-6 hours

---

## Phase 1: Critical (Do First - 1-2 hours)

These 5 actions will improve setup complexity from **65→85** with minimal effort.

### 1. Create `docker-compose.yml` in Root (15 min)

**File**: `c:\Users\blake\OneDrive\Documents\mythRL\docker-compose.yml`

**Content template**:
```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:latest
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      NEO4J_AUTH: neo4j/hololoom123
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - neo4j_data:/data
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:7474"]
      interval: 5s
      timeout: 3s
      retries: 5

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"  # HTTP API
      - "6334:6334"  # gRPC
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT_API_KEY: "qdrant123"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  neo4j_data:
  qdrant_data:

# Usage:
#   docker-compose up -d         # Start services
#   docker-compose logs -f        # View logs
#   docker-compose down          # Stop services
#   docker-compose down -v       # Stop + remove volumes
```

**Why**: Eliminates Docker guesswork, production blocker solved.
**Verification**: `docker-compose ps` should show both healthy.

---

### 2. Create `.env.example` (10 min)

**File**: `c:\Users\blake\OneDrive\Documents\mythRL\.env.example`

**Content**:
```bash
# HoloLoom Configuration

# Optional: Neo4j Backend (requires docker-compose)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=hololoom123

# Optional: Qdrant Backend (requires docker-compose)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=qdrant123

# Optional: Model Cache Location
# Default: ~/.cache/huggingface/
TRANSFORMERS_CACHE=~/.cache/huggingface/

# Optional: HoloLoom Cache
# Default: ~/.cache/hololoom/
HOLOLOOM_CACHE=~/.cache/hololoom/

# Development Settings
PYTHONPATH=.
ENVIRONMENT=development  # or staging, production

# Optional: LLM Provider (if using full features)
LLM_PROVIDER=ollama  # or anthropic, openai
```

**Why**: Reduces environment variable guessing.
**Steps**:
1. Copy to `.env` locally
2. Update values for your setup
3. `hololoom/config.py` already respects these

---

### 3. Create Installation Verification Script (20 min)

**File**: `c:\Users\blake\OneDrive\Documents\mythRL\verify_setup.py`

**Content**:
```python
#!/usr/bin/env python3
"""
HoloLoom Setup Verification Script

Checks: Python version, core dependencies, optional features, Docker
Outputs: Green/yellow/red status for each component
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python 3.10+"""
    version = sys.version_info
    required = (3, 10)
    status = "✅" if version >= required else "❌"
    print(f"{status} Python {version.major}.{version.minor} (required: {required[0]}.{required[1]}+)")
    return version >= required

def check_import(module_name, friendly_name=None):
    """Check if module can be imported"""
    try:
        __import__(module_name)
        print(f"  ✅ {friendly_name or module_name}")
        return True
    except ImportError as e:
        print(f"  ❌ {friendly_name or module_name} - {e}")
        return False

def check_dependencies():
    """Check all dependencies"""
    print("\n📦 Core Dependencies:")
    results = {
        'torch': check_import('torch', 'PyTorch'),
        'numpy': check_import('numpy', 'NumPy'),
        'networkx': check_import('networkx', 'NetworkX'),
        'sentence_transformers': check_import('sentence_transformers', 'Sentence Transformers'),
    }

    print("\n📦 Optional Dependencies:")
    optional = {
        'spacy': check_import('spacy', 'spaCy (NLP)'),
        'scipy': check_import('scipy', 'SciPy (Spectral)'),
        'neo4j': check_import('neo4j', 'Neo4j (Graph DB)'),
        'qdrant_client': check_import('qdrant_client', 'Qdrant (Vector DB)'),
    }

    print("\n📦 Testing Dependencies:")
    testing = {
        'pytest': check_import('pytest', 'pytest'),
        'pytest_asyncio': check_import('pytest_asyncio', 'pytest-asyncio'),
    }

    core_ok = all(results.values())
    optional_count = sum(optional.values())

    return core_ok, optional_count

def check_hololoom():
    """Check HoloLoom imports"""
    print("\n🧠 HoloLoom Imports:")
    try:
        from hololoom import HoloLoom
        print("  ✅ HoloLoom (main)")
    except ImportError as e:
        print(f"  ❌ HoloLoom - {e}")
        return False

    try:
        from hololoom.lite import HoloLoomLite
        print("  ✅ HoloLoomLite")
    except ImportError as e:
        print(f"  ❌ HoloLoomLite - {e}")

    return True

def check_docker():
    """Check Docker availability"""
    print("\n🐳 Docker:")
    try:
        result = subprocess.run(['docker', '--version'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"  ✅ Docker - {result.stdout.strip()}")

            # Check docker-compose
            result = subprocess.run(['docker', 'compose', '--version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"  ✅ Docker Compose - {result.stdout.strip()}")
            else:
                print(f"  ⚠️  Docker Compose not available")
        else:
            print(f"  ⚠️  Docker available but not responding")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"  ⚠️  Docker not installed (optional)")

def check_cache():
    """Check cache directories"""
    print("\n💾 Cache Directories:")
    cache_dir = Path.home() / '.cache' / 'huggingface'
    if cache_dir.exists():
        size_mb = sum(f.stat().st_size for f in cache_dir.rglob('*')) / 1024 / 1024
        print(f"  ✅ HuggingFace cache: {cache_dir} ({size_mb:.0f} MB)")
    else:
        print(f"  ℹ️  HuggingFace cache not yet created (normal on first run)")

def main():
    """Run all checks"""
    print("=" * 70)
    print("       HoloLoom Setup Verification")
    print("=" * 70)

    # Python version
    if not check_python_version():
        print("\n❌ Python version too old. Please upgrade to 3.10+")
        return False

    # Dependencies
    core_ok, optional_count = check_dependencies()

    if not core_ok:
        print("\n❌ Core dependencies missing. Run:")
        print("   pip install torch numpy networkx sentence-transformers")
        return False

    print(f"\n✅ Core dependencies OK")
    print(f"ℹ️  {optional_count}/4 optional dependencies installed")

    # HoloLoom
    if not check_hololoom():
        print("\n❌ HoloLoom import failed")
        return False

    # Docker
    check_docker()

    # Cache
    check_cache()

    # Final status
    print("\n" + "=" * 70)
    if core_ok:
        print("✅ HoloLoom is ready to use!")
        print("\nQuick start:")
        print("  python -m HoloLoom.lite repl")
        print("\nOr in Python:")
        print("  from hololoom import HoloLoomLite")
        print("  async with HoloLoomLite() as loom:")
        print("      await loom.experience('your knowledge')")
    else:
        print("❌ Setup incomplete. Check errors above.")
        return False

    print("=" * 70)
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
```

**Usage**:
```bash
python verify_setup.py
```

**Why**: Users know immediately what's working, what's missing.

---

### 4. Consolidate Documentation (20 min)

**Actions**:
1. Keep: `docs/getting-started/START_HERE.md` (entry point)
2. Keep: `docs/getting-started/installation.md` (reference)
3. Keep: `docs/getting-started/VISUAL_QUICK_START.md` (deep dive)
4. Archive: 8 other quick-start docs

**In START_HERE.md, add priority section**:
```markdown
## Choose Your Path

**First time?** Start here:
1. Run: `python verify_setup.py`
2. Read: [Installation Guide](installation.md) (5 min)
3. Try: `python -m HoloLoom.lite repl`

**Want to learn more?**
- Read: [Visual Quick Start](VISUAL_QUICK_START.md) (30+ min, comprehensive)
- Run: `PYTHONPATH=. python demos/demo_smart_routing.py`

**Production deployment?**
- Read: [Docker Setup](../production/docker-setup.md)
- Follow: [Self-Hosting Guide](../self-hosting/README.md)
```

**Why**: Clear learning path, no decision paralysis.

---

### 5. Add setup.py Entry Points (15 min)

**File**: Update `pyproject.toml`

**Current**:
```toml
[project.scripts]
hololoom = "hololoom.cli:main"
```

**Add these** (if not already present):
```toml
[project.scripts]
hololoom = "hololoom.cli:main"
hololoom-lite = "hololoom.lite:main"
hololoom-verify = "verify_setup:main"
```

**Why**: Users can run `hololoom-lite repl` instead of `PYTHONPATH=. python -m HoloLoom.lite repl`

**Note**: Requires `pip install -e .` for local development

---

## Phase 2: Important (Do Next - 4-6 hours)

These add polish and wider compatibility.

### 6. Publish to PyPI (2-3 hours)

**Steps**:
```bash
# 1. Build distribution
python -m build

# 2. Upload to TestPyPI first
python -m twine upload --repository testpypi dist/*

# 3. Test installation
pip install --index-url https://test.pypi.org/simple/ hololoom

# 4. Upload to PyPI
python -m twine upload dist/*
```

**Then users can**:
```bash
pip install hololoom              # Full system
pip install hololoom[lite]        # Lite only
pip install hololoom[production]  # + Neo4j/Qdrant
```

**Why**: Standard Python installation experience.

---

### 7. Create `.devcontainer/` Config (1-2 hours)

**File**: `.devcontainer/devcontainer.json`

**Benefits**: One-click VS Code setup with everything pre-installed.

```json
{
  "name": "HoloLoom Development",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "features": {
    "ghcr.io/devcontainers/features/docker-in-docker:2": {}
  },
  "postCreateCommand": "pip install -e '.[dev,production]' && docker-compose up -d",
  "customizations": {
    "vscode": {
      "extensions": ["ms-python.python", "ms-python.vscode-pylance"],
      "settings": {
        "python.defaultInterpreterPath": "${containerEnv:PYTHON_PATH}/bin/python"
      }
    }
  }
}
```

**Why**: New users click "Reopen in Container" → everything works.

---

### 8. Add Model Download Progress (1-2 hours)

**File**: `hololoom/embedding/spectral.py` (or wherever models are downloaded)

**Change**:
```python
# Before: Silent 30-second wait
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5')

# After: Show progress
from tqdm import tqdm
print("Downloading embedding model (137 MB)...")
with tqdm(total=137, unit='MB', desc='Downloading') as pbar:
    model = SentenceTransformer(
        'nomic-ai/nomic-embed-text-v1.5',
        cache_folder=cache_path
    )
    pbar.update(137)
```

**Why**: Users see "downloading" instead of mysterious hang.

---

## Phase 3: Nice-to-Have (2-4 weeks)

### 9. Create Setup Wizard (3-4 hours)

```bash
python -m HoloLoom.install
```

**Interactive flow**:
1. Detect platform (Linux/Mac/Windows)
2. Detect GPU (CUDA/ROCm/MPS/CPU)
3. Choose: Lite vs Full vs Production
4. Check dependencies
5. Download models
6. Run verification
7. Done! ✅

---

### 10. Lightweight Runtime Option (2-3 weeks)

Pure Python version without torch for:
- Serverless (AWS Lambda)
- Containerized (Docker)
- Embedded (Raspberry Pi)

---

## Implementation Checklist

### Phase 1 (Today - 2-3 hours)
- [ ] Create `docker-compose.yml` (15 min)
  - [ ] Test: `docker-compose ps` shows healthy
- [ ] Create `.env.example` (10 min)
- [ ] Create `verify_setup.py` (20 min)
  - [ ] Test: `python verify_setup.py` passes
- [ ] Consolidate docs (20 min)
  - [ ] Archive 8 duplicate quick-starts
  - [ ] Update START_HERE.md with clear paths
- [ ] Add setup.py entry points (15 min)
  - [ ] Test: `pip install -e .` works
  - [ ] Test: `hololoom-lite repl` works

### Phase 2 (This week - 4-6 hours)
- [ ] Publish to PyPI (2-3 hours)
  - [ ] Test TestPyPI first
  - [ ] Verify `pip install hololoom[lite]` works
- [ ] Create `.devcontainer/` config (1-2 hours)
  - [ ] Test: "Reopen in Container" works in VS Code
- [ ] Add model download progress (1-2 hours)
  - [ ] Test: See progress bar during first model download

### Phase 3 (This quarter - 2-4 weeks)
- [ ] Create setup wizard (3-4 hours)
- [ ] Build lightweight runtime (2-3 weeks)

---

## Success Metrics

**Setup Complexity Score**:
- **Before**: 65/100 (Full) | 90/100 (Lite)
- **After Phase 1**: 75/100 (Full) | 90/100 (Lite)
- **After Phase 2**: 85/100 (Full) | 92/100 (Lite)
- **After Phase 3**: 90/100 (Full) | 95/100 (Lite)

**User Experience**:
- **Before**: "Why is this hanging?" | "What's PYTHONPATH?"
- **After**: "Wow, that was easy!" | One-click setup

---

## Testing After Each Phase

### Phase 1 Verification
```bash
# 1. Run verification script
python verify_setup.py
# Expected: ✅ All core, some optional

# 2. Start Lite
python -m HoloLoom.lite repl
# Expected: Interactive prompt

# 3. Test Docker template
docker-compose ps
# Expected: neo4j and qdrant healthy
```

### Phase 2 Verification
```bash
# 1. Install from PyPI (TestPyPI first)
pip install --index-url https://test.pypi.org/simple/ hololoom[lite]

# 2. Open in DevContainer (VS Code)
# Expected: One-click, everything works

# 3. Watch model download progress
# Expected: See [████████░░] progress bar
```

### Phase 3 Verification
```bash
# 1. Run setup wizard
python -m HoloLoom.install
# Expected: Interactive, auto-detects platform

# 2. Try lightweight runtime
from hololoom.lite_minimal import HoloLoomMinimal
# Expected: Works without torch
```

---

## Notes

- **No code modifications required** for Phase 1 - pure infrastructure
- **Phase 2 requires minor changes** to entry points and model loading
- **Phase 3 is exploratory** - may discover blocking issues
- **All phases maintain backward compatibility**
- **Lite remains the recommended starting point**

---

## Expected Impact Timeline

| When | Action | Impact | Score |
|------|--------|--------|-------|
| Today | Phase 1 | Docker template, docs, verification script | 65 → 75 |
| This week | Phase 2 | PyPI, DevContainer, progress bar | 75 → 85 |
| Next month | Phase 3 | Setup wizard | 85 → 90+ |

**Bottom line**: 2-3 hours of work → 20-point improvement in setup complexity.

