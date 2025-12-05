# HoloLoom Directory Organization

**Last Updated:** 2025-10-25

---

## Root Directory Structure

```
mythRL/
├── HoloLoom/               # Core system (all modules)
├── demos/                  # Usage examples
├── tests/                  # Test files
├── config/                 # Configuration files
├── docs/                   # Documentation
│   ├── sessions/           # Development session logs
│   └── guides/             # Feature guides
├── archive/                # Archived files from cleanup
├── mcp_server/             # MCP server implementation
├── Promptly/               # Promptly CLI system
├── mythRL_core/            # Core RL components
├── dev/                    # Development utilities
├── synthesis_output/       # Synthesis artifacts
├── CLAUDE.md              # Complete developer guide
└── README.md              # Main project README
```

---

## Directory Descriptions

### Core Directories

**HoloLoom/**
- Complete weaving architecture
- All system modules (policy, embedding, memory, etc.)
- Main entry point: `unified_api.py`
- Orchestrators: `weaving_orchestrator.py`, `synthesis_bridge.py`

**demos/**
- Working usage examples
- `01_quickstart.py` - Basic usage
- `02_web_to_memory.py` - Web ingestion
- `03_conversational.py` - Chat interface
- `04_mcp_integration.py` - MCP setup

**tests/**
- All test files
- `test_*.py` - Various component tests
- `check_memory_status.py` - Memory diagnostics

**config/**
- Configuration files
- `docker-compose.yml` - Docker setup
- `claude_desktop_config*.json` - Claude Desktop configs

### Documentation

**docs/sessions/**
- Development session logs
- Phase completion documents
- Integration sprint documentation
- Session summaries

**docs/guides/**
- Feature guides
- Architecture documentation
- Quickstart guides
- Status reports

### Supporting Directories

**archive/**
- Files from cleanup operations
- Historical demos and tests
- Preserved for reference

**mcp_server/**
- MCP server implementation
- Protocol definitions
- Memory integration

**Promptly/**
- Promptly CLI system
- Separate project
- Skills and templates

**mythRL_core/**
- Core RL components
- Gymnasium environments
- Training utilities

---

## File Organization Rules

### Root Level
**ONLY essential files:**
- `README.md` - Main project README
- `CLAUDE.md` - Developer guide
- `HoloLoom.py` - Legacy entry point
- `example.prompty` - Promptly example
- `PR_DRAFT_v1.0.1.md` - PR template

### Documentation
**All docs go in `docs/`:**
- Session logs → `docs/sessions/`
- Feature guides → `docs/guides/`
- Architecture docs → `docs/guides/`

### Tests
**All tests go in `tests/`:**
- Unit tests
- Integration tests
- Diagnostic scripts

### Config
**All config goes in `config/`:**
- Docker configs
- Service configs
- Environment configs

### Demos
**All examples go in `demos/`:**
- Quickstart examples
- Feature demonstrations
- Integration examples

---

## What Was Moved

### To docs/sessions/ (32 files)
- All `PHASE*.md` files
- All `SESSION*.md` files
- `INTEGRATION_SPRINT*.md` files
- `CLEANUP_INVENTORY.md`
- Various session documentation

### To docs/guides/ (8 files)
- `HOLOLOOM_CLAUDE_DESKTOP_ARCHITECTURE.md`
- `HYBRID_MEMORY_STATUS.md`
- `MEDIUM_TERM_FEATURES_COMPLETE.md`
- `MEMORY_BACKEND_SYSTEM.md`
- `MULTIMODAL_WEB_SCRAPING.md`
- `RECURSIVE_CRAWLING_MATRYOSHKA.md`
- `TODAY_PROGRESS_OCT24.md`
- `QUICK_START_HOLOLOOM_SKILLS.md`

### To config/ (3 files)
- `claude_desktop_config_corrected.json`
- `claude_desktop_config_updated.json`
- `docker-compose.yml`

### To tests/ (5 files)
- `test_hybrid_memory.py`
- `test_warp_drive_complete.py`
- `test_web_memory.py`
- `test_web_scrape_simple.py`
- `check_memory_status.py`

### Deleted (6 files)
- `nul` (Windows artifact)
- `claude_docs_shards_full.txt` (temporary)
- `transcript_4g251atrdX8.txt` (temporary)
- `hello-world-HEAD.zip` (artifact)
- `load_beekeeping_data.cypher` (moved to archive)
- `run_unified_demo.bat` (obsolete)

---

## Quick Navigation

**Getting Started:**
- Start here: [README.md](../README.md)
- Developer guide: [CLAUDE.md](../CLAUDE.md)
- Quick demo: `python HoloLoom/unified_api.py`

**Core Code:**
- Unified API: [HoloLoom/unified_api.py](../HoloLoom/unified_api.py)
- Weaving: [HoloLoom/weaving_orchestrator.py](../HoloLoom/weaving_orchestrator.py)
- Synthesis: [HoloLoom/synthesis_bridge.py](../HoloLoom/synthesis_bridge.py)

**Examples:**
- Quickstart: [demos/01_quickstart.py](../demos/01_quickstart.py)
- Web ingestion: [demos/02_web_to_memory.py](../demos/02_web_to_memory.py)
- Chat: [demos/03_conversational.py](../demos/03_conversational.py)

**Documentation:**
- Integration sprint: [sessions/INTEGRATION_SPRINT_COMPLETE.md](sessions/INTEGRATION_SPRINT_COMPLETE.md)
- Phase docs: [sessions/PHASE*_COMPLETE.md](sessions/)
- Guides: [guides/](guides/)

---

## Maintenance

**Keep root clean:**
- No scattered test files
- No temporary files
- No session logs
- Only essential project files

**Use proper directories:**
- New tests → `tests/`
- New docs → `docs/sessions/` or `docs/guides/`
- New configs → `config/`
- New examples → `demos/`

**Archive when needed:**
- Obsolete files → `archive/`
- Preserve context for history
- Document what was archived

---

## Before vs After

### Before Cleanup
```
mythRL/
├── 40+ scattered MD files (sessions, guides, status)
├── 10+ test_*.py files in root
├── Multiple config files in root
├── Temporary files (nul, transcripts, etc.)
├── HoloLoom/
├── demos/
└── ... (total chaos)
```

### After Cleanup
```
mythRL/
├── HoloLoom/           # Core system
├── demos/              # Examples
├── tests/              # All tests
├── config/             # All configs
├── docs/               # All documentation
│   ├── sessions/       # 32 session logs
│   └── guides/         # 8 feature guides
├── CLAUDE.md
└── README.md
```

**Result:** Clean, professional structure with clear organization!

---

**Organized:** 2025-10-25
**Files Moved:** 48
**Files Deleted:** 6
**Directories Created:** 4

The root is now CLEAN and ORGANIZED!
