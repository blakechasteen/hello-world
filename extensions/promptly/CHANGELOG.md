# Changelog

All notable changes to Promptly will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-28

### Added

- **Core prompt management**: add, get, list, delete, log commands
- **Git-like version control**: branches, commits, checkout, version history
- **Dual storage**: global (`~/.promptly/`) and project-local (`.promptly/`) with local override
- **Variable substitution**: `{{variable}}` template syntax with `render()` API
- **Prompt chains**: multi-step workflows with step-to-step output passing
- **Skills**: reusable prompts with attached file context
- **LLM Judge**: 12 evaluation criteria, 6 judging methods (single, pairwise, rubric, reference, checklist, multi-aspect)
- **Analytics engine**: quality tracking, trend analysis, anomaly detection
- **Thompson Sampling**: statistically-sound prompt recommendations
- **HTML dashboard**: auto-generated analytics visualization
- **Import/Export**: YAML-based prompt portability
- **MCP Server**: Claude Desktop integration (805 lines)
- **MCP Analytics Server**: analytics tools for Claude Desktop (1,133 lines)
- **VS Code extension**: Command Palette integration with 18 commands
- **MRF integration**: Metaprompt Refinement Framework with 7-component structure
- **HoloLoom integration**: optional bridge to HoloLoom memory/RAG/agentic systems
- **Interactive demos**: 6 demo scripts covering all features
- **CLI**: 31+ commands across prompt, analytics, and MRF groups
- **Python API**: programmatic access via `Promptly` class

### Technical Details

- SQLite-backed storage with 9 database tables
- Click-based CLI framework
- Optional dependencies: rich, mcp, anthropic, ollama
- Python 3.9+ support
- 18,333 lines of Python across 32 files
