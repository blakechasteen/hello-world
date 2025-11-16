# Changelog

All notable changes to the Squad VS Code extension will be documented in this file.

## [1.0.0] - 2025-11-16

### Added
- Initial release of Squad extension
- **Code Intelligence Features**:
  - Explain Code: Detailed explanations with concepts and patterns
  - Find Similar Code: Semantic search across workspace
  - Generate Unit Tests: Automated test generation for pytest/jest/mocha
  - Add Documentation: Auto-generate docstrings and JSDoc
  - Suggest Refactorings: AI-powered code improvement suggestions
  - Review Changes: Git diff analysis with AI insights

- **Advanced Code Context**:
  - Tree-sitter AST parsing for Python, TypeScript, JavaScript
  - Intelligent context extraction (functions, classes, imports)
  - Minimal context sending (only relevant code)
  - Multi-file dependency analysis

- **Performance Optimizations**:
  - SQLite-backed embedding cache
  - LRU eviction policy
  - File watcher for auto-invalidation
  - Incremental updates (only re-embed changed code)
  - <1ms cache hit latency

- **HoloLoom Integration**:
  - Enhanced API client with type safety
  - Connection status monitoring
  - 4 reasoning modes: direct, verify, research, plan_execute
  - Workspace indexing and knowledge graph
  - Logic error detection with ML

- **Developer Experience**:
  - Context menu integration
  - Keyboard shortcuts (Ctrl+Alt+E/F/T)
  - Status bar indicator
  - Progress notifications
  - Markdown result panels
  - Comprehensive statistics dashboard

- **Configuration**:
  - Server URL configuration
  - Cache size and auto-indexing options
  - Reasoning mode selection
  - Context line limits

### Technical Details
- TypeScript 5.0+
- Tree-sitter for AST parsing
- Better-SQLite3 for caching
- Axios for HTTP client
- VS Code API 1.80+

### Documentation
- Comprehensive README with examples
- Architecture documentation
- Troubleshooting guide
- Development setup guide

---

## Future Releases

### [1.1.0] - Planned
- Inline code suggestions (like Copilot)
- Multi-cursor support
- Batch operations
- Custom prompt templates
- Code action provider integration

### [1.2.0] - Planned
- Language support expansion (Java, C++, Rust, Go)
- Visual Studio integration
- JetBrains IDE support
- Web-based dashboard
- Team collaboration features

### [2.0.0] - Planned
- Local model support (no backend required)
- Fine-tuned models for specific frameworks
- Real-time code analysis
- Project-specific learning
- Advanced refactoring tools
