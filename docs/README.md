# mythRL Documentation

This directory contains all documentation for the mythRL/HoloLoom project, organized by type for easy navigation.

## Directory Structure

### `/architecture/` (31 docs)
System architecture, design patterns, and technical specifications.

**Key Documents:**
- Architecture overviews and system design
- Roadmaps and implementation plans
- Framework separation and integration guides
- Semantic calculus and learning mathematics
- Performance optimization strategies

### `/guides/` (22 docs)
User guides, quickstarts, and how-to documentation.

**Key Documents:**
- `QUICKSTART.md` - Get started quickly
- `README.md` - Main project README
- `APP_DEVELOPMENT_GUIDE.md` - Building applications
- `SEMANTIC_NUDGING_QUICKSTART.md` - Using semantic nudging
- `WHEN_TO_USE_SEMANTIC_LEARNING.md` - Decision guide
- `SAFETY_CHECKLIST.md` - Safety guidelines
- `SECURITY_AUDIT.md` - Security considerations

### `/completion-logs/` (43 docs)
Development session summaries, completion reports, and progress tracking.

**Purpose:**
- Historical record of feature completions
- Session summaries and breakthrough documentation
- Phase completion reports
- Task completion tracking

### `/archive/` (9 docs)
Historical documentation and deprecated content.

**Contents:**
- Older implementation plans
- Superseded system designs
- Historical pitch decks and presentations

### `/sessions/`
Session-specific documentation (if present).

## Documentation Conventions

### File Naming
- **Architecture**: `FEATURE_ARCHITECTURE.md`, `SYSTEM_DESIGN.md`
- **Guides**: `FEATURE_GUIDE.md`, `HOW_TO_*.md`
- **Completion Logs**: `*_COMPLETE.md`, `PHASE_*.md`, `SESSION_*.md`

### Content Types
1. **Architecture Docs** - Technical deep-dives, system design, mathematics
2. **Guides** - Step-by-step instructions, quickstarts, tutorials
3. **Completion Logs** - "What we built and when" historical records
4. **Archive** - No longer current but kept for reference

## Finding Documentation

### I want to...

**Understand the system architecture**
→ Start with `/architecture/MYTHRL_ECOSYSTEM_ARCHITECTURE.md`

**Get started quickly**
→ See `/guides/QUICKSTART.md`

**Build an application**
→ Read `/guides/APP_DEVELOPMENT_GUIDE.md`

**Understand semantic learning**
→ Check `/architecture/SEMANTIC_LEARNING_MATHEMATICS.md`

**See what features exist**
→ Browse `/completion-logs/` for feature completion docs

**Understand a specific subsystem**
→ Search `/architecture/` for relevant docs

## Contributing to Documentation

When adding new documentation:

1. **Architecture changes** → `/architecture/`
2. **User guides** → `/guides/`
3. **Completion reports** → `/completion-logs/`
4. **Deprecated content** → `/archive/`

Keep the root directory clean - only `CLAUDE.md` and essential project files should remain in root.

## Documentation Quality

Good documentation should:
- Have clear headers and structure
- Include code examples where relevant
- Link to related documents
- Be dated for historical context
- Use consistent markdown formatting

## See Also

- Main project: `../CLAUDE.md` (AI assistant instructions)
- Code: `../HoloLoom/` (main package)
- Tests: `../tests/` (test suite)
- Demos: `../demos/` (example scripts)
- Experiments: `../experimental/` (research code)
