# HoloLoom Squad - AI-Powered Code Assistant

**Three AI personas working together to make you a better developer.**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/hololoom/squad)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![VS Code](https://img.shields.io/badge/VS%20Code-1.75%2B-blue.svg)](https://code.visualstudio.com/)

HoloLoom Squad brings three specialized AI assistants into your VS Code workspace, each with unique superpowers to help you write better code, faster.

---

## 🎭 Meet Your AI Team

### ⚡ **Proto** - The Code Orchestrator
*Your productivity accelerator*

Proto helps you **write and improve code** with surgical precision:
- **Explain Code**: Get schema-based explanations with 95%+ structured compliance
- **Refactor**: Surgical edits that preserve 90% of your original code
- **Research**: Multi-stage reasoning for complex questions (3-5x deeper analysis)
- **Verify**: Fact-check claims with evidence-based verification
- **Workflows**: Build visual pipelines with drag-and-drop agents

**Powered by**: Promptly AI reliability layer (6 problem solvers)

---

### 🔍 **Trough** - The Code Reviewer
*Your quality guardian*

Trough **finds and fixes issues** with 87% auto-fix success rate:
- **Analyze**: Detect 24 types of issues (15 AI slop + 9 ML logic bugs)
- **Auto-Fix**: xTerminator fixes issues with AST-based transformations
- **Validate**: 5-stage validation pipeline ensures fixes are safe
- **Learn**: Thompson Sampling improves fix strategies over time

**Powered by**: Trough detection algorithms + xTerminator auto-fixer

---

### 💡 **EdWIN** - The Research Companion
*Your knowledge navigator*

EdWIN helps you **understand and explore** your codebase:
- **Scene Analysis**: Understand code complexity and structure
- **Guidance**: Context-aware help (like Clippy, but actually useful)
- **Knowledge Graph**: Navigate code relationships with GraphRAG
- **Voice Chat**: Talk to your code (voice-to-text integration)

**Powered by**: Elle AR guide + GraphRAG + MultimodalRAG

---

## ✨ Bonus Features

### 📚 **EduVerse** - Learn While You Code
- Interactive tutorials tailored to your skill level
- Code explanations with built-in quizzes
- Learning path suggestions based on your codebase

### 💬 **ChatOps** - Conversational Interface
- Natural language commands with slash shortcuts
- Conversation history with context awareness
- Direct persona invocation from chat

**Available Commands**:
- `/analyze` - Run Trough code analysis
- `/explain` - Get EdWIN explanation
- `/refactor` - Trigger Proto refactoring
- `/workflow` - Open workflow builder
- `/graph` - Explore knowledge graph
- `/help` - Show all commands

---

## 🚀 Quick Start

### Prerequisites

1. **Backend Server** (required)
   ```bash
   cd mythRL
   export PYTHONPATH=.
   python HoloLoom/server/unified_server.py
   # Server starts at http://localhost:8000
   ```

2. **VS Code** 1.75 or higher

### Installation

**Option 1: From Source** (development)
```bash
cd squad
npm install
npm run compile
code --extensionDevelopmentPath=$(pwd)
```

**Option 2: From VSIX** (coming soon)
```bash
code --install-extension hololoom-squad-1.0.0.vsix
```

**Option 3: From Marketplace** (coming soon)
Search for "HoloLoom Squad" in VS Code Extensions

### First Steps

1. **Start the backend server** (see Prerequisites above)

2. **Configure backend URL** (if not localhost:8000):
   ```json
   // .vscode/settings.json
   {
     "hololoom.backendUrl": "http://your-backend:8000"
   }
   ```

3. **Try Quick Actions**:
   - Press `Ctrl+Shift+H` (or `Cmd+Shift+H` on Mac)
   - Select "⚡ Proto: Explain Code"
   - Select some code and see the magic!

4. **Open ChatOps**:
   - Press `Ctrl+Shift+C`
   - Type `/help` to see available commands
   - Try `/analyze` to scan your current file

---

## 📖 Features in Detail

### Proto Features

#### Explain Code
Select code and get a structured explanation with:
- Complexity assessment (simple/moderate/complex)
- Key concepts identified
- Structure breakdown
- Confidence scoring

**Modes**:
- `brief` - Quick summary
- `comprehensive` - Full analysis (default)
- `tutorial` - Step-by-step learning

**Usage**:
```
1. Select code
2. Right-click → ⚡ Proto → Explain Code
   OR press Ctrl+Shift+E
3. View explanation in side panel
```

#### Surgical Refactoring
Proto makes **minimal, targeted changes** while preserving your code's logic:
- 90%+ preservation rate
- Clear change descriptions
- Preview before applying
- Undo-friendly

**Goals**:
- `readability` - Improve code clarity
- `performance` - Optimize for speed
- `maintainability` - Enhance long-term quality

**Usage**:
```
1. Select code to refactor
2. Right-click → ⚡ Proto → Refactor Code
3. Choose goal (readability/performance/maintainability)
4. Review changes → Apply or Preview
```

#### Multi-Stage Research
Complex questions deserve thorough answers:
- Stage 1: Understanding the question
- Stage 2: Research and exploration
- Stage 3: Synthesis and final answer

**Usage**:
```
1. Ctrl+Shift+P → "Proto: Research Question"
2. Enter your question
3. Watch Proto work through 3 research stages
4. Get comprehensive answer with confidence score
```

#### Workflow Builder
Visual pipelines for complex tasks:
- Drag-and-drop agent palette
- 6 agent types (Explain, Refactor, Research, Verify, Synthesize, Output)
- Save/load workflows
- Execute with one click

**Usage**:
```
1. Right-click → ⚡ Proto → Open Workflow Builder
2. Drag agents onto canvas
3. Connect them in sequence
4. Click Execute
```

---

### Trough Features

#### Code Analysis
Comprehensive issue detection with 24 algorithms:

**AI Slop Detection** (15 types):
- Hardcoded secrets/values
- Missing error handling
- Resource leaks
- Security vulnerabilities
- Performance issues
- Dead code
- Incomplete implementations

**ML Logic Detection** (9 types):
- Division by zero
- Null dereferences
- Logic contradictions
- Missing returns
- Array bounds errors

**Severity Levels**:
- 🔴 CRITICAL - Fix immediately
- 🟠 HIGH - Fix soon
- 🟡 MEDIUM - Should fix
- 🟢 LOW - Nice to fix

**Usage**:
```
1. Open file to analyze
2. Press Ctrl+Shift+A
   OR Right-click → 🔍 Trough → Analyze File
3. View issues in sidebar tree view
4. Click issue to jump to location
```

#### Auto-Fix with xTerminator
87% success rate for automatic fixes:
- AST-based transformations (safe)
- 5-stage validation pipeline
- Git safety checks
- Automatic rollback on failure

**Validation Stages**:
1. Syntax validation
2. Import resolution
3. Test execution
4. Git safety checks
5. Rollback on failure

**Usage**:
```
1. After analysis, click "Auto-Fix Issues"
2. Review proposed fixes
3. Choose Apply/Preview/Cancel
4. Watch xTerminator work!
```

#### QA Statistics
Track your code quality over time:
- Total files analyzed
- Issues found by type
- Fix success rate
- Top recurring issues
- Thompson Sampling learning stats

**Usage**:
```
Right-click → 🔍 Trough → Show QA Statistics
```

---

### EdWIN Features

#### Scene Analysis
Understand code context at a glance:
- Complexity assessment
- Entity extraction (functions, classes, variables)
- Relationship mapping
- Focus area suggestions

**Usage**:
```
Right-click → 💡 EdWIN → Analyze Scene
```

#### Context-Aware Guidance
Get help based on what you're trying to do:

**Intent Types**:
- `seeking_guidance` - General help
- `debugging` - Fix issues
- `learning` - Understand code
- `refactoring` - Improve code

**Usage**:
```
1. Press Ctrl+Shift+G
2. EdWIN analyzes current context
3. Get personalized guidance + suggested actions
4. See focus areas highlighted in editor
```

#### Knowledge Graph Explorer
Navigate code relationships:
- Multi-hop graph traversal
- Entity relationship visualization
- Find connections between components
- Understand data flow

**Usage**:
```
1. Right-click → 💡 EdWIN → Explore Knowledge Graph
2. Enter entity name (e.g., "MyClass")
3. See all related entities and relationships
4. Click to navigate
```

---

### EduVerse Features

#### Interactive Tutorials
Learn by doing:
- Skill-level adaptive content
- Hands-on exercises
- Progress tracking
- Certificate of completion

**Topics**:
- TypeScript Fundamentals
- VS Code Extension Development
- HoloLoom Architecture
- RAG Systems
- Thompson Sampling

**Usage**:
```
Ctrl+Shift+P → "EduVerse: Start Tutorial"
```

#### Code Quizzes
Test your understanding:
- Auto-generated from code
- Multiple choice + coding challenges
- Instant feedback
- Track learning progress

**Usage**:
```
1. Select code
2. Right-click → EduVerse → Interactive Quiz
3. Answer questions
4. Get immediate feedback
```

---

### ChatOps Features

#### Conversational Interface
Talk to HoloLoom naturally:
- Natural language understanding
- Context-aware responses
- Conversation history
- Slash command shortcuts

**Usage**:
```
1. Press Ctrl+Shift+C
2. Type message or /command
3. Get instant response
4. Continue conversation
```

**Slash Commands**:
| Command | Action |
|---------|--------|
| `/analyze` | Run Trough analysis |
| `/explain` | Get EdWIN explanation |
| `/refactor` | Trigger Proto refactoring |
| `/workflow` | Open workflow builder |
| `/graph` | Explore knowledge graph |
| `/help` | Show all commands |

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Command | Description |
|----------|---------|-------------|
| `Ctrl+Shift+H` | Quick Actions | Show all HoloLoom actions |
| `Ctrl+Shift+E` | Proto Explain | Explain selected code |
| `Ctrl+Shift+A` | Trough Analyze | Analyze current file |
| `Ctrl+Shift+G` | EdWIN Guidance | Get context-aware guidance |
| `Ctrl+Shift+C` | Open Chat | Open ChatOps interface |

*Replace `Ctrl` with `Cmd` on macOS*

**Customize**: File → Preferences → Keyboard Shortcuts → Search "hololoom"

---

## ⚙️ Configuration

### Extension Settings

Configure HoloLoom Squad in VS Code settings:

```json
{
  // Backend connection
  "hololoom.backendUrl": "http://localhost:8000",

  // Enable/disable personas
  "hololoom.proto.enabled": true,
  "hololoom.trough.enabled": true,
  "hololoom.edwin.enabled": true,

  // Proto settings
  "hololoom.proto.defaultMode": "verify",  // verify | research | direct

  // Trough settings
  "hololoom.trough.autoFix": false,        // Auto-fix on detection
  "hololoom.trough.gitSafe": true,         // Enable Git safety checks

  // EdWIN settings
  "hololoom.edwin.autoTrigger": false,     // Show guidance while typing

  // Feature toggles
  "hololoom.eduverse.enabled": true,
  "hololoom.chatops.enabled": true,
  "hololoom.voice.enabled": true
}
```

### Workspace Configuration

Create `.hololoom/config.json` for project-specific settings:

```json
{
  "proto": {
    "workflows": {
      "autoSave": true,
      "templates": ["research", "safety-gated", "test-gen"]
    }
  },
  "trough": {
    "excludePatterns": ["*.test.ts", "*.spec.ts"],
    "severityThreshold": "MEDIUM"
  },
  "edwin": {
    "graph": {
      "maxDepth": 2,
      "autoVisualize": false
    }
  }
}
```

---

## 🎨 UI Elements

### Activity Bar
Click HoloLoom icon in Activity Bar (left sidebar) to access:
- **Trough Issues** tree view
- Quick access to all personas

### Status Bar
Bottom-right indicators:
- ⚡ Proto: Ready/Working/Error
- 🔍 Trough: Issue count
- 💡 EdWIN: Ready/Analyzing

### Tree Views
- **Trough Issues** - Hierarchical issue list with severity icons
  - Click to jump to issue location
  - Right-click for fix options

### Context Menus
Right-click in editor for submenu access:
- ⚡ **Proto** (Code Orchestrator)
- 🔍 **Trough** (Code Reviewer)
- 💡 **EdWIN** (Research Companion)

---

## 📊 What's Happening Under the Hood

### Architecture Overview

```
┌─────────────────────────────────────────┐
│  VS Code Extension (TypeScript)         │
│  - ProtoManager (500 lines)             │
│  - TroughManager (400 lines)            │
│  - EdWINManager (300 lines)             │
│  - HoloLoomIntegration (450 lines)      │
└─────────────┬───────────────────────────┘
              │ HTTP/WebSocket
              ▼
┌─────────────────────────────────────────┐
│  FastAPI Backend (Python)               │
│  - /promptly/* (Proto - 6 endpoints)    │
│  - /trough/*   (Trough - 4 endpoints)   │
│  - /elle/*     (EdWIN - 3 endpoints)    │
│  - /graph/*    (GraphRAG - 6 endpoints) │
│  - /voice/*    (Voice - 6 endpoints)    │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  HoloLoom Core (Python)                 │
│  - Memory Systems (11 types)            │
│  - Policy Engine (Thompson Sampling)    │
│  - Learning Loops (7 parallel)          │
│  - Alignment Framework                  │
└─────────────────────────────────────────┘
```

### Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| Code Explanation | ~150ms | Proto DIRECT mode |
| Code Analysis | ~100ms | Per file |
| Auto-Fix | ~500ms | Including validation |
| Knowledge Graph Query | ~50ms | Cached |
| Multi-Stage Research | ~900ms | 3 stages |

---

## 🔧 Troubleshooting

### Backend Connection Issues

**Symptom**: "Backend not available" errors

**Solutions**:
1. Check backend is running:
   ```bash
   curl http://localhost:8000/health
   ```
2. Verify URL in settings matches backend
3. Check firewall/antivirus isn't blocking port 8000

### No Issues Detected

**Symptom**: Trough analysis finds 0 issues in problematic code

**Solutions**:
1. Ensure backend has Trough components installed
2. Check file language is supported (Python, TypeScript, JavaScript)
3. Try restarting backend server

### Commands Not Appearing

**Symptom**: HoloLoom commands missing from Command Palette

**Solutions**:
1. Reload VS Code window (Ctrl+Shift+P → "Reload Window")
2. Check extension is activated (View → Output → HoloLoom Squad)
3. Reinstall extension

### Webview Panels Not Opening

**Symptom**: Clicking commands does nothing

**Solutions**:
1. Check VS Code console for errors (Help → Toggle Developer Tools)
2. Disable conflicting extensions
3. Update VS Code to latest version

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

1. **Clone repository**:
   ```bash
   git clone https://github.com/hololoom/squad.git
   cd squad
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start backend** (in separate terminal):
   ```bash
   cd ../mythRL
   export PYTHONPATH=.
   python HoloLoom/server/unified_server.py
   ```

4. **Compile TypeScript**:
   ```bash
   npm run compile
   ```

5. **Run extension**:
   ```bash
   code --extensionDevelopmentPath=$(pwd)
   ```

6. **Make changes** and test
   - Edit files in `src/`
   - Reload extension (Ctrl+Shift+P → "Reload Window")

### Project Structure

```
squad/
├── src/
│   ├── extension.ts              # Entry point
│   ├── ProtoManager.ts           # Proto persona
│   ├── TroughManager.ts          # Trough persona
│   ├── EdWINManager.ts           # EdWIN persona
│   └── HoloLoomIntegration.ts    # Integration layer
├── package.json                  # Extension manifest
├── tsconfig.json                 # TypeScript config
└── README.md                     # This file
```

### Testing

```bash
npm test
```

### Pull Request Process

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **HoloLoom Team** - Core architecture and backend
- **Anthropic** - Claude AI model
- **VS Code Team** - Excellent extension API
- **Community Contributors** - Feature requests and bug reports

---

## 📞 Support

- **GitHub Issues**: https://github.com/hololoom/squad/issues
- **Documentation**: https://docs.hololoom.ai
- **Discord**: https://discord.gg/hololoom (coming soon)
- **Email**: support@hololoom.ai

---

## 🗺️ Roadmap

### Version 1.1 (Q2 2025)
- [ ] Full voice chat implementation
- [ ] Real-time collaboration
- [ ] Multi-file refactoring
- [ ] Advanced workflow templates

### Version 1.2 (Q3 2025)
- [ ] Custom learning paths
- [ ] Team dashboards
- [ ] Integration with GitHub Copilot
- [ ] Mobile companion app

### Version 2.0 (Q4 2025)
- [ ] Multi-language support (beyond Python/TS/JS)
- [ ] Cloud sync for workflows
- [ ] Enterprise SSO integration
- [ ] Advanced analytics

---

## ⭐ Star History

If you find HoloLoom Squad useful, please star the repository!

---

**Built with ❤️ by the HoloLoom Team**

*Making developers more productive, one AI persona at a time.*

---

## 📸 Screenshots

> **Note**: Screenshots coming soon! Extension just shipped.

### Proto in Action
*(Screenshot: Code explanation panel)*

### Trough Issue Detection
*(Screenshot: Issues tree view with severity indicators)*

### EdWIN Knowledge Graph
*(Screenshot: Interactive graph visualization)*

### ChatOps Interface
*(Screenshot: Chat panel with slash commands)*

---

**Ready to supercharge your coding workflow? Install HoloLoom Squad today!** 🚀
