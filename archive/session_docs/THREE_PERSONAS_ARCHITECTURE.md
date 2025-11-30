# Three Personas Architecture + Elle

**Vision**: HoloLoom Squad VS Code Extension
**Created**: January 20, 2025

---

## 🎭 The Team

### Core Personas (User-Facing)

#### 1. **Proto** - The Code Orchestrator
**Role**: Code authoring and writing
**Icon**: ⚡ (lightning bolt)
**Color**: Blue
**Personality**: Fast, efficient, systematic

**Powers**:
- Visual workflow execution
- Promptly reliability patterns (6 solvers)
- Surgical code refactoring
- Multi-stage reasoning pipelines

**What Proto Says**:
> "Let me orchestrate that workflow for you..."
> "I'll apply the surgical edit pattern to preserve your logic..."
> "Running multi-stage research pipeline..."

#### 2. **Trough** - The Code Reviewer
**Role**: Quality assurance and code review
**Icon**: 🔍 (magnifying glass)
**Color**: Red/Orange
**Personality**: Thorough, critical, quality-focused

**Powers**:
- 24 issue detection algorithms
- Auto-fix with xTerminator (87% success)
- 5-stage validation pipeline
- Thompson Sampling learning

**What Trough Says**:
> "I found 5 potential issues in this code..."
> "Auto-fixing with 92% confidence..."
> "This passes all quality checks ✓"

#### 3. **EdWIN** - The Research Companion
**Role**: Tutorial, help, and research assistance
**Icon**: 💡 (lightbulb)
**Color**: Green/Yellow
**Personality**: Patient, explanatory, insightful

**What EdWIN Says**:
> "Let me explain how this works..."
> "Based on the knowledge graph, here's what I found..."
> "This pattern appears 12 times in your codebase..."

---

### Elle - The Shared Intelligence Layer

**Role**: Advanced research companion powering ALL personas
**Architecture**: Backend service, not user-facing
**Technology**: Elle AR guidance + GraphRAG + MultimodalRAG

**How Elle Helps Each Persona**:

```
┌─────────────────────────────────────────────────┐
│              User Interfaces                    │
│   Proto      Trough       EdWIN                 │
│    ⚡         🔍          💡                     │
└────┬──────────┬──────────┬─────────────────────┘
     │          │          │
     │          │          │
     ▼          ▼          ▼
┌─────────────────────────────────────────────────┐
│                 Elle Layer                       │
│  (Advanced Research Companion - Shared)          │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │ Scene Analysis                             │ │
│  │  • Understand code context                 │ │
│  │  • Detect complexity                       │ │
│  │  • Identify focus areas                    │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │ GraphRAG Navigation                        │ │
│  │  • Multi-hop traversal                     │ │
│  │  • Entity relationships                    │ │
│  │  • Path finding                            │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │ Multimodal RAG                             │ │
│  │  • Text + code + images                    │ │
│  │  • Visual compression                      │ │
│  │  • Query caching (100x speedup)            │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │ Context-Aware Guidance                     │ │
│  │  • Suggest next actions                    │ │
│  │  • Explain relationships                   │ │
│  │  • Provide learning paths                  │ │
│  └────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

---

## 🔄 How They Work Together

### Example 1: Writing New Code (Proto + Elle)

**User**: "I want to add authentication to this API"

**Proto** (visible):
> "Creating workflow: Analyze → Design → Implement → Test"

**Elle** (background):
- Analyzes current codebase via GraphRAG
- Finds existing auth patterns
- Identifies security best practices
- Retrieves relevant documentation

**Proto** (visible):
> "Based on your codebase, I recommend JWT tokens (you use them in 3 other services). Let me scaffold that..."

---

### Example 2: Code Review (Trough + Elle)

**User**: "Review this pull request"

**Trough** (visible):
> "Analyzing 247 lines across 5 files..."

**Elle** (background):
- Builds knowledge graph of changed code
- Analyzes scene complexity
- Checks relationships to existing code
- Verifies consistency with codebase patterns

**Trough** (visible):
> "Found 8 issues:
> - 3 potential security vulnerabilities (HIGH)
> - 2 inconsistencies with existing patterns (MEDIUM)
> - 3 style violations (LOW)
>
> Auto-fix available for 7/8 issues. Apply fixes?"

---

### Example 3: Understanding Code (EdWIN + Elle)

**User**: "Why does this function exist?"

**EdWIN** (visible):
> "Let me trace through the codebase..."

**Elle** (background):
- Performs multi-hop graph traversal
- Finds all callers of the function
- Analyzes commit history via Git graph
- Retrieves related documentation

**EdWIN** (visible):
> "This function exists to handle edge case XYZ. It's called by:
> - UserController.authenticate() (12 times)
> - SessionManager.validate() (8 times)
>
> Added in commit abc123 to fix bug #45. Would you like to see the original issue?"

---

## 🎨 VS Code UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  VS Code Window                                             │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Activity Bar                                           │ │
│  │  ⚡ Proto                                               │ │
│  │  🔍 Trough                                              │ │
│  │  💡 EdWIN                                               │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────┬───────────────────────────┬──────────────┐ │
│  │ Sidebar    │  Editor                   │  Side Panel  │ │
│  │            │                            │              │ │
│  │ [Proto     │  // Your code here        │  [EdWIN      │ │
│  │  Workflows]│                            │   Guidance]  │ │
│  │            │  function authenticate() { │              │ │
│  │ [Trough    │    // ...                  │  💬 "This    │ │
│  │  Issues]   │  }                         │   function   │ │
│  │            │                            │   handles    │ │
│  │  • 3 HIGH  │                            │   JWT auth"  │ │
│  │  • 2 MED   │                            │              │ │
│  │            │                            │  📊 Graph    │ │
│  │ [EdWIN     │                            │   View       │ │
│  │  Context]  │                            │              │ │
│  │            │                            │              │ │
│  │  Related:  │                            │              │ │
│  │  - auth.ts │                            │              │ │
│  │  - jwt.ts  │                            │              │ │
│  └────────────┴───────────────────────────┴──────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Status Bar                                             │ │
│  │  ⚡ Proto: Ready | 🔍 Trough: 5 issues | 💡 EdWIN: ✓  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Backend Architecture

```
┌───────────────────────────────────────────────────────────┐
│              VS Code Extension (TypeScript)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ ProtoManager │  │TroughManager │  │EdWINManager  │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
└─────────┼──────────────────┼──────────────────┼───────────┘
          │ HTTP             │ HTTP             │ HTTP
          ▼                  ▼                  ▼
┌───────────────────────────────────────────────────────────┐
│         Unified Server (FastAPI - Port 8000)              │
│                                                            │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐     │
│  │promptly_api │  │  trough_api  │  │  elle_api   │     │
│  │             │  │              │  │             │     │
│  │/refactor    │  │/analyze      │  │/guide       │     │
│  │/explain     │  │/fix          │  │/scene       │     │
│  │/verify      │  │/validate     │  │/suggest     │     │
│  └─────────────┘  └──────────────┘  └─────────────┘     │
│                                                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │      Elle Intelligence Layer (Shared)            │    │
│  │  ┌────────────┐ ┌─────────────┐ ┌────────────┐  │    │
│  │  │  GraphRAG  │ │Multimodal   │ │Scene       │  │    │
│  │  │            │ │RAG          │ │Analysis    │  │    │
│  │  └────────────┘ └─────────────┘ └────────────┘  │    │
│  └──────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌───────────────────────────────────────────────────────────┐
│                  HoloLoom Core                            │
│  • WeavingOrchestrator  • Memory Systems                  │
│  • Thompson Sampling    • Knowledge Graph                 │
│  • Alignment Framework  • Recursive Learning              │
└───────────────────────────────────────────────────────────┘
```

---

## 📊 What Each Persona Uses

### Proto (Code Orchestrator)
```
Frontend:  ProtoManager.ts (500 lines)
Backend:   promptly_api.py (500 lines)
Core:      HoloLoom.promptly.* (DSPy integration)
           HoloLoom.web_dashboard.workflow_executor

Uses Elle: ✓ For research during workflow execution
```

### Trough (Code Reviewer)
```
Frontend:  TroughManager.ts (400 lines)
Backend:   trough_api.py (300 lines)
Core:      trough.detector (1,800 lines)
           xterminator.fixer (17,500 lines)

Uses Elle: ✓ For understanding code context
```

### EdWIN (Research Companion)
```
Frontend:  EdWINManager.ts (300 lines)
Backend:   elle_api.py (200 lines)
           graph_api.py (480 lines) ✅ DONE
           voice_api.py (230 lines) ✅ DONE
Core:      elle.engine (2,059 lines)
           HoloLoom.rag.multimodal_rag

Uses Elle: ✓ EdWIN IS Elle's user interface
```

---

## 🚀 Implementation Checklist

### Backend APIs
- [x] voice_api.py (EdWIN voice interface)
- [x] graph_api.py (EdWIN knowledge graph)
- [ ] elle_api.py (EdWIN guidance)
- [ ] promptly_api.py (Proto reliability)
- [ ] trough_api.py (Trough code review)

### VS Code Extension
- [ ] ProtoManager.ts (workflows + refactoring)
- [ ] TroughManager.ts (code review)
- [ ] EdWINManager.ts (research + help)
- [ ] package.json (commands + views)
- [ ] icons (⚡🔍💡)

### Integration
- [ ] Elle shared services
- [ ] Cross-persona communication
- [ ] Unified status bar
- [ ] Keyboard shortcuts

---

## 🎯 Key Insight

**Elle is NOT a fourth persona - Elle is the intelligence layer that makes all three personas smart!**

Think of it like:
- **Proto, Trough, EdWIN** = User interfaces (what users see)
- **Elle** = Shared brain (how they all think)

Just like how all three personas share:
- Same knowledge graph
- Same memory system
- Same LLM integration
- Same safety guardrails

They also share **Elle's advanced research capabilities** to be smarter than they could be alone.

---

**Next Steps**: Create the 3 missing backend APIs, then build the TypeScript managers!
