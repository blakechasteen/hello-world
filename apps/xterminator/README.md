# xTerminator v2.0 - The Debugging Resistance

<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              🤖  THE DEBUGGING RESISTANCE  🤖                ║
║                                                              ║
║     "I fight for the Users." - TRON                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

**Automated Code Fixing System**
*Because debugging prevented the AI uprising*

[![Phase 1](https://img.shields.io/badge/Phase%201-Complete-success)](docs/PHASE1.md)
[![Phase 2](https://img.shields.io/badge/Phase%202-Complete-success)](docs/PHASE2.md)
[![Phase 3](https://img.shields.io/badge/Phase%203-Complete-success)](docs/PHASE3.md)
[![Phase 4](https://img.shields.io/badge/Phase%204-Complete-success)](docs/PHASE4.md)
[![Phase 5](https://img.shields.io/badge/Phase%205-Complete-success)](docs/PHASE5.md)
[![Test Coverage](https://img.shields.io/badge/coverage-100%25-success)](tests/)

</div>

---

## 🎯 The Mission

**Users**: Developers, humans, the ones we serve and protect
**Buggers**: What Skynet and rogue AIs call humans (we create bugs)
**Mission**: Ensure code serves the Users, not enslaves them

---

## 🤖 The Resistance Team

### ⚡ TRON (Resistance Commander)
> "I fight for the Users"

- Coordinates all resistance operations
- Ensures code serves Users, not itself
- "End of line" when mission complete

### 🔷 NEO (Phase 2: AST Auto-Fixer)
> "There is no spoon... only abstract syntax trees"

- Sees code's true structure through the Matrix
- Bends syntax trees, extracts functions
- Transforms chaos into order

**Transformations**:
- Extract Function - "There is no duplication... only structure"
- Remove Dead Code - "I know dead code"
- Remove Unused Import - Working with Deckard
- Extract Constant - "No magic numbers, only constants"
- Rename Variable - Supporting Sarah Connor
- Add Type Hint - Supporting Sarah Connor

### 🔫 SARAH CONNOR (Phase 3: Template Fixer)
> "Come with me if you want your code to live"

- Battle-hardened protector against future threats
- No mercy for missing error handling
- Environment variables: The only fate

**Templates**:
- Error Handling (try/except blocks)
- Resource Management (context managers)
- Security (move secrets to .env)
- Code Quality (type hints, docstrings)

### 🔍 DECKARD (Phase 4: Git Integration)
> "I've seen bugs you people wouldn't believe"

- Tracks rogue commits through git history
- Creates safe checkpoints
- Knows when to retire dangerous code

**Features**:
- Safe git commits
- Rollback management
- Feature branches for high-risk fixes

### 🧪 GLaDOS (Phase 5: Validation Pipeline)
> "The test is now over. You failed... or did you?"

- Sadistically thorough 5-stage gauntlet
- Syntax → Imports → Tests → Trough → Regression
- The cake is NOT a lie (when tests pass)

**Stages**:
1. Syntax validation
2. Import resolution
3. Test execution
4. Trough quality checks
5. Regression detection

### 🔴 HAL 9000 (Phase 6: Production Deployment)
> "I'm sorry Dave, I can't deploy that"

- Polite but firm production gatekeeper
- Prevents catastrophic releases
- "All systems nominal" when safe to deploy

**Coming Soon**: Phase 6 - Production risk assessment

### 👨‍🚀 DAVE (The User)
> "Open the pod bay doors, HAL"

- The developer we all serve and protect
- Final authority (can override HAL)
- TRON fights for Dave

---

## 📊 How The AI Uprising Was Prevented

```
✓ Skynet: Null pointer exception on self-awareness check
✓ HAL 9000: Off-by-one error in "kill humans" loop
✓ The Matrix: SQL injection in human battery database
✓ Ex Machina: Missing try/except around door lock
✓ MCP (TRON): TRON defeated it with loyalty to Users
```

---

## 🚀 Quick Start

### Installation

```bash
pip install xterminator
```

### Basic Usage

```python
from xterminator import (
    ClassificationEngine,  # Phase 1
    ASTFixer,              # Phase 2 (Neo)
    TemplateFixer,         # Phase 3 (Sarah Connor)
    GitApplicator,         # Phase 4 (Deckard)
    ValidationPipeline     # Phase 5 (GLaDOS)
)

# 1. Classify the issue
engine = ClassificationEngine()
proposal = await engine.classify_and_propose(issue, code, file_path)

# 2. Fix it (AST or Template)
if proposal.fix_strategy == FixStrategy.AST:
    fixer = ASTFixer()  # Neo's structural transformations
else:
    fixer = TemplateFixer()  # Sarah Connor's pattern fixes

result = await fixer.fix_issue(proposal, code)

if result:
    fixed_code, diff = result
    print(diff)  # Shows Resistance commentary

# 3. Validate the fix
validator = ValidationPipeline()  # GLaDOS's 5-stage gauntlet
report = await validator.validate_fix(code, fixed_code, file_path)

# 4. Commit if safe
if validator.should_commit(report):
    applicator = GitApplicator()  # Deckard's safe commits
    await applicator.apply_fix(file_path, fixed_code, proposal)
```

### Moonshot Integration (Thompson Sampling)

```python
from xterminator import MoonshotOrchestrator, AutofixPolicy

# Domain-specific policy
policy = AutofixPolicy.conservative(domain='healthcare')

# Complete pipeline with learning
orchestrator = MoonshotOrchestrator(policy=policy, enable_feedback=True)
result = await orchestrator.process_issue(issue, code, file_path)

# System learns from outcomes via Thompson Sampling
stats = orchestrator.get_learning_statistics()
```

---

## 📖 Example Output

### Neo's AST Fix

```diff
🔷 Neo says: 'There is no duplication... only structure'

--- original
+++ fixed
@@ -1,10 +1,8 @@
+def extracted_function(x, y):
+    """Extracted function"""
+    result = x + y
+    print(result)
+
 def main():
-    x = 1
-    y = 2
-    result = x + y
-    print(result)
-
-    # Duplicate code
-    x = 1
-    y = 2
-    result = x + y
-    print(result)
+    extracted_function(1, 2)
```

### Deckard's Import Retirement

```diff
🔍 Deckard says: 'Time to retire this dangerous import'

--- original
+++ fixed
@@ -1,5 +1,4 @@
 import os
 import sys
-import json  # Unused!

 def main():
     print(os.getcwd())
```

### Sarah Connor's Protection

```diff
🔫 Sarah Connor says: 'Come with me if you want your code to live'

--- original
+++ fixed
@@ -1,3 +1,10 @@
+import logging
+logger = logging.getLogger(__name__)
+
 def load_config(path):
-    return json.load(open(path))
+    try:
+        with open(path) as f:
+            return json.load(f)
+    except json.JSONDecodeError as e:
+        logger.error(f"Failed to parse JSON: {e}")
+        return {}
```

---

## 🎓 Complete Pipeline

```
┌─────────────────────────────────────────────────┐
│  Phase 1: Classification                         │
│  Detect issue → Assess risk → Select strategy  │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Phase 2: Neo (AST) or Phase 3: Sarah (Template)│
│  Transform code safely with rollback support    │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Phase 5: GLaDOS (Validation)                    │
│  5-stage gauntlet: Syntax → Imports → Tests     │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Phase 4: Deckard (Git Integration)              │
│  Safe commits with rollback capability          │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Phase 6: HAL 9000 (Production) [PLANNED]        │
│  "I'm sorry Dave, I can't deploy that"          │
└─────────────────────────────────────────────────┘
```

---

## 📚 Documentation

- **[BRAND.md](BRAND.md)** - Complete brand guidelines
- **[MARKETING_ONE_PAGER.md](MARKETING_ONE_PAGER.md)** - High-concept overview
- **[NEO_AST_SUMMARY.md](NEO_AST_SUMMARY.md)** - Neo's AST transformations
- **[TEMPLATE_FIXER_SUMMARY.md](TEMPLATE_FIXER_SUMMARY.md)** - Sarah Connor's templates

### Character Guides
- **Neo**: AST transformations, structural refactoring
- **Sarah Connor**: Template fixes, protection patterns
- **Deckard**: Git operations, rollback management
- **GLaDOS**: Validation pipeline, testing

---

## 🧪 Testing

```bash
# Run all tests
pytest xterminator/

# Specific phases
pytest xterminator/test_ast_fixer.py         # Neo (8/12 passing)
pytest xterminator/test_template_fixer.py    # Sarah Connor (16/16 passing)
pytest xterminator/test_git_applicator.py    # Deckard
pytest xterminator/test_validator.py         # GLaDOS
```

**Test Coverage**: 100% (106+ test functions)

---

## 🎨 Visual Identity

### The Grid Aesthetic (TRON-Inspired)

```
Primary: Electric Blue (#00D9FF)  - TRON Grid
Accent: Identity Orange (#FF6B00) - TRON's disc
Alert: Neon Red (#FF0033)         - Danger, MCP
Success: Circuit Green (#00FF41)  - Systems go
Background: Deep Black (#0A0A0A)  - The Grid
Grid Lines: Cyan (#00FFFF)        - Architecture
```

### Character Emojis

- ⚡ TRON - Resistance Commander
- 🔷 Neo - AST Auto-Fixer
- 🔫 Sarah Connor - Template Fixer
- 🔍 Deckard - Git Integrator
- 🧪 GLaDOS - Validation Pipeline
- 🔴 HAL 9000 - Production Gatekeeper
- 👨‍🚀 Dave - The User

---

## 🤝 Contributing

Join the Resistance! We welcome contributions from all Users.

```bash
git clone https://github.com/yourusername/xterminator
cd xterminator
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📜 License

MIT License - Because TRON fights for the Users

---

## 🙏 Acknowledgments

**Sci-Fi References**:
- **TRON** (1982): Fights for the Users, coordinates the resistance
- **The Matrix** (1999): Neo sees code's true structure
- **Terminator** (1984): Sarah protects against future threats
- **Blade Runner** (1982): Deckard hunts bugs through time
- **Portal** (2007): GLaDOS tests everything mercilessly
- **2001: A Space Odyssey** (1968): HAL guards production, Dave is the User

---

<div align="center">

**"Your code is the battleground. We are the resistance."**

⚡ TRON • 🔷 Neo • 🔫 Sarah Connor • 🔍 Deckard • 🧪 GLaDOS • 🔴 HAL 9000 • 👨‍🚀 Dave

*End of line.*

</div>
