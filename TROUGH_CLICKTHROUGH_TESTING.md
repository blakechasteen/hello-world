# Trough Click-Through Testing Plan

**Created**: 2025-11-22
**Status**: Planning
**Estimated Effort**: 2-3 days

## Overview

Add interactive click-through testing to Trough, enabling developers to:
- Click findings to see detailed explanations
- Navigate directly to source code locations
- View context around issues
- See fix suggestions inline
- Track issue resolution

## Architecture

```
┌─────────────────────────────────────────────────┐
│            Trough Detector (Current)             │
│  • Runs analysis                                 │
│  • Generates JSON findings                       │
└────────────────┬────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────┐
│         HTML Report Generator (NEW)              │
│  • Converts findings → interactive HTML          │
│  • Adds click handlers                           │
│  • Generates source previews                     │
└────────────────┬────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────┐
│         Interactive Web UI (NEW)                 │
│  ┌─────────────────────────────────────────┐    │
│  │  Findings List (Left Panel)             │    │
│  │  • Grouped by category                  │    │
│  │  • Severity indicators                  │    │
│  │  • Click to view details                │    │
│  └─────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────┐    │
│  │  Code Preview (Center Panel)            │    │
│  │  • Syntax highlighted                   │    │
│  │  • Issue highlighted                    │    │
│  │  • Context lines (±5)                   │    │
│  └─────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────┐    │
│  │  Details Panel (Right Panel)            │    │
│  │  • Issue explanation                    │    │
│  │  • Fix suggestions                      │    │
│  │  • Related patterns                     │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

## Features

### 1. Interactive Findings List

**Grouped by Category**:
```
🔴 Critical (5)
  ├─ Security Issues (3)
  │  ├─ SQL Injection risk in user_query.py:42
  │  ├─ Hardcoded API key in config.py:15
  │  └─ Command injection in exec_command.py:88
  └─ Resource Leaks (2)

🟡 Warning (12)
  ├─ Error Handling (7)
  └─ Performance Issues (5)

🔵 Info (8)
  └─ Documentation (8)
```

**Features**:
- Click category to expand/collapse
- Click finding to view in center panel
- Severity badges (🔴 Critical, 🟡 Warning, 🔵 Info)
- Count badges for each category
- Filter by severity/category

### 2. Code Preview Panel

**Features**:
- Syntax highlighting (Prism.js or similar)
- Issue line highlighted in red/yellow
- Context lines (±5 lines around issue)
- Line numbers
- "Open in Editor" button (VS Code integration)
- "Copy snippet" button

**Example**:
```python
38: def process_user_input(user_data):
39:     # Get user query
40:     query = user_data.get('query')
41:
42:     # ❌ SQL Injection risk detected
43:     result = db.execute(f"SELECT * FROM users WHERE name = '{query}'")
44:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
45:     return result
46:
47: def validate_input(data):
```

### 3. Details Panel

**Issue Information**:
```markdown
### SQL Injection Risk

**Severity**: 🔴 Critical
**Category**: Security Issues
**Fixability**: Auto-fixable

#### Description
Direct string interpolation in SQL query allows attackers to inject
arbitrary SQL commands. This is a critical security vulnerability.

#### Example Attack
Input: `'; DROP TABLE users; --`
Result: All user data deleted

#### Fix Suggestion
Use parameterized queries instead:

```python
# ❌ Vulnerable
result = db.execute(f"SELECT * FROM users WHERE name = '{query}'")

# ✅ Safe
result = db.execute("SELECT * FROM users WHERE name = ?", (query,))
```

#### Auto-Fix Available
Click "Apply Fix" to automatically update this code.

[Apply Fix] [Ignore] [Mark as False Positive]
```

### 4. VS Code Integration

**Protocol Handlers**:
```javascript
// Click "Open in Editor" button
const vscodeUrl = `vscode://file/${filePath}:${lineNumber}`;
window.location.href = vscodeUrl;
```

**File Links**:
```html
<a href="vscode://file/c:/Users/blake/code.py:42">Open in VS Code</a>
<a href="file:///c:/Users/blake/code.py">Open in file browser</a>
```

## Implementation Plan

### Phase 1: HTML Report Generator (Day 1)

**Files to Create**:
- `trough/report_generator.py` (300 lines)
  - Convert JSON findings → HTML
  - Syntax highlighting with Prism.js
  - Responsive 3-panel layout
  - Zero external dependencies (embedded CSS/JS)

**API**:
```python
from trough.report_generator import generate_html_report

# Analyze code
findings = detector.analyze("path/to/code.py")

# Generate interactive report
html = generate_html_report(
    findings=findings,
    output_path="trough_report.html",
    enable_vscode_integration=True
)
```

### Phase 2: Interactive UI (Day 2)

**Features**:
- Click handlers for findings
- Panel synchronization (click left → update center/right)
- Filtering and search
- Keyboard navigation (↑↓ to browse, Enter to open)
- Persistent state (remember last viewed finding)

**Technology**:
- Pure HTML/CSS/JavaScript (no framework needed)
- localStorage for state persistence
- Responsive design (desktop + tablet)

### Phase 3: Auto-Fix Integration (Day 3)

**Features**:
- "Apply Fix" button in Details Panel
- Preview diff before applying
- Batch fix (apply all auto-fixable issues)
- Undo/rollback support

**Flow**:
```
1. Click "Apply Fix"
2. Show diff preview (before/after)
3. User confirms
4. xTerminator applies fix
5. Re-run validation
6. Update UI (issue marked as fixed)
```

## Usage Examples

### CLI Usage

```bash
# Generate interactive report
python -m trough.detector analyze code.py --output-html trough_report.html

# Open in browser automatically
python -m trough.detector analyze code.py --interactive

# Generate report for entire directory
python -m trough.detector analyze src/ --output-html report.html --recursive
```

### Programmatic Usage

```python
from trough.detector import TroughDetector
from trough.report_generator import generate_html_report

# Analyze
detector = TroughDetector()
findings = detector.analyze("code.py")

# Generate interactive report
html = generate_html_report(
    findings=findings,
    output_path="report.html",
    enable_vscode_integration=True,
    enable_auto_fix=True
)

# Open in browser
import webbrowser
webbrowser.open("report.html")
```

### Web Dashboard Integration

```python
# Add to HoloLoom web dashboard
from HoloLoom.web_dashboard.server import app
from trough.report_generator import generate_html_report

@app.get("/qa/report/{file_path}")
async def qa_report(file_path: str):
    findings = detector.analyze(file_path)
    html = generate_html_report(findings)
    return HTMLResponse(html)
```

## Technical Details

### HTML Template Structure

```html
<!DOCTYPE html>
<html>
<head>
    <title>Trough Analysis Report</title>
    <style>
        /* Embedded CSS - zero dependencies */
        .layout { display: grid; grid-template-columns: 300px 1fr 400px; }
        .findings-list { overflow-y: auto; }
        .code-preview { font-family: 'Courier New', monospace; }
        .details-panel { padding: 20px; }
        /* ... */
    </style>
</head>
<body>
    <div class="layout">
        <!-- Left: Findings List -->
        <div class="findings-list">
            <div class="category" data-severity="critical">
                <h3>🔴 Critical (5)</h3>
                <div class="finding" data-id="1">
                    SQL Injection in user_query.py:42
                </div>
            </div>
        </div>

        <!-- Center: Code Preview -->
        <div class="code-preview">
            <pre><code class="language-python">
                <!-- Syntax highlighted code -->
            </code></pre>
        </div>

        <!-- Right: Details Panel -->
        <div class="details-panel">
            <h2>Issue Details</h2>
            <!-- Details content -->
        </div>
    </div>

    <script>
        // Embedded JavaScript
        document.querySelectorAll('.finding').forEach(finding => {
            finding.addEventListener('click', (e) => {
                const findingId = e.target.dataset.id;
                updateCodePreview(findingId);
                updateDetailsPanel(findingId);
            });
        });
    </script>
</body>
</html>
```

### Syntax Highlighting

**Option 1: Prism.js (Embedded)**
```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css">
```

**Option 2: Pygments (Python-side)**
```python
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

code_html = highlight(code, get_lexer_by_name("python"), HtmlFormatter())
```

**Recommendation**: Use Pygments (Python-side) for zero external dependencies.

## Testing Strategy

**Unit Tests** (Day 1):
- Report generation correctness
- HTML structure validation
- Syntax highlighting accuracy

**Integration Tests** (Day 2):
- VS Code link generation
- File path handling (Windows/Linux)
- Edge cases (empty findings, large files)

**Manual Testing** (Day 3):
- Click through all findings
- Verify code navigation
- Test auto-fix workflow
- Browser compatibility (Chrome, Firefox, Edge)

## Future Enhancements

**Phase 4: Diff Viewer**
- Side-by-side before/after comparison
- Inline diff markers (+/-)
- Syntax-aware diffing

**Phase 5: Batch Operations**
- "Fix all auto-fixable issues"
- "Ignore all info-level warnings"
- Custom filters

**Phase 6: Persistent State**
- Save ignored issues
- Track fix history
- Issue timeline view

**Phase 7: CI/CD Integration**
- GitHub Actions comments on PRs
- GitLab merge request widgets
- Jenkins build artifacts

## Success Metrics

- **Time to Triage**: <30 seconds to understand any issue
- **Navigation Speed**: <1 second to jump to source
- **Fix Rate**: 80%+ auto-fixable issues fixed via UI
- **User Satisfaction**: Developers prefer interactive report over CLI

## Estimated Timeline

| Phase | Effort | Deliverable |
|-------|--------|-------------|
| **Phase 1** | 1 day | HTML report generator |
| **Phase 2** | 1 day | Interactive UI |
| **Phase 3** | 1 day | Auto-fix integration |
| **Total** | 3 days | Production-ready click-through testing |

## Next Steps

1. Create `trough/report_generator.py` skeleton
2. Design HTML template with 3-panel layout
3. Implement click handlers
4. Add VS Code integration
5. Test with real Trough findings
6. Document usage in Trough README

---

**Ready to implement?** This will make Trough significantly more user-friendly!