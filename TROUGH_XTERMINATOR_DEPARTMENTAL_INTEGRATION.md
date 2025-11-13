# Trough + xTerminator: Quality Assurance Department Integration

**Date**: November 10, 2025
**Status**: Design Complete
**Architecture**: MCP-Based Departmental Integration

## Executive Summary

Trough (AI code quality detector) and xTerminator (automated code fixer) integrate into HoloLoom's departmental agent architecture as a unified **Quality Assurance Department**. Using Model Context Protocol (MCP), they provide detection and fixing tools to other departments while maintaining strict safety boundaries and session-aware context management.

## Department Architecture

### Quality Assurance Department

**Mission**: Detect code quality issues and safely fix them with human oversight

**Components**:
1. **Trough MCP Server** - Detection tools (15 AI slop categories + 9 ML logic algorithms)
2. **xTerminator MCP Server** - Fixing tools (classification → fix → validate → apply)
3. **Session Manager** - Cross-session quality metrics and learning

**40k Token Context Budget**:
- **10k**: Incoming requests from other departments
- **15k**: Code under analysis + detected issues
- **10k**: Fix proposals + validation results
- **5k**: Session memory (previous fixes, learned patterns)

### Integration Points

```
┌─────────────────────────────────────────────────────────┐
│                   Orchestration Dept                     │
│         (Routes tasks, manages departments)              │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
    ┌────▼───┐  ┌───▼────┐  ┌──▼─────┐
    │Master  │  │Verify  │  │Infra   │
    │Weaver  │  │Dept    │  │Dept    │
    └────┬───┘  └───┬────┘  └──┬─────┘
         │          │           │
         └──────────┼───────────┘
                    │
         ┌──────────▼──────────┐
         │  Quality Assurance   │
         │  ┌─────────────────┐ │
         │  │ Trough Server   │ │  Detection
         │  └─────────┬───────┘ │
         │            │         │
         │  ┌─────────▼───────┐ │
         │  │xTerminator Srvr │ │  Fixing
         │  └─────────────────┘ │
         └─────────────────────┘
```

**Department Interactions**:

1. **MasterWeaver → QA**: "Analyze this code for entity extraction quality"
2. **Verification → QA**: "Check this fix for regressions before approval"
3. **Infrastructure → QA**: "Store quality metrics in Neo4j"
4. **QA → Orchestration**: "High-risk fix detected, escalate to human"

## MCP Server: Trough Detection Tools

### Server Definition

```json
{
  "name": "trough-detection",
  "version": "1.0.0",
  "description": "AI code quality detection - 15 categories, 9 ML algorithms",
  "vendor": "mythRL",
  "tools": [
    "trough_scan_file",
    "trough_scan_directory",
    "trough_get_issue_details",
    "trough_classify_issues",
    "trough_get_metrics"
  ],
  "context_budget": 20000,
  "session_aware": true
}
```

### Tool: trough_scan_file

```python
{
  "name": "trough_scan_file",
  "description": "Scan a single file for AI slop and logic errors",
  "parameters": {
    "file_path": {
      "type": "string",
      "description": "Path to Python file to scan",
      "required": true
    },
    "categories": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Filter to specific categories (empty = all)",
      "default": []
    },
    "min_severity": {
      "type": "string",
      "enum": ["info", "low", "medium", "high", "critical"],
      "description": "Minimum severity to report",
      "default": "low"
    },
    "include_ml_logic": {
      "type": "boolean",
      "description": "Run ML logic detector (slower but deeper)",
      "default": true
    }
  },
  "returns": {
    "type": "object",
    "properties": {
      "file_path": {"type": "string"},
      "total_issues": {"type": "integer"},
      "issues_by_severity": {"type": "object"},
      "issues": {
        "type": "array",
        "items": {
          "category": "string",
          "severity": "string",
          "message": "string",
          "line_number": "integer",
          "code_snippet": "string",
          "suggestion": "string",
          "confidence": "float"
        }
      },
      "scan_duration_ms": {"type": "float"}
    }
  }
}
```

**Example Usage**:
```json
// Request from Verification Dept
{
  "tool": "trough_scan_file",
  "parameters": {
    "file_path": "HoloLoom/policy/unified.py",
    "categories": ["error_handling", "type_confusion"],
    "min_severity": "medium"
  }
}

// Response
{
  "file_path": "HoloLoom/policy/unified.py",
  "total_issues": 8,
  "issues_by_severity": {"medium": 5, "high": 3},
  "issues": [
    {
      "category": "error_handling",
      "severity": "high",
      "message": "Network call without error handling",
      "line_number": 245,
      "code_snippet": "response = requests.get(url)",
      "suggestion": "Wrap in try/except, handle ConnectionError",
      "confidence": 0.85
    }
  ],
  "scan_duration_ms": 123.4
}
```

### Tool: trough_scan_directory

Scans multiple files with parallel processing.

**Parameters**:
- `directory`: Path to scan (recursive)
- `max_files`: Limit (default 100)
- `file_pattern`: Glob pattern (default `**/*.py`)
- `min_severity`: Filter threshold
- `categories`: Category filter

**Returns**: Aggregated results with per-file breakdown

### Tool: trough_classify_issues

Classifies issues by fixability for xTerminator routing.

**Parameters**:
- `issues`: Array of SlopIssue objects

**Returns**:
```json
{
  "auto_fixable": [/* 40% - high confidence fixes */],
  "needs_review": [/* 24% - medium confidence */],
  "manual_only": [/* 1% - security issues */],
  "false_positives": [/* 35% - low confidence */]
}
```

## MCP Server: xTerminator Fixing Tools

### Server Definition

```json
{
  "name": "xterminator-fixer",
  "version": "1.0.0-alpha",
  "description": "Automated code fixing with safety-first approach",
  "vendor": "mythRL",
  "tools": [
    "xterminator_propose_fix",
    "xterminator_validate_fix",
    "xterminator_apply_fix",
    "xterminator_rollback_fix",
    "xterminator_batch_fix"
  ],
  "context_budget": 20000,
  "session_aware": true,
  "requires_human_approval": ["security", "high_risk"]
}
```

### Tool: xterminator_propose_fix

```python
{
  "name": "xterminator_propose_fix",
  "description": "Generate fix proposal for detected issue",
  "parameters": {
    "issue": {
      "type": "object",
      "description": "SlopIssue from Trough detection",
      "required": true
    },
    "fix_strategy": {
      "type": "string",
      "enum": ["ast", "template", "manual"],
      "description": "Override auto-selected strategy"
    },
    "safety_level": {
      "type": "string",
      "enum": ["aggressive", "balanced", "conservative"],
      "default": "balanced"
    }
  },
  "returns": {
    "type": "object",
    "properties": {
      "fix_id": {"type": "string"},
      "issue": {"type": "object"},
      "strategy": {"type": "string"},
      "confidence": {"type": "float"},
      "risk_level": {"type": "string"},
      "proposed_code": {"type": "string"},
      "diff": {"type": "string"},
      "validation_plan": {"type": "array"},
      "requires_approval": {"type": "boolean"}
    }
  }
}
```

**Example Usage**:
```json
// Request from Orchestration
{
  "tool": "xterminator_propose_fix",
  "parameters": {
    "issue": {
      "category": "copy_paste",
      "severity": "medium",
      "line_number": 150,
      "code_snippet": "def process_a(): ...\ndef process_b(): ...",
      "suggestion": "Extract common logic",
      "confidence": 0.85
    },
    "safety_level": "balanced"
  }
}

// Response
{
  "fix_id": "fix_20251110_001",
  "strategy": "ast_extract_function",
  "confidence": 0.87,
  "risk_level": "low",
  "proposed_code": "def _common_process(): ...\ndef process_a(): return _common_process(...)",
  "diff": "@@ -150,10 +150,8 @@ ...",
  "validation_plan": [
    "syntax_check",
    "test_existing_suite",
    "rescan_trough"
  ],
  "requires_approval": false
}
```

### Tool: xterminator_validate_fix

Runs validation pipeline before application.

**Validation Steps**:
1. **Syntax Check**: AST parse validation
2. **Test Execution**: Run existing test suite
3. **Re-scan**: Trough detection on fixed code
4. **Regression Check**: Compare before/after behavior

**Returns**:
```json
{
  "fix_id": "fix_20251110_001",
  "validation_passed": true,
  "steps": [
    {"step": "syntax_check", "passed": true, "duration_ms": 12},
    {"step": "test_execution", "passed": true, "tests_run": 15, "duration_ms": 342},
    {"step": "trough_rescan", "passed": true, "new_issues": 0},
    {"step": "regression_check", "passed": true}
  ],
  "safe_to_apply": true
}
```

### Tool: xterminator_apply_fix

Applies validated fix with Git integration.

**Safety Features**:
- Atomic Git commit per fix
- Rollback capability
- Branch creation for high-risk fixes
- Audit trail in commit message

**Returns**:
```json
{
  "fix_id": "fix_20251110_001",
  "applied": true,
  "git_commit": "a3f2c1b",
  "branch": "main",
  "rollback_command": "git revert a3f2c1b",
  "audit_trail": {
    "timestamp": "2025-11-10T14:32:15Z",
    "detector": "trough",
    "validator": "xterminator",
    "approval": "automated"
  }
}
```

## Cross-Department Integration Patterns

### Pattern 1: Verification-Driven Quality Check

```
Verification Dept detects low confidence in response
    ↓
Requests QA scan: trough_scan_file(file_path)
    ↓
QA returns 8 issues (3 high, 5 medium)
    ↓
Verification escalates to Orchestration
    ↓
Orchestration requests fix proposals
    ↓
QA (xTerminator) proposes fixes
    ↓
Verification validates against original query
    ↓
Approval/rejection routed back to QA
```

### Pattern 2: Infrastructure-Backed Pattern Learning

```
xTerminator applies fix successfully
    ↓
Logs outcome to Infrastructure (Neo4j)
    ↓
Infrastructure stores: issue_type → fix_strategy → success_rate
    ↓
QA queries historical patterns for similar issues
    ↓
Uses learned strategy for next occurrence
    ↓
Continuous improvement loop
```

### Pattern 3: MasterWeaver Entity Extraction Quality

```
MasterWeaver extracts entities from code
    ↓
Requests QA validation: trough_scan_file(focus="entity_extraction")
    ↓
QA detects: hardcoded values, magic numbers, unclear naming
    ↓
Proposes fixes to improve entity clarity
    ↓
MasterWeaver re-extracts with improved code
    ↓
Higher quality entity graph
```

## Session Management

### Session-Aware Context

**QA Session State** (5k token budget):
```python
{
  "session_id": "qa_session_20251110",
  "active_fixes": [
    {"fix_id": "fix_001", "status": "awaiting_validation"},
    {"fix_id": "fix_002", "status": "applied"}
  ],
  "learned_patterns": [
    {
      "issue_type": "copy_paste",
      "fix_strategy": "ast_extract_function",
      "success_rate": 0.92,
      "sample_count": 12
    }
  ],
  "quality_metrics": {
    "total_scans": 145,
    "issues_detected": 1246,
    "fixes_applied": 87,
    "rollbacks": 3,
    "false_positives": 436
  },
  "department_interactions": {
    "verification_requests": 23,
    "masterweaver_requests": 8,
    "orchestration_escalations": 5
  }
}
```

### Context Compaction Strategy

**When context approaches 40k tokens**:

1. **Archive old fixes** (>1 hour old) to Infrastructure
2. **Summarize patterns**: 12 examples → "ast_extract_function works for copy_paste (92%)"
3. **Keep recent**: Last 10 fixes, last 5 department interactions
4. **Preserve metrics**: Aggregate counts always retained

**Compaction Example**:
```
Before (12k tokens):
- 50 detailed fix records
- 200 issue detections with full snippets

After (5k tokens):
- 10 recent detailed fixes
- Summarized: "40 fixes applied (35 copy_paste, 5 error_handling)"
- Pattern learned: "copy_paste → ast_extract_function (92%)"
- Links to archived data in Neo4j
```

## Error Escalation

### Escalation Triggers

1. **High-Risk Fix**: Security category, confidence <0.7
2. **Validation Failure**: Tests fail after fix
3. **Multiple Rollbacks**: Same fix type fails 3+ times
4. **Department Conflict**: Verification rejects QA-approved fix

### Escalation Path

```
QA detects escalation trigger
    ↓
Notifies Orchestration with context
    ↓
Orchestration evaluates:
  - Can another department help? (MasterWeaver for entity context?)
  - Requires human approval?
  - System halt needed?
    ↓
Routes to:
  - Human-in-the-loop (alignment framework)
  - Alternative department
  - Manual queue
```

## Permission Model

### QA Department Permissions

**Allowed**:
- ✅ Read any file in codebase
- ✅ Propose fixes (requires validation)
- ✅ Apply low-risk fixes (automated)
- ✅ Write to audit logs
- ✅ Query Infrastructure for patterns

**Requires Approval**:
- ⚠️ Apply medium-risk fixes (Orchestration approval)
- ⚠️ Create new branches (Git integration)
- ⚠️ Modify test files (Verification approval)

**Forbidden**:
- ❌ Apply high-risk/security fixes (manual only)
- ❌ Modify production config files
- ❌ Delete code without backup
- ❌ Bypass validation pipeline

## Implementation Roadmap

### Phase 1: MCP Server Foundations (Week 1-2)
- Implement Trough MCP server
- Define tool schemas
- Basic session management
- Integration tests with mock departments

### Phase 2: xTerminator Alpha (Week 3-5)
- Classification engine
- Fix proposal generation
- Validation pipeline
- Git integration

### Phase 3: Cross-Department Integration (Week 6-7)
- Verification department integration
- Infrastructure logging
- MasterWeaver collaboration
- Orchestration escalation paths

### Phase 4: Learning & Optimization (Week 8-10)
- Pattern learning from Infrastructure
- Context compaction
- Session optimization
- Performance tuning

## Success Metrics

**Quality Metrics**:
- False positive rate <35% (current baseline)
- Auto-fix success rate >85%
- Rollback rate <5%

**Integration Metrics**:
- Avg response time <500ms (detection)
- Avg fix time <2s (proposal + validation)
- Cross-department latency <100ms

**Learning Metrics**:
- Pattern success rate improving >5% per 100 fixes
- Context compaction maintaining >95% relevant info
- Session continuity across department handoffs

## Conway's Law Application

> "Organizations which design systems are constrained to produce designs which are copies of the communication structures of these organizations."

**QA Department Structure Mirrors Code Quality Pipeline**:

1. **Detection Team** (Trough) ↔ Scanner/Analyzer code structure
2. **Classification Team** (xTerminator Classifier) ↔ Risk assessment logic
3. **Fixing Team** (xTerminator Fixer) ↔ AST/Template fix generators
4. **Validation Team** (xTerminator Validator) ↔ Multi-stage validation pipeline
5. **Application Team** (Git Integrator) ↔ Atomic commit system

**Communication Channels**:
- Detection → Classification: Issue objects with confidence scores
- Classification → Fixing: Risk-stratified fix requests
- Fixing → Validation: Proposed diffs with validation plans
- Validation → Application: Approved fixes with audit trails

**Organizational Benefits**:
- Clear ownership boundaries (no overlapping responsibilities)
- Parallel development (teams work independently on their module)
- Fault isolation (validation failure doesn't break detection)
- Scalability (add more fixers without changing validators)

## Summary

Trough + xTerminator integrate as a **Quality Assurance Department** using MCP, providing detection and fixing tools to HoloLoom's departmental architecture while maintaining:

- **Safety-first**: Human approval for high-risk fixes
- **Context-aware**: Session memory and learned patterns
- **Cross-department**: Collaborative quality improvement
- **Auditable**: Complete provenance in Git and Neo4j
- **Efficient**: 40k token budget with smart compaction

**Next**: Build xTerminator Phase 1 (Classification Engine) with MCP server foundation.

---

**Architecture Status**: ✅ Design Complete
**Implementation Status**: Ready to build
**Integration Points**: 4 departments (Verification, Infrastructure, MasterWeaver, Orchestration)
