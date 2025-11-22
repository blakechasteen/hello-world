# Claude Skills Architecture Patterns

**Philosophy**: Pure functions, clean boundaries, zero coupling

**Last Updated**: 2025-11-22

---

## Core Architecture Principle

```
┌─────────────────────────────────────────────────────────┐
│  "Skills are pure functions with contracts"             │
│                                                          │
│  Input Schema → Skill Logic → Output Schema             │
│                                                          │
│  No side effects. No state. No dependencies.            │
└─────────────────────────────────────────────────────────┘
```

---

## System Layers (4 Tiers)

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Skill Runtime                                  │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│ │ Claude Code │  │ Claude Web  │  │ Claude      │      │
│ │ (local)     │  │ (browser)   │  │ Desktop     │      │
│ └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                          │
│ Boundary: .skill file format (.zip with manifest.json)  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 2: Skill Packaging                                │
│ ┌──────────────────────────────────────────────┐        │
│ │ build_skill.py                               │        │
│ │ • Validates schema                           │        │
│ │ • Packages → .skill                          │        │
│ │ • Generates manifest.json                    │        │
│ └──────────────────────────────────────────────┘        │
│                                                          │
│ Boundary: manifest.json schema                          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 3: Skill Definition                               │
│ ┌──────────────────────────────────────────────┐        │
│ │ skill.markdown                               │        │
│ │ • Source of truth                            │        │
│ │ • Human-readable                             │        │
│ │ • Version controlled (git)                   │        │
│ └──────────────────────────────────────────────┘        │
│                                                          │
│ Boundary: Template compliance                           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 4: Skill Execution                                │
│ ┌──────────────────────────────────────────────┐        │
│ │ Claude interprets skill.markdown             │        │
│ │ • Parses input schema                        │        │
│ │ │ Executes prompt template                   │        │
│ │ • Returns output schema                      │        │
│ └──────────────────────────────────────────────┘        │
│                                                          │
│ Boundary: Input/output JSON schemas                     │
└─────────────────────────────────────────────────────────┘
```

**Key Insight**: Each layer has clean contracts. Changes at Layer 3 don't break Layer 1.

---

## Skill Types (3 Categories)

```
┌─────────────────────────────────────────────────────────┐
│                    Skill Taxonomy                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────┐     │
│  │ Meta-Skills (Orchestrate)                      │     │
│  │ • skill_security_analyzer                      │     │
│  │ • skill_tester                                 │     │
│  │ • skill_dependency_resolver                    │     │
│  │                                                │     │
│  │ Role: Manage other skills, workflows          │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
│  ┌────────────────────────────────────────────────┐     │
│  │ Domain-Skills (Execute)                        │     │
│  │ • hololoom_rag_helper                          │     │
│  │ • typescript_error_explainer                   │     │
│  │ • sql_query_optimizer                          │     │
│  │                                                │     │
│  │ Role: Perform specific domain tasks            │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
│  ┌────────────────────────────────────────────────┐     │
│  │ Utility-Skills (Support)                       │     │
│  │ • json_formatter                               │     │
│  │ • markdown_linter                              │     │
│  │ • regex_builder                                │     │
│  │                                                │     │
│  │ Role: General-purpose transformations          │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Rules**:
- Domain skills **never** call other domain skills (zero coupling)
- Meta-skills **orchestrate** domain skills
- Utility skills are **stateless transforms**

---

## Pattern 1: Pure Function Skill

```
┌─────────────────────────────────────────────────────────┐
│ Skill: typescript_error_explainer                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input:                                                  │
│  {                                                       │
│    "error_code": "TS2322",                              │
│    "snippet": "let x: number = 'hello';",               │
│    "context": "Assignment to variable x"                │
│  }                                                       │
│                                                          │
│  ↓ [Pure Function Logic]                                │
│                                                          │
│  Output:                                                 │
│  {                                                       │
│    "explanation": "Type mismatch: string ≠ number",     │
│    "fix_suggestions": [                                 │
│      "Change type to string: let x: string = ...",      │
│      "Change value to number: let x: number = 123"      │
│    ],                                                    │
│    "metadata": {                                         │
│      "confidence": 0.95,                                 │
│      "execution_time_ms": 450                            │
│    }                                                     │
│  }                                                       │
│                                                          │
└─────────────────────────────────────────────────────────┘

Characteristics:
✓ No state (same input → same output)
✓ No side effects (no file writes, no network calls)
✓ No dependencies (doesn't call other skills)
✓ Fast (<3s execution)
✓ Token-efficient (<700 tokens)
```

---

## Pattern 2: HoloLoom Integration Skill

```
┌─────────────────────────────────────────────────────────┐
│ Skill: memory_graph_navigator                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input:                                                  │
│  {                                                       │
│    "from_memory": "thompson_sampling",                  │
│    "direction": "sideways",                             │
│    "steps": 3                                           │
│  }                                                       │
│                                                          │
│  ↓ [HoloLoom API Call]                                  │
│                                                          │
│  from HoloLoom.memory.unified import NavigationDirection│
│  path = memory.navigate(                                │
│    from_memory=input["from_memory"],                    │
│    direction=NavigationDirection.SIDEWAYS,              │
│    steps=input["steps"]                                 │
│  )                                                       │
│                                                          │
│  ↓ [Transform to Output Schema]                         │
│                                                          │
│  Output:                                                 │
│  {                                                       │
│    "path": ["bayesian_methods", "UCB", "epsilon_greedy"],│
│    "insights": ["All exploration strategies"],          │
│    "metadata": {                                         │
│      "confidence": 0.88,                                 │
│      "execution_time_ms": 650                            │
│    }                                                     │
│  }                                                       │
│                                                          │
└─────────────────────────────────────────────────────────┘

Characteristics:
✓ Wraps HoloLoom API (thin wrapper)
✓ Graceful degradation (fallback if HoloLoom unavailable)
✓ Exposes HoloLoom features to users (simplified interface)
✓ Still pure function (input → API call → output)
```

---

## Pattern 3: Meta-Skill Orchestration

```
┌─────────────────────────────────────────────────────────┐
│ Meta-Skill: skill_dependency_resolver                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input:                                                  │
│  {                                                       │
│    "user_intent": "Optimize SQL query and explain",    │
│    "available_skills": [                                │
│      "sql_query_optimizer",                             │
│      "sql_explainer"                                    │
│    ]                                                     │
│  }                                                       │
│                                                          │
│  ↓ [Orchestration Logic]                                │
│                                                          │
│  1. Analyze intent: "optimize" + "explain"              │
│  2. Identify skills: [sql_query_optimizer, sql_explainer]│
│  3. Determine sequence: optimize → then explain         │
│  4. Build workflow DAG                                  │
│                                                          │
│  ↓ [Generate Workflow]                                  │
│                                                          │
│  Output:                                                 │
│  {                                                       │
│    "workflow": [                                         │
│      {                                                   │
│        "skill": "sql_query_optimizer",                  │
│        "inputs": {"query": "..."},                      │
│        "outputs": ["optimized_query"]                   │
│      },                                                  │
│      {                                                   │
│        "skill": "sql_explainer",                        │
│        "inputs": {"query": "$optimized_query"},         │
│        "outputs": ["explanation"]                       │
│      }                                                   │
│    ],                                                    │
│    "rationale": "Optimize first, then explain"          │
│  }                                                       │
│                                                          │
└─────────────────────────────────────────────────────────┘

Characteristics:
✓ Coordinates domain skills (doesn't execute domain logic)
✓ Returns workflow (doesn't execute it directly)
✓ Enables multi-skill composition
✓ DAG-based execution (parallel when possible)
```

---

## Pattern 4: Graceful Degradation

```
┌─────────────────────────────────────────────────────────┐
│ Skill: hololoom_rag_helper                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────┐                 │
│  │ Try: Full HoloLoom RAG             │                 │
│  │ • MultimodalRAG                    │                 │
│  │ • Agentic reasoning                │                 │
│  │ • Context packing                  │                 │
│  └────────────────────────────────────┘                 │
│         │                                                │
│         │ HoloLoom Available?                            │
│         ├─ YES → Return full result                     │
│         │                                                │
│         └─ NO  ↓                                         │
│                                                          │
│  ┌────────────────────────────────────┐                 │
│  │ Fallback: Basic Retrieval          │                 │
│  │ • Text similarity only             │                 │
│  │ • No graph traversal               │                 │
│  │ • No visual compression            │                 │
│  └────────────────────────────────────┘                 │
│         │                                                │
│         └─ Return degraded result + warning             │
│                                                          │
│  Output (degraded mode):                                │
│  {                                                       │
│    "answer": "Basic answer (reduced quality)",          │
│    "sources": ["text_only_sources"],                    │
│    "metadata": {                                         │
│      "confidence": 0.65,  // Lower confidence           │
│      "warnings": [                                       │
│        "HoloLoom unavailable, using fallback"           │
│      ]                                                   │
│    }                                                     │
│  }                                                       │
│                                                          │
└─────────────────────────────────────────────────────────┘

Philosophy: "Degrade gracefully, warn explicitly"
```

---

## Data Flow: Skill Invocation

```
User Query
    │
    ↓
┌───────────────────────────────────────────────────────┐
│ Step 1: Intent Classification                         │
│ • Is this a skill-suitable task?                      │
│ • Which skill(s) apply?                               │
└───────────────────────────────────────────────────────┘
    │
    ↓
┌───────────────────────────────────────────────────────┐
│ Step 2: Skill Discovery                               │
│ • List available skills (skills/dist/*.skill)         │
│ • Filter by capabilities                              │
│ • Rank by relevance                                   │
└───────────────────────────────────────────────────────┘
    │
    ↓
┌───────────────────────────────────────────────────────┐
│ Step 3: Input Preparation                             │
│ • Extract data from user query                        │
│ • Validate against input schema                       │
│ • Provide defaults for optional fields                │
└───────────────────────────────────────────────────────┘
    │
    ↓
┌───────────────────────────────────────────────────────┐
│ Step 4: Skill Execution                               │
│ • Load skill.markdown                                 │
│ • Substitute input data into prompt template          │
│ • Execute (Claude interprets)                         │
│ • Parse output                                        │
└───────────────────────────────────────────────────────┘
    │
    ↓
┌───────────────────────────────────────────────────────┐
│ Step 5: Output Validation                             │
│ • Validate against output schema                      │
│ • Check metadata (confidence, warnings)               │
│ • Log execution metrics                               │
└───────────────────────────────────────────────────────┘
    │
    ↓
┌───────────────────────────────────────────────────────┐
│ Step 6: Response Formatting                           │
│ • Format for user display                             │
│ • Include sources/reasoning                           │
│ • Provide follow-up suggestions                       │
└───────────────────────────────────────────────────────┘
    │
    ↓
User Response
```

---

## Skill Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│                   Skill Lifecycle                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Creation                                             │
│     • Copy template                                      │
│     • Fill in skill.markdown                             │
│     • Define schemas                                     │
│     └─> skills/domain/new_skill/skill.markdown          │
│                                                          │
│  2. Validation                                           │
│     • Run skill_security_analyzer                        │
│     • Run skill_tester                                   │
│     • Run token_budget_adviser                           │
│     └─> All pass? → Continue                             │
│                                                          │
│  3. Packaging                                            │
│     • python build_skill.py skills/domain/new_skill      │
│     • Generates manifest.json                            │
│     • Creates .zip → .skill                              │
│     └─> skills/dist/new_skill-1.0.0.skill                │
│                                                          │
│  4. Deployment                                           │
│     • Local: cp to ~/.claude/skills/                     │
│     • Web: Upload to MirrorCore marketplace             │
│     └─> Skill available for invocation                   │
│                                                          │
│  5. Execution (per-use)                                  │
│     • User invokes skill                                 │
│     • Runtime loads .skill                               │
│     • Executes prompt template                           │
│     └─> Returns output                                   │
│                                                          │
│  6. Monitoring                                           │
│     • Track usage metrics                                │
│     • Identify performance issues                        │
│     • Collect user feedback                              │
│     └─> Inform v1.1.0 improvements                       │
│                                                          │
│  7. Evolution                                            │
│     • Bump version (1.0.0 → 1.1.0)                       │
│     • Update skill.markdown                              │
│     • Rebuild and redeploy                               │
│     └─> Repeat cycle                                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Directory Structure (Clean)

```
mythRL/
├── skills/
│   ├── meta/                    # Meta-skills (orchestrate)
│   │   ├── continuous_learning_capture/
│   │   │   ├── skill.markdown
│   │   │   └── manifest.json
│   │   ├── skill_security_analyzer/
│   │   ├── skill_tester/
│   │   └── token_budget_adviser/
│   │
│   ├── domain/                  # Domain-skills (execute)
│   │   ├── hololoom_rag_helper/
│   │   ├── typescript_error_explainer/
│   │   ├── sql_query_optimizer/
│   │   └── [future domain skills]/
│   │
│   ├── utility/                 # Utility-skills (support)
│   │   └── [future utility skills]/
│   │
│   ├── templates/               # Templates
│   │   └── skill.markdown.template
│   │
│   ├── dist/                    # Built .skill files
│   │   ├── continuous_learning_capture-1.0.0.skill
│   │   ├── hololoom_rag_helper-1.0.0.skill
│   │   └── [all packaged skills]
│   │
│   └── archive/                 # Deprecated skills
│       ├── v1.x/                # Historical versions
│       └── deprecated/          # Retired skills
│
├── scripts/
│   └── build_skill.py           # Packaging automation
│
└── docs/
    └── skills_workflow.md       # Workflow guide
```

**Rules**:
- One directory per skill
- `skill.markdown` is source of truth
- `manifest.json` auto-generated by build_skill.py
- Built `.skill` files in `dist/`
- Never delete (archive instead)

---

## Quality Gates (4 Checks)

```
┌─────────────────────────────────────────────────────────┐
│               Skill Quality Pipeline                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Gate 1: Security                                        │
│  ┌────────────────────────────────────────────┐         │
│  │ skill_security_analyzer                    │         │
│  │ • Prompt injection detection               │         │
│  │ • Capability boundary checks               │         │
│  │ • Data privacy validation                  │         │
│  └────────────────────────────────────────────┘         │
│  ✅ Pass: No critical vulnerabilities                    │
│  ❌ Fail: Fix and resubmit                               │
│                                                          │
│  Gate 2: Testing                                         │
│  ┌────────────────────────────────────────────┐         │
│  │ skill_tester                               │         │
│  │ • Run all examples                         │         │
│  │ • Validate outputs match schemas           │         │
│  │ • Test edge cases                          │         │
│  └────────────────────────────────────────────┘         │
│  ✅ Pass: All tests succeed                              │
│  ❌ Fail: Fix failing tests                              │
│                                                          │
│  Gate 3: Token Efficiency                                │
│  ┌────────────────────────────────────────────┐         │
│  │ token_budget_adviser                       │         │
│  │ • Count tokens in prompt template          │         │
│  │ • Suggest optimizations                    │         │
│  │ • Enforce budget (<1000 tokens)            │         │
│  └────────────────────────────────────────────┘         │
│  ✅ Pass: < 1000 tokens                                  │
│  ⚠️  Warning: 700-1000 tokens (review)                   │
│  ❌ Fail: > 1000 tokens (refactor)                       │
│                                                          │
│  Gate 4: Schema Validation                               │
│  ┌────────────────────────────────────────────┐         │
│  │ build_skill.py --validate                  │         │
│  │ • Check template compliance                │         │
│  │ • Validate manifest.json                   │         │
│  │ • Ensure all sections present              │         │
│  └────────────────────────────────────────────┘         │
│  ✅ Pass: Schema valid                                   │
│  ❌ Fail: Fix schema errors                              │
│                                                          │
└─────────────────────────────────────────────────────────┘

All 4 gates must pass before deployment.
```

---

## Performance Budget

```
┌─────────────────────────────────────────────────────────┐
│                  Performance Targets                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Execution Time:                                         │
│  • Simple skills: < 1s                                   │
│  • Standard skills: < 3s                                 │
│  • Complex skills: < 5s                                  │
│  • Hard limit: 10s (timeout)                             │
│                                                          │
│  Token Budget:                                           │
│  • Minimal: 200-400 tokens                               │
│  • Standard: 500-700 tokens ⭐ Target                    │
│  • Complex: 800-1000 tokens                              │
│  • Hard limit: 1000 tokens                               │
│                                                          │
│  Memory:                                                 │
│  • Per skill instance: < 100MB                           │
│  • Total (100 skills): < 500MB                           │
│                                                          │
│  Packaging:                                              │
│  • Build time: < 1s per skill                            │
│  • Package size: < 50KB per .skill                       │
│  • Validation time: < 2s per skill                       │
│                                                          │
│  Scalability:                                            │
│  • Total skills: 100+ supported                          │
│  • Concurrent executions: 10+ in parallel                │
│  • Skill discovery: < 100ms                              │
│  • Lazy loading: Load only when invoked                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Versioning Strategy

```
Semantic Versioning: MAJOR.MINOR.PATCH

┌─────────────────────────────────────────────────────────┐
│  MAJOR (1.0.0 → 2.0.0)                                   │
│  • Breaking changes to input/output schemas              │
│  • Removed capabilities                                  │
│  • Incompatible API changes                              │
│  └─> Requires user migration                             │
│                                                          │
│  MINOR (1.0.0 → 1.1.0)                                   │
│  • New features (backward compatible)                    │
│  • New optional input fields                             │
│  • Enhanced output (additional fields)                   │
│  └─> Drop-in replacement (no migration)                  │
│                                                          │
│  PATCH (1.0.0 → 1.0.1)                                   │
│  • Bug fixes                                             │
│  • Performance improvements                              │
│  • Documentation updates                                 │
│  └─> Safe to auto-update                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘

Deprecation Policy:
• 6-month notice for MAJOR version changes
• Mark deprecated fields with warnings
• Provide migration guide (old → new)
• Support N-1 major version for 6 months
```

---

## Error Handling Pattern

```
┌─────────────────────────────────────────────────────────┐
│                  Standard Error Format                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  {                                                       │
│    "error": "Human-readable error message",             │
│    "error_code": "SKILL_ERROR_CODE",                    │
│    "details": {                                          │
│      "input_validation_failures": [...],                │
│      "execution_errors": [...],                         │
│      "stack_trace": "..." (optional, debug mode)        │
│    },                                                    │
│    "metadata": {                                         │
│      "confidence": 0.0,                                  │
│      "execution_time_ms": 125,                           │
│      "warnings": ["Warning message 1"]                   │
│    },                                                    │
│    "suggestions": [                                      │
│      "Try providing field X",                           │
│      "Check input format matches schema"                │
│    ]                                                     │
│  }                                                       │
│                                                          │
└─────────────────────────────────────────────────────────┘

Error Codes:
• INPUT_VALIDATION_ERROR - Invalid input schema
• EXECUTION_ERROR - Skill logic failed
• TIMEOUT_ERROR - Exceeded time limit
• CAPABILITY_DENIED - Missing required capability
• DEPENDENCY_UNAVAILABLE - External dependency missing
```

---

## Capability Declaration

```
┌─────────────────────────────────────────────────────────┐
│                Standard Capabilities                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  File System:                                            │
│  • file_read       - Read files                          │
│  • file_write      - Write files                         │
│  • file_delete     - Delete files (use sparingly)        │
│                                                          │
│  Code Execution:                                         │
│  • bash_exec       - Run bash commands                   │
│  • python_exec     - Run Python code                     │
│                                                          │
│  Network:                                                │
│  • network_fetch   - HTTP requests (read)                │
│  • network_post    - HTTP requests (write)               │
│  • web_search      - Search the web                      │
│                                                          │
│  User Interaction:                                       │
│  • user_prompt     - Ask user questions                  │
│  • user_confirm    - Request user confirmation           │
│                                                          │
│  External Services:                                      │
│  • mcp_server      - MCP server access                   │
│  • external_api    - External API calls                  │
│                                                          │
│  HoloLoom (Optional):                                    │
│  • hololoom_memory - HoloLoom memory system              │
│  • hololoom_rag    - HoloLoom RAG                        │
│  • hololoom_align  - HoloLoom alignment framework        │
│                                                          │
└─────────────────────────────────────────────────────────┘

Principle: "Least Privilege"
Only request capabilities you actually need.
```

---

## Testing Pattern

```
┌─────────────────────────────────────────────────────────┐
│               Standard Test Structure                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Test Case 1: Basic Usage                                │
│  ┌────────────────────────────────────────────┐         │
│  │ Input:                                     │         │
│  │   {"field": "valid_value"}                 │         │
│  │                                            │         │
│  │ Expected Output:                           │         │
│  │   {"result": "expected_result"}            │         │
│  │                                            │         │
│  │ Assertion:                                 │         │
│  │   output["result"] == "expected_result"    │         │
│  │   output["metadata"]["confidence"] > 0.8   │         │
│  └────────────────────────────────────────────┘         │
│                                                          │
│  Test Case 2: Edge Case                                  │
│  ┌────────────────────────────────────────────┐         │
│  │ Input:                                     │         │
│  │   {"field": "edge_case_value"}             │         │
│  │                                            │         │
│  │ Expected Output:                           │         │
│  │   {"result": "handled_gracefully"}         │         │
│  │   {"metadata": {"warnings": [...]}}        │         │
│  └────────────────────────────────────────────┘         │
│                                                          │
│  Test Case 3: Error Scenario                             │
│  ┌────────────────────────────────────────────┐         │
│  │ Input:                                     │         │
│  │   {"field": "invalid_value"}               │         │
│  │                                            │         │
│  │ Expected Output:                           │         │
│  │   {"error": "Error description"}           │         │
│  │   {"metadata": {"confidence": 0.0}}        │         │
│  └────────────────────────────────────────────┘         │
│                                                          │
└─────────────────────────────────────────────────────────┘

Minimum: 3 test cases per skill
• 1 happy path (basic usage)
• 1 edge case (boundary conditions)
• 1 error case (invalid input)
```

---

## Conclusion: Architectural Invariants

### ✅ Always True
1. Skills are pure functions (input → output)
2. Domain skills never depend on other domain skills
3. All skills have input/output schemas
4. All skills pass 4 quality gates
5. All skills are versioned (semver)

### ❌ Never True
1. Skills maintain state between invocations
2. Skills have side effects (without capability declaration)
3. Skills bypass quality gates
4. Skills are deployed without testing

### 🎯 Goal
**100+ skills, zero coupling, <3s execution, <700 tokens avg**

---

**Next Steps**:
1. Use these patterns when building Wave 1 skills
2. Validate patterns with first 6 domain skills
3. Refine patterns based on learnings
4. Document deviations and rationale

---

**Prepared by**: Claude Code Agent
**Date**: 2025-11-22
