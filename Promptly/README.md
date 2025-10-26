# Promptly - Prompt Composition & Testing Framework

**A powerful meta-prompt framework with loop composition, LLM judging, and HoloLoom integration**

---

## Quick Start

```bash
# Install
pip install -e promptly/

# Run CLI
python promptly/promptly_cli.py

# Or use shortcuts
./promptly/Promptly.ps1  # PowerShell
./promptly/promptly.bat  # Windows CMD
```

---

## Directory Structure

```
Promptly/
├── promptly/              # Main package
│   ├── promptly.py        # Core prompt composition engine
│   ├── promptly_cli.py    # Command-line interface
│   ├── execution_engine.py # Execution orchestration
│   ├── loop_composition.py # Loop DSL and composition
│   ├── recursive_loops.py  # Recursive loop support
│   ├── package_manager.py  # Skill package management
│   ├── advanced_integration.py # Advanced features
│   ├── skill_templates_extended.py # Skill scaffolding
│   │
│   ├── tools/             # Utility tools
│   │   ├── ab_testing.py      # A/B test framework
│   │   ├── cost_tracker.py    # LLM cost tracking
│   │   ├── diff_merge.py      # Diff/merge utilities
│   │   ├── llm_judge.py       # LLM-as-judge
│   │   ├── llm_judge_enhanced.py # Enhanced judge
│   │   ├── prompt_analytics.py # Analytics
│   │   ├── ultraprompt_llm.py # Ultraprompt integration
│   │   └── ultraprompt_ollama.py # Ollama ultraprompt
│   │
│   ├── integrations/      # External integrations
│   │   ├── hololoom_bridge.py # HoloLoom integration
│   │   └── mcp_server.py      # Model Context Protocol
│   │
│   ├── docs/              # Package documentation
│   │   ├── QUICKSTART.md
│   │   ├── PROJECT_OVERVIEW.md
│   │   ├── EXECUTION_GUIDE.md
│   │   ├── MCP_SETUP.md
│   │   ├── OLLAMA_SETUP.md
│   │   ├── SKILLS.md
│   │   └── ...
│   │
│   ├── examples/          # Example code
│   │   ├── example_skill_workflow.py
│   │   ├── test_ollama_debug.py
│   │   └── ...
│   │
│   ├── skill_templates/   # Skill templates
│   └── .promptly/         # User data (skills, configs)
│
├── demos/                 # Demo scripts
│   ├── demo_integration_showcase.py
│   ├── demo_ultimate_integration.py
│   ├── demo_rich_cli.py
│   ├── demo_analytics_live.py
│   └── ...
│
├── docs/                  # Project documentation
│   ├── INTEGRATION_COMPLETE.md
│   ├── PROMPTLY_PHASE*.md
│   ├── SESSION_SUMMARY.md
│   └── ...
│
├── tests/                 # Test suite
│   ├── test_mcp_tools.py
│   └── test_recursive_loops.py
│
├── templates/             # Project templates
└── config/                # Configuration files
```

---

## Core Features

### 1. Prompt Composition

```python
from promptly import Promptly

p = Promptly()

# Chain prompts
result = p.chain([
    "Analyze this code",
    "Find bugs",
    "Suggest fixes"
])
```

### 2. Loop Composition

```python
from promptly.loop_composition import LoopComposer

composer = LoopComposer()

# Define loop with DSL
loop = composer.parse("""
LOOP research
  INPUT: topic
  STEPS:
    - search(topic)
    - analyze(results)
    - summarize(findings)
  OUTPUT: summary
END
""")

result = loop.execute({"topic": "quantum computing"})
```

### 3. LLM Judge

```python
from promptly.tools.llm_judge_enhanced import EnhancedLLMJudge

judge = EnhancedLLMJudge()

# Evaluate responses
score = judge.evaluate(
    query="Explain AI",
    response="AI is artificial intelligence...",
    criteria=["accuracy", "clarity", "completeness"]
)
```

### 4. A/B Testing

```python
from promptly.tools.ab_testing import ABTester

tester = ABTester()

# Compare prompts
winner = tester.compare(
    prompt_a="Explain {topic} simply",
    prompt_b="Provide detailed {topic} overview",
    test_cases=[{"topic": "ML"}, {"topic": "NLP"}]
)
```

### 5. HoloLoom Integration

```python
from promptly.integrations.hololoom_bridge import HoloLoomBridge

bridge = HoloLoomBridge()

# Use HoloLoom memory
result = bridge.query_with_memory(
    query="What did we discuss about MCTS?",
    context_limit=5
)
```

---

## Key Components

### Execution Engine
Orchestrates prompt execution with:
- Retry logic
- Error handling
- Cost tracking
- Analytics

### Loop Composer
DSL for defining reusable workflows:
- Variable interpolation
- Conditional branching
- Nested loops
- Result aggregation

### Skill System
Modular, reusable prompt components:
- Templates
- Parameter validation
- Versioning
- Package management

### MCP Server
Model Context Protocol integration:
- Tool exposure
- Resource management
- Prompt templates
- Claude Desktop integration

---

## Quick Examples

### Example 1: Simple Chain
```python
from promptly import Promptly

p = Promptly()
result = p.chain([
    "List 5 ML topics",
    "Pick the most interesting",
    "Explain it in detail"
])
```

### Example 2: Loop with Variables
```python
from promptly.loop_composition import LoopComposer

composer = LoopComposer()
loop = composer.parse("""
LOOP process_topics
  INPUT: topics
  FOR topic IN topics:
    research = search(topic)
    summary = summarize(research)
    YIELD summary
  OUTPUT: summaries
END
""")
```

### Example 3: Judge Comparison
```python
from promptly.tools.llm_judge_enhanced import EnhancedLLMJudge

judge = EnhancedLLMJudge()

results = judge.batch_evaluate([
    {"query": "What is AI?", "response": "..."},
    {"query": "Explain ML", "response": "..."}
])
```

---

## Documentation

- **[QUICKSTART.md](promptly/docs/QUICKSTART.md)** - Get started in 5 minutes
- **[PROJECT_OVERVIEW.md](promptly/docs/PROJECT_OVERVIEW.md)** - Architecture overview
- **[EXECUTION_GUIDE.md](promptly/docs/EXECUTION_GUIDE.md)** - Execution engine details
- **[SKILLS.md](promptly/docs/SKILLS.md)** - Creating skills
- **[MCP_SETUP.md](promptly/docs/MCP_SETUP.md)** - MCP server setup

### Session Documentation
- **[INTEGRATION_COMPLETE.md](docs/INTEGRATION_COMPLETE.md)** - Latest integration work
- **[SESSION_SUMMARY.md](docs/SESSION_SUMMARY.md)** - Development session summary

---

## Demos

Run demos to see features in action:

```bash
# Integration showcase
python demos/demo_integration_showcase.py

# Rich CLI demo
python demos/demo_rich_cli.py

# Analytics dashboard
python demos/demo_analytics_live.py

# Ultimate meta-demo
python demos/demo_ultimate_meta.py
```

---

## Requirements

```
ollama      # Local LLM inference
rich        # Terminal formatting
pyyaml      # Config files
```

Optional:
```
anthropic   # Claude API
openai      # OpenAI API
```

---

## License

MIT License - See [promptly/LICENSE](promptly/LICENSE)

---

## Development Status

**Phase 4 Complete** - All core features operational:
- ✅ Prompt composition
- ✅ Loop DSL
- ✅ LLM judge
- ✅ A/B testing
- ✅ Skills system
- ✅ MCP server
- ✅ HoloLoom integration
- ✅ Rich CLI
- ✅ Analytics

---

## Contributing

1. Add skills to `promptly/.promptly/skills/`
2. Add tools to `promptly/tools/`
3. Add demos to `demos/`
4. Update docs in `docs/`

---

**Built for composable, testable, and analyzable prompt workflows.**