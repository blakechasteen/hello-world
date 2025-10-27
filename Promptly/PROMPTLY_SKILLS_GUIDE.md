# Promptly Skills for Claude Code

**Version**: v1.1
**Status**: Ready for use!
**Integration**: MCP (Model Context Protocol)

---

## What is Promptly Skills?

Promptly Skills integrates Promptly's **30+ prompt management tools** directly into Claude Desktop via the MCP protocol. Manage prompts, execute recursive loops, run A/B tests, and more - all from within your Claude conversations.

---

## Quick Start

### 1. Prerequisites

```bash
# Install MCP SDK
pip install mcp

# Ensure Promptly is installed
cd Promptly/promptly
pip install -e .
```

### 2. Configure Claude Desktop

Add to your Claude Desktop MCP settings (`claude_desktop_config.json`):

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "promptly": {
      "command": "python",
      "args": [
        "C:/Users/YOUR_USERNAME/Documents/mythRL/Promptly/promptly/integrations/mcp_server.py"
      ]
    }
  }
}
```

**Replace `YOUR_USERNAME` and adjust the path to match your installation!**

### 3. Restart Claude Desktop

Close and reopen Claude Desktop. Promptly tools should now be available!

---

## Available Tools (30+)

### Core Prompt Management

#### `promptly_add`
Add or update a prompt

```
promptly_add(
  name="SQL Optimizer",
  content="Analyze this SQL query: {query}\n\nSuggest optimizations for performance.",
  metadata={"tags": ["sql", "optimization"]}
)
```

#### `promptly_get`
Retrieve a prompt by name

```
promptly_get(name="SQL Optimizer", version=3)
```

#### `promptly_list`
List all prompts

```
promptly_list(branch="main")
```

---

### Skills Management

#### `promptly_skill_add`
Create a new skill

```
promptly_skill_add(
  name="Code Reviewer",
  description="Review code for bugs and improvements",
  metadata={"runtime": "claude", "tags": ["code", "quality"]}
)
```

#### `promptly_skill_add_file`
Attach files to a skill

```
promptly_skill_add_file(
  skill_name="Code Reviewer",
  filename="review_checklist.md",
  content="# Review Checklist\n- Check for bugs\n- Assess security\n...",
  filetype="md"
)
```

#### `promptly_execute_skill`
Execute a skill with input

```
promptly_execute_skill(
  skill_name="Code Reviewer",
  user_input="Review this Python function: def process(data)..."
)
```

---

### Recursive Loops & Composition

#### `promptly_refine_iteratively`
Refine output through self-critique

```
promptly_refine_iteratively(
  task="Write a blog post about AI ethics",
  initial_output="AI is important...",
  max_iterations=3,
  backend="ollama"
)
```

#### `promptly_hofstadter_loop`
Think recursively with meta-levels

```
promptly_hofstadter_loop(
  task="How can I improve my problem-solving skills?",
  levels=3,
  backend="ollama"
)
```

#### `promptly_compose_loops`
Execute composed pipeline

```
promptly_compose_loops(
  task="Design a distributed database",
  steps=[
    {"loop_type": "decompose", "max_iterations": 2},
    {"loop_type": "refine", "max_iterations": 3},
    {"loop_type": "verify", "max_iterations": 1}
  ],
  backend="ollama"
)
```

#### `promptly_decompose_refine_verify`
Common DRV pattern

```
promptly_decompose_refine_verify(
  task="Build a REST API for user authentication",
  backend="ollama"
)
```

---

### Analytics & Insights

#### `promptly_analytics_summary`
Get overall analytics

```
promptly_analytics_summary()
```

**Output**:
```
Total Executions: 340
Unique Prompts: 15
Success Rate: 94.1%
Average Quality Score: 0.87
Total Cost: $2.45
```

#### `promptly_analytics_prompt_stats`
Detailed stats for a prompt

```
promptly_analytics_prompt_stats(prompt_name="SQL Optimizer")
```

#### `promptly_analytics_top_prompts`
Top performing prompts

```
promptly_analytics_top_prompts(metric="quality", limit=5)
```

#### `promptly_analytics_recommendations`
AI recommendations

```
promptly_analytics_recommendations()
```

---

### A/B Testing

#### `promptly_ab_test`
Compare prompt variants

```
promptly_ab_test(
  test_name="SQL Optimizer Comparison",
  variants=["SQL Optimizer v1", "SQL Optimizer v2"],
  test_inputs=[
    "SELECT * FROM users WHERE active=1",
    "SELECT u.* FROM users u JOIN orders o ON u.id=o.user_id"
  ],
  backend="ollama"
)
```

---

### Package Management

#### `promptly_export_package`
Export prompts/skills to shareable package

```
promptly_export_package(
  package_name="sql-toolkit",
  prompts=["SQL Optimizer", "Query Explainer"],
  skills=["Database Advisor"],
  author="Your Name"
)
```

#### `promptly_import_package`
Import from package file

```
promptly_import_package(
  package_path="./packages/sql-toolkit.promptly",
  overwrite=false
)
```

---

### Version Control

#### `promptly_diff`
Compare prompt versions

```
promptly_diff(
  prompt_name="SQL Optimizer",
  version_a=1,
  version_b=3
)
```

#### `promptly_merge_branches`
Merge branches with conflict detection

```
promptly_merge_branches(
  source_branch="feature/new-prompts",
  target_branch="main",
  auto_resolve=true
)
```

---

### Cost Tracking

#### `promptly_cost_summary`
Get cost analysis

```
promptly_cost_summary(prompt_name="SQL Optimizer")
```

---

### NEW: Week 2 Smart Features 🎉

#### `promptly_auto_score` *(Coming Soon)*
Automatically score prompt quality

```
promptly_auto_score(
  prompt_name="SQL Optimizer",
  method="heuristic"  # or "llm" or "hybrid"
)
```

**Returns**: Clarity, Completeness, Effectiveness scores + feedback

#### `promptly_suggest_related` *(Coming Soon)*
Find related prompts using HoloLoom

```
promptly_suggest_related(
  prompt_name="SQL Optimizer",
  limit=5
)
```

**Returns**: Similar prompts with relevance scores

#### `promptly_auto_tag` *(Coming Soon)*
Extract tags from prompt content

```
promptly_auto_tag(
  prompt_name="SQL Optimizer",
  max_tags=10
)
```

**Returns**: Suggested tags with confidence scores

#### `promptly_detect_duplicates` *(Coming Soon)*
Find duplicate/similar prompts

```
promptly_detect_duplicates(
  min_similarity=0.90
)
```

**Returns**: Duplicate pairs with merge suggestions

#### `promptly_health_check` *(Coming Soon)*
Check system health

```
promptly_health_check()
```

**Returns**: Status of Database, HoloLoom, Neo4j, Qdrant, Redis

---

## Usage Examples

### Example 1: Create and Execute a Prompt

```
1. Create prompt:
   promptly_add(
     name="Bug Finder",
     content="Find bugs in this code: {code}"
   )

2. Execute it:
   promptly_execute_skill_real(
     skill_name="Bug Finder",
     user_input="code: def calc(x): return x/0",
     backend="ollama"
   )
```

### Example 2: Recursive Refinement

```
1. Get initial output (from any source)

2. Refine it:
   promptly_refine_iteratively(
     task="Explain quantum computing",
     initial_output="Quantum computers use qubits...",
     max_iterations=3
   )

3. Check analytics:
   promptly_analytics_prompt_stats(prompt_name="refine")
```

### Example 3: A/B Test Prompts

```
1. Create two variants:
   promptly_add(name="Explainer v1", content="Explain {topic} simply")
   promptly_add(name="Explainer v2", content="Explain {topic} in detail with examples")

2. Run A/B test:
   promptly_ab_test(
     test_name="Explainer Comparison",
     variants=["Explainer v1", "Explainer v2"],
     test_inputs=["neural networks", "blockchain"]
   )

3. Review results and pick winner
```

### Example 4: Package and Share

```
1. Export your best prompts:
   promptly_export_package(
     package_name="my-prompts",
     prompts=["Bug Finder", "Code Reviewer", "SQL Optimizer"]
   )

2. Share the .promptly file with team

3. Team imports:
   promptly_import_package(package_path="./my-prompts.promptly")
```

---

## Architecture

```
Claude Desktop
      ↓
   MCP Protocol
      ↓
Promptly MCP Server (mcp_server.py)
      ↓
   Promptly Core
      ├─ Prompt Management (Git-based versioning)
      ├─ Skills System (Reusable workflows)
      ├─ Execution Engine (Ollama, Claude API)
      ├─ Analytics (SQLite tracking)
      ├─ Recursive Loops (Meta-cognitive patterns)
      ├─ A/B Testing (Variant comparison)
      └─ HoloLoom Integration (Semantic memory)
```

---

## Benefits

### For Solo Developers
- **Prompt Library**: Store and version your best prompts
- **Quick Execution**: Run prompts without leaving Claude
- **Analytics**: Track what works best
- **Refinement**: Improve outputs iteratively

### For Teams
- **Sharing**: Export/import prompt packages
- **Versioning**: Git-based history
- **Collaboration**: Branch and merge prompts
- **Consistency**: Everyone uses tested prompts

### For Researchers
- **Recursive Intelligence**: Hofstadter loops, meta-reasoning
- **A/B Testing**: Scientific prompt comparison
- **HoloLoom Memory**: Semantic search and relationships
- **Cost Tracking**: Monitor API expenses

---

## Troubleshooting

### "MCP server not found"
- Check Claude Desktop config path is correct
- Ensure Python path in config is absolute
- Try using `python3` instead of `python`

### "Execution engine not available"
- Install: `pip install ollama anthropic`
- Start Ollama: `ollama serve`
- Set API key for Claude API backend

### "Analytics not available"
- Analytics DB auto-creates at `~/.promptly/analytics.db`
- Check write permissions to home directory

### "HoloLoom not available"
- HoloLoom is optional for semantic features
- Install if needed: `cd HoloLoom && pip install -e .`

---

## Roadmap

### v1.1 (Current)
- [x] 30+ MCP tools
- [x] Recursive loops
- [x] A/B testing
- [x] Analytics
- [ ] Week 2 smart features integration

### v1.2 (Next)
- [ ] VS Code extension
- [ ] Web dashboard
- [ ] Team collaboration features
- [ ] Multi-modal prompts

---

## Contributing

Promptly Skills is part of the **Promptly** project. See main repo for contribution guidelines.

---

## Support

- **Issues**: File at main Promptly repo
- **Questions**: Ask in Claude conversations using Promptly tools!
- **Docs**: See [main Promptly documentation](./README.md)

---

## Example Session

```
User: I need to create a prompt for code review

Claude: I'll help you create that! Let me use Promptly:

[Calls promptly_add with appropriate content]

✓ Created prompt "Code Reviewer v1"

Would you like me to:
1. Test it with a code sample?
2. Create a skill for reusable workflows?
3. Set up an A/B test with variants?

User: Test it with this code: [paste code]

Claude: Let me execute it:

[Calls promptly_execute_skill_real]

Here are the bugs found: [output]

I've also tracked this in analytics. Use promptly_analytics_prompt_stats
to see performance over time.
```

---

**Promptly Skills**: Prompt management, recursive intelligence, and analytics - all in Claude Desktop. 🚀