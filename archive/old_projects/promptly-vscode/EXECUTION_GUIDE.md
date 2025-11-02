# Promptly Execution Guide

## Overview

Promptly VS Code Extension now supports three powerful execution modes:
- **Skills**: Execute individual prompts/skills with user input
- **Chains**: Execute sequences of skills where output flows between steps
- **Loops**: Execute recursive reasoning loops with iterative refinement

## Getting Started

### Prerequisites

1. **Python Bridge Running**: The FastAPI bridge must be running on `localhost:8765`
   ```bash
   cd Promptly
   python promptly/vscode_bridge.py
   ```

2. **Ollama Installed**: Default backend uses Ollama for LLM execution
   - Download from https://ollama.ai
   - Pull a model: `ollama pull llama3.2:3b`

3. **Promptly Skills Created**: Create and manage skills using Promptly CLI

### Opening the Execution Panel

1. Click the Promptly icon in the VS Code activity bar
2. Click the **Play** (▶) button in the Prompt Library view title
3. Or use Command Palette: `Promptly: Open Execution Panel`

## Execution Modes

### 1. Skill Execution (⚡)

Execute a single skill with user input.

**Use Cases:**
- Test individual prompts
- Quick one-off completions
- Debugging prompt behavior

**How to Use:**
1. Select "Skill" mode
2. Enter the skill name (must exist in Promptly)
3. Enter your input text
4. Click "▶ Execute Skill"

**Example:**
```
Skill Name: summarize_text
User Input: Odysseus faced the Cyclops and overcame his pride.
            The journey home transformed him from warrior to wise king.
```

**Output:** Real-time progress bar → Final summary appears in output box

---

### 2. Chain Execution (🔗)

Execute multiple skills in sequence where each skill's output becomes the next skill's input.

**Use Cases:**
- Multi-step workflows (extract → analyze → generate)
- Progressive refinement pipelines
- Data transformation chains

**How to Use:**
1. Select "Chain" mode
2. Build your chain:
   - Enter first skill name
   - Click "+ Add Skill" for each additional step
   - Reorder or remove skills as needed
3. Enter initial input
4. Click "▶ Execute Chain"

**Example Chain:**
```
Skills:
  1. extract_entities
  2. analyze_relationships
  3. generate_knowledge_graph

Initial Input: [Your text to process]
```

**Output:** Shows progress through each step + final output after last skill

**Data Flow:**
```
Input → Skill 1 → Output 1 → Skill 2 → Output 2 → Skill 3 → Final Output
```

---

### 3. Loop Execution (🔄)

Execute recursive reasoning loops with iterative improvement.

**Use Cases:**
- Iterative refinement of outputs
- Self-critique and improvement
- Complex problem decomposition
- Quality-driven generation

**Loop Types:**

#### ⚡ Refine
Iteratively improve output based on quality scoring.

**Best For:** Text refinement, code improvement, argument strengthening

**How It Works:**
1. Generate initial output
2. Score quality (0-1)
3. If quality < threshold, refine and repeat
4. Stop when quality threshold reached or max iterations

**Example:**
```
Skill: improve_essay
Input: Write a compelling essay about AI ethics
Max Iterations: 5
Quality Threshold: 0.90
```

#### 🔍 Critique
Generate output → Self-critique → Improve based on critique

**Best For:** Critical analysis, finding flaws, comprehensive reviews

**How It Works:**
1. Generate initial response
2. Generate self-critique
3. Improve based on critique
4. Repeat until satisfied

#### 🧩 Decompose
Break problem into parts → Solve each → Combine results

**Best For:** Complex multi-part problems, structured analysis

**How It Works:**
1. Decompose problem into subproblems
2. Solve each subproblem
3. Synthesize solutions
4. Validate combined solution

#### ✓ Verify
Generate → Verify correctness → Fix issues → Repeat

**Best For:** Factual accuracy, logical consistency, code correctness

**How It Works:**
1. Generate solution
2. Verify against constraints/requirements
3. If issues found, fix and regenerate
4. Repeat until verification passes

#### 🌟 Explore
Try multiple approaches → Evaluate → Synthesize best ideas

**Best For:** Creative problems, multiple valid solutions, brainstorming

**How It Works:**
1. Generate N different approaches
2. Evaluate each approach
3. Synthesize best elements
4. Create final solution from synthesis

#### ∞ Hofstadter
Meta-level self-referential thinking (Strange Loops)

**Best For:** Philosophical problems, meta-reasoning, self-awareness

**How It Works:**
1. Reason about the reasoning process itself
2. Consider how thinking affects the problem
3. Apply meta-level insights
4. Iterate with self-referential awareness

---

## Real-Time Streaming

All execution modes support real-time progress updates via WebSocket.

**What You See:**
- ✅ Progress bar (0-100%)
- ✅ Current step description
- ✅ Iteration count (for loops)
- ✅ Quality score (for loops)
- ✅ Status indicator (running/completed/failed)

**Status Indicators:**
- 🔵 **Running**: Pulsing blue dot
- 🟢 **Completed**: Solid green dot
- 🔴 **Failed**: Solid red dot

---

## Configuration Options

### Backend Selection
- **Ollama** (default): Local LLM execution
- **Claude API**: Anthropic Claude (requires API key)
- **Custom**: Your own executor function

### Model Selection
Default: `llama3.2:3b`

Other popular models:
- `llama3.1:8b` - Larger, more capable
- `mistral:7b` - Fast and efficient
- `codellama:13b` - Optimized for code

### Loop Parameters

**Max Iterations** (1-10)
- Higher = more refinement
- Lower = faster completion
- Default: 5

**Quality Threshold** (0.5-1.0)
- Higher = stricter quality requirements
- Lower = faster completion
- Default: 0.9

---

## Examples

### Example 1: Document Analysis Chain

```
Mode: Chain
Skills: [extract_key_points, analyze_sentiment, generate_summary]
Input: [Long document text]

Flow:
1. extract_key_points → Bullet list of main points
2. analyze_sentiment → Sentiment analysis of each point
3. generate_summary → Final summary with sentiment context
```

### Example 2: Code Refactoring Loop

```
Mode: Loop (Refine)
Skill: refactor_code
Input: [Messy code snippet]
Max Iterations: 7
Quality Threshold: 0.85

Iterations:
1. Quality: 0.65 - Basic cleanup, remove duplicates
2. Quality: 0.72 - Extract functions, improve naming
3. Quality: 0.81 - Add type hints, docstrings
4. Quality: 0.87 - Final polish → DONE (threshold reached)
```

### Example 3: Multi-Perspective Analysis

```
Mode: Loop (Explore)
Skill: analyze_decision
Input: Should we pivot our startup's business model?
Max Iterations: 5

Iterations:
1. Generate 5 different perspectives (financial, customer, technical, team, market)
2. Evaluate pros/cons of each perspective
3. Synthesize insights across perspectives
4. Create comprehensive recommendation
```

---

## Troubleshooting

### "Promptly not available" Error
- Ensure Python bridge is running: `python promptly/vscode_bridge.py`
- Check bridge health: http://localhost:8765/health

### "Skill not found" Error
- Verify skill exists: `promptly list`
- Check skill name spelling (case-sensitive)

### Execution Hangs/No Progress
- Check Ollama is running: `ollama list`
- Verify model is pulled: `ollama pull llama3.2:3b`
- Check Python bridge logs for errors

### WebSocket Not Connecting
- Bridge may need restart
- Check firewall settings for localhost:8765
- Extension will retry automatically (5 attempts)

### Quality Threshold Never Reached
- Lower quality threshold to 0.7-0.8
- Increase max iterations
- Check if skill produces scorable output

---

## Best Practices

### Skill Design
✅ **DO:**
- Write clear, specific prompts
- Include examples in skill content
- Define expected output format
- Handle edge cases

❌ **DON'T:**
- Make prompts too vague
- Assume context the skill doesn't have
- Chain incompatible skill outputs

### Chain Design
✅ **DO:**
- Keep chains focused (3-5 steps max)
- Ensure output formats match input expectations
- Test skills individually first
- Document expected data flow

❌ **DON'T:**
- Create circular dependencies
- Chain skills with incompatible formats
- Make chains too long (slower, more fragile)

### Loop Configuration
✅ **DO:**
- Start with lower iterations (3-5)
- Use appropriate loop type for task
- Monitor quality scores to tune threshold
- Set reasonable timeouts

❌ **DON'T:**
- Set max iterations too high (expensive)
- Use overly strict quality thresholds (may never reach)
- Use refine loop for exploration tasks

---

## Performance Tips

1. **Use Faster Models for Iteration**
   - Development: `llama3.2:3b` (fastest)
   - Production: `llama3.1:8b` (balanced)
   - High-quality: Claude API (best quality)

2. **Optimize Loop Iterations**
   - Start low (3-4 iterations)
   - Increase only if needed
   - Monitor quality improvement per iteration

3. **Cache Intermediate Results**
   - Promptly automatically versions results
   - Use `promptly history` to review past executions

4. **Parallel Execution**
   - Run multiple independent tasks in separate panels
   - WebSocket handles concurrent executions

---

## Advanced: Custom Executors

You can create custom execution backends beyond Ollama and Claude API.

**Python Bridge Extension:**
```python
# In vscode_bridge.py
@app.post("/execute/custom")
async def execute_custom(request: CustomExecutionRequest):
    # Your custom logic here
    pass
```

**TypeScript Client Extension:**
```typescript
// In ExecutionClient.ts
async executeCustom(request: CustomRequest): Promise<ExecutionResponse> {
    return await this.client.post('/execute/custom', request);
}
```

---

## API Reference

### REST Endpoints

#### POST /execute/skill
Execute a single skill.

**Request:**
```json
{
  "skill_name": "summarize_text",
  "user_input": "Your input here...",
  "backend": "ollama",
  "model": "llama3.2:3b"
}
```

**Response:**
```json
{
  "execution_id": "uuid-here",
  "status": "queued",
  "message": "Execution started"
}
```

#### POST /execute/chain
Execute a chain of skills.

**Request:**
```json
{
  "skill_names": ["skill1", "skill2", "skill3"],
  "initial_input": "Starting input...",
  "backend": "ollama",
  "model": "llama3.2:3b"
}
```

#### POST /execute/loop
Execute a recursive loop.

**Request:**
```json
{
  "skill_name": "refine_output",
  "user_input": "Input to refine...",
  "loop_type": "refine",
  "max_iterations": 5,
  "quality_threshold": 0.9,
  "backend": "ollama",
  "model": "llama3.2:3b"
}
```

#### GET /execute/status/{execution_id}
Get execution status.

**Response:**
```json
{
  "execution_id": "uuid",
  "status": "running",
  "progress": 0.6,
  "current_step": "Iteration 3/5 (quality: 0.78)",
  "output": null,
  "metadata": {
    "iterations": 3,
    "improvement_history": [0.65, 0.72, 0.78]
  }
}
```

#### WebSocket /ws/execution
Real-time execution events.

**Events:**
- `status_update`: Progress and step updates
- `iteration_update`: Loop iteration progress
- `completed`: Final results
- `failed`: Error information

---

## Integration with HoloLoom

Promptly execution integrates with HoloLoom's narrative intelligence system:

- **Memory Storage**: Execution results stored in knowledge graph
- **Learning**: Loop performance analytics inform future executions
- **Context**: Retrieved similar past executions for meta-learning
- **Provenance**: Full execution trace for debugging

See `HoloLoom/integrations/hololoom_bridge.py` for details.

---

## Keyboard Shortcuts

*Coming in v1.1:*
- `Ctrl+Shift+E`: Open Execution Panel
- `Ctrl+Enter`: Execute current mode
- `Escape`: Stop running execution

---

## Roadmap

**v1.1 (Next Release):**
- [ ] Execution history viewer
- [ ] Skill templates library
- [ ] Chain/loop save & load
- [ ] Performance analytics dashboard
- [ ] Claude API integration
- [ ] Keyboard shortcuts

**v1.2 (Future):**
- [ ] Visual chain composer (drag & drop)
- [ ] Loop reasoning scratchpad viewer
- [ ] A/B testing framework
- [ ] Collaborative execution (team sharing)
- [ ] Export execution reports

---

## Contributing

Found a bug? Have a feature request?
- GitHub Issues: https://github.com/anthropics/promptly/issues
- Discussions: https://github.com/anthropics/promptly/discussions

---

## License

MIT License - See LICENSE file for details.

---

**Generated with ⚡ Promptly Execution Engine**
