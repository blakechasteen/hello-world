# Promptly Execution - Quick Start

Get up and running with chains and loops in 5 minutes.

## Setup (One-Time)

### 1. Install Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

### 2. Pull a Model
```bash
ollama pull llama3.2:3b
```

### 3. Start Python Bridge
```bash
cd Promptly
python promptly/vscode_bridge.py
```

You should see:
```
Starting Promptly VS Code Bridge on http://localhost:8765
```

### 4. Open Execution Panel in VS Code
1. Click Promptly icon in activity bar
2. Click the **Play** (▶) button
3. The execution panel opens!

---

## Your First Execution

### Skill Execution (Simple)

1. **Select Mode**: Click "Skill" (⚡)
2. **Enter Details**:
   ```
   Skill Name: test_prompt
   User Input: Write a haiku about coding
   ```
3. **Execute**: Click "▶ Execute Skill"
4. **Watch**: Progress bar → Result appears

**Done!** You just executed your first skill.

---

## Your First Chain

### Example: Text → Summary → Bullets

1. **Select Mode**: Click "Chain" (🔗)
2. **Build Chain**:
   ```
   Skill 1: summarize_text
   Skill 2: extract_key_points
   ```
   *(Click "+ Add Skill" to add Skill 2)*
3. **Enter Input**:
   ```
   Odysseus faced the Cyclops and overcame his pride.
   After many trials, the journey home transformed him
   from warrior to wise king, teaching him patience
   and humility through suffering.
   ```
4. **Execute**: Click "▶ Execute Chain"
5. **Watch**:
   - Step 1/2: Summarizing...
   - Step 2/2: Extracting points...
   - Final output appears!

**Data Flow:**
```
Original Text → Summary → Key Points (bullets)
```

---

## Your First Loop

### Example: Iterative Essay Refinement

1. **Select Mode**: Click "Loop" (🔄)
2. **Configure**:
   ```
   Skill Name: improve_essay
   Loop Type: ⚡ Refine (click to select)
   Max Iterations: 5 (slider)
   Quality Threshold: 0.90 (slider)
   User Input: Write an essay about AI ethics
   ```
3. **Execute**: Click "▶ Execute Loop"
4. **Watch** real-time refinement:
   ```
   Iteration 1/5 (quality: 0.65) ▓▓▓▓░░░░░░
   Iteration 2/5 (quality: 0.74) ▓▓▓▓▓▓░░░░
   Iteration 3/5 (quality: 0.83) ▓▓▓▓▓▓▓▓░░
   Iteration 4/5 (quality: 0.91) ▓▓▓▓▓▓▓▓▓░ ✓ Done!
   ```

**Result**: Progressively improved essay!

---

## Loop Types Cheat Sheet

| Type | Icon | Use When | Example |
|------|------|----------|---------|
| **Refine** | ⚡ | Iterative improvement | Essay writing, code refactoring |
| **Critique** | 🔍 | Self-evaluation | Critical analysis, finding flaws |
| **Decompose** | 🧩 | Complex problems | Multi-part questions, structured analysis |
| **Verify** | ✓ | Correctness checks | Fact-checking, logical validation |
| **Explore** | 🌟 | Multiple solutions | Brainstorming, creative problems |
| **Hofstadter** | ∞ | Meta-reasoning | Philosophy, self-referential thinking |

---

## Common Patterns

### Pattern 1: Extract → Analyze → Generate
```
Chain: [extract_data, analyze_patterns, generate_insights]
Use: Data analysis workflows
```

### Pattern 2: Draft → Critique → Refine
```
Loop: Critique (3-4 iterations)
Use: High-quality content creation
```

### Pattern 3: Decompose → Solve → Synthesize
```
Loop: Decompose (2-3 iterations)
Use: Complex problem solving
```

### Pattern 4: Generate → Verify → Fix
```
Loop: Verify (5-7 iterations)
Use: Ensuring correctness
```

---

## Troubleshooting

**Problem: "Skill not found"**
```bash
# List available skills
cd Promptly
python -c "from promptly import Promptly; p = Promptly(); print([s['name'] for s in p.list_prompts()])"
```

**Problem: "Promptly not available"**
```bash
# Check bridge is running
curl http://localhost:8765/health

# Should return: {"status":"healthy","promptly_available":true}
```

**Problem: Execution hangs**
```bash
# Check Ollama is running
ollama list

# Test model directly
ollama run llama3.2:3b "Hello, world!"
```

**Problem: WebSocket not connecting**
- Restart Python bridge (Ctrl+C, then restart)
- Check VS Code Output panel for errors

---

## Next Steps

1. **Read Full Guide**: See [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)
2. **Create Custom Skills**: Use Promptly CLI
3. **Build Complex Chains**: Combine multiple skills
4. **Experiment with Loops**: Try different loop types
5. **Share Your Workflows**: Export and share chains

---

## Quick Tips

💡 **Tip 1**: Start with Skill mode to test individual prompts

💡 **Tip 2**: Keep chains to 3-5 skills for best performance

💡 **Tip 3**: Lower quality threshold (0.7-0.8) for faster loops

💡 **Tip 4**: Use Refine loop for most iterative improvement tasks

💡 **Tip 5**: Watch quality scores to tune your loop configuration

---

## Examples Library

### Example 1: Code Review Chain
```
Skills: [analyze_code, find_bugs, suggest_improvements]
Input: [Your code]
Output: Comprehensive review with suggestions
```

### Example 2: Creative Writing Loop
```
Type: Refine
Iterations: 7
Threshold: 0.85
Input: Story premise
Output: Polished narrative
```

### Example 3: Research Analysis
```
Type: Decompose
Iterations: 4
Input: Research question
Output: Structured analysis with citations
```

---

## Resources

- **Full Guide**: [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)
- **Promptly Docs**: [README.md](../Promptly/README.md)
- **VS Code Extension**: [README.md](README.md)
- **Demo Video**: *Coming soon*

---

**Ready to build? Open the execution panel and start creating! 🚀**
