# Impressive Promptly Demos for Claude Desktop

## Setup
1. Make sure Claude Desktop is running
2. Restart it to load the Promptly MCP server
3. The server should auto-connect (check Claude Desktop settings)

---

## Demo 1: Hofstadter Strange Loop - "What is creativity?"

**Prompt to Claude Desktop:**
```
Use the promptly_hofstadter_loop tool to think deeply about: "What is creativity?"

Run 4 meta-levels and show me the complete thought progression.
```

**What makes this impressive:**
- Each level thinks about the previous level's thinking
- Creates self-referential insights (strange loops)
- Shows how meta-reasoning reveals hidden assumptions
- Final synthesis integrates all levels of abstraction

**Expected output:** Deep philosophical analysis showing:
- Level 1: Direct answer
- Level 2: Thinking about that answer
- Level 3: Thinking about thinking (meta-cognition)
- Level 4: Self-referential synthesis

---

## Demo 2: Iterative Refinement - Code Review

**Prompt to Claude Desktop:**
```
Use the promptly_refine_iteratively tool to improve this code:

```python
def calc(x, y):
    return x + y * 2
```

Task: "Make this code production-ready with proper naming, documentation, type hints, and error handling"

Run up to 5 iterations with Ollama backend.
```

**What makes this impressive:**
- Automatically critiques and improves code
- Tracks quality scores at each iteration
- Shows complete scratchpad of reasoning
- Stops when quality threshold reached or no improvement

**Expected output:**
- Iteration 1: Add docstring and type hints
- Iteration 2: Better function name, add validation
- Iteration 3: Error handling, edge cases
- Final: Production-ready code with quality score

---

## Demo 3: Multi-Prompt A/B Testing

**Prompt to Claude Desktop:**
```
Use the promptly_ab_test tool to compare these two prompts:

Variant A: "Explain quantum entanglement"
Variant B: "Explain quantum entanglement to a curious 12-year-old using everyday analogies"

Test on 3 test cases:
1. "What is quantum entanglement?"
2. "How does it work?"
3. "Why is it important?"

Use Ollama backend and determine the winner.
```

**What makes this impressive:**
- Scientific A/B testing of prompts
- Multiple evaluation metrics (quality, clarity, length)
- Statistical winner determination
- Shows concrete improvement percentages

**Expected output:**
- Variant A vs B results for each test
- Quality scores, latency, token counts
- Winner: Variant B (likely - simpler language wins)
- Confidence score and improvement percentage

---

## Demo 4: Recursive Prompt Engineering

**Prompt to Claude Desktop:**
```
First, use promptly_suggest to get advice on: "Writing a prompt for code review"

Then, create that prompt using promptly_add_prompt with the suggested improvements.

Finally, use promptly_refine_iteratively to improve it further.
```

**What makes this impressive:**
- Chains multiple AI tools together
- Uses AI to improve AI prompts (meta!)
- Shows complete workflow from idea → refinement
- Demonstrates full platform capabilities

**Expected workflow:**
1. AI advisor suggests best practices
2. Create initial prompt with suggestions
3. Iteratively refine until excellent
4. Final prompt is battle-tested

---

## Demo 5: Hofstadter Loop on AI Itself

**Prompt to Claude Desktop:**
```
Use promptly_hofstadter_loop with 5 levels to analyze:

"Can an AI truly understand the concept of understanding?"

This is a self-referential question about AI thinking about AI thinking.
```

**What makes this impressive:**
- Deeply philosophical
- Self-referential strange loop (AI thinking about AI)
- Shows emergence of insight at higher meta-levels
- Perfect demonstration of Hofstadter's GEB concepts

**Expected output:**
- Level 1: Direct answer about AI understanding
- Level 2: Realizes the paradox of AI discussing understanding
- Level 3: Strange loop appears - the discussion IS the understanding
- Level 4: Meta-insight about self-reference
- Level 5: Deep synthesis revealing the recursive nature

---

## Demo 6: Package Export/Import Workflow

**Prompt to Claude Desktop:**
```
1. Use promptly_install_template to install the "code_reviewer" skill template

2. Use promptly_execute_skill to run it on some sample code

3. Use promptly_export_package to create a shareable .promptly file with:
   - The code_reviewer skill
   - Any prompts created during review

4. Show me the package contents
```

**What makes this impressive:**
- Complete workflow from template → execution → sharing
- Shows skills system in action
- Creates shareable package
- Professional collaboration features

---

## Demo 7: Cost-Tracked Execution Chain

**Prompt to Claude Desktop:**
```
Execute this chain and track costs:

1. Use promptly_execute_prompt with Claude API: "Write a Python function for binary search"

2. Use promptly_refine_iteratively to improve it (3 iterations)

3. Use promptly_get_costs to show me total API cost breakdown

Calculate cost per iteration and total tokens used.
```

**What makes this impressive:**
- Real API cost tracking
- Per-token pricing analysis
- Shows ROI of iterative improvement
- Professional cost management

---

## Demo 8: Hofstadter Loop - "What is a strange loop?"

**Prompt to Claude Desktop:**
```
Use promptly_hofstadter_loop to answer the question: "What is a strange loop?"

This is perfectly meta - using a strange loop to explain strange loops!

Run 4 levels.
```

**What makes this impressive:**
- Ultimate meta-demonstration
- Tool explains itself by being itself
- Beautiful recursive self-reference
- Shows Hofstadter's concepts in action

**Expected output:**
- Level 1: Definition of strange loops
- Level 2: Realizes THIS IS a strange loop
- Level 3: The explanation becomes what it's explaining
- Level 4: Complete self-referential synthesis

---

## Demo 9: Multi-Iteration Code Evolution

**Prompt to Claude Desktop:**
```
Use promptly_refine_iteratively to evolve this code through multiple iterations:

Initial code:
```python
def f(n):
    if n < 2: return n
    return f(n-1) + f(n-2)
```

Task: "Transform this into a production-ready Fibonacci implementation with memoization, type hints, docstrings, and comprehensive error handling"

Show me the scratchpad of improvements at each iteration.
```

**What makes this impressive:**
- Dramatic visible improvement
- Clear quality progression
- Complete thought process visible
- Shows AI learning and improving

**Expected evolution:**
- Iteration 1: Add memoization
- Iteration 2: Type hints and docstring
- Iteration 3: Error handling and validation
- Iteration 4: Performance optimization
- Final: Production-ready with quality score 0.9+

---

## Demo 10: The Ultimate Meta Test

**Prompt to Claude Desktop:**
```
Use promptly_hofstadter_loop (4 levels) to think about:

"Can recursive self-improvement lead to artificial general intelligence?"

Then use promptly_refine_iteratively (3 iterations) to improve the final synthesis.

Show both the meta-level thinking AND the iterative refinement of the conclusion.
```

**What makes this impressive:**
- Combines TWO recursive systems
- Deep philosophical + practical refinement
- Shows full platform power
- Ultimate demonstration of recursive intelligence

**Expected flow:**
1. Hofstadter loop explores meta-levels of AGI recursion
2. Each level reveals deeper insights about self-improvement
3. Final synthesis is then iteratively refined
4. Result: Deeply considered, well-polished answer

---

## Demo 11: Skill Template Showcase

**Prompt to Claude Desktop:**
```
Show me all available skill templates with promptly_list_templates

Then install and execute the "prompt_engineer" template to create a prompt for:
"Analyzing code for security vulnerabilities"

Finally, use that generated prompt to analyze some sample code.
```

**What makes this impressive:**
- Shows 8 professional templates
- Template generates custom prompt
- Prompt is immediately useful
- Complete meta-workflow

---

## Demo 12: Comparative Analysis with A/B Testing

**Prompt to Claude Desktop:**
```
Use promptly_ab_test to scientifically compare:

Variant A (Simple): "List 5 Python best practices"
Variant B (Detailed): "List 5 Python best practices with code examples, explanations of why they matter, and common mistakes to avoid"

Test cases:
1. "Error handling"
2. "Code organization"
3. "Performance"

Determine which prompt produces better results.
```

**What makes this impressive:**
- Scientific prompt comparison
- Multiple evaluation dimensions
- Clear winner with evidence
- Shows value of detailed prompts

---

## Quick Fire Demos (1-2 minutes each)

### Demo A: Instant Skill Creation
```
Use promptly_add_skill to create a skill named "json_validator" that validates JSON structure.
Then execute it on some sample JSON.
```

### Demo B: Prompt Branching
```
Use promptly_add_prompt to create a prompt for "API design"
Then use promptly_diff_prompts to compare it with the built-in api_designer template
```

### Demo C: Cost Analysis
```
Execute 5 different prompts with Claude API
Then use promptly_get_costs to show total spending and cost per prompt
```

### Demo D: Strange Loop Inception
```
Use promptly_hofstadter_loop to think about:
"What happens when you use a Hofstadter loop to think about Hofstadter loops?"
```

---

## Most Impressive Single Demo

**THE RECURSION SHOWCASE:**

```
Use promptly_hofstadter_loop (5 levels) to deeply analyze:

"Is consciousness a strange loop?"

This combines:
- Hofstadter's core thesis from Gödel, Escher, Bach
- Self-referential reasoning about consciousness
- AI thinking about thinking about thinking
- Perfect demonstration of recursive intelligence

Request the full scratchpad to see each meta-level's insights.
```

**Why this is the best:**
- Philosophical depth
- Perfect use case for strange loops
- Shows emergence of understanding
- Each level builds beautifully on previous
- Final synthesis is genuinely insightful
- Demonstrates the "I Am a Strange Loop" concept

---

## Pro Tips for Impressive Demos

1. **Always request the scratchpad** - Shows the thinking process
2. **Use 4-5 meta-levels for Hofstadter** - Sweet spot for deep insights
3. **Self-referential questions** - "Can AI be creative?" asked TO an AI
4. **Compare iterative improvement** - Show before/after quality scores
5. **Chain multiple tools** - Advisor → Create → Refine → Test
6. **Meta questions about the tool** - "Can recursive loops solve recursive problems?"

---

## Expected Performance

- **Hofstadter loops (4 levels):** ~25-35 seconds with Ollama
- **Refinement (3 iterations):** ~15-25 seconds with Ollama
- **A/B testing:** ~30-45 seconds with Ollama
- **Claude API:** Faster but costs ~$0.01-0.05 per request

---

## Troubleshooting

If tools aren't available:
1. Check Claude Desktop settings → MCP Servers
2. Restart Claude Desktop
3. Check `C:\Users\blake\AppData\Roaming\Claude\claude_desktop_config.json`
4. Look for "promptly" server entry

If Ollama backend fails:
- Install Ollama: https://ollama.ai
- Run: `ollama pull llama3.2:3b`
- Ensure Ollama is running

---

**Start with Demo 8 or Demo 10 for maximum impact!** 🚀
