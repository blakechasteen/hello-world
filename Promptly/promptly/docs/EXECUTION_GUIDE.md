# Promptly Execution Guide

Quick reference for executing skills and prompts with real LLMs.

## Quick Start

### Execute a Skill (Ollama - Free!)
```
User (in Claude Desktop):
"Execute the code_reviewer skill with this code:
def calculate(x, y):
    return x + y
"

Claude will:
1. Use promptly_execute_skill_real tool
2. Run with Ollama (local, free)
3. Return code review from llama3.2:3b
```

### Execute a Prompt
```
User:
"Execute my 'summarizer' prompt with input='Long article text...'"

Claude will:
1. Get the prompt from database
2. Format with your input
3. Execute with Ollama
4. Return summary
```

### A/B Test Variants
```
User:
"A/B test 'summarizer_v1' vs 'summarizer_v2' on these articles:
- Article 1 text
- Article 2 text
- Article 3 text"

Claude will:
1. Run both variants on all inputs
2. Compare metrics (latency, quality)
3. Declare a winner
4. Show confidence score
```

## MCP Tools Reference

### promptly_execute_skill_real
Actually execute a skill with a real LLM.

**Parameters:**
- `skill_name` (required) - Name of skill to execute
- `user_input` (required) - Task description
- `backend` (optional) - "ollama" (default) or "claude_api"
- `model` (optional) - Model name (default: llama3.2:3b for Ollama)
- `api_key` (optional) - API key for Claude backend

**Example:**
```json
{
  "skill_name": "code_reviewer",
  "user_input": "Review this Python function: def add(a, b): return a + b",
  "backend": "ollama",
  "model": "llama3.2:3b"
}
```

**Returns:**
- Backend used
- Model name
- Success status
- Execution time
- Token count (if available)
- LLM output or error

### promptly_execute_prompt
Execute any prompt with variable substitution.

**Parameters:**
- `prompt_name` (required) - Name of prompt to execute
- `inputs` (optional) - Object with variables to substitute
- `backend` (optional) - "ollama" or "claude_api"
- `model` (optional) - Model name

**Example:**
```json
{
  "prompt_name": "summarizer",
  "inputs": {
    "text": "Long article to summarize...",
    "max_length": "100 words"
  },
  "backend": "ollama"
}
```

**Returns:**
- Execution results
- Formatted output
- Performance metrics

### promptly_ab_test
Compare prompt variants with A/B testing.

**Parameters:**
- `test_name` (required) - Name for this test
- `variants` (required) - Array of prompt names
- `test_inputs` (required) - Array of test inputs
- `backend` (optional) - Execution backend

**Example:**
```json
{
  "test_name": "summarizer_comparison",
  "variants": ["summarizer_v1", "summarizer_v2"],
  "test_inputs": [
    "First test article...",
    "Second test article...",
    "Third test article..."
  ],
  "backend": "ollama"
}
```

**Returns:**
- Full A/B test report
- Winner declaration
- Confidence score
- Metrics per variant
- Detailed comparison

## Backends

### Ollama (Local, Free)
**Pros:**
- ✅ Free, unlimited usage
- ✅ Privacy (runs locally)
- ✅ No API keys needed
- ✅ Works offline

**Cons:**
- ❌ Requires installation
- ❌ Uses local compute
- ❌ Smaller models available

**Installation:**
```bash
# Download from https://ollama.ai
# Or use winget:
winget install Ollama.Ollama

# Pull a model:
ollama pull llama3.2:3b
```

**Recommended Models:**
- `llama3.2:3b` - General purpose (default)
- `llama3.2:1b` - Fastest
- `qwen2.5-coder:3b` - Code tasks
- `llama3.1:8b` - Higher quality

### Claude API (Cloud)
**Pros:**
- ✅ Highest quality output
- ✅ No local compute needed
- ✅ Large context windows
- ✅ Multimodal support

**Cons:**
- ❌ Requires API key
- ❌ Costs money per token
- ❌ Requires internet

**Setup:**
```python
# Get API key from https://console.anthropic.com
# Pass as parameter:
{
  "skill_name": "code_reviewer",
  "user_input": "...",
  "backend": "claude_api",
  "model": "claude-3-5-sonnet-20241022",
  "api_key": "sk-ant-..."
}
```

**Recommended Models:**
- `claude-3-5-sonnet-20241022` - Best overall (default)
- `claude-3-haiku` - Fast & economical
- `claude-3-opus` - Maximum quality

## Common Workflows

### Workflow 1: Develop a Skill
```
1. "Install the code_reviewer template"
2. "Execute code_reviewer on sample code"
3. Review output
4. "Modify the review_checklist.md file"
5. Execute again to see improvements
```

### Workflow 2: Compare Prompt Variants
```
1. Create variants:
   "Add prompt summarizer_short: Summarize in 50 words: {text}"
   "Add prompt summarizer_detailed: Provide detailed summary: {text}"

2. A/B test:
   "A/B test summarizer_short vs summarizer_detailed on:
    - Sample article 1
    - Sample article 2
    - Sample article 3"

3. Review winner:
   Claude shows which variant performed better

4. Use winner:
   "Execute [winner] on new content"
```

### Workflow 3: Build a Chain
```python
# In Python (not Claude Desktop yet):
from promptly import Promptly
from execution_engine import ExecutionEngine, ChainExecutor, ExecutionConfig

p = Promptly()
engine = ExecutionEngine(ExecutionConfig())
chain = ChainExecutor(engine)

# Get skills
skills = [
    p.prepare_skill_payload("api_designer"),
    p.prepare_skill_payload("sql_designer"),
    p.prepare_skill_payload("code_reviewer")
]

# Execute chain
results = chain.execute_chain(skills, "Design a blog platform")

# results[0] = API design
# results[1] = DB schema for that API
# results[2] = Code review of both
```

## A/B Testing Metrics

### Accuracy
- Success rate (% of non-error executions)
- Range: 0.0 to 1.0
- Higher is better

### Latency
- Average execution time in seconds
- Includes LLM processing time
- Lower is better

### Token Efficiency
- Average tokens used per execution
- Only available for Claude API
- Lower is usually better (cheaper)

### Quality Score
- Optional, requires evaluator function
- Range: 0.0 to 1.0
- Higher is better

### Winner Determination
- Primary metric (default: quality_score)
- Minimum improvement threshold (default: 5%)
- Confidence based on margin of victory

## Built-in Evaluators

### exact_match_evaluator
Perfect string match (1.0 or 0.0).
```python
score = exact_match_evaluator(output, expected)
# Returns 1.0 if exact match, 0.0 otherwise
```

### contains_evaluator
Check if output contains expected string.
```python
score = contains_evaluator(output, "expected phrase")
# Returns 1.0 if found, 0.0 otherwise
```

### word_overlap_evaluator
Score based on word overlap (0.0 to 1.0).
```python
score = word_overlap_evaluator(output, expected)
# Returns % of expected words found in output
```

### length_similarity_evaluator
Score based on length similarity.
```python
score = length_similarity_evaluator(output, expected)
# Returns ratio of min/max length
```

## Programmatic Usage

### Execute a Skill
```python
from execution_engine import execute_with_ollama

result = execute_with_ollama(
    prompt="Review this code: def add(a, b): return a + b",
    model="llama3.2:3b"
)

print(f"Success: {result.success}")
print(f"Output: {result.output}")
print(f"Time: {result.execution_time}s")
```

### Run A/B Test
```python
from promptly import Promptly
from execution_engine import ExecutionEngine, ExecutionConfig
from ab_testing import ABTestRunner, TestCase

p = Promptly()
engine = ExecutionEngine(ExecutionConfig())
runner = ABTestRunner(p, engine)

# Create test
test_cases = [
    TestCase(input="First article..."),
    TestCase(input="Second article..."),
]

test = runner.create_test(
    test_name="summarizer_test",
    variants=["summarizer_v1", "summarizer_v2"],
    test_cases=test_cases
)

# Run
result = runner.run_test(test)

# Get report
print(result.to_report())
```

### Execute a Chain
```python
from execution_engine import ChainExecutor

chain = ChainExecutor(engine)

skills = [
    p.prepare_skill_payload("skill1"),
    p.prepare_skill_payload("skill2"),
]

results = chain.execute_chain(skills, "Initial input")

for i, r in enumerate(results):
    print(f"Step {i+1}: {r.success} in {r.execution_time}s")
```

## Error Handling

All execution errors return structured results:
```python
ExecutionResult(
    skill_name="...",
    success=False,
    output="",
    error="Error description",
    execution_time=0.0,
    model="...",
    backend="..."
)
```

Common errors:
- **Ollama not found** - Install from https://ollama.ai
- **Model not available** - Run `ollama pull model-name`
- **API key missing** - Provide api_key parameter
- **Timeout** - Increase timeout in config
- **Network error** - Check internet connection

## Performance Tips

1. **Use Ollama for development** - Free & fast iteration
2. **Switch to Claude for production** - Higher quality
3. **Cache similar requests** - Avoid redundant executions
4. **Use smaller models** - Faster for simple tasks
5. **Batch test cases** - Run A/B tests in bulk
6. **Set timeouts** - Prevent hanging executions

## Troubleshooting

### Skill won't execute
```
✗ Check if skill exists: "Get skill details for skill_name"
✗ Check if Ollama is running: Run `ollama list` in terminal
✗ Check model is downloaded: Run `ollama pull llama3.2:3b`
```

### A/B test returns no winner
```
✗ Variants perform similarly (< 5% difference)
✗ Try more test cases for clearer signal
✗ Use different primary metric
✗ Lower minimum improvement threshold
```

### Execution times out
```
✗ Increase timeout in config (default: 120s)
✗ Use smaller model
✗ Simplify prompt
✗ Check system resources
```

## Best Practices

1. **Start with templates** - Use built-in skill templates
2. **Test locally first** - Use Ollama before Claude API
3. **Run A/B tests** - Don't guess, measure
4. **Track metrics** - Monitor latency and tokens
5. **Iterate rapidly** - Quick execution cycle
6. **Version prompts** - Use branches for experiments
7. **Document results** - Save A/B test reports

## Next Steps

1. **Try execution** - "Execute code_reviewer on sample code"
2. **Run A/B test** - Compare two prompt variants
3. **Build chains** - Combine multiple skills
4. **Track metrics** - Monitor performance over time
5. **Share results** - Export successful prompts

---

**Need help?** Check:
- [EXECUTION_GUIDE.md](./EXECUTION_GUIDE.md) - This file
- [SKILL_TEMPLATES.md](./SKILL_TEMPLATES.md) - Template library
- [MCP_SETUP.md](./MCP_SETUP.md) - MCP configuration
- [PHASE2_COMPLETE.md](../PROMPTLY_PHASE2_COMPLETE.md) - Technical details
