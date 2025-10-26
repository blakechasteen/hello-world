# MCP Server Update - Composition & Analytics

## Summary

Updated the Promptly MCP server with 6 new tools for loop composition and prompt analytics, making these powerful features available directly in Claude Desktop.

## New Tools Added

### Loop Composition Tools

#### 1. `promptly_compose_loops`
Execute custom pipelines of multiple loop types in sequence.

**Parameters:**
- `task` (required): The task to process
- `steps` (required): Array of composition steps with:
  - `loop_type`: refine, critique, decompose, verify, explore, or hofstadter
  - `max_iterations`: Maximum iterations for this step
  - `description`: What this step does
- `backend` (optional): ollama or claude_api (default: ollama)

**Example:**
```json
{
  "task": "Optimize this SQL query: SELECT * FROM users WHERE active = 1",
  "steps": [
    {
      "loop_type": "critique",
      "max_iterations": 1,
      "description": "Identify query issues"
    },
    {
      "loop_type": "refine",
      "max_iterations": 3,
      "description": "Optimize based on critique"
    },
    {
      "loop_type": "verify",
      "max_iterations": 1,
      "description": "Verify optimization correctness"
    }
  ]
}
```

**Output:**
- Total steps and iterations
- Results for each pipeline step
- Final optimized output

#### 2. `promptly_decompose_refine_verify`
Common DRV pattern: Decompose problem → Refine solution → Verify correctness.

**Parameters:**
- `task` (required): The problem to solve
- `backend` (optional): ollama or claude_api

**Use Cases:**
- Problem solving
- Code optimization
- Essay writing
- Design tasks

### Analytics Tools

#### 3. `promptly_analytics_summary`
Get overall analytics for all prompt executions.

**Output:**
- Total executions
- Unique prompts
- Success rate
- Average execution time
- Total cost
- Average quality score

#### 4. `promptly_analytics_prompt_stats`
Get detailed statistics for a specific prompt.

**Parameters:**
- `prompt_name` (required): Name of the prompt

**Output:**
- Total executions
- Success rate
- Average execution time
- Average quality score
- Quality trend (improving/stable/degrading)
- Total cost
- Last executed timestamp

#### 5. `promptly_analytics_recommendations`
Get AI-powered recommendations to improve prompts based on analytics.

**Output:**
- List of recommendations with:
  - Prompt name
  - Issue identified
  - Recommended action
  - Priority level

**Recommendation Types:**
- High error rate → Review prompt clarity
- Slow execution → Simplify instructions
- Degrading quality → Update prompt for consistency
- Low usage → Consider archiving

#### 6. `promptly_analytics_top_prompts`
Get top-performing prompts by quality, speed, or cost efficiency.

**Parameters:**
- `metric` (optional): quality, speed, or cost_efficiency (default: quality)
- `limit` (optional): Number of prompts to return (default: 5)

**Output:**
- Ranked list of prompts
- Execution count and success rate
- Metric-specific values

## Integration Features

### Automatic Analytics Recording
Composed loops automatically record execution metrics:
- Quality scores from improvement history
- Total iterations and steps
- Execution metadata

### Error Handling
All tools gracefully handle:
- Missing dependencies
- Empty analytics database
- Invalid parameters
- Execution failures

## Usage in Claude Desktop

### Example 1: Optimize Code with Composition
```
Use promptly_compose_loops to optimize this Python function:
[paste code]

Steps:
1. Critique the code (1 iteration)
2. Refine based on critique (3 iterations)
3. Verify correctness (1 iteration)
```

### Example 2: Check Analytics
```
Use promptly_analytics_summary to see overall performance statistics
```

### Example 3: Get Recommendations
```
Use promptly_analytics_recommendations to see how I can improve my prompts
```

### Example 4: Find Best Prompts
```
Use promptly_analytics_top_prompts with metric="speed" to find my fastest prompts
```

## Technical Details

### New Imports
- `loop_composition`: LoopComposer, CompositionStep, CompositionResult
- `prompt_analytics`: PromptAnalytics, PromptExecution

### Global Instances
- `analytics`: Global PromptAnalytics instance for tracking executions

### Availability Flags
- `LOOP_COMPOSITION_AVAILABLE`: True if loop_composition module loaded
- `ANALYTICS_AVAILABLE`: True if prompt_analytics module loaded

## Total Tools Count

**Before:** 21 tools
**After:** 27 tools

## Files Modified

- `promptly/mcp_server.py` (1,660 lines)
  - Added 6 new tool definitions
  - Added 6 new tool handlers
  - Added imports for composition and analytics
  - Updated docstring

## What's Next

Users can now:
1. Build custom reasoning pipelines in Claude Desktop
2. Track prompt performance over time
3. Get data-driven recommendations
4. Identify top performers for reuse
5. Compose complex multi-stage workflows

All without leaving Claude Desktop!
