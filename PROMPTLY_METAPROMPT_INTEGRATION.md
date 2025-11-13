# Promptly Meta-Prompting Integration Design

**Date:** November 10, 2025
**Status:** Design Phase
**Goal:** Integrate GPT-5 meta-prompting as a proto-LLM preprocessing layer

---

## Executive Summary

Add **meta-prompting as a first-class feature** in Promptly:
- Proto-LLM call preprocesses casual prompts → structured prompts
- Built-in `meta-prompt` skill implementing 7-component framework
- Optional auto-enhancement for all LLM calls
- Tracks improvement metrics (before/after quality)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Promptly Flow                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  User Input: "help me prepare for tomorrow's meeting"  │
│       ↓                                                 │
│  [1. Proto-LLM Call] ← Meta-Prompt Skill               │
│       ↓                                                 │
│  Structured Brief:                                      │
│    • Role: Meeting preparation consultant              │
│    • Objective: Create actionable prep plan            │
│    • Process: Clarify → Structure → Surface points     │
│    • Format: Meeting prep sheet                        │
│    • Constraints: Ask questions if unclear             │
│       ↓                                                 │
│  [2. Main LLM Call] ← Actual execution                 │
│       ↓                                                 │
│  Enhanced Result + Quality Metrics                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Component 1: Proto-LLM Call

**File:** `promptly/execution_engine.py`

**New Method:**
```python
async def proto_llm_call(
    self,
    user_prompt: str,
    config: ExecutionConfig,
    meta_prompt_template: str = None
) -> tuple[str, dict]:
    """
    Preprocessing step: expand casual prompt into structured prompt.

    Args:
        user_prompt: User's raw/casual input
        config: Execution config (model, backend, etc.)
        meta_prompt_template: Optional custom template

    Returns:
        tuple of (structured_prompt, metadata)

    Example:
        >>> proto = ExecutionEngine()
        >>> enhanced, meta = await proto.proto_llm_call(
        ...     "help me prepare for meeting",
        ...     config
        ... )
        >>> print(enhanced)
        '''
        Role: Meeting preparation consultant
        Objective: Create actionable meeting prep plan

        Context: User needs help preparing for a meeting

        Please:
        1. Clarify the meeting type, attendees, desired outcome
        2. Structure preparation into clear sections
        3. Surface 2-3 likely talking points

        Format: Meeting prep sheet with context, message, questions

        If information is insufficient, ask specific questions.
        '''
    """
    # Load meta-prompt skill
    if meta_prompt_template is None:
        meta_prompt_template = self._load_default_metaprompt()

    # Format template with user's casual prompt
    preprocessing_prompt = meta_prompt_template.format(
        user_request=user_prompt
    )

    # Execute with fast/cheap model (Haiku for speed)
    preprocessing_config = ExecutionConfig(
        backend=config.backend,
        model="claude-3-5-haiku-20241022",  # Fast, cheap
        max_tokens=2048,
        temperature=0.3  # Lower temp for structured output
    )

    start_time = time.time()
    structured_prompt = await self.execute_prompt(
        preprocessing_prompt,
        preprocessing_config
    )
    preprocessing_time = time.time() - start_time

    # Extract metadata
    metadata = {
        "preprocessing_time": preprocessing_time,
        "original_prompt": user_prompt,
        "enhancement": "meta-prompt",
        "template_version": "gpt5_v1"
    }

    return structured_prompt, metadata
```

**Key Features:**
- Uses **Haiku** for speed/cost (preprocessing should be fast)
- Returns both structured prompt + metadata
- Template customizable per use case
- Tracks preprocessing time for analytics

---

## Component 2: Meta-Prompt Skill

**File:** `promptly/skills/meta_prompt/`

**Structure:**
```
meta_prompt/
├── skill.yaml          # Skill metadata
├── README.md           # Documentation
├── template_gpt5.md    # GPT-5 optimized template
└── examples.json       # Example transformations
```

**skill.yaml:**
```yaml
name: meta-prompt
version: 1.0.0
description: Transform casual prompts into structured GPT-5 ready prompts
author: Promptly Team
tags: [meta, preprocessing, gpt5, structure]

config:
  model: claude-3-5-haiku-20241022
  temperature: 0.3
  max_tokens: 2048

components:
  - role_extraction
  - objective_clarification
  - process_methodology
  - format_specification
  - constraint_handling
  - uncertainty_protocols
  - validation_criteria
```

**template_gpt5.md:**
```markdown
# Meta-Prompt Template (GPT-5 Optimized)

## Instructions

Transform the following user request into a comprehensive, structured prompt:

**User Request:** {user_request}

## Your Task

### Step 1: Interpret the Request
Analyze what the user is actually asking for:
- **Type of output**: What would help them most?
- **Expertise needed**: What domain knowledge is relevant?
- **Format preference**: What structure makes sense?
- **Detail level**: How deep should the response go?

### Step 2: Structure the Prompt

Create a structured prompt with these 7 components:

#### 1. Role (Expertise Routing)
Define the expert role needed:
```
Role: [specific expertise]
```

#### 2. Objective Framework
State the primary goal explicitly:
```
Objective: [clear, specific goal]
Primary: [main objective]
Secondary: [supporting objective]
When in doubt, prioritize: [priority]
```

#### 3. Process Methodology
Provide step-by-step methodology:
```
Process:
1. [first step]
2. [second step]
3. [third step]
```

#### 4. Format Expectations
Specify output structure:
```
Format: [description of expected output]
Structure:
- [section 1]
- [section 2]
- [section 3]
```

#### 5. Boundaries & Limitations
Define constraints:
```
Constraints:
- Do NOT [anti-pattern 1]
- Avoid [anti-pattern 2]
- Limit to [constraint]
```

#### 6. Uncertainty Handling
Specify fallback behavior:
```
If unclear or insufficient data:
- Ask: [specific questions]
- Do NOT: assume or fabricate
- Instead: [fallback behavior]
```

#### 7. Validation Criteria
Define success metrics:
```
Check your output for:
✓ [criterion 1]
✓ [criterion 2]
✓ [criterion 3]
```

### Step 3: Output

Provide:
1. The structured prompt (all 7 components)
2. Brief justification for your choices
3. Any clarifying questions if request is ambiguous

---

## Output Format

```yaml
structured_prompt: |
  [Full structured prompt with all 7 components]

justification: |
  [Why you chose this structure]

clarifying_questions:
  - [question 1 if needed]
  - [question 2 if needed]
```
```

**examples.json:**
```json
{
  "examples": [
    {
      "casual": "help me prepare for tomorrow's meeting",
      "structured": "Role: Meeting preparation consultant with expertise in stakeholder management\n\nObjective:\nPrimary: Create a concrete, actionable meeting preparation plan\nSecondary: Anticipate likely discussion points and objections\nWhen in doubt, prioritize: Asking clarifying questions over assumptions\n\nProcess:\n1. Clarify the meeting context (type, attendees, desired outcome)\n2. Structure preparation into manageable sections\n3. Surface 2-3 likely discussion points based on context\n\nFormat: Meeting prep sheet\nStructure:\n- Context recap (what we know)\n- Core message (what to communicate)\n- Key questions to ask\n- Anticipated objections\n- Next steps\n\nConstraints:\n- Do NOT fabricate details about attendees or context\n- Avoid generic advice disconnected from specific situation\n- Limit assumptions - ask when unclear\n\nIf unclear or insufficient data:\n- Ask: What kind of meeting? Who's attending? What outcome do you need?\n- Do NOT: Assume meeting type, fabricate stakeholder concerns\n- Instead: Provide template with blanks to fill\n\nCheck your output for:\n✓ Specific, actionable items\n✓ Questions asked when context missing\n✓ No fabricated facts or assumptions\n",
      "quality_improvement": 0.73
    },
    {
      "casual": "write a Python function",
      "structured": "Role: Senior Python developer with expertise in clean code and best practices\n\nObjective:\nPrimary: Write a well-documented, tested Python function\nSecondary: Explain design decisions and potential edge cases\nWhen in doubt, prioritize: Clarity and maintainability over cleverness\n\nProcess:\n1. Clarify function requirements (inputs, outputs, edge cases)\n2. Write function with docstring and type hints\n3. Provide example usage and test cases\n4. Explain design decisions\n\nFormat: Complete Python code block with documentation\nStructure:\n- Function signature with type hints\n- Comprehensive docstring (Google style)\n- Implementation\n- Example usage\n- Test cases (pytest format)\n- Design notes\n\nConstraints:\n- Do NOT use deprecated Python features\n- Avoid premature optimization\n- Limit dependencies to standard library unless specified\n\nIf unclear or insufficient data:\n- Ask: What should the function do? What are the inputs/outputs? Any constraints?\n- Do NOT: Assume requirements or make up functionality\n- Instead: Provide template with TODOs\n\nCheck your output for:\n✓ Type hints on all parameters and return\n✓ Comprehensive docstring\n✓ Edge cases handled\n✓ Example usage provided\n",
      "quality_improvement": 0.85
    },
    {
      "casual": "explain this code to me",
      "structured": "Role: Technical educator specializing in code explanation and mentorship\n\nObjective:\nPrimary: Explain code in a clear, pedagogical way appropriate for the audience\nSecondary: Highlight important patterns, gotchas, and learning opportunities\nWhen in doubt, prioritize: Clarity and educational value over exhaustive detail\n\nProcess:\n1. Analyze code complexity and likely audience level\n2. Provide high-level overview before diving into details\n3. Explain line-by-line with context and reasoning\n4. Highlight key patterns and learning points\n\nFormat: Structured code explanation\nStructure:\n- Overview (what does this do?)\n- Line-by-line walkthrough with annotations\n- Key concepts explained\n- Potential gotchas or edge cases\n- Suggested improvements (if any)\n\nConstraints:\n- Do NOT assume advanced knowledge unless evident from context\n- Avoid jargon without explanation\n- Limit explanation to relevant details (not every syntax element)\n\nIf unclear or insufficient data:\n- Ask: What's your experience level? What specifically is confusing? What's the goal?\n- Do NOT: Assume expertise level or guess at confusion\n- Instead: Provide multi-level explanation (beginner → advanced)\n\nCheck your output for:\n✓ Clear high-level overview\n✓ Jargon explained when used\n✓ Examples where helpful\n✓ Learning points highlighted\n",
      "quality_improvement": 0.68
    }
  ]
}
```

---

## Component 3: Auto-Enhancement Flag

**Usage:**

```python
# Manual meta-prompting
executor = ExecutionEngine()
structured, meta = await executor.proto_llm_call(
    "help me prepare for meeting",
    config
)
result = await executor.execute_prompt(structured, config)

# Auto-enhancement (set flag)
config = ExecutionConfig(
    auto_enhance=True,  # ← NEW FLAG
    enhancement_strategy="meta-prompt"
)
result = await executor.execute_prompt(
    "help me prepare for meeting",  # Casual prompt
    config  # Auto-enhances before execution
)
```

**Implementation in `execution_engine.py`:**

```python
async def execute_prompt(
    self,
    prompt: str,
    config: ExecutionConfig
) -> ExecutionResult:
    """Execute prompt with optional auto-enhancement"""

    start_time = time.time()
    original_prompt = prompt

    # Auto-enhancement if enabled
    if config.auto_enhance:
        prompt, enhancement_meta = await self.proto_llm_call(
            prompt,
            config,
            meta_prompt_template=self._get_enhancement_template(
                config.enhancement_strategy
            )
        )
    else:
        enhancement_meta = None

    # Execute actual prompt
    if config.backend == ExecutionBackend.CLAUDE_API:
        result = self._execute_claude(prompt, config)
    elif config.backend == ExecutionBackend.OLLAMA:
        result = self._execute_ollama(prompt, config)
    else:
        result = self._execute_custom(prompt, config)

    # Add enhancement metadata
    if enhancement_meta:
        result.metadata['enhancement'] = enhancement_meta
        result.metadata['original_prompt'] = original_prompt
        result.metadata['enhanced_prompt'] = prompt

    result.execution_time = time.time() - start_time

    return result
```

---

## Analytics Integration

Track meta-prompting effectiveness:

**New Analytics Metrics:**
```python
@dataclass
class MetaPromptMetrics:
    """Track meta-prompting performance"""
    total_enhancements: int
    avg_preprocessing_time: float
    avg_quality_improvement: float  # Before/after scores
    cost_savings: float  # Tokens saved via better prompts
    success_rate_improvement: float  # % improvement in success

    # Breakdown by prompt type
    by_type: Dict[str, MetricsByType]
```

**Database Schema Addition:**
```sql
CREATE TABLE meta_prompt_analytics (
    id INTEGER PRIMARY KEY,
    execution_id INTEGER,
    original_prompt TEXT,
    enhanced_prompt TEXT,
    preprocessing_time REAL,
    quality_before REAL,
    quality_after REAL,
    improvement REAL,
    template_version TEXT,
    created_at TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES executions(id)
);
```

**Analytics Dashboard:**
- Chart: Meta-prompt quality improvement over time
- Chart: Preprocessing time distribution
- Chart: Success rate: meta-prompted vs raw
- Table: Top improvements (biggest quality gains)

---

## CLI Integration

**New Commands:**

```bash
# Use meta-prompting explicitly
promptly enhance "help me with meeting"

# Execute with auto-enhancement
promptly run "help me with meeting" --enhance

# Configure default enhancement
promptly config set auto_enhance true

# View enhancement analytics
promptly analytics meta-prompt

# Create custom meta-prompt template
promptly meta-prompt create my-custom-template

# Test enhancement on existing prompt
promptly meta-prompt test my-prompt
```

---

## Testing Strategy

### Unit Tests
```python
# test_meta_prompt.py

async def test_proto_llm_call():
    """Test basic meta-prompting"""
    executor = ExecutionEngine()
    enhanced, meta = await executor.proto_llm_call(
        "help me with SQL",
        ExecutionConfig()
    )

    # Should have 7 components
    assert "Role:" in enhanced
    assert "Objective:" in enhanced
    assert "Process:" in enhanced
    assert "Format:" in enhanced
    assert "Constraints:" in enhanced
    assert "If unclear" in enhanced
    assert "Check your output" in enhanced

async def test_quality_improvement():
    """Test that meta-prompting improves results"""
    executor = ExecutionEngine()

    # Without enhancement
    result_raw = await executor.execute_prompt(
        "help me with SQL",
        ExecutionConfig(auto_enhance=False)
    )

    # With enhancement
    result_enhanced = await executor.execute_prompt(
        "help me with SQL",
        ExecutionConfig(auto_enhance=True)
    )

    # Enhanced should be better (by some metric)
    assert result_enhanced.quality_score > result_raw.quality_score
```

### Integration Tests
```python
async def test_full_flow():
    """Test complete meta-prompt → execute → track flow"""
    executor = ExecutionEngine()
    analytics = PromptAnalytics()

    result = await executor.execute_prompt(
        "optimize this query: SELECT * FROM users",
        ExecutionConfig(auto_enhance=True)
    )

    # Should have enhancement metadata
    assert result.metadata['enhancement'] is not None
    assert 'original_prompt' in result.metadata
    assert 'enhanced_prompt' in result.metadata

    # Should be tracked in analytics
    metrics = analytics.get_meta_prompt_metrics()
    assert metrics.total_enhancements > 0
```

---

## Performance Considerations

### Speed
- **Proto-LLM call overhead:** ~500ms (Haiku is fast)
- **Cost:** ~$0.001 per enhancement (Haiku is cheap)
- **Net benefit:** Better prompts = fewer retries = cost savings

### Cost Analysis
```
Without Meta-Prompting:
- Raw prompt → poor result → retry 2-3 times
- Cost: 3 × $0.015 = $0.045
- Time: 3 × 2s = 6s

With Meta-Prompting:
- Enhancement: $0.001, 0.5s
- Good prompt → good result first try
- Cost: $0.001 + $0.015 = $0.016
- Time: 0.5s + 2s = 2.5s

Savings: 65% cost, 58% time
```

---

## Roadmap

### Phase 1: Core Implementation (Week 1)
- [ ] Add `proto_llm_call()` to execution engine
- [ ] Create meta-prompt skill template
- [ ] Add auto-enhancement flag
- [ ] Basic analytics tracking

### Phase 2: Analytics & Optimization (Week 2)
- [ ] Dashboard for meta-prompt metrics
- [ ] A/B testing meta-prompt vs raw
- [ ] Template optimization based on data
- [ ] Cost/benefit tracking

### Phase 3: Advanced Features (Week 3)
- [ ] Custom templates per domain (SQL, code, writing, etc.)
- [ ] Learning: auto-improve templates based on outcomes
- [ ] Multi-stage enhancement (refinement loops)
- [ ] Integration with HoloLoom memory

---

## Success Metrics

**Target Improvements:**
- Quality scores: +30% average improvement
- Success rate: +25% first-try success
- Cost reduction: 50% fewer retries
- User satisfaction: "I can write casual prompts now"

**Analytics to Track:**
- Preprocessing time distribution
- Quality improvement distribution
- Success rate: enhanced vs raw
- Cost per enhancement
- ROI (cost savings from better prompts)

---

## Example Usage

### Before (Raw Prompt)
```python
result = await executor.execute_prompt(
    "help me with meeting",
    ExecutionConfig()
)
# → Generic, unfocused response
# → Quality: 0.45
# → User has to iterate 3+ times
```

### After (Meta-Prompted)
```python
result = await executor.execute_prompt(
    "help me with meeting",
    ExecutionConfig(auto_enhance=True)
)
# → Asks clarifying questions
# → Provides structured prep sheet
# → Quality: 0.87
# → User gets what they need first try
```

---

## Conclusion

This integration brings GPT-5 era meta-prompting to Promptly as a **first-class feature**:

✅ **Proto-LLM preprocessing** - automatic prompt enhancement
✅ **Built-in skill** - 7-component GPT-5 framework
✅ **Auto-enhancement** - optional flag for all calls
✅ **Analytics** - track improvement metrics
✅ **Cost-effective** - Haiku for preprocessing, net cost savings

**Result:** Users can write casual prompts and get GPT-5 quality results automatically.

---

## Next Steps

1. **Review this design** - feedback on architecture?
2. **Prioritize phases** - Phase 1 first, or all at once?
3. **Choose implementation location** - Revive Promptly in archive, or new location?
4. **Decide on HoloLoom integration** - Keep separate or integrate?

**Ready to build?** 🚀
