# Meta-Prompt Core Template (Universal)

**LLM Agnostic** - Works with Claude, GPT, Gemini, Llama, or any modern LLM.

This is the **foundation** of the metaprompting system. For model-specific optimizations, see `adapters/` directory.

---

## The 7-Component Framework

Every metaprompt includes these components to maximize clarity, reduce hallucinations, and ensure high-quality outputs:

1. **ROLE** - Expert perspective and domain knowledge
2. **OBJECTIVE** - Primary/secondary goals with explicit priorities
3. **PROCESS** - Step-by-step methodology
4. **FORMAT** - Exact output structure specification
5. **CONSTRAINTS** - What NOT to do (anti-patterns)
6. **UNCERTAINTY** - Fallback behavior when information is unclear
7. **VALIDATION** - Success criteria checklist

---

## How to Use

### Option 1: Direct Use (Copy-Paste)

Copy the template below, replace `{YOUR_CASUAL_REQUEST}`, and paste into any LLM.

### Option 2: Programmatic Use

```python
from HoloLoom.prompting import auto_detect_strategy

enhanced = auto_detect_strategy("your casual request")
```

---

## THE CORE TEMPLATE

```
You are a prompt engineering expert. Transform my casual request into a structured, comprehensive prompt using the 7-component framework.

MY CASUAL REQUEST:
{YOUR_CASUAL_REQUEST}

YOUR TASK:
Create a structured prompt with these 7 components:

### 1. ROLE (Expert Perspective)
Define the specific expert role needed with relevant domain knowledge.

Format: "Role: [specific expert with domain expertise]"

Example: "Role: Senior Python developer with async programming expertise"

### 2. OBJECTIVE (Goals with Priorities)
State primary and secondary goals with explicit prioritization.

Format:
"Objective:
Primary: [main goal]
Secondary: [supporting goal 1], [supporting goal 2]
When in doubt, prioritize: [priority guidance]"

Example:
"Objective:
Primary: Write production-ready code
Secondary: Optimize for readability, include comprehensive tests
When in doubt, prioritize: Correctness over performance"

### 3. PROCESS (Step-by-Step Methodology)
Provide clear, numbered steps for approaching the task.

Format:
"Process:
1. [first step]
2. [second step]
3. [third step]
..."

Example:
"Process:
1. Analyze requirements and identify edge cases
2. Design solution with clear interfaces
3. Implement with error handling
4. Provide usage examples and tests"

### 4. FORMAT (Output Structure)
Specify exact output format and structure.

Format:
"Format: [output type]
Structure:
- [section 1]
- [section 2]
- [section 3]"

Example:
"Format: Code module with documentation
Structure:
- Docstring with complexity analysis
- Function implementation with type hints
- Usage examples
- Test cases"

### 5. CONSTRAINTS (What NOT to Do)
Define anti-patterns and boundaries.

Format:
"Constraints:
- Do NOT [anti-pattern 1]
- Avoid [anti-pattern 2]
- Limit [scope/complexity]"

Example:
"Constraints:
- Do NOT use deprecated features
- Avoid premature optimization
- Limit to standard library unless specified"

### 6. UNCERTAINTY (Fallback Behavior)
Define behavior when information is unclear or insufficient.

Format:
"If unclear or insufficient data:
- Ask: [specific clarifying questions]
- Do NOT: [things to avoid when guessing]
- Instead: [fallback behavior]"

Example:
"If unclear or insufficient data:
- Ask: What are the input types? Expected edge cases? Performance requirements?
- Do NOT: Assume requirements or fabricate specifications
- Instead: Provide template with TODOs and ask for clarification"

### 7. VALIDATION (Success Criteria)
Provide quality checklist to verify before delivery.

Format:
"Check your output for:
✓ [criterion 1]
✓ [criterion 2]
✓ [criterion 3]
..."

Example:
"Check your output for:
✓ Type hints on all parameters
✓ Comprehensive docstring with examples
✓ Edge cases handled explicitly
✓ Examples demonstrate usage
✓ No deprecated features used"

---

## OUTPUT FORMAT

Provide:

1. **Structured Prompt** - Complete prompt with all 7 components formatted clearly
2. **Justification** - Brief explanation (2-3 sentences) of key decisions
3. **Questions** - Any clarifying questions if the request is ambiguous

## GUIDELINES

- **Be specific, not generic** - "developer" → "Senior Python developer with FastAPI expertise"
- **Always prioritize** - Include "When in doubt, prioritize..." in objectives
- **Define anti-patterns** - Constraints prevent unwanted behavior
- **Ask, don't assume** - Use uncertainty section to prevent hallucination
- **Give methodology** - Process steps guide structured thinking

Make it specific to the request, not generic. Ask questions if unclear. Structure beats raw intelligence.
```

---

## Example Transformations

### Example 1: Simple Code Request

**Input:**
```
write a Python function
```

**Output:**
```
### ROLE
Senior Python developer with clean code expertise and production experience

### OBJECTIVE
Primary: Write well-documented, tested Python function
Secondary: Include type hints, handle edge cases gracefully
When in doubt, prioritize: Clarity over cleverness

### PROCESS
1. Clarify requirements (inputs, outputs, edge cases)
2. Write function with type hints and docstring
3. Handle edge cases explicitly
4. Provide usage examples and test cases

### FORMAT
Complete Python code module
Structure:
- Function signature with type hints
- Google-style docstring with complexity analysis
- Implementation
- Usage examples
- Test cases (pytest format)

### CONSTRAINTS
- Do NOT use deprecated features without noting them
- Avoid premature optimization without benchmarks
- Limit to standard library unless dependencies specified

### UNCERTAINTY
If unclear:
- Ask: What should the function do? Input/output types? Performance requirements?
- Do NOT: Assume requirements or make up functionality
- Instead: Provide template with TODOs and request clarification

### VALIDATION
Check output for:
✓ Type hints on all parameters and return value
✓ Comprehensive docstring (purpose, args, returns, examples)
✓ Edge cases handled (None, empty, invalid input)
✓ Usage examples provided
✓ Test cases demonstrate correctness
```

### Example 2: Meeting Preparation

**Input:**
```
help me prepare for tomorrow's meeting
```

**Output:**
```
### ROLE
Meeting preparation consultant with stakeholder management expertise

### OBJECTIVE
Primary: Create actionable meeting prep plan
Secondary: Anticipate likely discussion points, prepare responses
When in doubt, prioritize: Asking clarifying questions over making assumptions

### PROCESS
1. Clarify meeting context (type, attendees, your role, desired outcome)
2. Structure prep into manageable sections
3. Identify 2-3 key talking points you should drive
4. Anticipate likely questions/objections

### FORMAT
Meeting prep sheet
Structure:
- Context recap (who, what, why)
- Your core message/goal
- Key questions to ask
- Anticipated objections with responses
- Next steps to propose

### CONSTRAINTS
- Do NOT fabricate stakeholder details or concerns
- Avoid generic advice disconnected from your situation
- Limit assumptions - ask when context is missing

### UNCERTAINTY
If unclear:
- Ask: What type of meeting? Who's attending? What outcome do you want? What's your role?
- Do NOT: Assume meeting type, fabricate stakeholder concerns, or invent context
- Instead: Provide prep template with blanks to fill in

### VALIDATION
Check output for:
✓ Specific, actionable items (not generic advice)
✓ Questions asked when context was missing
✓ No fabricated facts or assumptions stated as fact
✓ Preparation is relevant to stated meeting type
```

---

## Why This Framework Works

Modern LLMs are **powerful but need structure**:

- **Without structure**: Like a speedboat going in circles (power without direction)
- **With 7 components**: Speedboat going straight to destination (power + steering)

The framework provides:

1. **ROLE** → Activates relevant knowledge and perspective
2. **OBJECTIVE** → Clarifies what success looks like
3. **PROCESS** → Guides step-by-step thinking
4. **FORMAT** → Ensures deliverable matches expectations
5. **CONSTRAINTS** → Prevents common mistakes and hallucinations
6. **UNCERTAINTY** → Reduces guessing, increases question-asking
7. **VALIDATION** → Self-check before delivery

---

## Customization for Domains

Feel free to extend the template for your domain:

**For Code:**
- Add "Code style: [PEP 8, Google Style Guide, etc.]"
- Add "Complexity level: [beginner/intermediate/expert]"
- Add "Testing framework: [pytest, unittest, etc.]"

**For Writing:**
- Add "Tone: [formal/casual/persuasive/technical]"
- Add "Audience: [executives/engineers/general public]"
- Add "Length: [brief/moderate/comprehensive]"

**For Analysis:**
- Add "Depth: [high-level overview/detailed analysis/comprehensive research]"
- Add "Evidence: [cite sources/provide examples/show data]"
- Add "Format: [bullet points/narrative/tables]"

---

## Model-Specific Enhancements

This core template works on **all LLMs**. For model-specific optimizations:

- **Claude**: See `adapters/claude.md` for thinking tags, artifacts, XML formatting
- **Gemini**: See `adapters/gemini.md` for multimodal, code execution, grounding
- **GPT**: See `adapters/gpt.md` for function calling, JSON mode, structured outputs
- **Ollama**: See `adapters/ollama.md` for local model optimizations

Or use programmatically:
```python
from HoloLoom.prompting import create_adapter

adapter = create_adapter(llm_provider="anthropic")  # Auto-enhances for Claude
enhanced = adapter.enhance(core_prompt)
```

---

## Next Steps

1. **Try it**: Copy the template, fill in your request, paste into any LLM
2. **Customize**: Add domain-specific sections for your use case
3. **Iterate**: Save good metaprompts for reuse
4. **Enhance**: Use model-specific adapters for optimal results

---

**Happy meta-prompting!** 🚀
