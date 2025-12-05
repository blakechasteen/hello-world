# Meta-Prompt (Claude Desktop Custom Prompt)

**For Claude Desktop users:** Add this as a custom prompt!

---

## Installation Instructions

### Option 1: Via Claude Desktop Settings

1. Open Claude Desktop
2. Go to Settings → Custom Prompts
3. Click "New Custom Prompt"
4. Name: `Meta-Prompt`
5. Copy the prompt text below
6. Save

### Option 2: Via Config File

1. Locate your Claude Desktop config:
   - **macOS:** `~/Library/Application Support/Claude/prompts/`
   - **Windows:** `%APPDATA%\Claude\prompts\`
   - **Linux:** `~/.config/Claude/prompts/`

2. Create file: `meta-prompt.md`
3. Copy the prompt text below
4. Restart Claude Desktop

---

## THE PROMPT TEXT (Copy this)

```markdown
Transform my casual request into a comprehensive, structured prompt using the 7-component framework for optimal Claude performance.

## The Request

{$PROMPT}

## Your Task

Create a structured prompt with these components:

### 1. Role (Expertise)
Define the specific expert role needed with relevant domain knowledge.

Example: "Role: Senior Python developer with async programming expertise"

### 2. Objective (Goals)
State primary and secondary goals with explicit prioritization.

Example:
"Objective:
Primary: Write production-ready code
Secondary: Optimize for readability
When in doubt, prioritize: Correctness over performance"

### 3. Process (Methodology)
Provide step-by-step approach.

Example:
"Process:
1. Analyze requirements
2. Design solution
3. Implement with error handling
4. Provide examples"

### 4. Format (Output Structure)
Specify exact output format.

Example:
"Format: Code module with documentation
Structure:
- Docstring
- Implementation
- Examples
- Tests"

### 5. Constraints (Boundaries)
Define what NOT to do.

Example:
"Constraints:
- Do NOT use deprecated features
- Avoid external dependencies
- Limit complexity to intermediate level"

### 6. Uncertainty (Fallback)
Define behavior for unclear situations.

Example:
"If unclear:
- Ask: [specific questions]
- Do NOT: assume or fabricate
- Instead: provide template with TODOs"

### 7. Validation (Success Criteria)
Provide quality checklist.

Example:
"Check for:
✓ Type hints present
✓ Edge cases handled
✓ Examples provided"

## Output

Provide:
1. **Structured Prompt** - Complete prompt with all 7 components
2. **Justification** - Brief explanation (2-3 sentences)
3. **Questions** - Any clarifying questions if request is ambiguous

## Guidelines

- Be specific, not generic
- Always include priorities in objectives
- Use constraints to prevent unwanted behavior
- Ask questions rather than assume
- Structure guides thinking better than raw intelligence

Transform the request now.
```

---

## How to Use in Claude Desktop

Once installed, invoke with:

```
@meta-prompt help me prepare for a client meeting tomorrow
```

Or:

```
@meta-prompt write a Python function to sort data
```

Claude will transform your casual request into a structured prompt, then you can use that structured prompt for your actual query!

---

## Advanced Usage: Two-Step Flow

### Step 1: Generate structured prompt
```
@meta-prompt optimize this SQL query
```

### Step 2: Use the structured prompt
```
[Paste the structured prompt Claude gave you]

Here's the query to optimize:
SELECT * FROM users WHERE active = 1
```

This two-step flow ensures maximum quality!

---

## Pro Tips for Claude Desktop

1. **Save good meta-prompts** - When you get a great structured prompt, save it as its own custom prompt for reuse

2. **Chain with other prompts** - Use meta-prompt output as input to specialized prompts

3. **Create domain-specific variants** - Make custom meta-prompts for code, writing, analysis, etc.

4. **Use with projects** - Apply meta-prompting to project-specific queries for better context

---

## Example Session

**You:**
```
@meta-prompt help me write a FastAPI endpoint
```

**Claude:**
```
STRUCTURED PROMPT:

Role: Senior Python developer with FastAPI and REST API expertise

Objective:
Primary: Create production-ready FastAPI endpoint with validation
Secondary: Include documentation and error handling
When in doubt, prioritize: Security and data validation over brevity

Process:
1. Define endpoint path and HTTP method
2. Create Pydantic models for request/response
3. Implement endpoint with error handling
4. Add OpenAPI documentation
5. Provide usage examples

Format: Complete Python code module
Structure:
- Imports
- Pydantic models
- Endpoint function
- Error handling
- Usage examples

Constraints:
- Do NOT skip input validation
- Avoid complex business logic in endpoint
- Limit to FastAPI best practices

If unclear:
- Ask: What data does endpoint receive/return? What's the use case?
- Do NOT: Assume database schema or business logic
- Instead: Provide template with TODOs

Check for:
✓ Pydantic validation present
✓ HTTP status codes appropriate
✓ Error responses defined
✓ OpenAPI docs complete

---

CLARIFYING QUESTIONS:
- What should this endpoint do?
- What data format (JSON, form, etc.)?
- Any authentication required?
```

**You can then use that structured prompt for your actual implementation!**

---

## Compatibility

- ✅ Claude Desktop (all versions)
- ✅ Claude Web (copy-paste prompt)
- ✅ Claude API (use as system prompt)
- ✅ Works with all Claude models (Opus, Sonnet, Haiku)

---

**Enjoy better prompting with Claude!** 🚀
