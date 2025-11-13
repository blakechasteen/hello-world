# Meta-Prompt Template (Standalone)

**Use this template anywhere** - just copy-paste into ChatGPT, Claude, or any LLM!

---

## How to Use

1. Copy everything below the line
2. Replace `{YOUR_CASUAL_REQUEST}` with your actual request
3. Paste into any LLM
4. Get a structured, GPT-5-ready prompt back!

---

## THE TEMPLATE (Copy from here)

```
You are a prompt engineering expert. Transform my casual request into a structured, comprehensive prompt using the 7-component framework for modern LLMs.

MY CASUAL REQUEST:
{YOUR_CASUAL_REQUEST}

YOUR TASK:
Create a structured prompt with these 7 components:

1. ROLE - Define the expert role/expertise needed
   Format: "Role: [specific expert with relevant domain knowledge]"

2. OBJECTIVE - State goals with explicit priorities
   Format:
   "Objective:
   Primary: [main goal]
   Secondary: [supporting goal]
   When in doubt, prioritize: [priority]"

3. PROCESS - Provide step-by-step methodology
   Format:
   "Process:
   1. [step 1]
   2. [step 2]
   3. [step 3]"

4. FORMAT - Specify exact output structure
   Format:
   "Format: [description]
   Structure:
   - [section 1]
   - [section 2]"

5. CONSTRAINTS - Define what NOT to do
   Format:
   "Constraints:
   - Do NOT [anti-pattern 1]
   - Avoid [anti-pattern 2]"

6. UNCERTAINTY - Fallback behavior for unclear situations
   Format:
   "If unclear or insufficient data:
   - Ask: [specific questions]
   - Do NOT: [things to avoid]
   - Instead: [fallback behavior]"

7. VALIDATION - Success criteria checklist
   Format:
   "Check your output for:
   ✓ [criterion 1]
   ✓ [criterion 2]
   ✓ [criterion 3]"

OUTPUT FORMAT:
Provide:
1. The complete structured prompt (all 7 components)
2. Brief justification (2-3 sentences)
3. Clarifying questions if my request is ambiguous

Make it specific, not generic. Ask questions if unclear. Structure beats intelligence.
```

---

## Quick Examples

### Example 1: Meeting Prep

**Casual request:**
```
help me prepare for tomorrow's meeting
```

**Enhanced prompt you'd get back:**
```
Role: Meeting preparation consultant with stakeholder management expertise

Objective:
Primary: Create actionable meeting prep plan
Secondary: Anticipate likely discussion points
When in doubt, prioritize: Asking clarifying questions over assumptions

Process:
1. Clarify meeting context (type, attendees, goal)
2. Structure prep into manageable sections
3. Surface 2-3 likely talking points

Format: Meeting prep sheet
Structure:
- Context recap
- Core message
- Key questions to ask
- Anticipated objections
- Next steps

Constraints:
- Do NOT fabricate stakeholder details
- Avoid generic advice disconnected from situation
- Limit assumptions - ask when unclear

If unclear or insufficient data:
- Ask: What kind of meeting? Who's attending? What outcome?
- Do NOT: Assume meeting type or fabricate concerns
- Instead: Provide template with blanks to fill

Check your output for:
✓ Specific, actionable items
✓ Questions asked when context missing
✓ No fabricated facts
```

### Example 2: Code Request

**Casual request:**
```
write a Python function
```

**Enhanced prompt you'd get back:**
```
Role: Senior Python developer with clean code expertise

Objective:
Primary: Write well-documented, tested Python function
Secondary: Explain design decisions
When in doubt, prioritize: Clarity over cleverness

Process:
1. Clarify requirements (inputs, outputs, edge cases)
2. Write function with docstring and type hints
3. Provide examples and test cases
4. Explain design decisions

Format: Complete Python code block
Structure:
- Function signature with type hints
- Google-style docstring
- Implementation
- Usage examples
- Test cases (pytest)

Constraints:
- Do NOT use deprecated features
- Avoid premature optimization
- Limit to standard library unless specified

If unclear or insufficient data:
- Ask: What should function do? Inputs/outputs? Constraints?
- Do NOT: Assume requirements or make up functionality
- Instead: Provide template with TODOs

Check your output for:
✓ Type hints on all parameters
✓ Comprehensive docstring
✓ Edge cases handled
✓ Examples provided
```

---

## Pro Tips

1. **Be specific** - "developer" → "Senior Python developer with async expertise"
2. **Always prioritize** - Include "When in doubt, prioritize..." in objectives
3. **Define anti-patterns** - Constraints are critical for GPT-5!
4. **Ask, don't assume** - Use uncertainty section to prevent hallucination
5. **Give methodology** - Process steps guide the model's thinking

---

## Why This Works

Modern LLMs (GPT-5, Claude Opus) are like **speedboats with big rudders**:
- They have massive power/capability
- They need strong steering/structure
- Casual prompts = speedboat going in circles
- Structured prompts = speedboat going straight to destination

The 7-component framework provides that steering!

---

## Adapt This Template

Feel free to customize for your domain:

**For code:**
- Add "Code style: [PEP 8, Google, etc.]"
- Add "Complexity limit: [beginner/intermediate/advanced]"

**For writing:**
- Add "Tone: [formal/casual/persuasive]"
- Add "Audience: [executives/technical/general]"

**For analysis:**
- Add "Depth: [high-level overview / detailed analysis]"
- Add "Evidence: [cite sources / provide examples]"

---

**Happy meta-prompting!** 🚀
