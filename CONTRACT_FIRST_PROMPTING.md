# Contract-First Prompting

**Status**: ✅ Production Ready (November 2025)
**Location**: `HoloLoom/prompting/`
**Philosophy**: "Clarity of intent before execution"

## Overview

Contract-First Prompting is a systematic approach to achieving clarity of intent when working with LLMs. Like engineering teams write contracts for how microservices interact, we establish tight technical shared understanding with the LLM before starting work.

### The Problem

Almost every prompt that fails fails because **intent wasn't clearly communicated**. Human language is really rough on intent. We bring tremendous domain expertise, passion, and experience to a subject, but struggle to convey it clearly in words to an LLM.

**Common failure mode**: "Just ask clarifying questions"
- **Why it's insufficient**: Scattershot and unprofessional
- **The issue**: LLM swims in a sea of ambiguity with free reign to pick questions
- **Missing**: No structure or parameters to ensure understanding

### The Solution

Contract-First Prompting provides a structured sequence:

1. **Mission**: Clear goal for the interaction
2. **Gap Identification**: Silently scan and list every fact/constraint needed
3. **Iterative Questioning**: One question at a time until 95% confidence
4. **Echo Check**: Crisp summary stating deliverable, includes, and constraints
5. **User Control**: Yes/lock, edit, blueprint, risks, reset
6. **Self-Testing**: Review before delivery

## Core Philosophy

> **"We need to get to a point where we have very tight technical shared understanding with the LLM of the meaningful work we want to do together before it starts to work."**

### Key Principles

1. **Assume humans are humans** - We don't have perfect, complete intent initially
2. **Structured ambiguity resolution** - Not random clarifying questions
3. **Iterative refinement** - One question at a time, building understanding
4. **Explicit contract** - Both parties agree before execution
5. **User control** - Multiple paths forward (yes, edit, blueprint, risks)

## The Contract-First Template

```
Your goal is to turn my rough idea into a very clear work order. You will deliver the work only after both of us agree it's right.

# Step 0: Gap Analysis (Silent)
When I describe what I need, silently scan and list every fact or constraint you still need to know. Don't show me this list yet—it's your internal checklist.

# Step 1: Dig Until 95% Confidence
Ask me ONE question at a time, focusing on the biggest unknowns first.

Examples of places to dig:
- Purpose: Why does this exist? What problem does it solve?
- Audience: Who will use/read/see this?
- Facts: What specific data, examples, or constraints apply?
- Success criteria: How will we know it's good enough?
- Length/scope: How much detail? How many parts?
- Tech stack (if code): Languages, frameworks, libraries?
- Edge cases: What unusual situations must it handle?
- Risk tolerance: Should it be conservative or experimental?

Keep asking until you're 95% confident you can ship the correct result.

# Step 2: Echo Check
When you think you're close, reply with ONE crisp sentence that states:
1. The deliverable (what you'll build)
2. Something you know it must include
3. One hard constraint (to show you understood the boundaries)

Then ask: "Is this correct? Reply: yes (to lock it in), edit (to change), blueprint (to see the outline), or risks (to call out potential issues)."

# Step 3: Handle Response
- If I say "yes": Lock the contract and proceed to build
- If I say "edit": Ask what to change, then echo check again
- If I say "blueprint": Show me a structured outline before building
- If I say "risks": List the top 3 risks and how you'll handle them

# Step 4: Build and Self-Test
Build the deliverable. Before showing me:
- Review for completeness against the contract
- If code: Check for bugs, edge cases, and best practices
- If document: Check structure, clarity, and completeness
- Self-assess: Does this meet the 95% confidence bar?

# Ready
I'm ready. What do you need?
```

## Usage Examples

### Example 1: Ambiguous Historical Summary

**User Initial Request**:
```
I need a 500-word summary of the history of the Balkans since 1660.
```

**What the LLM Figured Out**:
The LLM identified that a key leverage point was **how to handle the evolution of political entities and their naming conventions** across that time period. It asked 3-4 rounds of questions about:
- Which political entities to include
- How to handle border changes
- Naming conventions (Ottoman Empire → Yugoslavia → modern states)
- Scope of "Balkans" (which countries count?)

**Result**: Solid summary that handled political complexity appropriately.

### Example 2: Software Project

**User Initial Request**:
```
I want to build a tool to centralize comments in live streams across multiple channels.
```

**What the LLM Dug Into**:
- How many channels? Which platforms?
- What counts as a "comment"? (Donations? Reactions? Polls?)
- What's an MVP? What's nice-to-have?
- Expected number of users/channels?
- Tech stack preferences?
- Real-time requirements?

**Result**: Clear PRD (Product Requirements Document) with aligned intent.

### Example 3: Code Implementation

**User Initial Request**:
```
I need a function to validate email addresses.
```

**What the LLM Asked**:
- Level of validation? (Regex only? DNS lookup? SMTP verification?)
- International support? (Unicode domains? IDN?)
- Error handling? (Throw exception? Return boolean? Return detailed error?)
- Performance requirements? (Single email? Batch processing?)
- Edge cases? (Plus addressing? Subdomain validation?)

**Result**: Production-ready validator with appropriate trade-offs.

## Integration with HoloLoom

HoloLoom's contract-first prompting system integrates with the existing architecture:

### Quick Start

```python
from HoloLoom.prompting import ContractFirstPrompting

async with ContractFirstPrompting() as cfp:
    # Step 1: User provides rough idea
    await cfp.start("I need a system to track user engagement metrics")

    # Step 2: LLM asks clarifying questions (one at a time)
    # User answers interactively

    # Step 3: Echo check
    contract = await cfp.get_contract()
    print(contract.deliverable)
    print(contract.constraints)

    # Step 4: User approves ("yes", "edit", "blueprint", "risks")
    if await cfp.approve():
        # Step 5: Build and deliver
        result = await cfp.execute()
        print(result)
```

### Advanced Usage

```python
from HoloLoom.prompting import ContractFirstPrompting, DiggingStrategy

async with ContractFirstPrompting(
    confidence_threshold=0.95,
    max_questions=10,
    digging_strategy=DiggingStrategy.BREADTH_FIRST
) as cfp:
    # Custom gap identification
    await cfp.start(
        rough_idea="Build a dashboard for ML model monitoring",
        context={
            "tech_stack": ["Python", "FastAPI", "React"],
            "audience": "ML engineers",
            "deployment": "AWS"
        }
    )

    # Interactive questioning loop
    while not cfp.is_confident():
        question = await cfp.next_question()
        answer = input(f"{question}\n> ")
        await cfp.answer(answer)

    # Review contract
    contract = await cfp.get_contract()

    # Request blueprint before approval
    blueprint = await cfp.blueprint()
    print(blueprint)

    # Approve and execute
    if input("Approve? (yes/edit/risks): ") == "yes":
        result = await cfp.execute()
```

## When to Use Contract-First Prompting

### ✅ Use Contract-First When:

1. **Complex work** - Multi-step, high-stakes, or ambiguous requirements
2. **Unclear intent** - You have a vague idea but not crisp requirements
3. **Domain expertise gap** - You know the domain, LLM doesn't (or vice versa)
4. **High cost of failure** - Getting it wrong is expensive
5. **Collaborative refinement** - You want to think through requirements together

### 🟡 Maybe Use Contract-First When:

1. **Medium complexity** - Some ambiguity but mostly clear
2. **Iterative work** - Planning to refine in multiple rounds anyway
3. **Learning mode** - You want to understand what questions to ask

### ❌ Don't Use Contract-First When:

1. **Simple, clear tasks** - "Write a function to add two numbers"
2. **Extremely urgent** - No time for iterative refinement
3. **Exploratory work** - You want to see what the LLM generates first
4. **You already have a PRD** - Intent is already crystal clear

## Gap Identification Dimensions

The LLM should dig across these dimensions:

### Core Dimensions

1. **Purpose** - Why does this exist? What problem does it solve?
2. **Audience** - Who will use/read/see this?
3. **Success Criteria** - How will we know it's good enough?
4. **Scope** - How much detail? How many parts?

### Technical Dimensions (Code)

5. **Tech Stack** - Languages, frameworks, libraries?
6. **Edge Cases** - What unusual situations must it handle?
7. **Error Handling** - How should errors be handled?
8. **Performance** - Latency, throughput, resource requirements?
9. **Testing** - What tests are needed?
10. **Security** - What threats must be considered?

### Content Dimensions (Documents)

11. **Tone** - Formal, casual, technical, friendly?
12. **Structure** - Sections, headings, flow?
13. **Examples** - Concrete examples needed?
14. **Citations** - Sources required?

### Operational Dimensions

15. **Timeline** - When is this needed?
16. **Risk Tolerance** - Conservative or experimental?
17. **Dependencies** - What else must exist first?
18. **Constraints** - Hard limits (budget, time, resources)?

## Echo Check Format

After iterative questioning, the LLM provides an echo check:

```
I will create [DELIVERABLE] that [KEY INCLUDE].
It must [HARD CONSTRAINT].

Is this correct? Reply:
- yes (to lock it in)
- edit (to change something)
- blueprint (to see the outline)
- risks (to call out potential issues)
```

**Example**:
```
I will create a Python function to validate email addresses that supports
international domains and Unicode. It must return a detailed error object
(not just boolean) and handle edge cases like plus addressing.

Is this correct? Reply: yes, edit, blueprint, or risks
```

## Control Flow Options

### 1. Yes (Lock)

User says "yes" → LLM locks the contract and proceeds to build.

### 2. Edit

User says "edit" or specifies changes:
- LLM asks what to change
- Iterates on specific aspects
- Returns to echo check

### 3. Blueprint

User says "blueprint":
- LLM provides structured outline before building
- Shows architecture, sections, or components
- User can approve or request changes

**Example Blueprint** (Code):
```
Blueprint:
1. Input validation layer
   - Check for null/empty
   - Basic format check (@ symbol, domain)
2. International support
   - Unicode normalization
   - IDN (Internationalized Domain Names) handling
3. DNS validation (optional)
   - MX record lookup
   - Timeout handling
4. Error object
   - Error type enum
   - Detailed message
   - Suggested fix
5. Test suite
   - Valid emails (50 cases)
   - Invalid emails (50 cases)
   - Edge cases (20 cases)
```

### 4. Risks

User says "risks":
- LLM lists top 3-5 risks
- Explains mitigation strategy for each
- User can approve or request changes

**Example Risks**:
```
Risks:
1. DNS validation adds latency (~500ms per email)
   - Mitigation: Make DNS check optional, default off
2. Unicode edge cases may cause false negatives
   - Mitigation: Comprehensive test suite with IDN examples
3. Regex complexity may impact performance for batch processing
   - Mitigation: Pre-compile regex, consider batch optimization
```

## Self-Testing Requirements

Before delivering, the LLM must self-test:

### For Code:
- ✅ Runs without errors
- ✅ Handles specified edge cases
- ✅ Follows best practices
- ✅ Includes error handling
- ✅ Has clear documentation
- ✅ Meets performance requirements

### For Documents:
- ✅ Covers all required sections
- ✅ Matches specified tone and audience
- ✅ Includes requested examples
- ✅ Meets length requirements
- ✅ Follows structural guidelines
- ✅ Has clear, logical flow

### For PRDs/Specs:
- ✅ Defines success criteria
- ✅ Lists all requirements
- ✅ Identifies risks and mitigations
- ✅ Specifies timeline and milestones
- ✅ Calls out dependencies
- ✅ Addresses stakeholder concerns

## Performance Characteristics

| Phase | Duration | Notes |
|-------|----------|-------|
| Gap Identification | <1s | Silent, internal |
| Iterative Questioning | 5-20 questions | 1-2 min total |
| Echo Check | <1s | Single response |
| User Approval | Variable | User decision time |
| Blueprint Generation | 2-5s | If requested |
| Risk Analysis | 2-5s | If requested |
| Execution | Variable | Depends on work |
| Self-Testing | 1-3s | Before delivery |

**Total overhead**: 1-3 minutes for clarification, massive time savings on rework.

## Integration with HoloLoom Components

### Memory Integration

Contract-first prompting integrates with HoloLoom's memory systems:

```python
from HoloLoom import HoloLoom
from HoloLoom.prompting import ContractFirstPrompting

async with HoloLoom() as loom:
    async with ContractFirstPrompting(memory=loom) as cfp:
        # LLM can recall past contracts and patterns
        await cfp.start("Build another dashboard like last time")
        # Memory automatically provides context from past contracts
```

### Reflection Integration

Contracts are stored for reflection and learning:

```python
async with ContractFirstPrompting(enable_reflection=True) as cfp:
    result = await cfp.execute()

    # Store contract + result for learning
    await cfp.reflect(feedback={"successful": True, "time_saved_hours": 4})
```

### Agentic Integration

Contract-first prompting works with agentic reasoning:

```python
from HoloLoom.agentic import AgenticOrchestrator
from HoloLoom.prompting import ContractFirstPrompting

async with AgenticOrchestrator() as orchestrator:
    async with ContractFirstPrompting(orchestrator=orchestrator) as cfp:
        # Use agentic reasoning for complex gap identification
        await cfp.start(
            "Build a multi-agent system for code review",
            mode=ReasoningMode.RESEARCH
        )
```

## Best Practices

### 1. Start with Rough Ideas

Don't over-prepare. The whole point is to work from ambiguous initial ideas:

```python
# ✅ Good - Start rough
await cfp.start("I need to track user engagement somehow")

# ❌ Bad - Over-specified (just execute directly)
await cfp.start("Build a PostgreSQL-backed engagement tracking system with...")
```

### 2. Trust the Process

Let the LLM ask questions. Don't jump ahead:

```python
# ✅ Good - Answer one question at a time
question = await cfp.next_question()
# "What metrics do you want to track?"
await cfp.answer("Page views, time on site, and conversions")

# ❌ Bad - Info dumping
await cfp.answer("Page views, time on site, conversions, also I want PostgreSQL and...")
```

### 3. Use Blueprint for Complex Work

For multi-step work, always request a blueprint:

```python
contract = await cfp.get_contract()
blueprint = await cfp.blueprint()  # Review structure first
await cfp.approve("yes")
```

### 4. Don't Skip Risk Analysis

For high-stakes work, review risks:

```python
risks = await cfp.analyze_risks()
# Review before approval
if all(r.mitigation for r in risks):
    await cfp.approve("yes")
```

### 5. Learn from Patterns

Track successful contracts for reuse:

```python
async with ContractFirstPrompting(learn_from_history=True) as cfp:
    # System learns what questions work for what domains
    await cfp.start("Build another ML monitoring dashboard")
    # Automatically applies patterns from past contracts
```

## Common Patterns

### Pattern 1: Code Implementation

```
User: "I need a function to [purpose]"
LLM: "What programming language?"
User: "Python"
LLM: "What should it return when [edge case]?"
User: "[error handling strategy]"
LLM: "Should it [performance question]?"
User: "[performance requirement]"
[... continue until 95% confidence ...]
LLM: [Echo check]
```

### Pattern 2: Document Creation

```
User: "I need a document about [topic]"
LLM: "Who is the audience?"
User: "[audience description]"
LLM: "What's the purpose? (Educate, persuade, document?)"
User: "[purpose]"
LLM: "How long should it be?"
User: "[length]"
LLM: "What tone? (Formal, casual, technical?)"
User: "[tone]"
[... continue until 95% confidence ...]
LLM: [Echo check]
```

### Pattern 3: System Architecture

```
User: "I need to design a system for [purpose]"
LLM: "What's the expected scale? (Users, requests, data size?)"
User: "[scale]"
LLM: "What are the latency requirements?"
User: "[latency]"
LLM: "What's the risk tolerance for downtime?"
User: "[availability requirements]"
LLM: "What's the tech stack constraint?"
User: "[tech stack]"
[... continue until 95% confidence ...]
LLM: [Echo check]
```

## Troubleshooting

### Issue: Too Many Questions

**Symptom**: LLM asks 15+ questions, feels like interrogation

**Solutions**:
1. Lower confidence threshold: `confidence_threshold=0.85`
2. Provide more context upfront: Use `context` parameter
3. Set max questions: `max_questions=10`
4. Use `DiggingStrategy.ESSENTIAL_ONLY`

### Issue: Wrong Questions

**Symptom**: LLM asks irrelevant questions

**Solutions**:
1. Add domain context: `domain="machine_learning"`
2. Provide examples: `similar_to="previous_project_id"`
3. Guide with constraints: `constraints={"must_use": ["Python", "FastAPI"]}`

### Issue: Vague Echo Check

**Symptom**: Echo check doesn't show deep understanding

**Solutions**:
1. Request detailed echo: `echo_detail_level="comprehensive"`
2. Ask for blueprint immediately: User says "blueprint" instead of "yes"
3. Review gap analysis: `cfp.show_gap_analysis()`

### Issue: Execution Doesn't Match Contract

**Symptom**: Final result doesn't match agreed contract

**Solutions**:
1. Enable strict mode: `strict_contract_adherence=True`
2. Request blueprint review before execution
3. Use checkpoints: `enable_checkpoints=True` for multi-step work

## Future Enhancements

Roadmap for contract-first prompting (Phase 6+):

1. **Contract Templates** - Pre-built templates for common domains
2. **Visual Contract Builder** - Drag-and-drop contract construction
3. **Contract Diff** - Show what changed between iterations
4. **Multi-Party Contracts** - Multiple stakeholders agree on contract
5. **Contract Versioning** - Track evolution of contracts over time
6. **Automated Testing** - Generate tests from contract specifications
7. **Contract Marketplace** - Share successful contracts across teams

## Documentation Files

- **This file**: Complete framework overview
- `HoloLoom/prompting/README.md`: API reference
- `HoloLoom/prompting/EXAMPLES.md`: Extended examples
- `demos/demo_contract_first.py`: Interactive demo

## References

- Original video transcript: [Contract-First Prompting Video]
- HoloLoom architecture: [HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md]
- Prompt engineering best practices: [HoloLoom/prompting/BEST_PRACTICES.md]

---

**Created**: 2025-11-18
**Author**: HoloLoom Team
**Status**: Production Ready
