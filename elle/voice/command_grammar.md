# Elle Voice Command Grammar

**Milestone 2: Command Mode + Structured Grammar**
**Created**: November 22, 2025
**Status**: Draft Specification

## Overview

This document defines the structured command grammar for Elle's Command Mode, enabling concise voice commands with shortcuts, navigation, and task delegation.

## Grammar Philosophy

**Conversational Mode** (Milestone 1): Natural language, verbose
- "Hey Elle, start a new thread for baking bread"
- "Can you switch to the biochar thread please?"

**Command Mode** (Milestone 2): Structured, concise
- "thread 3" or "t3"
- "back" or "b"
- "run analyze_code"

## BNF-Like Grammar Specification

```bnf
<command> ::= <navigation> | <thread_op> | <task_op> | <query>

# Navigation Commands
<navigation> ::= "back" | "b" | "next" | "n" | "home" | "h" | "up"

# Thread Operations
<thread_op> ::= <thread_switch> | <thread_list> | <thread_create> | <thread_delete>

<thread_switch> ::= "thread" <number>
                  | "t" <number>
                  | "#" <number>
                  | <thread_name>

<thread_list> ::= "threads" | "ts" | "list"

<thread_create> ::= "new" <topic>
                  | "create" <topic>
                  | "+" <topic>

<thread_delete> ::= "delete" <number>
                  | "del" <number>
                  | "-" <number>

# Task Operations
<task_op> ::= <task_run> | <task_stop> | <task_pause> | <task_resume> | <task_status>

<task_run> ::= "run" <task_name>
             | "exec" <task_name>
             | ">" <task_name>

<task_stop> ::= "stop" | "s" | "halt"

<task_pause> ::= "pause" | "p"

<task_resume> ::= "resume" | "continue" | "c"

<task_status> ::= "status" | "st" | "?"

# Query Operations
<query> ::= <question> | <reference_query>

<reference_query> ::= "@" <entity_name>  # Quick entity lookup
                    | "find" <entity_name>

# Primitives
<number> ::= [0-9]+
<topic> ::= [a-zA-Z0-9_ ]+
<task_name> ::= [a-zA-Z0-9_]+
<entity_name> ::= [a-zA-Z0-9_]+
<thread_name> ::= [a-zA-Z0-9_ ]+
<question> ::= [natural language text]
```

## Command Examples

### Navigation
```
"back"           → Go to previous thread
"b"              → Short for back
"next"           → Go to next thread
"n"              → Short for next
"home"           → Go to default/main thread
"h"              → Short for home
"up"             → Go to parent context (if nested)
```

### Thread Operations
```
# Switch
"thread 3"       → Switch to thread ID 3
"t3"             → Switch to thread 3 (compact)
"#3"             → Switch to thread 3 (reference style)
"baking"         → Switch to thread matching "baking" (fuzzy)

# List
"threads"        → List all threads
"ts"             → List all threads (compact)
"list"           → List all threads

# Create
"new baking"     → Create thread named "baking"
"create biochar" → Create thread named "biochar"
"+ greenhouse"   → Create thread named "greenhouse" (operator style)

# Delete
"delete 3"       → Delete thread 3
"del 3"          → Delete thread 3 (compact)
"- 3"            → Delete thread 3 (operator style)
```

### Task Operations
```
# Run
"run analyze_code"     → Execute task "analyze_code"
"exec test_suite"      → Execute task "test_suite"
"> build_project"      → Execute task "build_project" (shell style)

# Control
"stop"           → Stop current task
"s"              → Short for stop
"pause"          → Pause current task
"p"              → Short for pause
"resume"         → Resume paused task
"c"              → Short for continue
"status"         → Show task status
"?"              → Show task status (compact)
```

### Quick References
```
"@Thompson"      → Look up entity "Thompson" (Thompson Sampling)
"@attention"     → Look up entity "attention" (attention mechanism)
"find Matryoshka" → Search for "Matryoshka" in knowledge base
```

## Context-Aware Disambiguation

When multiple interpretations exist, Elle uses context to disambiguate:

**Ambiguous**: "3"
- In thread list context → Switch to thread 3
- In task context → Resume task 3
- No context → Ask for clarification

**Ambiguous**: "back"
- In nested context (e.g., inside thread → task → subtask) → Go up one level
- In thread context → Previous thread in history
- In navigation history → Browser-style back

## Command Chaining

Multiple commands can be chained with semicolons:

```
"t3; run analyze"    → Switch to thread 3, then run analyze task
"new research; > search_papers" → Create thread, then run task
"threads; #2"        → List threads, then switch to #2
```

## Natural Language Fallback

If a command doesn't match the structured grammar, Elle falls back to Conversational Mode (Milestone 1):

```
"What's the weather?" → Not a structured command → Process as natural query
"Tell me about..." → Not a structured command → Conversational mode
```

## Implementation Phases

### Phase 1.1: Navigation + Thread Shortcuts (Week 5-6)
- Implement navigation commands (back, next, home)
- Implement thread shortcuts (t3, #3)
- Thread creation/deletion shortcuts

### Phase 1.2: Task Operations (Week 7-8)
- Implement task execution commands
- Task control (stop, pause, resume)
- Task status queries

### Phase 1.3: Quick References + Chaining (Week 9-10)
- Entity lookup (@mention syntax)
- Command chaining with semicolons
- Context-aware disambiguation

### Phase 1.4: Testing + Refinement (Week 11-12)
- Comprehensive test suite
- Edge case handling
- User feedback integration

## Parser Implementation Strategy

### Option 1: Regex-Based (Fast, Simple)
- Pattern matching for each command type
- Fallback to Conversational Mode for unmatched
- **Pros**: Fast, easy to debug
- **Cons**: Limited composability

### Option 2: PEG Parser (Powerful, Complex)
- Parsing Expression Grammar
- Full BNF support
- **Pros**: Compositional, extensible
- **Cons**: More complex, slower

### Recommended: Hybrid Approach
- Regex for simple commands (navigation, shortcuts)
- PEG for complex patterns (chaining, nested)
- Fallback to Conversational Mode

## Backward Compatibility

All Milestone 1 (Conversational Mode) commands remain supported:

```
# Milestone 1 (Verbose)
"start a new thread for baking"  ✓ Still works

# Milestone 2 (Concise)
"+ baking"  ✓ New shortcut

Both are valid and produce the same result.
```

## Voice Feedback

Elle provides immediate voice feedback for structured commands:

```
User: "t3"
Elle: "Thread 3" (brief confirmation, <500ms)

User: "threads"
Elle: "3 threads: Baking, Biochar, Greenhouse" (concise summary)

User: "run analyze"
Elle: "Running analyze" (immediate acknowledgment)
       [pause for task execution]
       "Analysis complete" (completion notification)
```

## Next Steps

1. ✅ Create grammar specification (this document)
2. ⬜ Implement CommandGrammarParser class
3. ⬜ Extend LLMParser with structured patterns
4. ⬜ Add context-aware disambiguation
5. ⬜ Implement command chaining
6. ⬜ Create comprehensive test suite
7. ⬜ Integrate with VoiceAssistant

---

**Related Documents**:
- `voice_ux_metaprompt_refined.md` - Complete Voice UX specification
- `elle/voice/llm_parser.py` - Milestone 1 parser (conversational)
- `elle/voice/threads.py` - Thread management system
