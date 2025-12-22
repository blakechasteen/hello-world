---
description: Invoke Promptly prompts, chains, and microprompts for prompt management
---

# Promptly Slash Command

You have access to the Promptly prompt management system through the `promptly` CLI. Use these capabilities to help the user manage and execute their prompts.

## Available Actions

### Prompt Management

**List all prompts:**
```bash
python -m promptly.cli.main list
```

**Get a prompt by name:**
```bash
python -m promptly.cli.main get <name> [--version <version>]
```

**Add or update a prompt:**
```bash
python -m promptly.cli.main add <name> "<content>" [--tags tag1,tag2]
```

**View prompt history:**
```bash
python -m promptly.cli.main log <name>
```

### Branch Management

**Create a branch:**
```bash
python -m promptly.cli.main branch <name>
```

**Switch branches:**
```bash
python -m promptly.cli.main checkout <branch>
```

**List branches:**
```bash
python -m promptly.cli.main branches
```

### Skill Management

**Add a skill (reusable parameterized prompt):**
```bash
python -m promptly.cli.main skill add <name> "<template>" --description "<desc>" --inputs '{"var": "description"}'
```

**Run a skill:**
```bash
python -m promptly.cli.main skill run <name> --inputs '{"var": "value"}'
```

**List skills:**
```bash
python -m promptly.cli.main skill list
```

### Chain Execution

**Create a chain (multi-step prompt pipeline):**
```bash
python -m promptly.cli.main chain create <name> --steps '[{"prompt": "step1"}, {"prompt": "step2"}]'
```

**Run a chain:**
```bash
python -m promptly.cli.main chain run <name> --inputs '{"var": "value"}'
```

### Evaluation with LLM Judge

**Evaluate a prompt:**
```bash
python -m promptly.cli.main eval <name> [--criteria clarity,accuracy,relevance]
```

## User Request Handling

Based on the user's request, determine the appropriate action:

1. **"list prompts"** / **"show prompts"** / **"what prompts do I have"**
   - Run `list` command

2. **"get prompt X"** / **"show prompt X"** / **"what's in X"**
   - Run `get <name>` command

3. **"add prompt X"** / **"create prompt X"** / **"save this as X"**
   - Run `add <name> "<content>"` command
   - Ask user for content if not provided

4. **"run prompt X"** / **"execute X"** / **"use prompt X"**
   - First `get` the prompt, then show it to user or execute with provided inputs

5. **"evaluate prompt X"** / **"check quality of X"** / **"rate prompt X"**
   - Run `eval <name>` command

6. **"create skill"** / **"make this reusable"**
   - Run `skill add` command with appropriate template

7. **"run skill X"** / **"use skill X with..."**
   - Run `skill run <name> --inputs '{...}'`

8. **"create chain"** / **"multi-step workflow"**
   - Run `chain create` with appropriate steps

9. **"history of X"** / **"versions of X"**
   - Run `log <name>` command

10. **"switch to branch X"** / **"use branch X"**
    - Run `checkout <branch>` command

## Storage Locations

- **Global prompts:** `~/.promptly/prompts.db`
- **Project-local prompts:** `.promptly/prompts.db` (inherits from global)
- **Project-local overrides global** when prompt names conflict

## Examples

### Example 1: User wants to save a code review prompt
User: "Save this as my code-review prompt: Review this code for bugs, security issues, and best practices"

Action:
```bash
python -m promptly.cli.main add "code-review" "Review this code for bugs, security issues, and best practices" --tags code,review
```

### Example 2: User wants to create a reusable summarization skill
User: "Create a summarization skill that takes text and length"

Action:
```bash
python -m promptly.cli.main skill add "summarize" "Summarize the following text in {length} sentences:\n\n{text}" --description "Summarize text to specified length" --inputs '{"text": "The text to summarize", "length": "Number of sentences"}'
```

### Example 3: User wants to evaluate a prompt
User: "How good is my code-review prompt?"

Action:
```bash
python -m promptly.cli.main eval "code-review" --criteria clarity,accuracy,relevance,completeness
```

## Notes

- All commands use the `python -m promptly.cli.main` pattern
- JSON arguments should be properly escaped for the shell
- The system auto-detects LLM backend: tries Ollama first, falls back to Claude API
- Use `--scope global` to explicitly use global storage, `--scope local` for project-local