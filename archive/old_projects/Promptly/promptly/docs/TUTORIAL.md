# Promptly Tutorial: Real-World Examples

## Scenario 1: Content Creation Pipeline

```bash
promptly init

# Create prompts for each stage
promptly add research "Research: {topic}"
promptly add outline "Create outline: {output}"
promptly add draft "Write from outline: {output}"
promptly add edit "Edit and improve: {output}"

# Create chain
promptly chain create content-pipeline research outline draft edit

# Run it
echo "topic: 'AI Ethics'" > input.yaml
promptly chain run content-pipeline input.yaml
```

## Scenario 2: A/B Testing

```bash
# Main branch - baseline
promptly add ad-copy "Write ad for: {product}"

# Variant A - emotional
promptly branch variant-a
promptly checkout variant-a
promptly add ad-copy "Write emotional ad for: {product}"

# Variant B - data-driven
promptly checkout main
promptly branch variant-b
promptly checkout variant-b
promptly add ad-copy "Write data-driven ad for: {product}"

# Test both
promptly checkout variant-a
promptly eval run ad-copy tests.json

promptly checkout variant-b
promptly eval run ad-copy tests.json
```

## Scenario 3: Iterative Improvement

```bash
# V1: Basic
promptly add reviewer "Review code: {code}"

# V2: More specific
promptly add reviewer "Review for bugs and style: {code}"

# V3: Structured
promptly add reviewer "Review code for:
1. Bugs
2. Performance
3. Style
Code: {code}"

# Track progress
promptly log --name reviewer
```

## Scenario 4: Team Collaboration

```bash
# Alice's feature
promptly branch alice-feature
promptly checkout alice-feature
promptly add new-prompt "Feature A: {input}"

# Bob's feature
promptly checkout main
promptly branch bob-feature
promptly checkout bob-feature
promptly add different-prompt "Feature B: {input}"

# Merge best to main
promptly checkout main
```

## Scenario 5: Production Deployment

```bash
# Mark production version
promptly add customer-support "Respond: {inquiry}" \
  --metadata '{"env": "production", "version": "v1.0"}'

# Test new version in dev
promptly branch development
promptly checkout development
promptly add customer-support "New version: {inquiry}"

# Evaluate
promptly eval run customer-support tests.json

# Deploy if good
promptly checkout main
promptly add customer-support "New version: {inquiry}" \
  --metadata '{"env": "production", "version": "v1.1"}'

# Rollback if needed
promptly get customer-support --version 1
```

## Integration Patterns

### Pattern 1: Python Library

```python
from promptly import Promptly

class PromptManager:
    def __init__(self, llm_client):
        self.promptly = Promptly()
        self.llm = llm_client
    
    def execute(self, prompt_name, **kwargs):
        prompt_data = self.promptly.get(prompt_name)
        formatted = prompt_data['content'].format(**kwargs)
        return self.llm.complete(formatted)
```

### Pattern 2: CLI Wrapper

```bash
#!/bin/bash
PROMPT_NAME=$1
INPUT=$2

CONTENT=$(promptly get $PROMPT_NAME)
# Use with your LLM API
```

### Pattern 3: CI/CD

```yaml
# .github/workflows/test-prompts.yml
name: Test Prompts
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: promptly eval run all-prompts tests.json
```

## Pro Tips

1. **Descriptive Names**: `summarize-news-article` not `sum1`
2. **Use Metadata**: Track model, temp, use case
3. **Branch Conventions**: `feature/`, `experiment/`, `hotfix/`
4. **Regular Testing**: Run eval before changes
5. **Document Changes**: Use metadata for change reasons

## Common Workflows

### Daily Workflow
```bash
promptly log --limit 5        # Check what's new
promptly add new "..."        # Add new prompt
promptly eval run new tests.json  # Test it
```

### Feature Development
```bash
promptly branch feature-x
promptly checkout feature-x
# Develop and test
promptly eval run prompt tests.json
# Merge to main when ready
```

### Production Deploy
```bash
promptly branch staging
# Test in staging
promptly eval run prompt prod-tests.json
# Deploy to main
promptly checkout main
```

---

**Ready to build? Start with: `promptly init` 🚀**
