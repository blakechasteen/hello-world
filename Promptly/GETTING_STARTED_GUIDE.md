# Promptly Getting Started Guide

**Welcome to Promptly!** This guide will take you from zero to productive in 30 minutes.

---

## Table of Contents

1. [Quick Start (10 Minutes)](#quick-start-10-minutes)
2. [Installation](#installation)
3. [Your First Prompts](#your-first-prompts)
4. [Common Workflows](#common-workflows)
5. [Configuration Guide](#configuration-guide)
6. [Troubleshooting FAQ](#troubleshooting-faq)
7. [Next Steps](#next-steps)

---

## Quick Start (10 Minutes)

### Prerequisites
- Python 3.7+
- pip

### Installation
```bash
# Clone repository
cd /path/to/promptly

# Install core dependencies
pip install click PyYAML

# Optional: Install enhanced features
pip install rich prompt_toolkit textual jinja2 fastapi uvicorn
```

### First Steps
```bash
# Create a project directory
mkdir my-prompts
cd my-prompts

# Initialize Promptly repository
python -m promptly.promptly init

# Add your first prompt
python -m promptly.promptly add greeter "Hello, {name}!"

# Get the prompt back
python -m promptly.promptly get greeter

# List all prompts
python -m promptly.promptly list

# Show commit history
python -m promptly.promptly log
```

**Congratulations!** You've created your first Promptly repository. 🎉

---

## Installation

### Development Installation

#### 1. Clone and Setup
```bash
# Clone repository
git clone <repository-url>
cd Promptly

# Navigate to promptly module
cd promptly

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install click PyYAML
```

#### 2. Install Optional Dependencies

**Rich CLI Features:**
```bash
pip install rich prompt_toolkit pygments textual
```

**Template Engine:**
```bash
pip install jinja2
```

**REST API:**
```bash
pip install fastapi uvicorn pydantic python-multipart
```

**Analytics:**
```bash
pip install psutil pandas matplotlib
```

**Advanced Evaluators:**
```bash
pip install sentence-transformers scikit-learn spacy
python -m spacy download en_core_web_sm
```

**All Features:**
```bash
pip install click PyYAML rich prompt_toolkit pygments textual \
  jinja2 fastapi uvicorn pydantic python-multipart \
  psutil pandas matplotlib sentence-transformers scikit-learn spacy
```

#### 3. Verify Installation
```bash
# Check basic CLI works
python -m promptly.promptly --version

# Run verification script
./verify_installation.sh

# Test imports
python -c "from promptly import Promptly; print('✓ Core working')"
python -c "from promptly.templates.engine import TemplateEngine; print('✓ Templates working')"
python -c "from promptly.api.main import app; print('✓ API working')"
```

### Production Installation

#### Using pip (recommended)
```bash
# Install from source
pip install -e .

# Or install from PyPI (when published)
pip install promptly

# Verify
promptly --version
```

#### Using Docker
```bash
# Build image
docker build -t promptly:latest .

# Run container
docker run -d -p 8000:8000 \
  -v $(pwd)/.promptly:/app/.promptly \
  promptly:latest

# Use CLI in container
docker exec -it <container-id> promptly list
```

#### System-wide Installation
```bash
# Install globally
sudo pip install -e .

# Add to PATH
export PATH="$PATH:/path/to/promptly/bin"

# Install shell completion
cd shell_completion
sudo ./install.sh
```

---

## Your First Prompts

### Tutorial 1: Basic Prompt Management

```python
from promptly import Promptly

# Initialize
p = Promptly()
p.init()

# Add a simple prompt
p.add("greeter", "Hello, {name}!")

# Get it back
prompt = p.get("greeter")
print(prompt['content'])  # "Hello, {name}!"
print(prompt['version'])  # 1

# Update the prompt (creates version 2)
p.add("greeter", "Hey there, {name}! How are you?")

# Get latest version
latest = p.get("greeter")
print(latest['version'])  # 2

# Get specific version
v1 = p.get("greeter", version=1)
print(v1['content'])  # "Hello, {name}!"

# List all prompts
prompts = p.list_prompts()
for prompt in prompts:
    print(f"{prompt['name']} v{prompt['version']}")

# View history
history = p.log(name="greeter")
for commit in history:
    print(f"{commit['commit_hash']}: {commit['name']} v{commit['version']}")
```

### Tutorial 2: Working with Branches

```python
from promptly import Promptly

p = Promptly()
p.init()

# Create prompts on main branch
p.add("summarizer", "Summarize: {text}")
p.add("translator", "Translate to {language}: {text}")

# Create experimental branch
p.branch("experiment", from_branch="main")

# Switch to experiment
p.checkout("experiment")

# Modify prompts on experiment
p.add("summarizer", "Provide a concise summary of: {text}")

# Check which branch we're on
current = p._get_current_branch()
print(f"Current branch: {current}")  # experiment

# Switch back to main
p.checkout("main")

# Get summarizer on main (unchanged)
main_prompt = p.get("summarizer")
print(main_prompt['content'])  # "Summarize: {text}"

# Get summarizer on experiment
p.checkout("experiment")
exp_prompt = p.get("summarizer")
print(exp_prompt['content'])  # "Provide a concise summary of: {text}"
```

### Tutorial 3: Prompt Evaluation

```python
from promptly import Promptly

p = Promptly()
p.init()
p.add("sentiment", "Classify sentiment as positive/negative/neutral: {text}")

# Define test cases
test_cases = [
    {
        'inputs': {'text': 'I love this product!'},
        'expected': 'positive',
        'evaluator': lambda actual, expected: 1.0 if expected.lower() in actual.lower() else 0.0
    },
    {
        'inputs': {'text': 'This is terrible.'},
        'expected': 'negative',
        'evaluator': lambda actual, expected: 1.0 if expected.lower() in actual.lower() else 0.0
    },
    {
        'inputs': {'text': 'It\'s okay.'},
        'expected': 'neutral',
        'evaluator': lambda actual, expected: 1.0 if expected.lower() in actual.lower() else 0.0
    }
]

# Mock model function (replace with your actual model)
def mock_model(prompt):
    # Simple keyword-based classification
    if 'love' in prompt or 'great' in prompt:
        return 'positive'
    elif 'terrible' in prompt or 'awful' in prompt:
        return 'negative'
    else:
        return 'neutral'

# Run evaluation
results = p.eval_prompt("sentiment", test_cases, model_func=mock_model)

# Analyze results
total_score = sum(r['score'] for r in results)
avg_score = total_score / len(results)

print(f"Total tests: {len(results)}")
print(f"Average score: {avg_score:.2f}")

for i, result in enumerate(results):
    print(f"\nTest {i+1}:")
    print(f"  Prompt: {result['formatted_prompt'][:50]}...")
    print(f"  Output: {result['actual']}")
    print(f"  Score: {result['score']}")
```

### Tutorial 4: Prompt Chains

```python
from promptly import Promptly

p = Promptly()
p.init()

# Create individual prompts for each step
p.add("extract", "Extract key entities from: {text}")
p.add("categorize", "Categorize these entities: {output}")
p.add("summarize", "Summarize the categories: {output}")

# Create a chain
p.create_chain(
    name="entity_pipeline",
    steps=["extract", "categorize", "summarize"],
    description="Extract entities, categorize them, then summarize"
)

# Mock model function
def mock_model(prompt):
    if "Extract" in prompt:
        return "Apple Inc., iPhone, Tim Cook"
    elif "Categorize" in prompt:
        return "Company: Apple Inc., Product: iPhone, Person: Tim Cook"
    elif "Summarize" in prompt:
        return "Technology company with product and leadership mentions"
    return "Unknown"

# Execute chain
initial_input = {
    'text': 'Apple Inc. CEO Tim Cook announced the new iPhone today.'
}

results = p.execute_chain("entity_pipeline", initial_input, model_func=mock_model)

# View results
print("Chain Execution Results:")
for i, step in enumerate(results):
    print(f"\nStep {i+1}: {step['step']}")
    print(f"  Prompt: {step['prompt'][:60]}...")
    print(f"  Output: {step['output']}")
```

### Tutorial 5: Using Templates

```python
from promptly.templates.engine import TemplateEngine

engine = TemplateEngine()

# Create a template
template = """
System: You are a {{ role }} assistant.

{% if examples %}
Examples:
{{ examples | few_shot }}
{% endif %}

User: {{ query }}
Assistant:
"""

# Render for different roles
for role in ['helpful', 'creative', 'analytical']:
    result = engine.render_string(
        template,
        role=role,
        query="What is machine learning?",
        examples=[
            {'input': 'What is AI?', 'output': 'Artificial Intelligence is...'},
            {'input': 'What is ML?', 'output': 'Machine Learning is...'}
        ]
    )
    print(f"\n{role.upper()} VERSION:")
    print(result)
```

---

## Common Workflows

### Workflow 1: Feature Development

```python
from promptly import Promptly

p = Promptly()

# 1. Create feature branch
p.branch("feature/new-summarizer")
p.checkout("feature/new-summarizer")

# 2. Develop new prompt
p.add("summarizer", "Extract key points from: {text}")

# 3. Test extensively
test_cases = [...]  # Your test cases
results = p.eval_prompt("summarizer", test_cases, model_func=your_model)

# 4. If passing, prepare for merge
avg_score = sum(r['score'] for r in results) / len(results)
if avg_score > 0.8:
    # Get the new content
    new_prompt = p.get("summarizer")

    # Switch to main and update
    p.checkout("main")
    p.add("summarizer", new_prompt['content'], metadata={'merged_from': 'feature/new-summarizer'})

print("Feature merged to main!")
```

### Workflow 2: A/B Testing

```python
from promptly import Promptly
import statistics

p = Promptly()

# Create two variants
p.checkout("main")
p.add("prompt", "Variant A content")

p.branch("variant_b")
p.checkout("variant_b")
p.add("prompt", "Variant B content")

# Test both
test_cases = [...]  # Your test cases

# Test variant A
p.checkout("main")
results_a = p.eval_prompt("prompt", test_cases, model_func=your_model)
scores_a = [r['score'] for r in results_a]

# Test variant B
p.checkout("variant_b")
results_b = p.eval_prompt("prompt", test_cases, model_func=your_model)
scores_b = [r['score'] for r in results_b]

# Compare
mean_a = statistics.mean(scores_a)
mean_b = statistics.mean(scores_b)

print(f"Variant A: {mean_a:.3f}")
print(f"Variant B: {mean_b:.3f}")

# Choose winner
if mean_b > mean_a:
    print("Variant B wins! Merging...")
    variant_b_prompt = p.get("prompt")
    p.checkout("main")
    p.add("prompt", variant_b_prompt['content'])
else:
    print("Variant A wins! Keeping main.")
```

### Workflow 3: Progressive Refinement

```python
from promptly import Promptly

p = Promptly()

# Start with simple prompt
p.add("classifier", "Classify: {text}")

# Iteratively improve
versions = [
    "Classify the sentiment: {text}",
    "Classify the sentiment (positive/negative/neutral): {text}",
    "Analyze the sentiment and classify as positive, negative, or neutral:\n{text}",
]

for i, content in enumerate(versions):
    # Update prompt
    p.add("classifier", content)

    # Evaluate
    results = p.eval_prompt("classifier", test_cases, model_func=your_model)
    avg_score = sum(r['score'] for r in results) / len(results)

    print(f"Version {i+2}: Score = {avg_score:.3f}")

    # Stop if good enough
    if avg_score > 0.9:
        print(f"Optimal version found: v{i+2}")
        break
```

### Workflow 4: Environment Promotion

```python
from promptly import Promptly

p = Promptly()

# Development → Staging
p.checkout("development")
dev_prompts = p.list_prompts()

# Thorough testing in dev
for prompt in dev_prompts:
    results = p.eval_prompt(prompt['name'], test_cases, model_func=your_model)
    avg_score = sum(r['score'] for r in results) / len(results)

    if avg_score < 0.8:
        print(f"WARNING: {prompt['name']} scored {avg_score:.2f}")
        continue

# If all pass, promote to staging
p.checkout("staging")
for prompt in dev_prompts:
    dev_content = p.get(prompt['name'], branch="development")
    p.add(prompt['name'], dev_content['content'])

# Staging → Production (after manual approval)
p.checkout("production")
staging_prompts = p.list_prompts(branch="staging")
for prompt in staging_prompts:
    staging_content = p.get(prompt['name'], branch="staging")
    p.add(prompt['name'], staging_content['content'])
```

### Workflow 5: API Integration

```python
from promptly.sdk import PromptlyClient

# Initialize client
client = PromptlyClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Create prompt via API
response = client.create_prompt(
    name="summarizer",
    content="Summarize: {text}",
    metadata={'author': 'alice'}
)

# Use in application
def summarize_text(text):
    # Get latest prompt
    prompt_data = client.get_prompt("summarizer")

    # Format with input
    formatted = prompt_data['content'].format(text=text)

    # Call your model
    summary = your_model(formatted)

    return summary

# Use it
result = summarize_text("Long article text...")
print(result)
```

---

## Configuration Guide

### Repository Configuration

Create `.promptly/config.yaml`:

```yaml
# Repository settings
repository:
  default_branch: main
  auto_commit: true
  require_metadata: false

# Storage backend
storage:
  backend: sqlite  # Options: sqlite, postgresql, mongodb, redis, json, git
  path: .promptly/promptly.db

  # PostgreSQL config (if using)
  postgresql:
    host: localhost
    port: 5432
    database: promptly
    user: promptly_user
    password: ${PROMPTLY_DB_PASSWORD}

# Evaluation settings
evaluation:
  default_evaluator: keyword
  parallel_execution: true
  max_workers: 4
  timeout_seconds: 30

# Chain processing
chains:
  max_steps: 50
  timeout_seconds: 300
  continue_on_error: false
  trace_execution: true

# Template engine
templates:
  template_dirs:
    - .promptly/templates
    - templates
  autoescape: false
  cache_size: 400

# Analytics
analytics:
  enabled: true
  retention_days: 30
  sample_interval_seconds: 60
  performance_monitoring: true
  quality_tracking: true
  usage_analytics: true

# API settings
api:
  host: 0.0.0.0
  port: 8000
  reload: false
  workers: 4
  log_level: info
  cors_origins:
    - http://localhost:3000
    - https://your-domain.com
  rate_limit_per_minute: 60
  rate_limit_burst: 10
```

### User Configuration

Create `~/.promptly/config.yaml`:

```yaml
# User preferences
user:
  name: Alice
  email: alice@example.com
  editor: vim

# CLI settings
cli:
  color: true
  pager: less
  table_format: rich

# API defaults
api:
  default_base_url: http://localhost:8000
  default_api_key: ${PROMPTLY_API_KEY}
  timeout: 30
  max_retries: 3
```

### Environment Variables

```bash
# Required
export PROMPTLY_API_KEY="your-api-key-here"

# Optional
export PROMPTLY_DB_PASSWORD="db-password"
export PROMPTLY_BASE_URL="http://localhost:8000"
export PROMPTLY_LOG_LEVEL="info"
export PROMPTLY_STORAGE_BACKEND="postgresql"

# Add to ~/.bashrc or ~/.zshrc
echo 'export PROMPTLY_API_KEY="your-key"' >> ~/.bashrc
```

---

## Troubleshooting FAQ

### Q: "Repository not initialized" error

**A:** Run `promptly init` in your project directory.

```bash
cd /path/to/your/project
python -m promptly.promptly init
```

### Q: Import errors for optional dependencies

**A:** Install optional features you need:

```bash
# For templates
pip install jinja2

# For rich CLI
pip install rich prompt_toolkit

# For API
pip install fastapi uvicorn

# For all features
pip install click PyYAML jinja2 rich prompt_toolkit fastapi uvicorn
```

### Q: SQLite database locked

**A:** Close other Promptly processes or use PostgreSQL for concurrent access:

```yaml
# .promptly/config.yaml
storage:
  backend: postgresql
  postgresql:
    host: localhost
    database: promptly
```

### Q: API key authentication failing

**A:** Ensure API key is set correctly:

```bash
# Set environment variable
export PROMPTLY_API_KEY="dev-test-key-1234567890"

# Or pass directly
client = PromptlyClient(api_key="dev-test-key-1234567890")
```

### Q: Template rendering fails

**A:** Check template syntax and install Jinja2:

```bash
pip install jinja2

# Debug template
from promptly.templates.engine import TemplateEngine
engine = TemplateEngine()
try:
    result = engine.render_string(template_string, **vars)
except Exception as e:
    print(f"Error: {e}")
```

### Q: Evaluation scores are all zero

**A:** Ensure evaluator function is correct:

```python
# Correct evaluator signature
def my_evaluator(actual, expected):
    # Compare actual vs expected
    score = compute_score(actual, expected)
    return score  # Must be between 0.0 and 1.0

# Use in test case
test_case = {
    'inputs': {'text': 'input'},
    'expected': 'expected output',
    'evaluator': my_evaluator  # Pass function, not result
}
```

### Q: Chain execution hangs

**A:** Check for circular dependencies or set timeout:

```python
# Set timeout in DSL
chain_yaml = """
name: my_chain
timeout_seconds: 60
steps:
  - name: step1
    ...
"""
```

### Q: Out of memory during large operations

**A:** Use batch processing and pagination:

```python
# Process in batches
batch_size = 100
prompts = p.list_prompts()

for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i+batch_size]
    process_batch(batch)
```

### Q: Unicode/encoding errors

**A:** Ensure UTF-8 encoding:

```python
# When writing prompts with special characters
p.add("greeting", "Hello! 👋 こんにちは")

# Set file encoding
with open('prompts.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(data, f)
```

### Q: API server won't start

**A:** Check port availability and dependencies:

```bash
# Check if port 8000 is in use
lsof -i :8000

# Use different port
uvicorn promptly.api.main:app --port 8001

# Install dependencies
pip install fastapi uvicorn
```

### Q: Shell completion not working

**A:** Reinstall completion and reload shell:

```bash
cd shell_completion
./install.sh

# Reload shell
source ~/.bashrc  # or ~/.zshrc
```

---

## Next Steps

### Beginner Track
1. ✅ Complete all tutorials in this guide
2. 📖 Read **COMPLETE_FEATURE_GUIDE.md** for in-depth features
3. 🎯 Build a simple prompt library for your use case
4. 🧪 Set up evaluation for your prompts
5. 🌿 Learn branching for experimentation

### Intermediate Track
1. 🔗 Create multi-step chains for complex workflows
2. 📝 Use templates for prompt composition
3. 🔌 Explore plugin system for custom evaluators
4. 📊 Set up analytics and monitoring
5. 🌐 Deploy REST API for team access

### Advanced Track
1. 🏗️ Build custom storage backend
2. 🧠 Integrate with HoloLoom for neural evaluation
3. 🔄 Implement CI/CD for prompt updates
4. 📈 Create advanced analytics dashboards
5. 🚀 Deploy production system with PostgreSQL + Redis

### Resources
- **Complete Feature Guide** - All features with examples
- **API Reference** - Complete API documentation
- **Extension Development Guide** - Build plugins
- **Production Handbook** - Deploy and scale
- **GitHub Discussions** - Ask questions and share ideas
- **Discord Community** - Real-time help and discussions

---

## Need Help?

- 📚 **Documentation**: See other guides in `/Promptly/docs`
- 🐛 **Bug Reports**: Create GitHub issue
- 💡 **Feature Requests**: Open discussion
- 💬 **Community**: Join Discord server
- 📧 **Email**: support@promptly.dev

**Happy prompting!** 🎉
