# Promptly Skills for Claude

Skills are versioned, composable units of functionality that can include code, prompts, data files, and metadata. They're designed to work seamlessly with Claude and other AI models.

## Overview

**Skills** extend Promptly's prompt management with:
- **Version control**: Track skill evolution across branches
- **Multi-file support**: Attach Python scripts, .sip files, JSON configs, docs, etc.
- **Runtime metadata**: Tag skills with runtime info (e.g., `runtime: "claude"`)
- **Execution scaffolding**: Prepare skill payloads and execute with LLM runners

## Quick Start

### 1. Initialize Promptly (if not already done)

```bash
python promptly_cli.py init
```

### 2. Create a Skill

```bash
# Create a skill with description
python promptly_cli.py skill add "data_analyzer" "Analyze CSV data and generate insights"

# Or from Python
from promptly import Promptly
p = Promptly()
p.add_skill("data_analyzer", "Analyze CSV data and generate insights", 
            metadata={"runtime": "claude", "tags": ["data", "analytics"]})
```

### 3. Attach Files to Skills

```bash
# Attach a Python script
python promptly_cli.py skill add-file "data_analyzer" ./analyzer.py

# Attach configuration
python promptly_cli.py skill add-file "data_analyzer" ./config.json

# Attach a .sip file (or any other file type)
python promptly_cli.py skill add-file "data_analyzer" ./schema.sip
```

### 4. List Your Skills

```bash
python promptly_cli.py skill list
```

Output:
```
Name                           Version    Commit          Created
--------------------------------------------------------------------------------
data_analyzer                  v1         a3f9d2e10bc4    2025-10-23 14:32:11
text_summarizer                v2         b8e4a1c92def    2025-10-23 15:10:45
```

### 5. Get Skill Details

```bash
python promptly_cli.py skill get "data_analyzer"
```

Output:
```
Name: data_analyzer
Description: Analyze CSV data and generate insights
Version: 1
Branch: main
Commit: a3f9d2e10bc4
Created: 2025-10-23 14:32:11
Metadata: {'runtime': 'claude', 'tags': ['data', 'analytics']}
```

### 6. View Skill Files

```bash
python promptly_cli.py skill files "data_analyzer"
```

Output:
```
Filename                                 Type       Created
--------------------------------------------------------------------------------
analyzer.py                              py         2025-10-23 14:33:20
config.json                              json       2025-10-23 14:34:05
schema.sip                               sip        2025-10-23 14:35:12
```

## Python API

### Basic Skill Management

```python
from promptly import Promptly

p = Promptly()

# Add a skill
p.add_skill(
    name="text_processor",
    description="Process and transform text data",
    metadata={
        "runtime": "claude",
        "version": "1.0",
        "tags": ["nlp", "text"]
    }
)

# Get a skill
skill = p.get_skill("text_processor")
print(skill)
# {
#   'name': 'text_processor',
#   'description': 'Process and transform text data',
#   'version': 1,
#   'branch': 'main',
#   'commit_hash': 'abc123def456',
#   'metadata': {'runtime': 'claude', ...}
# }

# List all skills
skills = p.list_skills()
for skill in skills:
    print(f"{skill['name']} v{skill['version']}")
```

### Attaching Files

```python
# Attach files to a skill
p.add_skill_file("text_processor", "./processor.py")
p.add_skill_file("text_processor", "./test_data.json")
p.add_skill_file("text_processor", "./prompt_template.txt")

# Get all files for a skill
files = p.get_skill_files("text_processor")
for f in files:
    print(f"{f['filename']} ({f['filetype']})")
```

### Claude-Specific Helpers

```python
# Set runtime to Claude
p.set_skill_runtime("text_processor", runtime="claude")

# Validate skill is Claude-compatible
is_valid = p.validate_skill_for_claude("text_processor")
if is_valid:
    print("Skill is ready for Claude!")

# Prepare skill payload for execution
payload = p.prepare_skill_payload("text_processor")
print(payload)
# {
#   'skill_name': 'text_processor',
#   'description': 'Process and transform text data',
#   'version': 1,
#   'files': [
#     {'filename': 'processor.py', 'filetype': 'py', 'content': '...'},
#     {'filename': 'test_data.json', 'filetype': 'json', 'content': '...'}
#   ],
#   'metadata': {'runtime': 'claude', ...}
# }
```

### Executing Skills

```python
# Define an executor function (example with Ollama)
def ollama_executor(prompt, model):
    from ultraprompt_ollama import ollama_run
    return ollama_run(prompt, model=model)

# Execute a skill
result = p.execute_skill(
    skill_name="text_processor",
    user_input="Process this text: 'Hello, World!'",
    model="llama3.2:3b",
    executor_func=ollama_executor
)

print(result['result'])
```

### Execution with Claude (via API)

```python
# Example executor for Claude API
def claude_executor(prompt, model):
    import anthropic
    client = anthropic.Anthropic(api_key="your-api-key")
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# Execute skill with Claude
result = p.execute_skill(
    skill_name="data_analyzer",
    user_input="Analyze the attached CSV data",
    model="claude-3-5-sonnet-20241022",
    executor_func=claude_executor
)
```

## Use Cases

### 1. Code Analysis Skills

```python
# Create a skill for code review
p.add_skill(
    "code_reviewer",
    "Review Python code for best practices and bugs",
    metadata={"runtime": "claude", "language": "python"}
)

# Attach code files
p.add_skill_file("code_reviewer", "./target_code.py")
p.add_skill_file("code_reviewer", "./style_guide.md")

# Execute review
result = p.execute_skill(
    "code_reviewer",
    user_input="Review this code for security issues",
    executor_func=claude_executor
)
```

### 2. Data Processing Skills

```python
# Create data processing skill with schema
p.add_skill(
    "etl_processor",
    "Extract, transform, and load data according to schema",
    metadata={"runtime": "claude", "type": "data-processing"}
)

p.add_skill_file("etl_processor", "./schema.json")
p.add_skill_file("etl_processor", "./sample_data.csv")
p.add_skill_file("etl_processor", "./transformation_rules.py")
```

### 3. Documentation Generation

```python
p.add_skill(
    "doc_generator",
    "Generate comprehensive documentation from code",
    metadata={"runtime": "claude", "output": "markdown"}
)

p.add_skill_file("doc_generator", "./codebase_summary.txt")
p.add_skill_file("doc_generator", "./api_examples.py")
```

## Versioning Skills

Skills use the same versioning system as prompts:

```python
# Create initial version
p.add_skill("my_skill", "Version 1 description")

# Update skill (creates version 2)
p.add_skill("my_skill", "Version 2 with improvements",
            metadata={"runtime": "claude", "version": "2.0"})

# Get specific version
skill_v1 = p.get_skill("my_skill", version=1)
skill_v2 = p.get_skill("my_skill", version=2)

# Execute specific version
result = p.execute_skill("my_skill", version=1, executor_func=executor)
```

## Branching Skills

```python
# Create a branch for experimental features
p.branch("experimental")
p.checkout("experimental")

# Add/modify skills on experimental branch
p.add_skill("new_feature", "Experimental feature")

# Switch back to main
p.checkout("main")

# Skills on main branch remain unchanged
```

## Metadata Best Practices

Recommended metadata fields:

```python
metadata = {
    "runtime": "claude",           # Target runtime
    "version": "1.0.0",            # Semantic versioning
    "tags": ["nlp", "analysis"],   # Categorization
    "author": "Your Name",         # Attribution
    "license": "MIT",              # License info
    "requires": ["pandas", "numpy"], # Dependencies
    "model": "claude-3-5-sonnet",  # Recommended model
    "max_tokens": 4096,            # Token limits
    "temperature": 0.7             # Model parameters
}

p.add_skill("my_skill", "Description", metadata=metadata)
```

## Integration with Ultraprompt

Combine skills with ultraprompt for enhanced prompting:

```python
# Create a skill with ultraprompt template
p.add_skill(
    "enhanced_analyzer",
    "Data analysis with ultraprompt enhancement"
)

# Attach ultraprompt template
p.add_skill_file("enhanced_analyzer", "./ultraprompt_template.txt")

# Execute with ultraprompt expansion
from ultraprompt_ollama import ultraprompt_with_ollama

def enhanced_executor(prompt, model):
    # First expand with ultraprompt
    expanded = ultraprompt_with_ollama(prompt, model=model)
    return expanded

result = p.execute_skill(
    "enhanced_analyzer",
    user_input="Analyze sales data",
    executor_func=enhanced_executor
)
```

## Advanced: Skill Chains

Execute multiple skills in sequence:

```python
# Skill 1: Extract data
p.add_skill("extractor", "Extract data from sources")
p.add_skill_file("extractor", "./extraction_rules.py")

# Skill 2: Transform data
p.add_skill("transformer", "Transform extracted data")
p.add_skill_file("transformer", "./transformation_schema.json")

# Skill 3: Generate report
p.add_skill("reporter", "Generate analysis report")
p.add_skill_file("reporter", "./report_template.md")

# Execute in sequence
extract_result = p.execute_skill("extractor", user_input="Extract from API", 
                                 executor_func=executor)

transform_result = p.execute_skill("transformer", 
                                   user_input=extract_result['result'],
                                   executor_func=executor)

final_report = p.execute_skill("reporter", 
                               user_input=transform_result['result'],
                               executor_func=executor)
```

## File Type Support

Skills support any file type:

- **Code**: `.py`, `.js`, `.ts`, `.java`, `.cpp`, etc.
- **Data**: `.json`, `.csv`, `.yaml`, `.xml`, `.parquet`
- **Configs**: `.toml`, `.ini`, `.env`, `.conf`
- **Docs**: `.md`, `.txt`, `.rst`, `.pdf` (as binary)
- **Custom**: `.sip` or any proprietary format
- **Binary**: Stored with metadata, execution handles gracefully

## Storage Structure

Skills are stored in `.promptly/skills/`:

```
.promptly/
├── skills/
│   ├── data_analyzer/
│   │   ├── skill.yaml          # Skill metadata
│   │   ├── analyzer.py         # Attached files
│   │   ├── config.json
│   │   └── schema.sip
│   └── text_processor/
│       ├── skill.yaml
│       └── processor.py
└── promptly.db                 # Database with versions
```

## Command Reference

### CLI Commands

```bash
# Skill management
python promptly_cli.py skill add <name> [description]
python promptly_cli.py skill list
python promptly_cli.py skill get <name> [version]

# File management
python promptly_cli.py skill add-file <skill> <filepath>
python promptly_cli.py skill files <skill>
```

### Python API

```python
# Core methods
p.add_skill(name, description, metadata)
p.get_skill(name, version, commit_hash)
p.list_skills(branch)
p.add_skill_file(skill_name, filepath, filetype)
p.get_skill_files(skill_name)

# Claude helpers
p.set_skill_runtime(skill_name, runtime)
p.validate_skill_for_claude(skill_name)
p.prepare_skill_payload(skill_name, version)
p.execute_skill(skill_name, user_input, model, executor_func, version)
```

## Next Steps

1. **Create your first skill**: Start with a simple text processing or data analysis skill
2. **Attach files**: Add scripts, configs, and data files to your skills
3. **Set up execution**: Create an executor function for your preferred model (Claude, Ollama, etc.)
4. **Version and iterate**: Update skills as you refine them, branch for experiments
5. **Share skills**: Export skill directories to share with team members

## Tips

- Use descriptive skill names (e.g., `csv_analyzer` not `skill1`)
- Include comprehensive descriptions for future reference
- Tag skills with metadata for easy filtering and discovery
- Version skills semantically (1.0, 1.1, 2.0) via metadata
- Store sample data with skills for testing
- Document expected inputs/outputs in skill description
- Use branches for experimental features
- Keep skills focused (single responsibility principle)

## Troubleshooting

**Skill not found**: Ensure you've called `p.add_skill()` and are on the correct branch

**File read errors**: Binary files are handled gracefully; check file paths are absolute or relative to working directory

**Execution errors**: Ensure your executor function matches signature: `executor(prompt: str, model: str) -> str`

**Runtime validation**: Skills without `runtime` metadata are considered universal; explicitly set for best results

---

For more information, see:
- [README.md](./README.md) - Promptly overview
- [TUTORIAL.md](./TUTORIAL.md) - General Promptly tutorial
- [QUICKSTART_OLLAMA.md](./QUICKSTART_OLLAMA.md) - Ollama integration
