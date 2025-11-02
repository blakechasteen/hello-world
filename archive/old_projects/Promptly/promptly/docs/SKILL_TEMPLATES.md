# Promptly Skill Templates

Pre-built, production-ready skills for common AI tasks. Install instantly via MCP or CLI!

## Available Templates (8)

### 1. 🔍 code_reviewer
**Description:** Review code for best practices, bugs, and security issues

**What's included:**
- `review_checklist.md` - Comprehensive review checklist (security, quality, performance, maintainability)
- `reviewer.py` - Template code review function

**Use cases:**
- Pre-commit code reviews
- Security audits
- Best practices enforcement
- Onboarding new developers

**Install:**
```bash
# Via MCP in Claude Desktop
"Install the code_reviewer template"

# Via CLI
python promptly_cli.py skill add code_reviewer "Review code for best practices..."
```

---

### 2. 🌐 api_designer
**Description:** Design RESTful APIs with best practices and OpenAPI specs

**What's included:**
- `design_principles.md` - REST API best practices, naming conventions, status codes
- `openapi_template.yaml` - OpenAPI 3.0 specification template

**Use cases:**
- API architecture design
- OpenAPI documentation generation
- REST API standards compliance
- Microservices design

**Install:**
```bash
"Install the api_designer template"
```

---

### 3. 📊 data_analyzer
**Description:** Analyze datasets and generate insights with visualizations

**What's included:**
- `analysis_template.py` - Pandas-based data analysis functions
- `viz_examples.md` - Visualization patterns (distributions, heatmaps, time series)

**Use cases:**
- Exploratory data analysis
- Dataset profiling
- Statistical summaries
- Pattern detection

**Requires:** pandas, matplotlib

**Install:**
```bash
"Install the data_analyzer template"
```

---

### 4. 📝 documentation_writer
**Description:** Generate comprehensive documentation from code

**What's included:**
- `doc_structure.md` - Documentation structure templates (README, API docs, architecture)
- `doc_generator.py` - Automated README generator

**Use cases:**
- README generation
- API documentation
- Architecture docs
- User guides

**Install:**
```bash
"Install the documentation_writer template"
```

---

### 5. ✅ test_generator
**Description:** Generate comprehensive unit tests for code

**What's included:**
- `test_template.py` - pytest test templates with AAA pattern
- `test_guidelines.md` - Testing best practices and coverage goals

**Use cases:**
- Unit test generation
- Test coverage improvement
- TDD workflows
- Regression testing

**Requires:** pytest

**Install:**
```bash
"Install the test_generator template"
```

---

### 6. 🎯 prompt_engineer
**Description:** Optimize and improve prompts for better AI responses

**What's included:**
- `prompt_patterns.md` - Chain of thought, few-shot learning, role-based prompting, structured output
- `optimizer.py` - Prompt optimization functions

**Use cases:**
- Prompt improvement
- Prompt library curation
- AI workflow optimization
- Instruction tuning

**Install:**
```bash
"Install the prompt_engineer template"
```

---

### 7. 🗄️ sql_designer
**Description:** Design database schemas and write optimized SQL queries

**What's included:**
- `schema_template.sql` - Database schema with indexes, foreign keys
- `query_patterns.md` - Efficient joins, aggregations, pagination, subqueries

**Use cases:**
- Database schema design
- Query optimization
- Migration planning
- Index strategy

**Install:**
```bash
"Install the sql_designer template"
```

---

### 8. ⚠️ error_handler
**Description:** Design robust error handling and logging strategies

**What's included:**
- `error_patterns.py` - Custom exceptions, retry decorators, safe execution
- `logging_config.yaml` - Production logging configuration

**Use cases:**
- Error handling architecture
- Logging infrastructure
- Retry logic
- Exception management

**Install:**
```bash
"Install the error_handler template"
```

---

## Quick Reference

### List All Templates
```bash
# In Claude Desktop
"Show me all skill templates"

# Via MCP tool
promptly_template_list
```

### Install a Template
```bash
# In Claude Desktop
"Install the code_reviewer template"
"Install api_designer as my_api_tool"

# Via MCP tool
promptly_template_install(template_name="code_reviewer")
promptly_template_install(template_name="api_designer", skill_name="my_api_tool")
```

### View Installed Skills
```bash
# In Claude Desktop
"List all my skills"

# Via CLI
python promptly_cli.py skill list
```

### Execute a Skill
```bash
# In Claude Desktop
"Execute the code_reviewer skill with this code: [paste code]"

# Via MCP tool
promptly_execute_skill(skill_name="code_reviewer", user_input="[code]")
```

## Customizing Templates

After installing a template, you can:

1. **View the files:**
   ```bash
   python promptly_cli.py skill files code_reviewer
   ```

2. **Add more files:**
   ```bash
   python promptly_cli.py skill add-file code_reviewer ./custom_rules.md
   ```

3. **Update metadata:**
   ```python
   from promptly import Promptly
   p = Promptly()
   skill = p.get_skill("code_reviewer")
   skill['metadata']['custom_field'] = 'value'
   p.add_skill("code_reviewer", skill['description'], metadata=skill['metadata'])
   ```

## Template Combinations

Combine templates for powerful workflows:

### Full Stack Development
1. `api_designer` - Design the API
2. `sql_designer` - Design the database
3. `code_reviewer` - Review implementation
4. `test_generator` - Generate tests
5. `documentation_writer` - Generate docs
6. `error_handler` - Add error handling

### Data Science Pipeline
1. `data_analyzer` - Explore dataset
2. `sql_designer` - Design data warehouse
3. `documentation_writer` - Document findings
4. `test_generator` - Test transformations

### AI/ML Workflow
1. `prompt_engineer` - Design prompts
2. `data_analyzer` - Analyze results
3. `test_generator` - Test edge cases
4. `documentation_writer` - Document models

## Creating Custom Templates

Want to add your own template? Edit `skill_templates/__init__.py`:

```python
TEMPLATES = {
    "your_template": {
        "description": "Your description",
        "metadata": {
            "runtime": "claude",
            "tags": ["tag1", "tag2"]
        },
        "files": {
            "file1.py": "# Python code...",
            "file2.md": "# Documentation..."
        }
    }
}
```

Then restart Claude Desktop or re-import the MCP server.

## Benefits

✅ **Instant setup** - No manual file creation
✅ **Best practices** - Pre-configured with industry standards
✅ **Consistent** - Same structure across all skills
✅ **Extensible** - Customize after installation
✅ **Reusable** - Install across multiple projects
✅ **Versioned** - Track changes with Promptly's version control

---

**Start using templates:** *"Show me all skill templates"* in Claude Desktop!
