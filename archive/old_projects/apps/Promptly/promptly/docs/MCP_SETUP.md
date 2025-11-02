# Promptly MCP Server Setup for Claude Desktop

This guide shows you how to enable Promptly as an MCP (Model Context Protocol) server in Claude Desktop, giving Claude direct access to your prompt management system.

## What You Get

Once enabled, Claude Desktop can:
- ✅ **Browse prompts & skills** as resources in the UI
- ✅ **Add and update prompts** in your Promptly repository
- ✅ **Install skill templates** from built-in library (8 pre-built skills!)
- ✅ **Get AI suggestions** to improve your prompts
- ✅ **Create and manage skills** with multi-file support
- ✅ **Attach files to skills** (code, configs, data)
- ✅ **Execute skills** with context

## Prerequisites

1. **Claude Desktop** installed
2. **MCP package** installed: `pip install mcp`
3. **Promptly** installed (you have this!)

## Installation Steps

### 1. MCP Server Already Created

The MCP server is located at:
```
c:/Users/blake/Documents/mythRL/Promptly/promptly/mcp_server.py
```

### 2. Configuration Already Added

Your Claude Desktop config has been updated at:
```
C:\Users\blake\AppData\Roaming\Claude\claude_desktop_config.json
```

It now includes the "promptly" MCP server configuration.

### 3. Restart Claude Desktop

**Important:** You must restart Claude Desktop for the changes to take effect.

1. Completely quit Claude Desktop
2. Reopen Claude Desktop
3. Look for the MCP server icon/indicator (usually a hammer or tools icon)

## Available Features

### 🗂️ Resources (Browse in UI)

Claude Desktop will show all your prompts and skills as browsable resources:
- **Prompts**: `promptly://prompt/{name}` - Browse and read prompt content
- **Skills**: `promptly://skill/{name}` - Browse skills with attached files

Click on any resource in Claude Desktop to view its full content!

## Available Tools

Once connected, Claude Desktop will have access to these tools:

### Prompt Management

**promptly_add** - Add or update a prompt
```json
{
  "name": "summarizer",
  "content": "Summarize the following text: {text}",
  "metadata": {"tags": ["nlp", "summary"]}
}
```

**promptly_get** - Retrieve a prompt
```json
{
  "name": "summarizer",
  "version": 1  // optional
}
```

**promptly_list** - List all prompts
```json
{
  "branch": "main"  // optional
}
```

### Skill Management

**promptly_skill_add** - Create a skill
```json
{
  "name": "data_analyzer",
  "description": "Analyze CSV data and generate insights",
  "metadata": {"runtime": "claude", "tags": ["data"]}
}
```

**promptly_skill_get** - Get skill details
```json
{
  "name": "data_analyzer",
  "version": 1  // optional
}
```

**promptly_skill_list** - List all skills
```json
{
  "branch": "main"  // optional
}
```

**promptly_skill_add_file** - Attach file to skill
```json
{
  "skill_name": "data_analyzer",
  "filename": "analyzer.py",
  "content": "# Python code here...",
  "filetype": "py"
}
```

**promptly_execute_skill** - Execute a skill
```json
{
  "skill_name": "data_analyzer",
  "user_input": "Analyze sales data from Q3",
  "version": 1  // optional
}
```

### 🎨 Skill Templates (NEW!)

**promptly_template_list** - List all available templates
```json
{}
```

Returns 8 pre-built skill templates:
- **code_reviewer** - Review code for best practices, bugs, and security
- **api_designer** - Design RESTful APIs with OpenAPI specs
- **data_analyzer** - Analyze datasets and generate insights
- **documentation_writer** - Generate comprehensive docs
- **test_generator** - Create comprehensive unit tests
- **prompt_engineer** - Optimize and improve prompts
- **sql_designer** - Design database schemas and queries
- **error_handler** - Design robust error handling strategies

**promptly_template_install** - Install a skill from template
```json
{
  "template_name": "code_reviewer",
  "skill_name": "my_code_reviewer"  // optional, defaults to template name
}
```

### 💡 Prompt Suggestions (NEW!)

**promptly_prompt_suggest** - Get AI suggestions to improve a prompt
```json
{
  "prompt_name": "summarizer",
  "goal": "better structure and clarity"  // optional
}
```

Returns analysis and suggestions for improving your prompt!

## Example Usage in Claude Desktop

Once the server is running, you can ask Claude:

### 🎨 Quick Start with Templates
> "Show me all the skill templates"

Claude will use `promptly_template_list` to show 8 pre-built templates.

> "Install the code_reviewer template"

Claude will use `promptly_template_install` to create a fully-configured skill with all files!

> "Install the api_designer template as my_api_tool"

Custom name for the skill.

### 💡 Prompt Improvement
> "Analyze my 'summarizer' prompt and suggest improvements"

Claude will use `promptly_prompt_suggest` to provide AI-powered suggestions.

> "Suggest improvements for my 'code_reviewer' prompt with goal: better security focus"

Goal-specific suggestions.

### 🗂️ Browse Resources
Just open the Resources panel in Claude Desktop - you'll see all your:
- Prompts listed as `promptly://prompt/{name}`
- Skills listed as `promptly://skill/{name}`

Click any resource to view its full content!

### Basic Prompts
> "Can you add a prompt called 'summarizer' that summarizes text?"

Claude will use `promptly_add` to create the prompt.

> "Show me all my prompts"

Claude will use `promptly_list` to display them.

### Skills Workflow
> "Create a skill called 'api_designer' that helps design REST APIs"

Claude will use `promptly_skill_add`.

> "Attach an OpenAPI schema template to the 'api_designer' skill"

Claude will use `promptly_skill_add_file` with the schema content.

> "Execute the 'api_designer' skill with input: 'Create an API for a blog platform'"

Claude will use `promptly_execute_skill` to prepare the skill for execution.

## Troubleshooting

### Server Not Showing Up

1. **Check config file syntax:**
   ```bash
   python -m json.tool C:\Users\blake\AppData\Roaming\Claude\claude_desktop_config.json
   ```

2. **Verify Python path:**
   ```bash
   C:/Users/blake/Documents/mythRL/.venv/Scripts/python.exe --version
   ```

3. **Test MCP server manually:**
   ```bash
   cd c:/Users/blake/Documents/mythRL/Promptly/promptly
   python mcp_server.py
   ```
   (Press Ctrl+C to exit)

### Import Errors

If you see import errors, ensure Promptly is properly installed:
```bash
cd c:/Users/blake/Documents/mythRL/Promptly/promptly
python -c "from promptly import Promptly; print('OK')"
```

### MCP Package Missing

Install the MCP SDK:
```bash
pip install mcp
```

### Server Crashes

Check Claude Desktop logs (usually in `%APPDATA%\Claude\logs\`) for error messages.

## Configuration Reference

Your Claude Desktop config entry:
```json
{
  "mcpServers": {
    "promptly": {
      "command": "C:/Users/blake/Documents/mythRL/.venv/Scripts/python.exe",
      "args": [
        "c:/Users/blake/Documents/mythRL/Promptly/promptly/mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "c:/Users/blake/Documents/mythRL/Promptly/promptly"
      }
    }
  }
}
```

## Advanced Usage

### Initialize Promptly Repository

If you haven't already, initialize a Promptly repository:
```bash
cd c:/Users/blake/Documents/mythRL/Promptly/promptly
python promptly_cli.py init
```

### Branching

Create branches for experimentation:
```bash
python promptly_cli.py branch experimental
python promptly_cli.py checkout experimental
```

Claude can then work with different branches via the tools.

### Skills with Multiple Files

You can build complex skills with multiple attached files:
1. Create the skill
2. Attach Python scripts
3. Attach config files (JSON, YAML)
4. Attach data samples
5. Attach documentation

All files become part of the skill payload when executed.

## Integration with Other Tools

Promptly MCP server works alongside:
- **HoloLoom Memory** (already configured in your setup)
- **Other MCP servers** (add to the same config file)

Claude Desktop can use multiple MCP servers simultaneously!

## Next Steps

1. **Restart Claude Desktop** to activate the server
2. **Test with a simple request**: "Add a prompt called 'test' with content 'Hello {name}'"
3. **Build your first skill**: Ask Claude to create a code review skill
4. **Explore the tools**: Try listing, retrieving, and updating prompts

## Benefits of MCP Integration

✅ **Persistent prompts** - Save and reuse your best prompts
✅ **Version control** - Track prompt evolution over time
✅ **Skills library** - Build reusable AI workflows
✅ **Team sharing** - Export and share prompt repositories
✅ **Workflow automation** - Chain prompts for complex tasks

---

**Ready!** Restart Claude Desktop and start using Promptly through MCP.

For questions or issues, check:
- [SKILLS.md](./SKILLS.md) - Skills system documentation
- [README.md](./README.md) - Promptly overview
- [QUICKSTART.md](./QUICKSTART.md) - Getting started guide