#!/usr/bin/env python3
"""
MCP Server for Promptly
========================
Exposes Promptly prompt management to Claude Desktop via MCP protocol.

Tools provided:
- promptly_add: Add or update a prompt
- promptly_get: Retrieve a prompt
- promptly_list: List all prompts
- promptly_skill_add: Create a new skill
- promptly_skill_get: Get skill details
- promptly_skill_list: List all skills
- promptly_skill_add_file: Attach file to skill
- promptly_execute_skill: Execute a skill with user input
- promptly_compose_loops: Execute composed pipeline of multiple loop types
- promptly_decompose_refine_verify: Common DRV pattern
- promptly_analytics_summary: Get overall analytics summary
- promptly_analytics_prompt_stats: Get detailed stats for a prompt
- promptly_analytics_recommendations: Get AI recommendations
- promptly_analytics_top_prompts: Get top-performing prompts
"""

import asyncio
import sys
import io
import json
from pathlib import Path
from typing import Any, Sequence

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add Promptly to path
promptly_dir = Path(__file__).parent
sys.path.insert(0, str(promptly_dir))

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, Resource
except ImportError:
    print("ERROR: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Import Promptly
try:
    from promptly import Promptly
except ImportError:
    print("ERROR: Could not import Promptly. Check installation.", file=sys.stderr)
    sys.exit(1)

# Import skill templates
try:
    from skill_templates import get_template, list_templates, TEMPLATES
except ImportError:
    print("WARNING: Skill templates not available", file=sys.stderr)
    TEMPLATES = {}
    def get_template(name): return None
    def list_templates(): return []

# Import execution engine
try:
    from execution_engine import (
        ExecutionEngine, ExecutionConfig, ExecutionBackend,
        ChainExecutor, execute_with_ollama
    )
    EXECUTION_AVAILABLE = True
except ImportError:
    print("WARNING: Execution engine not available", file=sys.stderr)
    EXECUTION_AVAILABLE = False

# Import A/B testing
try:
    from ab_testing import (
        ABTestRunner, TestCase, ABTest,
        exact_match_evaluator, contains_evaluator, word_overlap_evaluator
    )
    AB_TESTING_AVAILABLE = True
except ImportError:
    print("WARNING: A/B testing not available", file=sys.stderr)
    AB_TESTING_AVAILABLE = False

# Import LLM judge
try:
    from llm_judge import LLMJudge, JudgeCriteria, get_quality_config, get_comprehensive_config
    LLM_JUDGE_AVAILABLE = True
except ImportError:
    print("WARNING: LLM judge not available", file=sys.stderr)
    LLM_JUDGE_AVAILABLE = False

# Import package manager
try:
    from package_manager import PackageManager, quick_export, quick_import
    PACKAGE_MANAGER_AVAILABLE = True
except ImportError:
    print("WARNING: Package manager not available", file=sys.stderr)
    PACKAGE_MANAGER_AVAILABLE = False

# Import diff/merge tools
try:
    from diff_merge import DiffTool, MergeTool, diff_versions, quick_merge
    DIFF_MERGE_AVAILABLE = True
except ImportError:
    print("WARNING: Diff/merge tools not available", file=sys.stderr)
    DIFF_MERGE_AVAILABLE = False

# Import cost tracker
try:
    from cost_tracker import CostTracker, ModelProvider, estimate_prompt_cost
    COST_TRACKER_AVAILABLE = True
    # Initialize global cost tracker
    cost_tracker = CostTracker()
except ImportError:
    print("WARNING: Cost tracker not available", file=sys.stderr)
    COST_TRACKER_AVAILABLE = False
    cost_tracker = None

# Import recursive loops
try:
    from recursive_loops import RecursiveEngine, LoopConfig, LoopType, refine_iteratively, think_recursively
    RECURSIVE_LOOPS_AVAILABLE = True
except ImportError:
    print("WARNING: Recursive loops not available", file=sys.stderr)
    RECURSIVE_LOOPS_AVAILABLE = False

# Import loop composition
try:
    from loop_composition import LoopComposer, CompositionStep, CompositionResult
    LOOP_COMPOSITION_AVAILABLE = True
except ImportError:
    print("WARNING: Loop composition not available", file=sys.stderr)
    LOOP_COMPOSITION_AVAILABLE = False

# Import prompt analytics
try:
    from prompt_analytics import PromptAnalytics, PromptExecution
    ANALYTICS_AVAILABLE = True
    # Initialize global analytics instance
    analytics = PromptAnalytics()
except ImportError:
    print("WARNING: Prompt analytics not available", file=sys.stderr)
    ANALYTICS_AVAILABLE = False
    analytics = None


# ============================================================================
# MCP Server Setup
# ============================================================================

app = Server("promptly-mcp")
promptly_instance = None


def get_promptly():
    """Get or create Promptly instance."""
    global promptly_instance
    if promptly_instance is None:
        promptly_instance = Promptly()
    return promptly_instance


# ============================================================================
# MCP Resource Handlers
# ============================================================================

@app.list_resources()
async def list_resources() -> list[Resource]:
    """List all prompts and skills as browsable resources."""
    p = get_promptly()
    resources = []

    try:
        # Add all prompts as resources
        prompts = p.list()
        for prompt in prompts:
            uri = f"promptly://prompt/{prompt['name']}"
            resources.append(Resource(
                uri=uri,
                name=f"Prompt: {prompt['name']}",
                description=f"v{prompt['version']} on {prompt['branch']} branch",
                mimeType="text/plain"
            ))

        # Add all skills as resources
        skills = p.list_skills()
        for skill in skills:
            uri = f"promptly://skill/{skill['name']}"
            resources.append(Resource(
                uri=uri,
                name=f"Skill: {skill['name']}",
                description=skill.get('description', 'No description'),
                mimeType="application/json"
            ))

    except Exception as e:
        print(f"Error listing resources: {e}", file=sys.stderr)

    return resources


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read the content of a prompt or skill resource."""
    p = get_promptly()

    try:
        if uri.startswith("promptly://prompt/"):
            prompt_name = uri.replace("promptly://prompt/", "")
            prompt_data = p.get(prompt_name)

            # Format prompt data nicely
            content = f"""# Prompt: {prompt_data['name']}

**Version:** {prompt_data['version']}
**Branch:** {prompt_data['branch']}
**Commit:** {prompt_data['commit_hash'][:12]}

## Content
```
{prompt_data['content']}
```

## Metadata
{json.dumps(prompt_data.get('metadata', {}), indent=2)}
"""
            return content

        elif uri.startswith("promptly://skill/"):
            skill_name = uri.replace("promptly://skill/", "")
            skill = p.get_skill(skill_name)
            files = p.get_skill_files(skill_name)

            # Format skill data nicely
            content = f"""# Skill: {skill['name']}

**Description:** {skill['description']}
**Version:** {skill['version']}
**Branch:** {skill['branch']}
**Runtime:** {skill.get('metadata', {}).get('runtime', 'any')}

## Metadata
{json.dumps(skill.get('metadata', {}), indent=2)}

## Attached Files
"""
            for file_info in files:
                content += f"\n### {file_info['filename']} ({file_info['filetype']})\n"
                content += f"```\n{file_info.get('content', 'No content available')[:500]}...\n```\n"

            return content
        else:
            return f"Unknown resource URI: {uri}"

    except Exception as e:
        return f"Error reading resource: {str(e)}"


# ============================================================================
# MCP Tool Handlers
# ============================================================================

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available Promptly tools."""
    return [
        Tool(
            name="promptly_add",
            description="Add or update a prompt in Promptly",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the prompt"
                    },
                    "content": {
                        "type": "string",
                        "description": "Prompt content (can include {variables})"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata (tags, version info, etc.)",
                        "default": {}
                    }
                },
                "required": ["name", "content"]
            }
        ),
        Tool(
            name="promptly_get",
            description="Retrieve a prompt by name",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the prompt to retrieve"
                    },
                    "version": {
                        "type": "integer",
                        "description": "Specific version (optional, defaults to latest)"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="promptly_list",
            description="List all prompts in the repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "branch": {
                        "type": "string",
                        "description": "Branch name (optional, defaults to current)"
                    }
                }
            }
        ),
        Tool(
            name="promptly_skill_add",
            description="Create a new skill or update existing skill",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name"
                    },
                    "description": {
                        "type": "string",
                        "description": "Skill description"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Skill metadata (runtime, tags, etc.)",
                        "default": {"runtime": "claude"}
                    }
                },
                "required": ["name", "description"]
            }
        ),
        Tool(
            name="promptly_skill_get",
            description="Get skill details including attached files",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name"
                    },
                    "version": {
                        "type": "integer",
                        "description": "Specific version (optional)"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="promptly_skill_list",
            description="List all available skills",
            inputSchema={
                "type": "object",
                "properties": {
                    "branch": {
                        "type": "string",
                        "description": "Branch name (optional)"
                    }
                }
            }
        ),
        Tool(
            name="promptly_skill_add_file",
            description="Attach a file to a skill (code, config, data, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Name for the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "File content"
                    },
                    "filetype": {
                        "type": "string",
                        "description": "File type/extension (py, json, md, etc.)"
                    }
                },
                "required": ["skill_name", "filename", "content"]
            }
        ),
        Tool(
            name="promptly_execute_skill",
            description="Execute a skill with user input (returns formatted prompt for execution)",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill to execute"
                    },
                    "user_input": {
                        "type": "string",
                        "description": "User input/request for the skill"
                    },
                    "version": {
                        "type": "integer",
                        "description": "Skill version (optional)"
                    }
                },
                "required": ["skill_name", "user_input"]
            }
        ),
        Tool(
            name="promptly_template_list",
            description="List all available skill templates",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="promptly_template_install",
            description="Install a skill from template library",
            inputSchema={
                "type": "object",
                "properties": {
                    "template_name": {
                        "type": "string",
                        "description": "Name of the template to install"
                    },
                    "skill_name": {
                        "type": "string",
                        "description": "Custom name for the skill (optional, defaults to template name)"
                    }
                },
                "required": ["template_name"]
            }
        ),
        Tool(
            name="promptly_prompt_suggest",
            description="Get AI suggestions to improve a prompt",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_name": {
                        "type": "string",
                        "description": "Name of the prompt to analyze"
                    },
                    "goal": {
                        "type": "string",
                        "description": "What you want the prompt to achieve"
                    }
                },
                "required": ["prompt_name"]
            }
        ),
        Tool(
            name="promptly_execute_skill_real",
            description="Actually execute a skill with Ollama or Claude API (returns real LLM output)",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill to execute"
                    },
                    "user_input": {
                        "type": "string",
                        "description": "User input/request for the skill"
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["ollama", "claude_api"],
                        "description": "Execution backend (default: ollama)"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (e.g., llama3.2:3b, claude-3-5-sonnet-20241022)"
                    },
                    "api_key": {
                        "type": "string",
                        "description": "API key (required for Claude API backend)"
                    }
                },
                "required": ["skill_name", "user_input"]
            }
        ),
        Tool(
            name="promptly_execute_prompt",
            description="Execute a prompt directly with Ollama or Claude API",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_name": {
                        "type": "string",
                        "description": "Name of the prompt to execute"
                    },
                    "inputs": {
                        "type": "object",
                        "description": "Variables to substitute in the prompt"
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["ollama", "claude_api"],
                        "description": "Execution backend (default: ollama)"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use"
                    }
                },
                "required": ["prompt_name"]
            }
        ),
        Tool(
            name="promptly_ab_test",
            description="Run A/B test comparing prompt variants",
            inputSchema={
                "type": "object",
                "properties": {
                    "test_name": {
                        "type": "string",
                        "description": "Name for this A/B test"
                    },
                    "variants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of prompt names to compare"
                    },
                    "test_inputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of test inputs to try"
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["ollama", "claude_api"],
                        "description": "Execution backend (default: ollama)"
                    }
                },
                "required": ["test_name", "variants", "test_inputs"]
            }
        ),
        Tool(
            name="promptly_export_package",
            description="Export prompts and skills to a shareable package file",
            inputSchema={
                "type": "object",
                "properties": {
                    "package_name": {
                        "type": "string",
                        "description": "Name for the package"
                    },
                    "prompts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of prompt names to export"
                    },
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of skill names to export"
                    },
                    "author": {
                        "type": "string",
                        "description": "Package author name"
                    }
                },
                "required": ["package_name"]
            }
        ),
        Tool(
            name="promptly_import_package",
            description="Import prompts and skills from a package file",
            inputSchema={
                "type": "object",
                "properties": {
                    "package_path": {
                        "type": "string",
                        "description": "Path to .promptly package file"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Overwrite existing prompts/skills (default: false)"
                    }
                },
                "required": ["package_path"]
            }
        ),
        Tool(
            name="promptly_diff",
            description="Compare two prompt versions and show differences",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_name": {
                        "type": "string",
                        "description": "Name of the prompt"
                    },
                    "version_a": {
                        "type": "integer",
                        "description": "First version"
                    },
                    "version_b": {
                        "type": "integer",
                        "description": "Second version"
                    }
                },
                "required": ["prompt_name", "version_a", "version_b"]
            }
        ),
        Tool(
            name="promptly_merge_branches",
            description="Merge one branch into another with conflict detection",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_branch": {
                        "type": "string",
                        "description": "Branch to merge from"
                    },
                    "target_branch": {
                        "type": "string",
                        "description": "Branch to merge into"
                    },
                    "auto_resolve": {
                        "type": "boolean",
                        "description": "Automatically resolve simple conflicts (default: true)"
                    }
                },
                "required": ["source_branch", "target_branch"]
            }
        ),
        Tool(
            name="promptly_cost_summary",
            description="Get cost summary for prompt executions",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_name": {
                        "type": "string",
                        "description": "Filter by prompt name (optional)"
                    },
                    "model": {
                        "type": "string",
                        "description": "Filter by model (optional)"
                    }
                }
            }
        ),
        Tool(
            name="promptly_refine_iteratively",
            description="Iteratively refine output through self-critique loops",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The original task/question"
                    },
                    "initial_output": {
                        "type": "string",
                        "description": "Starting output to refine"
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum refinement iterations (default: 3)"
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["ollama", "claude_api"],
                        "description": "Execution backend (default: ollama)"
                    }
                },
                "required": ["task", "initial_output"]
            }
        ),
        Tool(
            name="promptly_hofstadter_loop",
            description="Think recursively with self-referential meta-levels (Hofstadter strange loop)",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task to think about recursively"
                    },
                    "levels": {
                        "type": "integer",
                        "description": "Number of meta-levels (default: 3)"
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["ollama", "claude_api"],
                        "description": "Execution backend (default: ollama)"
                    }
                },
                "required": ["task"]
            }
        ),
        Tool(
            name="promptly_compose_loops",
            description="Execute a composed pipeline of multiple loop types (Critique -> Refine -> Verify, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task to process through the pipeline"
                    },
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "loop_type": {
                                    "type": "string",
                                    "enum": ["refine", "critique", "decompose", "verify", "explore", "hofstadter"],
                                    "description": "Type of loop for this step"
                                },
                                "max_iterations": {
                                    "type": "integer",
                                    "description": "Maximum iterations for this step"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Description of what this step does"
                                }
                            },
                            "required": ["loop_type", "max_iterations"]
                        },
                        "description": "Pipeline steps to execute in sequence"
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["ollama", "claude_api"],
                        "description": "Execution backend (default: ollama)"
                    }
                },
                "required": ["task", "steps"]
            }
        ),
        Tool(
            name="promptly_decompose_refine_verify",
            description="Common pattern: Decompose problem -> Refine solution -> Verify correctness",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The problem to solve"
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["ollama", "claude_api"],
                        "description": "Execution backend (default: ollama)"
                    }
                },
                "required": ["task"]
            }
        ),
        Tool(
            name="promptly_analytics_summary",
            description="Get overall analytics summary for all prompt executions",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="promptly_analytics_prompt_stats",
            description="Get detailed statistics for a specific prompt",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_name": {
                        "type": "string",
                        "description": "Name of the prompt to analyze"
                    }
                },
                "required": ["prompt_name"]
            }
        ),
        Tool(
            name="promptly_analytics_recommendations",
            description="Get AI-powered recommendations to improve prompts based on analytics",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="promptly_analytics_top_prompts",
            description="Get top-performing prompts by quality, speed, or cost efficiency",
            inputSchema={
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["quality", "speed", "cost_efficiency"],
                        "description": "Metric to rank by (default: quality)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of prompts to return (default: 5)"
                    }
                }
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    """Handle tool calls."""
    p = get_promptly()

    try:
        if name == "promptly_add":
            prompt_name = arguments["name"]
            content = arguments["content"]
            metadata = arguments.get("metadata", {})

            commit = p.add(prompt_name, content, metadata=metadata)

            return [TextContent(
                type="text",
                text=f"✓ Prompt '{prompt_name}' added/updated\nCommit: {commit[:12]}"
            )]

        elif name == "promptly_get":
            prompt_name = arguments["name"]
            version = arguments.get("version")

            prompt_data = p.get(prompt_name, version=version)

            result = {
                "name": prompt_data["name"],
                "content": prompt_data["content"],
                "version": prompt_data["version"],
                "branch": prompt_data["branch"],
                "commit": prompt_data["commit_hash"],
                "metadata": prompt_data.get("metadata", {})
            }

            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        elif name == "promptly_list":
            branch = arguments.get("branch")
            prompts = p.list(branch=branch)

            if not prompts:
                return [TextContent(type="text", text="No prompts found")]

            lines = ["Available Prompts:", "=" * 60]
            for prompt in prompts:
                lines.append(f"• {prompt['name']} (v{prompt['version']}) - {prompt['branch']}")
                if prompt.get('metadata'):
                    lines.append(f"  Metadata: {json.dumps(prompt['metadata'])}")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "promptly_skill_add":
            skill_name = arguments["name"]
            description = arguments["description"]
            metadata = arguments.get("metadata", {"runtime": "claude"})

            p.add_skill(skill_name, description, metadata=metadata)

            return [TextContent(
                type="text",
                text=f"✓ Skill '{skill_name}' created\nDescription: {description}"
            )]

        elif name == "promptly_skill_get":
            skill_name = arguments["name"]
            version = arguments.get("version")

            skill = p.get_skill(skill_name, version=version)
            files = p.get_skill_files(skill_name)

            result = {
                "name": skill["name"],
                "description": skill["description"],
                "version": skill["version"],
                "branch": skill["branch"],
                "commit": skill["commit_hash"],
                "metadata": skill.get("metadata", {}),
                "files": [
                    {"filename": f["filename"], "filetype": f["filetype"]}
                    for f in files
                ]
            }

            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        elif name == "promptly_skill_list":
            branch = arguments.get("branch")
            skills = p.list_skills(branch=branch)

            if not skills:
                return [TextContent(type="text", text="No skills found")]

            lines = ["Available Skills:", "=" * 60]
            for skill in skills:
                lines.append(f"• {skill['name']} (v{skill['version']})")
                lines.append(f"  {skill['description']}")
                if skill.get('metadata'):
                    lines.append(f"  Runtime: {skill['metadata'].get('runtime', 'any')}")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "promptly_skill_add_file":
            skill_name = arguments["skill_name"]
            filename = arguments["filename"]
            content = arguments["content"]
            filetype = arguments.get("filetype")

            # Infer filetype from filename if not provided
            if not filetype:
                filetype = Path(filename).suffix.lstrip('.')

            # Create temporary file and add it
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'.{filetype}') as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                p.add_skill_file(skill_name, tmp_path, filetype=filetype)
            finally:
                Path(tmp_path).unlink()

            return [TextContent(
                type="text",
                text=f"✓ File '{filename}' attached to skill '{skill_name}'"
            )]

        elif name == "promptly_execute_skill":
            skill_name = arguments["skill_name"]
            user_input = arguments["user_input"]
            version = arguments.get("version")

            # Prepare skill payload
            payload = p.prepare_skill_payload(skill_name, version=version)

            # Format for execution
            lines = [
                f"Executing Skill: {skill_name}",
                f"Description: {payload['description']}",
                f"User Input: {user_input}",
                "",
                "Attached Files:",
            ]

            for file_info in payload["files"]:
                lines.append(f"- {file_info['filename']} ({file_info['filetype']})")
                lines.append(f"  Content preview: {file_info['content'][:200]}...")

            lines.append("")
            lines.append("Ready for execution with Claude or other LLM")

            return [TextContent(
                type="text",
                text="\n".join(lines)
            )]

        elif name == "promptly_template_list":
            templates = list_templates()

            if not templates:
                return [TextContent(type="text", text="No templates available")]

            lines = ["Available Skill Templates:", "=" * 60]
            for template in templates:
                lines.append(f"\n• {template['name']}")
                lines.append(f"  {template['description']}")
                lines.append(f"  Tags: {', '.join(template['tags'])}")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "promptly_template_install":
            template_name = arguments["template_name"]
            skill_name = arguments.get("skill_name", template_name)

            # Get template
            template = get_template(template_name)
            if not template:
                return [TextContent(
                    type="text",
                    text=f"Template '{template_name}' not found. Use promptly_template_list to see available templates."
                )]

            # Create skill from template
            p.add_skill(
                skill_name,
                template["description"],
                metadata=template["metadata"]
            )

            # Add all template files
            import tempfile
            files_added = []
            for filename, content in template["files"].items():
                # Determine filetype
                filetype = Path(filename).suffix.lstrip('.') or 'txt'

                # Create temp file
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'.{filetype}', encoding='utf-8') as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                try:
                    p.add_skill_file(skill_name, tmp_path, filetype=filetype)
                    files_added.append(filename)
                finally:
                    Path(tmp_path).unlink()

            return [TextContent(
                type="text",
                text=f"✓ Skill '{skill_name}' installed from template '{template_name}'\nFiles added: {', '.join(files_added)}"
            )]

        elif name == "promptly_prompt_suggest":
            prompt_name = arguments["prompt_name"]
            goal = arguments.get("goal", "general improvement")

            # Get the prompt
            try:
                prompt_data = p.get(prompt_name)
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Error: Could not find prompt '{prompt_name}'"
                )]

            # Generate suggestions
            current_content = prompt_data['content']
            suggestions = []

            # Check for common improvements
            if "{" not in current_content:
                suggestions.append("💡 Add variables using {variable_name} for flexibility")

            if len(current_content) < 50:
                suggestions.append("💡 Consider adding more context or instructions for clarity")

            if not any(word in current_content.lower() for word in ["please", "you are", "task", "goal"]):
                suggestions.append("💡 Add role-setting or task framing (e.g., 'You are an expert...')")

            if "step" not in current_content.lower() and "chain of thought" not in current_content.lower():
                suggestions.append("💡 Consider adding 'think step-by-step' for complex reasoning")

            if not any(word in current_content.lower() for word in ["format", "json", "markdown", "structure"]):
                suggestions.append("💡 Specify output format (JSON, markdown, etc.)")

            # Check length
            if len(current_content) > 500:
                suggestions.append("⚠️  Prompt is quite long - consider breaking into smaller parts")

            # Add goal-specific suggestion
            if goal and goal != "general improvement":
                suggestions.append(f"🎯 For goal '{goal}': Consider adding specific success criteria")

            result = f"""# Prompt Analysis: {prompt_name}

**Current Content:**
```
{current_content}
```

**Goal:** {goal}

## Suggestions for Improvement

{chr(10).join(suggestions)}

## Recommended Pattern

```
You are an expert assistant specialized in [domain].

Your task: {current_content}

Please provide:
1. [Expected output element 1]
2. [Expected output element 2]

Format your response as [format type].
```
"""

            return [TextContent(type="text", text=result)]

        elif name == "promptly_execute_skill_real":
            if not EXECUTION_AVAILABLE:
                return [TextContent(
                    type="text",
                    text="Execution engine not available. Check installation."
                )]

            skill_name = arguments["skill_name"]
            user_input = arguments["user_input"]
            backend_str = arguments.get("backend", "ollama")
            model = arguments.get("model")
            api_key = arguments.get("api_key")

            # Setup execution config
            backend = ExecutionBackend.OLLAMA if backend_str == "ollama" else ExecutionBackend.CLAUDE_API

            if not model:
                model = "llama3.2:3b" if backend == ExecutionBackend.OLLAMA else "claude-3-5-sonnet-20241022"

            config = ExecutionConfig(
                backend=backend,
                model=model,
                api_key=api_key
            )

            try:
                # Get skill payload
                payload = p.prepare_skill_payload(skill_name)

                # Execute
                engine = ExecutionEngine(config)
                result = engine.execute_skill(payload, user_input=user_input)

                # Track cost
                if COST_TRACKER_AVAILABLE and result.tokens_used and backend == ExecutionBackend.CLAUDE_API:
                    provider = ModelProvider.ANTHROPIC if "claude" in model.lower() else ModelProvider.OPENAI
                    # Assume roughly 40% input, 60% output split
                    input_tokens = int(result.tokens_used * 0.4)
                    output_tokens = int(result.tokens_used * 0.6)
                    cost_tracker.record_execution(
                        prompt_name=skill_name,
                        model=model,
                        provider=provider,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        execution_time=result.execution_time
                    )

                # Format response
                response_text = f"""# Execution Result: {skill_name}

**Backend:** {result.backend}
**Model:** {result.model}
**Success:** {'✓' if result.success else '✗'}
**Time:** {result.execution_time:.2f}s
"""
                if result.tokens_used:
                    response_text += f"**Tokens:** {result.tokens_used}\n"

                if result.success:
                    response_text += f"\n## Output\n{result.output}"
                else:
                    response_text += f"\n## Error\n{result.error}"

                return [TextContent(type="text", text=response_text)]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Error executing skill: {e}"
                )]

        elif name == "promptly_execute_prompt":
            if not EXECUTION_AVAILABLE:
                return [TextContent(
                    type="text",
                    text="Execution engine not available. Check installation."
                )]

            prompt_name = arguments["prompt_name"]
            inputs = arguments.get("inputs", {})
            backend_str = arguments.get("backend", "ollama")
            model = arguments.get("model")

            # Setup execution config
            backend = ExecutionBackend.OLLAMA if backend_str == "ollama" else ExecutionBackend.CLAUDE_API

            if not model:
                model = "llama3.2:3b" if backend == ExecutionBackend.OLLAMA else "claude-3-5-sonnet-20241022"

            config = ExecutionConfig(
                backend=backend,
                model=model,
                api_key=arguments.get("api_key")
            )

            try:
                # Get prompt
                prompt_data = p.get(prompt_name)

                # Format with inputs
                formatted_prompt = prompt_data['content'].format(**inputs)

                # Execute
                engine = ExecutionEngine(config)
                result = engine.execute_prompt(formatted_prompt, skill_name=prompt_name)

                # Format response
                response_text = f"""# Execution Result: {prompt_name}

**Backend:** {result.backend}
**Model:** {result.model}
**Success:** {'✓' if result.success else '✗'}
**Time:** {result.execution_time:.2f}s
"""
                if result.tokens_used:
                    response_text += f"**Tokens:** {result.tokens_used}\n"

                if result.success:
                    response_text += f"\n## Output\n{result.output}"
                else:
                    response_text += f"\n## Error\n{result.error}"

                return [TextContent(type="text", text=response_text)]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Error executing prompt: {e}"
                )]

        elif name == "promptly_ab_test":
            if not EXECUTION_AVAILABLE or not AB_TESTING_AVAILABLE:
                return [TextContent(
                    type="text",
                    text="A/B testing not available. Check installation."
                )]

            test_name = arguments["test_name"]
            variants = arguments["variants"]
            test_inputs = arguments["test_inputs"]
            backend_str = arguments.get("backend", "ollama")

            # Setup execution config
            backend = ExecutionBackend.OLLAMA if backend_str == "ollama" else ExecutionBackend.CLAUDE_API
            model = "llama3.2:3b" if backend == ExecutionBackend.OLLAMA else "claude-3-5-sonnet-20241022"

            config = ExecutionConfig(
                backend=backend,
                model=model,
                api_key=arguments.get("api_key")
            )

            try:
                # Create test cases
                test_cases = [TestCase(input=inp) for inp in test_inputs]

                # Setup runner
                engine = ExecutionEngine(config)
                runner = ABTestRunner(p, engine)

                # Create and run test
                ab_test = runner.create_test(test_name, variants, test_cases)
                ab_test = runner.run_test(ab_test)

                # Generate report
                report = ab_test.to_report()

                return [TextContent(type="text", text=report)]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Error running A/B test: {e}"
                )]

        elif name == "promptly_export_package":
            if not PACKAGE_MANAGER_AVAILABLE:
                return [TextContent(type="text", text="Package manager not available")]

            package_name = arguments["package_name"]
            prompts = arguments.get("prompts", [])
            skills = arguments.get("skills", [])
            author = arguments.get("author")

            try:
                path = quick_export(p, package_name, prompts, skills)
                return [TextContent(
                    type="text",
                    text=f"✓ Package exported to: {path}\nPrompts: {len(prompts)}, Skills: {len(skills)}"
                )]
            except Exception as e:
                return [TextContent(type="text", text=f"Error exporting: {e}")]

        elif name == "promptly_import_package":
            if not PACKAGE_MANAGER_AVAILABLE:
                return [TextContent(type="text", text="Package manager not available")]

            package_path = arguments["package_path"]
            overwrite = arguments.get("overwrite", False)

            try:
                results = quick_import(p, package_path, overwrite=overwrite)
                return [TextContent(
                    type="text",
                    text=f"✓ Import complete\n" +
                         f"Prompts: {results['prompts_imported']} imported, {results['prompts_skipped']} skipped\n" +
                         f"Skills: {results['skills_imported']} imported, {results['skills_skipped']} skipped"
                )]
            except Exception as e:
                return [TextContent(type="text", text=f"Error importing: {e}")]

        elif name == "promptly_diff":
            if not DIFF_MERGE_AVAILABLE:
                return [TextContent(type="text", text="Diff tools not available")]

            prompt_name = arguments["prompt_name"]
            version_a = arguments["version_a"]
            version_b = arguments["version_b"]

            try:
                diff_text = diff_versions(p, prompt_name, version_a, version_b)
                return [TextContent(type="text", text=diff_text)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error generating diff: {e}")]

        elif name == "promptly_merge_branches":
            if not DIFF_MERGE_AVAILABLE:
                return [TextContent(type="text", text="Merge tools not available")]

            source = arguments["source_branch"]
            target = arguments["target_branch"]
            auto_resolve = arguments.get("auto_resolve", True)

            try:
                result = quick_merge(p, source, target, auto_resolve=auto_resolve)
                return [TextContent(type="text", text=result.to_report())]
            except Exception as e:
                return [TextContent(type="text", text=f"Error merging: {e}")]

        elif name == "promptly_cost_summary":
            if not COST_TRACKER_AVAILABLE:
                return [TextContent(type="text", text="Cost tracker not available")]

            prompt_name = arguments.get("prompt_name")
            model = arguments.get("model")

            try:
                summary = cost_tracker.get_summary(
                    prompt_name=prompt_name,
                    model=model
                )
                return [TextContent(type="text", text=summary.to_report())]
            except Exception as e:
                return [TextContent(type="text", text=f"Error getting summary: {e}")]

        elif name == "promptly_refine_iteratively":
            if not RECURSIVE_LOOPS_AVAILABLE or not EXECUTION_AVAILABLE:
                return [TextContent(type="text", text="Recursive loops not available")]

            task = arguments["task"]
            initial_output = arguments["initial_output"]
            max_iterations = arguments.get("max_iterations", 3)
            backend_str = arguments.get("backend", "ollama")

            # Setup executor
            backend = ExecutionBackend.OLLAMA if backend_str == "ollama" else ExecutionBackend.CLAUDE_API
            model = "llama3.2:3b" if backend == ExecutionBackend.OLLAMA else "claude-3-5-sonnet-20241022"
            config_exec = ExecutionConfig(backend=backend, model=model, api_key=arguments.get("api_key"))
            engine_exec = ExecutionEngine(config_exec)
            executor = lambda p: engine_exec.execute_prompt(p, skill_name="refine").output

            try:
                # Execute refinement loop
                engine = RecursiveEngine(executor)
                config = LoopConfig(loop_type=LoopType.REFINE, max_iterations=max_iterations)
                result = engine.execute_refine_loop(task, initial_output, config)

                return [TextContent(type="text", text=result.to_report())]
            except Exception as e:
                return [TextContent(type="text", text=f"Error in refinement loop: {e}")]

        elif name == "promptly_hofstadter_loop":
            if not RECURSIVE_LOOPS_AVAILABLE or not EXECUTION_AVAILABLE:
                return [TextContent(type="text", text="Recursive loops not available")]

            task = arguments["task"]
            levels = arguments.get("levels", 3)
            backend_str = arguments.get("backend", "ollama")

            # Setup executor
            backend = ExecutionBackend.OLLAMA if backend_str == "ollama" else ExecutionBackend.CLAUDE_API
            model = "llama3.2:3b" if backend == ExecutionBackend.OLLAMA else "claude-3-5-sonnet-20241022"
            config_exec = ExecutionConfig(backend=backend, model=model, api_key=arguments.get("api_key"))
            engine_exec = ExecutionEngine(config_exec)
            executor = lambda p: engine_exec.execute_prompt(p, skill_name="hofstadter").output

            try:
                # Execute Hofstadter loop
                engine = RecursiveEngine(executor)
                config = LoopConfig(loop_type=LoopType.HOFSTADTER, max_iterations=levels)
                result = engine.execute_hofstadter_loop(task, config)

                return [TextContent(type="text", text=result.to_report())]
            except Exception as e:
                return [TextContent(type="text", text=f"Error in Hofstadter loop: {e}")]

        elif name == "promptly_compose_loops":
            if not LOOP_COMPOSITION_AVAILABLE or not EXECUTION_AVAILABLE:
                return [TextContent(type="text", text="Loop composition not available")]

            task = arguments["task"]
            steps_data = arguments["steps"]
            backend_str = arguments.get("backend", "ollama")

            # Setup executor
            backend = ExecutionBackend.OLLAMA if backend_str == "ollama" else ExecutionBackend.CLAUDE_API
            model = "llama3.2:3b" if backend == ExecutionBackend.OLLAMA else "claude-3-5-sonnet-20241022"
            config_exec = ExecutionConfig(backend=backend, model=model, api_key=arguments.get("api_key"))
            engine_exec = ExecutionEngine(config_exec)
            executor = lambda p: engine_exec.execute_prompt(p, skill_name="composed").output

            try:
                # Create composition steps
                steps = []
                for step_data in steps_data:
                    loop_type_str = step_data["loop_type"].upper()
                    loop_type = LoopType[loop_type_str]
                    max_iterations = step_data["max_iterations"]
                    description = step_data.get("description", f"{loop_type_str} step")

                    config = LoopConfig(
                        loop_type=loop_type,
                        max_iterations=max_iterations
                    )

                    steps.append(CompositionStep(
                        loop_type=loop_type,
                        config=config,
                        description=description
                    ))

                # Execute composition
                composer = LoopComposer(executor)
                result = composer.compose(task, steps, initial_output=task)

                # Record in analytics
                if ANALYTICS_AVAILABLE:
                    quality_scores = []
                    for _, step_result in result.steps:
                        if hasattr(step_result, 'improvement_history') and step_result.improvement_history:
                            quality_scores.extend(step_result.improvement_history)

                    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None

                    analytics.record_execution(PromptExecution(
                        prompt_id="composed_loop",
                        prompt_name="composed_pipeline",
                        execution_time=0.0,  # Would need to track
                        quality_score=avg_quality,
                        success=True,
                        model=model,
                        backend=backend_str,
                        metadata={
                            "loop_type": "composed",
                            "steps": len(result.steps),
                            "total_iterations": result.total_iterations
                        }
                    ))

                # Format report
                report = f"""# Composed Loop Pipeline

**Total Steps:** {len(result.steps)}
**Total Iterations:** {result.total_iterations}

## Pipeline Steps

"""
                for i, (desc, step_result) in enumerate(result.steps, 1):
                    report += f"{i}. **{desc}**\n"
                    report += f"   - Iterations: {step_result.iterations}\n"
                    report += f"   - Stop reason: {step_result.stop_reason}\n\n"

                report += f"\n## Final Output\n\n{result.final_output}"

                return [TextContent(type="text", text=report)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error in composed loop: {e}")]

        elif name == "promptly_decompose_refine_verify":
            if not LOOP_COMPOSITION_AVAILABLE or not EXECUTION_AVAILABLE:
                return [TextContent(type="text", text="Loop composition not available")]

            task = arguments["task"]
            backend_str = arguments.get("backend", "ollama")

            # Setup executor
            backend = ExecutionBackend.OLLAMA if backend_str == "ollama" else ExecutionBackend.CLAUDE_API
            model = "llama3.2:3b" if backend == ExecutionBackend.OLLAMA else "claude-3-5-sonnet-20241022"
            config_exec = ExecutionConfig(backend=backend, model=model, api_key=arguments.get("api_key"))
            engine_exec = ExecutionEngine(config_exec)
            executor = lambda p: engine_exec.execute_prompt(p, skill_name="drv_pattern").output

            try:
                composer = LoopComposer(executor)
                result = composer.decompose_refine_verify(
                    task=task,
                    decompose_iterations=2,
                    refine_iterations=3,
                    verify_iterations=1
                )

                return [TextContent(type="text", text=f"""# Decompose -> Refine -> Verify

**Task:** {task}

**Total Iterations:** {result.total_iterations}

## Results

{result.final_output}

## Pipeline Completed
- Decompose: {result.steps[0][1].iterations} iterations
- Refine: {result.steps[1][1].iterations} iterations
- Verify: {result.steps[2][1].iterations} iterations
""")]
            except Exception as e:
                return [TextContent(type="text", text=f"Error in DRV pattern: {e}")]

        elif name == "promptly_analytics_summary":
            if not ANALYTICS_AVAILABLE:
                return [TextContent(type="text", text="Analytics not available")]

            try:
                summary = analytics.get_summary()

                report = f"""# Prompt Analytics Summary

**Total Executions:** {summary['total_executions']}
**Unique Prompts:** {summary['unique_prompts']}
**Success Rate:** {summary['success_rate']:.1f}%
**Average Execution Time:** {summary['avg_execution_time']:.2f}s
**Total Cost:** ${summary['total_cost']:.2f}
"""

                if summary.get('avg_quality_score'):
                    report += f"**Average Quality Score:** {summary['avg_quality_score']:.2f}\n"

                return [TextContent(type="text", text=report)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error getting analytics: {e}")]

        elif name == "promptly_analytics_prompt_stats":
            if not ANALYTICS_AVAILABLE:
                return [TextContent(type="text", text="Analytics not available")]

            prompt_name = arguments["prompt_name"]

            try:
                stats = analytics.get_prompt_stats(prompt_name)

                if not stats:
                    return [TextContent(type="text", text=f"No statistics found for prompt '{prompt_name}'")]

                report = f"""# Statistics for '{prompt_name}'

**Total Executions:** {stats.total_executions}
**Success Rate:** {stats.success_rate:.1f}%
**Average Execution Time:** {stats.avg_execution_time:.2f}s
**Total Cost:** ${stats.total_cost:.2f}
"""

                if stats.avg_quality_score:
                    report += f"**Average Quality Score:** {stats.avg_quality_score:.2f}\n"
                    report += f"**Quality Trend:** {stats.quality_trend}\n"

                if stats.last_executed:
                    report += f"**Last Executed:** {stats.last_executed}\n"

                return [TextContent(type="text", text=report)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error getting prompt stats: {e}")]

        elif name == "promptly_analytics_recommendations":
            if not ANALYTICS_AVAILABLE:
                return [TextContent(type="text", text="Analytics not available")]

            try:
                recommendations = analytics.get_recommendations()

                if not recommendations:
                    return [TextContent(type="text", text="No recommendations available yet. Run more prompts to generate insights.")]

                report = "# Analytics Recommendations\n\n"
                for rec in recommendations:
                    report += f"## {rec['prompt_name']}\n"
                    report += f"**Issue:** {rec['issue']}\n"
                    report += f"**Recommendation:** {rec['recommendation']}\n"
                    report += f"**Priority:** {rec['priority']}\n\n"

                return [TextContent(type="text", text=report)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error getting recommendations: {e}")]

        elif name == "promptly_analytics_top_prompts":
            if not ANALYTICS_AVAILABLE:
                return [TextContent(type="text", text="Analytics not available")]

            metric = arguments.get("metric", "quality")
            limit = arguments.get("limit", 5)

            try:
                top_prompts = analytics.get_top_prompts(metric=metric, limit=limit)

                if not top_prompts:
                    return [TextContent(type="text", text=f"No prompts found for metric '{metric}'")]

                report = f"# Top {limit} Prompts by {metric.title()}\n\n"

                for i, stats in enumerate(top_prompts, 1):
                    report += f"{i}. **{stats.prompt_name}**\n"
                    report += f"   - Executions: {stats.total_executions}\n"
                    report += f"   - Success rate: {stats.success_rate:.1f}%\n"

                    if metric == "quality" and stats.avg_quality_score:
                        report += f"   - Quality score: {stats.avg_quality_score:.2f}\n"
                    elif metric == "speed":
                        report += f"   - Avg time: {stats.avg_execution_time:.2f}s\n"
                    elif metric == "cost_efficiency" and stats.total_cost > 0:
                        cost_per_exec = stats.total_cost / stats.total_executions
                        report += f"   - Cost per execution: ${cost_per_exec:.4f}\n"

                    report += "\n"

                return [TextContent(type="text", text=report)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error getting top prompts: {e}")]

        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())