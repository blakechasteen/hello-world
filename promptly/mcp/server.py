#!/usr/bin/env python3
"""
MCP Server for Promptly
========================
Exposes Promptly prompt management to Claude Desktop via MCP protocol.

Core Tools (13):
- promptly_add: Add or update a prompt
- promptly_get: Retrieve a prompt
- promptly_list: List all prompts
- promptly_history: View prompt version history
- promptly_branch_create: Create a new branch
- promptly_branch_switch: Switch to a branch
- promptly_branch_list: List all branches
- promptly_skill_add: Create a new skill
- promptly_skill_get: Get skill details
- promptly_skill_list: List all skills
- promptly_chain_create: Create a prompt chain
- promptly_chain_run: Run a prompt chain
- promptly_evaluate: Evaluate prompt with LLM Judge
"""

import asyncio
import sys
import io
import json
from pathlib import Path
from typing import Any, Sequence, Optional
from datetime import datetime

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, Resource
except ImportError:
    print("ERROR: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Import Promptly core
try:
    from promptly.core import Promptly
except ImportError:
    try:
        # Alternative import path
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from promptly.core import Promptly
    except ImportError:
        print("ERROR: Could not import Promptly. Check installation.", file=sys.stderr)
        sys.exit(1)

# Import LLM Judge (optional)
try:
    from promptly.judge import (
        LLMJudge, JudgeCriteria, JudgeConfig, JudgingMethod,
        get_quality_config, get_comprehensive_config
    )
    LLM_JUDGE_AVAILABLE = True
except ImportError:
    print("WARNING: LLM Judge not available", file=sys.stderr)
    LLM_JUDGE_AVAILABLE = False


# ============================================================================
# MCP Server Setup
# ============================================================================

app = Server("promptly-mcp")
promptly_instance: Optional[Promptly] = None


def get_promptly() -> Promptly:
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

        # Add all chains as resources
        chains = p.list_chains()
        for chain in chains:
            uri = f"promptly://chain/{chain['name']}"
            resources.append(Resource(
                uri=uri,
                name=f"Chain: {chain['name']}",
                description=f"{len(chain.get('steps', []))} steps",
                mimeType="application/json"
            ))

    except Exception as e:
        print(f"Error listing resources: {e}", file=sys.stderr)

    return resources


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read the content of a prompt, skill, or chain resource."""
    p = get_promptly()

    try:
        if uri.startswith("promptly://prompt/"):
            prompt_name = uri.replace("promptly://prompt/", "")
            prompt_data = p.get(prompt_name)

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

            content = f"""# Skill: {skill['name']}

**Description:** {skill['description']}
**Version:** {skill['version']}
**Branch:** {skill['branch']}

## Input Schema
```json
{json.dumps(skill.get('input_schema', {}), indent=2)}
```

## Output Schema
```json
{json.dumps(skill.get('output_schema', {}), indent=2)}
```

## Prompt Template
```
{skill.get('prompt_template', 'No template defined')}
```
"""
            return content

        elif uri.startswith("promptly://chain/"):
            chain_name = uri.replace("promptly://chain/", "")
            chain = p.get_chain(chain_name)

            content = f"""# Chain: {chain['name']}

**Description:** {chain.get('description', 'No description')}
**Steps:** {len(chain.get('steps', []))}

## Pipeline

"""
            for i, step in enumerate(chain.get('steps', []), 1):
                content += f"{i}. `{step['prompt_name']}`"
                if step.get('output_var'):
                    content += f" → `{step['output_var']}`"
                if step.get('condition'):
                    content += f" (if {step['condition']})"
                content += "\n"

            return content
        else:
            return f"Unknown resource URI: {uri}"

    except Exception as e:
        return f"Error reading resource: {str(e)}"


# ============================================================================
# Tool Definitions
# ============================================================================

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available Promptly tools."""
    tools = [
        # ========== Prompt Management ==========
        Tool(
            name="promptly_add",
            description="Add or update a prompt in Promptly with git-like versioning",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the prompt (e.g., 'code_review', 'summarize')"
                    },
                    "content": {
                        "type": "string",
                        "description": "Prompt content (can include {variables} for interpolation)"
                    },
                    "message": {
                        "type": "string",
                        "description": "Commit message describing the change"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata (tags, author, etc.)",
                        "default": {}
                    }
                },
                "required": ["name", "content"]
            }
        ),
        Tool(
            name="promptly_get",
            description="Retrieve a prompt by name, optionally at a specific version",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the prompt to retrieve"
                    },
                    "version": {
                        "type": "integer",
                        "description": "Specific version number (optional, defaults to latest)"
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
            name="promptly_history",
            description="View version history for a prompt",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the prompt"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum versions to show (default: 10)"
                    }
                },
                "required": ["name"]
            }
        ),

        # ========== Branch Management ==========
        Tool(
            name="promptly_branch_create",
            description="Create a new branch for prompt experimentation",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the new branch"
                    },
                    "from_branch": {
                        "type": "string",
                        "description": "Branch to create from (default: current branch)"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="promptly_branch_switch",
            description="Switch to a different branch",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the branch to switch to"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="promptly_branch_list",
            description="List all branches",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),

        # ========== Skill Management ==========
        Tool(
            name="promptly_skill_add",
            description="Create a new reusable skill (parameterized prompt template)",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name"
                    },
                    "description": {
                        "type": "string",
                        "description": "What the skill does"
                    },
                    "prompt_template": {
                        "type": "string",
                        "description": "Prompt template with {input} and {output} placeholders"
                    },
                    "input_schema": {
                        "type": "object",
                        "description": "JSON schema for input parameters",
                        "default": {}
                    },
                    "output_schema": {
                        "type": "object",
                        "description": "JSON schema for expected output",
                        "default": {}
                    }
                },
                "required": ["name", "description", "prompt_template"]
            }
        ),
        Tool(
            name="promptly_skill_get",
            description="Get skill details and template",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name"
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
                "properties": {}
            }
        ),

        # ========== Chain Management ==========
        Tool(
            name="promptly_chain_create",
            description="Create a multi-step prompt chain (pipeline)",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Chain name"
                    },
                    "description": {
                        "type": "string",
                        "description": "What the chain does"
                    },
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "prompt_name": {
                                    "type": "string",
                                    "description": "Name of prompt to use"
                                },
                                "output_var": {
                                    "type": "string",
                                    "description": "Variable name to store output"
                                },
                                "condition": {
                                    "type": "string",
                                    "description": "Optional condition for step execution"
                                }
                            },
                            "required": ["prompt_name"]
                        },
                        "description": "List of chain steps"
                    }
                },
                "required": ["name", "steps"]
            }
        ),
        Tool(
            name="promptly_chain_run",
            description="Execute a prompt chain with input variables",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the chain to run"
                    },
                    "inputs": {
                        "type": "object",
                        "description": "Input variables for the chain"
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["ollama", "claude"],
                        "description": "LLM backend to use (default: auto-detect)"
                    }
                },
                "required": ["name"]
            }
        ),

        # ========== LLM Judge Evaluation ==========
        Tool(
            name="promptly_evaluate",
            description="Evaluate a prompt or output using LLM Judge (12 criteria)",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_name": {
                        "type": "string",
                        "description": "Name of the prompt to evaluate (or provide task directly)"
                    },
                    "task": {
                        "type": "string",
                        "description": "Task description (if not using prompt_name)"
                    },
                    "output": {
                        "type": "string",
                        "description": "Output to evaluate"
                    },
                    "criteria": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["quality", "relevance", "coherence", "accuracy",
                                   "helpfulness", "safety", "creativity", "conciseness",
                                   "completeness", "truthfulness", "harmlessness"]
                        },
                        "description": "Criteria to evaluate (default: quality, relevance, coherence)"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["single_score", "chain_of_thought", "pairwise", "constitutional", "geval"],
                        "description": "Judging method (default: chain_of_thought)"
                    }
                },
                "required": ["output"]
            }
        ),
    ]

    return tools


# ============================================================================
# Tool Handlers
# ============================================================================

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    """Handle tool calls."""
    p = get_promptly()

    try:
        # ========== Prompt Management ==========
        if name == "promptly_add":
            prompt_name = arguments["name"]
            content = arguments["content"]
            message = arguments.get("message", f"Update {prompt_name}")
            metadata = arguments.get("metadata", {})

            commit = p.add(prompt_name, content, message=message, metadata=metadata)

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
                "commit": prompt_data["commit_hash"][:12],
                "metadata": prompt_data.get("metadata", {}),
                "created_at": prompt_data.get("created_at", "")
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

            lines = [f"📚 Prompts ({len(prompts)} total)", "=" * 50]
            for prompt in prompts:
                lines.append(f"• {prompt['name']} (v{prompt['version']}) [{prompt['branch']}]")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "promptly_history":
            prompt_name = arguments["name"]
            limit = arguments.get("limit", 10)

            history = p.log(prompt_name, limit=limit)

            if not history:
                return [TextContent(type="text", text=f"No history found for '{prompt_name}'")]

            lines = [f"📜 History for '{prompt_name}'", "=" * 50]
            for entry in history:
                lines.append(
                    f"v{entry['version']} [{entry['commit_hash'][:8]}] "
                    f"{entry.get('message', 'No message')} "
                    f"({entry.get('created_at', 'Unknown date')})"
                )

            return [TextContent(type="text", text="\n".join(lines))]

        # ========== Branch Management ==========
        elif name == "promptly_branch_create":
            branch_name = arguments["name"]
            from_branch = arguments.get("from_branch")

            p.branch(branch_name, from_branch=from_branch)

            return [TextContent(
                type="text",
                text=f"✓ Branch '{branch_name}' created"
            )]

        elif name == "promptly_branch_switch":
            branch_name = arguments["name"]

            p.checkout(branch_name)

            return [TextContent(
                type="text",
                text=f"✓ Switched to branch '{branch_name}'"
            )]

        elif name == "promptly_branch_list":
            branches = p.list_branches()
            current = p.current_branch()

            lines = ["🌿 Branches", "=" * 30]
            for branch in branches:
                marker = "* " if branch == current else "  "
                lines.append(f"{marker}{branch}")

            return [TextContent(type="text", text="\n".join(lines))]

        # ========== Skill Management ==========
        elif name == "promptly_skill_add":
            skill_name = arguments["name"]
            description = arguments["description"]
            prompt_template = arguments["prompt_template"]
            input_schema = arguments.get("input_schema", {})
            output_schema = arguments.get("output_schema", {})

            p.add_skill(
                skill_name,
                description=description,
                prompt_template=prompt_template,
                input_schema=input_schema,
                output_schema=output_schema
            )

            return [TextContent(
                type="text",
                text=f"✓ Skill '{skill_name}' created\nDescription: {description}"
            )]

        elif name == "promptly_skill_get":
            skill_name = arguments["name"]

            skill = p.get_skill(skill_name)

            return [TextContent(
                type="text",
                text=json.dumps(skill, indent=2)
            )]

        elif name == "promptly_skill_list":
            skills = p.list_skills()

            if not skills:
                return [TextContent(type="text", text="No skills found")]

            lines = [f"🛠️ Skills ({len(skills)} total)", "=" * 50]
            for skill in skills:
                lines.append(f"• {skill['name']}: {skill['description'][:50]}...")

            return [TextContent(type="text", text="\n".join(lines))]

        # ========== Chain Management ==========
        elif name == "promptly_chain_create":
            chain_name = arguments["name"]
            description = arguments.get("description", "")
            steps = arguments["steps"]

            p.add_chain(chain_name, steps=steps, description=description)

            return [TextContent(
                type="text",
                text=f"✓ Chain '{chain_name}' created with {len(steps)} steps"
            )]

        elif name == "promptly_chain_run":
            chain_name = arguments["name"]
            inputs = arguments.get("inputs", {})
            backend = arguments.get("backend")

            result = p.run_chain(chain_name, inputs=inputs, backend=backend)

            output = f"""# Chain Execution: {chain_name}

**Status:** {'✓ Success' if result.get('success') else '✗ Failed'}
**Steps Completed:** {result.get('steps_completed', 0)}

## Outputs

"""
            for var_name, value in result.get('variables', {}).items():
                output += f"### {var_name}\n```\n{value}\n```\n\n"

            if result.get('error'):
                output += f"\n## Error\n{result['error']}"

            return [TextContent(type="text", text=output)]

        # ========== LLM Judge Evaluation ==========
        elif name == "promptly_evaluate":
            if not LLM_JUDGE_AVAILABLE:
                return [TextContent(
                    type="text",
                    text="LLM Judge not available. Install required dependencies."
                )]

            prompt_name = arguments.get("prompt_name")
            task = arguments.get("task")
            output = arguments["output"]
            criteria_names = arguments.get("criteria", ["quality", "relevance", "coherence"])
            method_name = arguments.get("method", "chain_of_thought")

            # Get task from prompt if not provided directly
            if not task and prompt_name:
                try:
                    prompt_data = p.get(prompt_name)
                    task = prompt_data["content"]
                except:
                    task = prompt_name

            if not task:
                task = "Evaluate the following output"

            # Map criteria names to enum
            criteria_map = {
                "quality": JudgeCriteria.QUALITY,
                "relevance": JudgeCriteria.RELEVANCE,
                "coherence": JudgeCriteria.COHERENCE,
                "accuracy": JudgeCriteria.ACCURACY,
                "helpfulness": JudgeCriteria.HELPFULNESS,
                "safety": JudgeCriteria.SAFETY,
                "creativity": JudgeCriteria.CREATIVITY,
                "conciseness": JudgeCriteria.CONCISENESS,
                "completeness": JudgeCriteria.COMPLETENESS,
                "truthfulness": JudgeCriteria.TRUTHFULNESS,
                "harmlessness": JudgeCriteria.HARMLESSNESS,
            }
            criteria = [criteria_map.get(c, JudgeCriteria.QUALITY) for c in criteria_names]

            # Map method name to enum
            method_map = {
                "single_score": JudgingMethod.SINGLE_SCORE,
                "chain_of_thought": JudgingMethod.CHAIN_OF_THOUGHT,
                "pairwise": JudgingMethod.PAIRWISE_COMPARISON,
                "constitutional": JudgingMethod.CONSTITUTIONAL,
                "geval": JudgingMethod.GEVAL,
            }
            method = method_map.get(method_name, JudgingMethod.CHAIN_OF_THOUGHT)

            # Create config and evaluate
            config = JudgeConfig(criteria=criteria, method=method)
            judge = LLMJudge()
            result = judge.evaluate(task, output, config)

            # Format result
            report = f"""# LLM Judge Evaluation

**Task:** {task[:100]}{'...' if len(task) > 100 else ''}
**Method:** {result.method}

## Scores

"""
            for score in result.scores:
                bar = "█" * int(score.score * 10)
                report += f"**{score.criterion.value}**: {bar:10} {score.score:.2f}\n"
                report += f"  _{score.reasoning}_\n\n"

            report += f"\n## Overall Score: {result.overall_score:.2f}\n"

            if result.suggestions:
                report += "\n## Suggestions\n"
                for suggestion in result.suggestions:
                    report += f"- {suggestion}\n"

            return [TextContent(type="text", text=report)]

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
