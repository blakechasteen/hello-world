#!/usr/bin/env python3
"""
HoloLoom Promptly MCP Server
============================

Exposes HoloLoom's complete Promptly integration to Claude Desktop via MCP.

Phases Integrated:
- Phase 1: Recursive reasoning (6 strategies)
- Phase 2: Analytics tracking (performance metrics)
- Phase 3: Professional skills (13 agent templates)

Features:
- 13 professional skill tools (code-reviewer, bug-detective, etc.)
- Recursive weaving with strategy selection
- Memory operations (experience, recall, reflect)
- Analytics (summaries, trends, recommendations)
- Prompt refinement (7-component metaprompt framework)
- Complete reasoning provenance

Usage:
    python -m HoloLoom.mcp_server_promptly

Configuration for Claude Desktop:
    {
      "mcpServers": {
        "hololoom-promptly": {
          "command": "python",
          "args": ["-m", "HoloLoom.mcp_server_promptly"],
          "env": {
            "PYTHONPATH": "/path/to/mythRL"
          }
        }
      }
    }

Created: 2025-11-16
Integration: Phases 1-3 → Claude Desktop
"""

import asyncio
import logging
import json
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP not installed. Run: pip install mcp")
    sys.exit(1)

# HoloLoom imports
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.weaving_orchestrator_recursive import RecursiveWeavingOrchestrator
from HoloLoom.protocols.recursive_reasoning import ReasoningStrategy
from HoloLoom.agentic.skill_agents import (
    SkillRegistry,
    SkillExecutor,
    list_available_skills
)
from HoloLoom.prompting.metaprompt import create_metaprompt_auto, enhance_request

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Create MCP server
server = Server("hololoom-promptly")

# Global state
skill_registry: Optional[SkillRegistry] = None
orchestrator: Optional[RecursiveWeavingOrchestrator] = None
config: Optional[Config] = None


# ============================================================================
# Initialization
# ============================================================================

async def initialize_hololoom():
    """Initialize HoloLoom components."""
    global skill_registry, orchestrator, config

    logger.info("Initializing HoloLoom Promptly MCP Server...")

    # Create configuration
    config = Config.fast()
    logger.info(f"Configuration: {config.mode.value} mode")

    # Load skill registry
    skill_registry = SkillRegistry()
    await skill_registry.load_all_skills()
    logger.info(f"Loaded {len(skill_registry.skills)} professional skills")

    # Create orchestrator (we'll create fresh ones per request for now)
    logger.info("HoloLoom Promptly MCP Server ready")


# ============================================================================
# Tool Definitions
# ============================================================================

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available MCP tools."""
    tools = []

    # Core HoloLoom tools
    tools.extend([
        Tool(
            name="hololoom_experience",
            description="Form a memory from any input (text, code, etc.). Store in knowledge graph with entities and motifs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to remember (text, code, notes, etc.)"
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context or metadata"
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="hololoom_recall",
            description="Retrieve memories related to a query using semantic search + knowledge graph traversal.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of memories to return (default: 5)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="hololoom_weave",
            description="Weave a query with full recursive reasoning. Auto-selects strategy and refines until quality threshold met.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to weave"
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["refine", "critique", "decompose", "explore", "verify", "hofstadter", "adaptive"],
                        "description": "Reasoning strategy (default: adaptive)"
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum refinement iterations (default: 3)"
                    },
                    "quality_threshold": {
                        "type": "number",
                        "description": "Quality threshold for refinement (default: 0.85)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="hololoom_analytics_summary",
            description="Get analytics summary: strategy performance, quality improvements, costs, recommendations.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="refine_prompt",
            description="Refine a casual prompt into a structured, high-quality prompt using the 7-component metaprompt framework (ROLE, OBJECTIVE, PROCESS, FORMAT, CONSTRAINTS, UNCERTAINTY, VALIDATION) with model-specific optimizations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Casual prompt to refine (e.g., 'write a Python function', 'help me prepare for meeting')"
                    },
                    "provider": {
                        "type": "string",
                        "enum": ["anthropic", "google", "openai", "auto"],
                        "description": "LLM provider for model-specific optimizations. 'anthropic' adds Claude thinking tags (+30% quality), 'google' adds Gemini multimodal (+25%), 'openai' adds GPT structured outputs (+20%). Default: auto (uses current config)"
                    },
                    "apply_strategy": {
                        "type": "boolean",
                        "description": "Auto-detect and apply prompting strategy (verify, critique, decompose, etc.) if confidence > threshold. Default: true"
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence (0.0-1.0) to apply auto-detected strategy. Default: 0.7"
                    }
                },
                "required": ["prompt"]
            }
        ),
    ])

    # Professional skill tools (13 skills)
    skill_tools = [
        Tool(
            name="skill_code_reviewer",
            description="Review code for best practices, bugs, and quality improvements. Uses CRITIQUE strategy for thorough analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Source code to review"},
                    "language": {"type": "string", "description": "Programming language (python, javascript, etc.)"},
                    "filename": {"type": "string", "description": "Filename for context (optional)"},
                    "focus_areas": {"type": "string", "description": "Specific areas to focus on (optional)"}
                },
                "required": ["code", "language"]
            }
        ),
        Tool(
            name="skill_bug_detective",
            description="Systematically debug code and find root causes. Uses DECOMPOSE strategy for step-by-step analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Buggy code"},
                    "bug_description": {"type": "string", "description": "Description of the bug"},
                    "language": {"type": "string", "description": "Programming language"},
                    "error_message": {"type": "string", "description": "Error message or stack trace (optional)"},
                    "expected_behavior": {"type": "string", "description": "What should happen (optional)"},
                    "actual_behavior": {"type": "string", "description": "What actually happens (optional)"}
                },
                "required": ["code", "bug_description", "language"]
            }
        ),
        Tool(
            name="skill_test_generator",
            description="Generate comprehensive test cases with high coverage. Uses EXPLORE strategy to find edge cases.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to generate tests for"},
                    "language": {"type": "string", "description": "Programming language"},
                    "framework": {"type": "string", "description": "Testing framework (pytest, jest, etc.)"},
                    "happy_path": {"type": "boolean", "description": "Generate happy path tests (default: true)"},
                    "edge_cases": {"type": "boolean", "description": "Generate edge case tests (default: true)"},
                    "error_handling": {"type": "boolean", "description": "Generate error handling tests (default: true)"}
                },
                "required": ["code", "language"]
            }
        ),
        Tool(
            name="skill_api_designer",
            description="Design RESTful APIs with best practices. Uses REFINE strategy for iterative improvement.",
            inputSchema={
                "type": "object",
                "properties": {
                    "requirements": {"type": "string", "description": "API requirements and functionality"},
                    "version_strategy": {"type": "string", "description": "Versioning strategy (uri, header, etc.)"},
                    "auth_type": {"type": "string", "description": "Authentication type (jwt, oauth2, etc.)"}
                },
                "required": ["requirements"]
            }
        ),
        Tool(
            name="skill_documentation_writer",
            description="Generate comprehensive documentation for code. Uses REFINE strategy for clarity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to document"},
                    "language": {"type": "string", "description": "Programming language"},
                    "doc_type": {"type": "string", "description": "Type: readme, api, tutorial, reference"}
                },
                "required": ["code", "language"]
            }
        ),
        Tool(
            name="skill_performance_profiler",
            description="Analyze code performance and suggest optimizations. Uses DECOMPOSE strategy.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to profile"},
                    "language": {"type": "string", "description": "Programming language"},
                    "profile_data": {"type": "string", "description": "Profiling data if available (optional)"}
                },
                "required": ["code", "language"]
            }
        ),
        Tool(
            name="skill_architecture_advisor",
            description="Provide system architecture guidance and design decisions. Uses HOFSTADTER meta-reasoning.",
            inputSchema={
                "type": "object",
                "properties": {
                    "requirements": {"type": "string", "description": "System requirements"},
                    "scale": {"type": "string", "description": "Expected scale (small, medium, large)"},
                    "constraints": {"type": "string", "description": "Constraints (budget, timeline, etc.)"}
                },
                "required": ["requirements"]
            }
        ),
        Tool(
            name="skill_migration_planner",
            description="Plan technology migrations with step-by-step strategy. Uses DECOMPOSE for planning.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Current technology/framework"},
                    "target": {"type": "string", "description": "Target technology/framework"},
                    "context": {"type": "string", "description": "Additional context (optional)"}
                },
                "required": ["source", "target"]
            }
        ),
        Tool(
            name="skill_code_explainer",
            description="Explain complex code in simple terms. Uses REFINE for clarity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to explain"},
                    "language": {"type": "string", "description": "Programming language"},
                    "level": {"type": "string", "description": "Audience level: beginner, intermediate, advanced"}
                },
                "required": ["code", "language"]
            }
        ),
        Tool(
            name="skill_naming_consultant",
            description="Suggest better names for variables, functions, and classes. Uses CRITIQUE strategy.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code with names to improve"},
                    "language": {"type": "string", "description": "Programming language"},
                    "context": {"type": "string", "description": "Domain/context (optional)"}
                },
                "required": ["code", "language"]
            }
        ),
        Tool(
            name="skill_sql_optimizer",
            description="Optimize SQL queries for performance and readability. Uses REFINE strategy.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to optimize"},
                    "database_type": {"type": "string", "description": "Database type (postgresql, mysql, etc.)"},
                    "schema": {"type": "string", "description": "Database schema info (optional)"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="skill_refactoring_expert",
            description="Refactor code for better maintainability and performance. Uses CRITIQUE strategy.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to refactor"},
                    "language": {"type": "string", "description": "Programming language"},
                    "focus": {"type": "string", "description": "Focus: maintainability, performance, testability"}
                },
                "required": ["code", "language"]
            }
        ),
        Tool(
            name="skill_security_auditor",
            description="Audit code for OWASP vulnerabilities and security issues. Uses VERIFY strategy.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to audit"},
                    "language": {"type": "string", "description": "Programming language"},
                    "focus_areas": {"type": "string", "description": "Security focus areas (optional)"}
                },
                "required": ["code", "language"]
            }
        ),
    ]

    tools.extend(skill_tools)

    logger.info(f"Serving {len(tools)} MCP tools")
    return tools


# ============================================================================
# Tool Implementations
# ============================================================================

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    logger.info(f"Tool called: {name} with args: {list(arguments.keys())}")

    try:
        # Core HoloLoom tools
        if name == "hololoom_experience":
            return await handle_experience(arguments)
        elif name == "hololoom_recall":
            return await handle_recall(arguments)
        elif name == "hololoom_weave":
            return await handle_weave(arguments)
        elif name == "hololoom_analytics_summary":
            return await handle_analytics_summary(arguments)
        elif name == "refine_prompt":
            return await handle_refine_prompt(arguments)

        # Professional skill tools
        elif name.startswith("skill_"):
            skill_name = name.replace("skill_", "").replace("_", "-")
            return await handle_skill_execution(skill_name, arguments)

        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]

    except Exception as e:
        logger.error(f"Error in {name}: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def handle_experience(args: Dict[str, Any]) -> List[TextContent]:
    """Handle hololoom_experience tool."""
    from HoloLoom.hololoom import HoloLoom

    content = args["content"]
    context = args.get("context", "")

    async with HoloLoom(config=config) as loom:
        memory = await loom.experience(content)

        result = {
            "status": "success",
            "content": content[:100] + "..." if len(content) > 100 else content,
            "entities": memory.entities if hasattr(memory, 'entities') else [],
            "timestamp": datetime.now().isoformat()
        }

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]


async def handle_recall(args: Dict[str, Any]) -> List[TextContent]:
    """Handle hololoom_recall tool."""
    from HoloLoom.hololoom import HoloLoom

    query = args["query"]
    limit = args.get("limit", 5)

    async with HoloLoom(config=config) as loom:
        memories = await loom.recall(query, k=limit)

        result = {
            "status": "success",
            "query": query,
            "memories_found": len(memories),
            "memories": [
                {
                    "content": m.content[:200] + "..." if len(m.content) > 200 else m.content,
                    "score": getattr(m, 'score', 0.0)
                }
                for m in memories[:limit]
            ]
        }

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]


async def handle_weave(args: Dict[str, Any]) -> List[TextContent]:
    """Handle hololoom_weave tool with recursive reasoning."""
    query_text = args["query"]
    strategy_name = args.get("strategy", "adaptive")
    max_iterations = args.get("max_iterations", 3)
    quality_threshold = args.get("quality_threshold", 0.85)

    # Parse strategy
    strategy_map = {
        "refine": ReasoningStrategy.REFINE,
        "critique": ReasoningStrategy.CRITIQUE,
        "decompose": ReasoningStrategy.DECOMPOSE,
        "explore": ReasoningStrategy.EXPLORE,
        "verify": ReasoningStrategy.VERIFY,
        "hofstadter": ReasoningStrategy.HOFSTADTER,
        "adaptive": ReasoningStrategy.ADAPTIVE
    }
    strategy = strategy_map.get(strategy_name.lower(), ReasoningStrategy.ADAPTIVE)

    # Create orchestrator
    async with RecursiveWeavingOrchestrator(
        cfg=config,
        shards=[],
        enable_recursive=True,
        enable_analytics=True,
        quality_threshold=quality_threshold,
        max_iterations=max_iterations,
        default_strategy=strategy
    ) as orch:

        # Execute weaving
        query = Query(text=query_text)
        spacetime = await orch.weave_with_strategy(
            query=query,
            strategy=strategy,
            max_iterations=max_iterations,
            quality_threshold=quality_threshold
        )

        # Build result
        result = {
            "status": "success",
            "response": spacetime.response,
            "confidence": spacetime.confidence,
            "iterations": spacetime.iterations,
            "strategy_used": spacetime.strategy_used.value if spacetime.strategy_used else strategy_name,
            "reasoning_journal": spacetime.reasoning_journal.get_history() if spacetime.reasoning_journal else None
        }

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]


async def handle_analytics_summary(args: Dict[str, Any]) -> List[TextContent]:
    """Handle hololoom_analytics_summary tool."""
    # Create temporary orchestrator to access analytics
    async with RecursiveWeavingOrchestrator(
        cfg=config,
        shards=[],
        enable_analytics=True
    ) as orch:

        summary = orch.get_analytics_summary()

        return [TextContent(
            type="text",
            text=json.dumps(summary, indent=2)
        )]


async def handle_refine_prompt(args: Dict[str, Any]) -> List[TextContent]:
    """
    Handle refine_prompt tool.

    Transforms casual prompts into structured 7-component metaprompts with
    model-specific optimizations (thinking tags for Claude, multimodal for Gemini, etc.).
    """
    # Extract parameters
    prompt = args["prompt"]
    provider = args.get("provider", "auto")
    apply_strategy = args.get("apply_strategy", True)
    confidence_threshold = args.get("confidence_threshold", 0.7)

    try:
        # Determine provider
        if provider == "auto":
            # Use config's provider or default to anthropic
            provider_to_use = config.llm_provider if hasattr(config, 'llm_provider') and config.llm_provider else "anthropic"
        else:
            provider_to_use = provider

        # Create temporary config with specified provider
        temp_config = Config.fast()
        temp_config.llm_provider = provider_to_use

        # Refine prompt
        if apply_strategy:
            # Use auto-detection with strategy
            refined = create_metaprompt_auto(
                request=prompt,
                config=temp_config,
                confidence_threshold=confidence_threshold
            )

            logger.info(
                f"Refined prompt with strategy auto-detection "
                f"(provider: {provider_to_use}, threshold: {confidence_threshold})"
            )
        else:
            # Simple refinement without strategy
            refined = enhance_request(prompt, provider=provider_to_use)

            logger.info(f"Refined prompt without strategy (provider: {provider_to_use})")

        # Build result with metadata
        result = {
            "status": "success",
            "original_prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "refined_prompt": refined,
            "provider": provider_to_use,
            "strategy_applied": apply_strategy,
            "confidence_threshold": confidence_threshold if apply_strategy else None,
            "framework": "7-component (ROLE, OBJECTIVE, PROCESS, FORMAT, CONSTRAINTS, UNCERTAINTY, VALIDATION)",
            "expansion_ratio": round(len(refined) / len(prompt), 1) if len(prompt) > 0 else 0,
        }

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    except FileNotFoundError as e:
        # CORE_TEMPLATE.md not found
        logger.error(f"Meta-prompt template not found: {e}")
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "error": "CORE_TEMPLATE.md not found",
                "message": str(e),
                "help": "Ensure promptly_skills/meta_prompt/CORE_TEMPLATE.md exists in repository"
            }, indent=2)
        )]

    except Exception as e:
        # Generic error - return original prompt with warning
        logger.error(f"Error refining prompt: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "warning",
                "error": str(e),
                "message": "Refinement failed, returning original prompt",
                "original_prompt": prompt
            }, indent=2)
        )]


async def handle_skill_execution(skill_name: str, args: Dict[str, Any]) -> List[TextContent]:
    """Handle professional skill execution."""
    from HoloLoom.agentic.skill_agents import execute_skill

    # Execute skill
    result = await execute_skill(
        skill_name=skill_name,
        parameters=args,
        config=config,
        enable_analytics=True
    )

    # Build response
    response = {
        "status": "success" if result.success else "error",
        "skill": skill_name,
        "output": result.output,
        "confidence": result.confidence,
        "iterations": result.iterations,
        "strategy_used": result.strategy_used,
        "execution_time_ms": result.execution_time_ms,
        "error": result.error
    }

    return [TextContent(
        type="text",
        text=json.dumps(response, indent=2)
    )]


# ============================================================================
# Main Server
# ============================================================================

async def main():
    """Run MCP server."""
    logger.info("Starting HoloLoom Promptly MCP Server...")

    # Initialize HoloLoom
    await initialize_hololoom()

    # Run server
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Server running on stdio")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
