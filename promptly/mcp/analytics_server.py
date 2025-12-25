#!/usr/bin/env python3
"""
MCP Server for Promptly Analytics
==================================
Exposes Promptly analytics and HoloLoom integration tools to Claude Desktop via MCP protocol.

Analytics Tools (12):
- promptly_analytics_stats: Get comprehensive prompt analytics
- promptly_analytics_compare: Compare two prompts statistically
- promptly_recommend: Get Thompson Sampling recommendation
- promptly_analytics_trend: Get quality trend over time
- promptly_analytics_underperforming: Find underperforming prompts
- promptly_hololoom_enhance: Enhance prompt with RAG context
- promptly_hololoom_agentic: Execute with agentic reasoning
- promptly_hololoom_similar: Find similar prompts
- promptly_get: Retrieve a prompt
- promptly_list: List prompts
- promptly_create: Create new prompt
- promptly_record_execution: Record execution for analytics

MRF Tools (5):
- promptly_mrf_enhance: Enhance prompt with 7-component metaprompt framework
- promptly_mrf_refine: Refine a response using MRF strategies
- promptly_mrf_recommend: Get Thompson Sampling strategy recommendation
- promptly_mrf_analytics: Get MRF refinement analytics
- promptly_mrf_history: Get MRF refinement history

Usage:
    python -m promptly.mcp.analytics_server

Configuration in Claude Desktop:
    "mcpServers": {
        "promptly-analytics": {
            "command": "python",
            "args": ["-m", "promptly.mcp.analytics_server"]
        }
    }
"""

import asyncio
import sys
import io
import json
from pathlib import Path
from typing import Any, Sequence, Optional, Dict, List
from datetime import datetime

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("ERROR: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Import Promptly core
try:
    from promptly.core import Promptly
    from promptly.core.database import PromptDatabase
except ImportError:
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from promptly.core import Promptly
        from promptly.core.database import PromptDatabase
    except ImportError:
        print("ERROR: Could not import Promptly. Check installation.", file=sys.stderr)
        sys.exit(1)

# Import analytics engine
try:
    from promptly.analytics.engine import PromptAnalyticsEngine, create_analytics_engine
    ANALYTICS_AVAILABLE = True
except ImportError:
    print("WARNING: Analytics engine not available", file=sys.stderr)
    ANALYTICS_AVAILABLE = False

# Import HoloLoom integrations
try:
    from promptly.integrations.hololoom_bridge import HoloLoomBridge
    from promptly.integrations.rag_integration import PromptRAGIntegration
    from promptly.integrations.agentic_integration import AgenticPromptExecutor
    HOLOLOOM_AVAILABLE = True
except ImportError:
    print("WARNING: HoloLoom integrations not available", file=sys.stderr)
    HOLOLOOM_AVAILABLE = False

# Import MRF integration
try:
    from promptly.integrations.mrf_integration import (
        MRFBridge,
        get_mrf_bridge,
        RefinementStrategy,
        ModelProvider
    )
    MRF_AVAILABLE = True
except ImportError:
    print("WARNING: MRF integration not available", file=sys.stderr)
    MRF_AVAILABLE = False


# ============================================================================
# MCP Server Setup
# ============================================================================

app = Server("promptly-analytics")
promptly_instance: Optional[Promptly] = None
analytics_engine: Optional[PromptAnalyticsEngine] = None
hololoom_bridge: Optional[HoloLoomBridge] = None
rag_integration: Optional[PromptRAGIntegration] = None
agentic_executor: Optional[AgenticPromptExecutor] = None
mrf_bridge: Optional[MRFBridge] = None


def get_promptly() -> Promptly:
    """Get or create Promptly instance."""
    global promptly_instance
    if promptly_instance is None:
        promptly_instance = Promptly()
    return promptly_instance


def get_analytics() -> Optional[PromptAnalyticsEngine]:
    """Get or create analytics engine."""
    global analytics_engine
    if analytics_engine is None and ANALYTICS_AVAILABLE:
        try:
            p = get_promptly()
            db = PromptDatabase(p.repo_path / "prompts.db")
            analytics_engine = create_analytics_engine(db)
        except Exception as e:
            print(f"WARNING: Could not create analytics engine: {e}", file=sys.stderr)
    return analytics_engine


def get_hololoom_bridge() -> Optional[HoloLoomBridge]:
    """Get or create HoloLoom bridge."""
    global hololoom_bridge
    if hololoom_bridge is None and HOLOLOOM_AVAILABLE:
        try:
            hololoom_bridge = HoloLoomBridge()
        except Exception as e:
            print(f"WARNING: Could not create HoloLoom bridge: {e}", file=sys.stderr)
    return hololoom_bridge


async def get_rag_integration() -> Optional[PromptRAGIntegration]:
    """Get or create RAG integration."""
    global rag_integration
    if rag_integration is None and HOLOLOOM_AVAILABLE:
        try:
            rag_integration = PromptRAGIntegration()
        except Exception as e:
            print(f"WARNING: Could not create RAG integration: {e}", file=sys.stderr)
    return rag_integration


async def get_agentic_executor() -> Optional[AgenticPromptExecutor]:
    """Get or create agentic executor."""
    global agentic_executor
    if agentic_executor is None and HOLOLOOM_AVAILABLE:
        try:
            agentic_executor = AgenticPromptExecutor()
        except Exception as e:
            print(f"WARNING: Could not create agentic executor: {e}", file=sys.stderr)
    return agentic_executor


def get_mrf() -> Optional[MRFBridge]:
    """Get or create MRF bridge."""
    global mrf_bridge
    if mrf_bridge is None and MRF_AVAILABLE:
        try:
            mrf_bridge = get_mrf_bridge()
        except Exception as e:
            print(f"WARNING: Could not create MRF bridge: {e}", file=sys.stderr)
    return mrf_bridge


# ============================================================================
# Tool Definitions
# ============================================================================

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available analytics tools."""
    tools = [
        # ========== Analytics Tools ==========
        Tool(
            name="promptly_analytics_stats",
            description="Get comprehensive analytics for a prompt including quality metrics, trends, Thompson Sampling expected quality, and recommendations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_name": {
                        "type": "string",
                        "description": "Name of the prompt to analyze"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to analyze (default: 30)",
                        "default": 30,
                        "minimum": 1,
                        "maximum": 365
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format",
                        "enum": ["json", "text", "html"],
                        "default": "json"
                    }
                },
                "required": ["prompt_name"]
            }
        ),
        Tool(
            name="promptly_analytics_compare",
            description="Compare two prompts statistically with significance testing. Returns winner, statistical significance, and detailed metrics comparison.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_a": {
                        "type": "string",
                        "description": "First prompt name to compare"
                    },
                    "prompt_b": {
                        "type": "string",
                        "description": "Second prompt name to compare"
                    },
                    "task_type": {
                        "type": "string",
                        "description": "Optional task type to filter comparison"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to analyze (default: 30)",
                        "default": 30
                    }
                },
                "required": ["prompt_a", "prompt_b"]
            }
        ),
        Tool(
            name="promptly_recommend",
            description="Get Thompson Sampling recommendation for the best prompt for a task type. Uses Bayesian optimization to balance exploration and exploitation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "description": "Type of task (e.g., summarization, code_review, explanation)"
                    },
                    "candidates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of candidate prompt names to consider. If not provided, considers all prompts."
                    }
                },
                "required": ["task_type"]
            }
        ),
        Tool(
            name="promptly_analytics_trend",
            description="Get quality trend over time for a prompt. Returns time series data with quality scores and execution counts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_name": {
                        "type": "string",
                        "description": "Prompt to analyze"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to analyze (default: 30)",
                        "default": 30
                    },
                    "granularity": {
                        "type": "string",
                        "description": "Time granularity for data points",
                        "enum": ["hourly", "daily", "weekly"],
                        "default": "daily"
                    }
                },
                "required": ["prompt_name"]
            }
        ),
        Tool(
            name="promptly_analytics_underperforming",
            description="Find prompts performing below quality threshold. Returns list with reasons and improvement suggestions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "description": "Quality threshold (0.0-1.0). Prompts below this are flagged. Default: 0.6",
                        "default": 0.6,
                        "minimum": 0,
                        "maximum": 1
                    },
                    "days": {
                        "type": "integer",
                        "description": "Analysis window in days (default: 30)",
                        "default": 30
                    }
                }
            }
        ),

        # ========== HoloLoom Integration Tools ==========
        Tool(
            name="promptly_hololoom_enhance",
            description="Enhance a prompt with contextual information from HoloLoom memory using RAG (Retrieval-Augmented Generation).",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_name": {
                        "type": "string",
                        "description": "Name of prompt to enhance"
                    },
                    "task_description": {
                        "type": "string",
                        "description": "Description of the task for context retrieval"
                    },
                    "context_k": {
                        "type": "integer",
                        "description": "Number of context items to retrieve (default: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "injection_method": {
                        "type": "string",
                        "description": "How to inject context into prompt",
                        "enum": ["prefix", "suffix", "inline"],
                        "default": "prefix"
                    }
                },
                "required": ["prompt_name"]
            }
        ),
        Tool(
            name="promptly_hololoom_agentic",
            description="Execute a prompt with agentic reasoning for verification, research, or multi-step planning.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_name": {
                        "type": "string",
                        "description": "Name of prompt to execute"
                    },
                    "mode": {
                        "type": "string",
                        "description": "Reasoning mode: direct (fast), verify (check accuracy), research (explore), plan_execute (decompose)",
                        "enum": ["direct", "verify", "research", "plan_execute"]
                    },
                    "variables": {
                        "type": "object",
                        "description": "Variables to substitute in the prompt template",
                        "additionalProperties": {"type": "string"}
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum reasoning steps (default: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["prompt_name", "mode"]
            }
        ),
        Tool(
            name="promptly_hololoom_similar",
            description="Find similar prompts based on semantic similarity to a query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query or concept to find similar prompts for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "min_quality": {
                        "type": "number",
                        "description": "Minimum quality threshold for results (default: 0.7)",
                        "default": 0.7,
                        "minimum": 0,
                        "maximum": 1
                    }
                },
                "required": ["query"]
            }
        ),

        # ========== Prompt Management ==========
        Tool(
            name="promptly_get",
            description="Retrieve a prompt by name, optionally at a specific version.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Prompt name"
                    },
                    "version": {
                        "type": "integer",
                        "description": "Optional version number. If not specified, returns latest version.",
                        "minimum": 1
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="promptly_list",
            description="List available prompts with optional filtering.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Optional glob pattern to filter (e.g., 'code_*' for prompts starting with 'code_')"
                    },
                    "tag": {
                        "type": "string",
                        "description": "Optional tag to filter by"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 50)",
                        "default": 50
                    }
                }
            }
        ),
        Tool(
            name="promptly_create",
            description="Create a new prompt. Supports variable placeholders using {variable_name} syntax.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Prompt name (alphanumeric and underscores only)",
                        "pattern": "^[a-zA-Z][a-zA-Z0-9_]*$"
                    },
                    "content": {
                        "type": "string",
                        "description": "Prompt content. Use {variable_name} for placeholders."
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata",
                        "properties": {
                            "description": {"type": "string"},
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "author": {"type": "string"}
                        }
                    }
                },
                "required": ["name", "content"]
            }
        ),
        Tool(
            name="promptly_record_execution",
            description="Record a prompt execution for analytics tracking. This updates Thompson Sampling priors.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_name": {
                        "type": "string",
                        "description": "Name of the executed prompt"
                    },
                    "version": {
                        "type": "integer",
                        "description": "Version that was executed"
                    },
                    "task_type": {
                        "type": "string",
                        "description": "Type of task (e.g., 'summarization', 'code_review')"
                    },
                    "input_data": {
                        "type": "string",
                        "description": "Input provided to the prompt"
                    },
                    "output": {
                        "type": "string",
                        "description": "Generated output"
                    },
                    "quality_score": {
                        "type": "number",
                        "description": "Quality score (0.0-1.0). Score >= 0.7 is considered success.",
                        "minimum": 0,
                        "maximum": 1
                    },
                    "latency_ms": {
                        "type": "number",
                        "description": "Execution latency in milliseconds",
                        "minimum": 0
                    },
                    "llm_provider": {
                        "type": "string",
                        "description": "LLM provider used (e.g., 'anthropic', 'openai', 'ollama')"
                    },
                    "llm_model": {
                        "type": "string",
                        "description": "Model name used"
                    },
                    "token_count": {
                        "type": "integer",
                        "description": "Total tokens used",
                        "minimum": 0
                    }
                },
                "required": ["prompt_name", "quality_score"]
            }
        ),

        # ========== MRF Tools ==========
        Tool(
            name="promptly_mrf_enhance",
            description="Enhance a prompt with the 7-component MRF metaprompt framework (ROLE → OBJECTIVE → PROCESS → FORMAT → CONSTRAINTS → UNCERTAINTY → VALIDATION).",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_name": {
                        "type": "string",
                        "description": "Name identifier for the prompt (for tracking)"
                    },
                    "prompt_content": {
                        "type": "string",
                        "description": "The prompt content to enhance"
                    },
                    "task_type": {
                        "type": "string",
                        "description": "Type of task for optimization (e.g., 'code_review', 'summarization', 'general')"
                    },
                    "model_provider": {
                        "type": "string",
                        "description": "Target LLM provider for optimization",
                        "enum": ["claude", "gemini", "gpt", "ollama"],
                        "default": "claude"
                    }
                },
                "required": ["prompt_name", "prompt_content"]
            }
        ),
        Tool(
            name="promptly_mrf_refine",
            description="Refine a response using MRF strategies. Supports 5 refinement strategies: REFINE, CRITIQUE, VERIFY, ELEGANCE, HOFSTADTER (or AUTO for auto-selection).",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_name": {
                        "type": "string",
                        "description": "Name of the prompt that generated the response"
                    },
                    "query": {
                        "type": "string",
                        "description": "Original query/input"
                    },
                    "response": {
                        "type": "string",
                        "description": "Response to refine"
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Refinement strategy to use",
                        "enum": ["refine", "critique", "verify", "elegance", "hofstadter", "auto"],
                        "default": "auto"
                    },
                    "query_type": {
                        "type": "string",
                        "description": "Type of query for strategy selection (e.g., 'factual', 'analytical', 'creative')"
                    }
                },
                "required": ["prompt_name", "query", "response"]
            }
        ),
        Tool(
            name="promptly_mrf_recommend",
            description="Get Thompson Sampling recommendation for the best MRF refinement strategy for a query type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "description": "Type of query (e.g., 'factual', 'analytical', 'creative', 'procedural')"
                    }
                },
                "required": ["query_type"]
            }
        ),
        Tool(
            name="promptly_mrf_analytics",
            description="Get MRF Thompson Sampling learning statistics showing expected quality and sample counts for each refinement strategy per query type.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="promptly_mrf_history",
            description="Get available MRF task templates (code_review, summarization, explanation, general, etc.).",
            inputSchema={
                "type": "object",
                "properties": {}
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
        # ========== Analytics Tools ==========
        if name == "promptly_analytics_stats":
            engine = get_analytics()
            if not engine:
                return [TextContent(
                    type="text",
                    text="Analytics engine not available. Check installation."
                )]

            prompt_name = arguments["prompt_name"]
            days = arguments.get("days", 30)
            output_format = arguments.get("format", "json")

            analytics = engine.get_prompt_analytics(prompt_name, window_days=days)

            if output_format == "json":
                result = {
                    "prompt_name": analytics.prompt_name,
                    "total_executions": analytics.total_executions,
                    "avg_quality": analytics.avg_quality,
                    "success_rate": analytics.success_rate,
                    "avg_latency_ms": analytics.avg_latency_ms,
                    "quality_trend": analytics.quality_trend,
                    "task_type_distribution": analytics.task_type_distribution,
                    "version_performance": analytics.version_performance,
                    "thompson_expected_quality": analytics.thompson_expected_quality,
                    "recommendation": analytics.recommendation
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            else:
                # Text format
                lines = [
                    f"📊 Analytics for '{prompt_name}'",
                    "=" * 50,
                    f"Total Executions: {analytics.total_executions}",
                    f"Average Quality: {analytics.avg_quality:.2%}",
                    f"Success Rate: {analytics.success_rate:.2%}",
                    f"Average Latency: {analytics.avg_latency_ms:.1f}ms",
                    f"Quality Trend: {analytics.quality_trend}",
                    f"Thompson Expected: {analytics.thompson_expected_quality:.2%}",
                    "",
                    f"Recommendation: {analytics.recommendation}"
                ]
                return [TextContent(type="text", text="\n".join(lines))]

        elif name == "promptly_analytics_compare":
            engine = get_analytics()
            if not engine:
                return [TextContent(
                    type="text",
                    text="Analytics engine not available. Check installation."
                )]

            prompt_a = arguments["prompt_a"]
            prompt_b = arguments["prompt_b"]
            task_type = arguments.get("task_type")
            days = arguments.get("days", 30)

            result = engine.compare_prompts(prompt_a, prompt_b, task_type=task_type, window_days=days)

            return [TextContent(type="text", text=json.dumps({
                "prompt_a": {
                    "name": result.prompt_a,
                    "avg_quality": result.prompt_a_stats.avg_quality,
                    "executions": result.prompt_a_stats.total_executions,
                    "success_rate": result.prompt_a_stats.success_rate
                },
                "prompt_b": {
                    "name": result.prompt_b,
                    "avg_quality": result.prompt_b_stats.avg_quality,
                    "executions": result.prompt_b_stats.total_executions,
                    "success_rate": result.prompt_b_stats.success_rate
                },
                "winner": result.winner,
                "quality_difference": result.prompt_b_stats.avg_quality - result.prompt_a_stats.avg_quality,
                "statistical_significance": result.statistical_significance,
                "recommendation": result.recommendation
            }, indent=2))]

        elif name == "promptly_recommend":
            engine = get_analytics()
            if not engine:
                return [TextContent(
                    type="text",
                    text="Analytics engine not available. Check installation."
                )]

            task_type = arguments["task_type"]
            candidates = arguments.get("candidates")

            recommendation = engine.get_thompson_recommendation(task_type, candidates=candidates)

            return [TextContent(type="text", text=json.dumps(recommendation, indent=2))]

        elif name == "promptly_analytics_trend":
            engine = get_analytics()
            if not engine:
                return [TextContent(
                    type="text",
                    text="Analytics engine not available. Check installation."
                )]

            prompt_name = arguments["prompt_name"]
            days = arguments.get("days", 30)
            granularity = arguments.get("granularity", "daily")

            trend_data = engine.get_quality_trend(prompt_name, window_days=days, granularity=granularity)

            return [TextContent(type="text", text=json.dumps({
                "prompt_name": prompt_name,
                "period": {
                    "days": days,
                    "granularity": granularity
                },
                "trend_data": trend_data
            }, indent=2))]

        elif name == "promptly_analytics_underperforming":
            engine = get_analytics()
            if not engine:
                return [TextContent(
                    type="text",
                    text="Analytics engine not available. Check installation."
                )]

            threshold = arguments.get("threshold", 0.6)
            days = arguments.get("days", 30)

            underperforming = engine.identify_underperforming(threshold=threshold, window_days=days)

            return [TextContent(type="text", text=json.dumps({
                "threshold": threshold,
                "underperforming_prompts": underperforming,
                "total_underperforming": len(underperforming)
            }, indent=2))]

        # ========== HoloLoom Integration ==========
        elif name == "promptly_hololoom_enhance":
            rag = await get_rag_integration()
            if not rag:
                return [TextContent(
                    type="text",
                    text="HoloLoom RAG integration not available. Check installation."
                )]

            prompt_name = arguments["prompt_name"]
            task_description = arguments.get("task_description", "")
            context_k = arguments.get("context_k", 5)
            injection_method = arguments.get("injection_method", "prefix")

            # Get original prompt
            prompt_data = p.get(prompt_name)
            original_content = prompt_data["content"]

            # Enhance with RAG
            result = await rag.enhance_prompt(
                prompt_content=original_content,
                task_description=task_description,
                context_k=context_k,
                injection_method=injection_method
            )

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "promptly_hololoom_agentic":
            executor = await get_agentic_executor()
            if not executor:
                return [TextContent(
                    type="text",
                    text="HoloLoom agentic executor not available. Check installation."
                )]

            prompt_name = arguments["prompt_name"]
            mode = arguments["mode"]
            variables = arguments.get("variables", {})
            max_steps = arguments.get("max_steps", 5)

            # Get prompt
            prompt_data = p.get(prompt_name)
            prompt_content = prompt_data["content"]

            # Execute with agentic reasoning
            result = await executor.execute(
                prompt_content=prompt_content,
                variables=variables,
                mode=mode,
                max_steps=max_steps
            )

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "promptly_hololoom_similar":
            bridge = get_hololoom_bridge()
            if not bridge:
                return [TextContent(
                    type="text",
                    text="HoloLoom bridge not available. Check installation."
                )]

            query = arguments["query"]
            limit = arguments.get("limit", 5)
            min_quality = arguments.get("min_quality", 0.7)

            # Get analytics engine for quality filtering
            engine = get_analytics()

            # Find similar prompts
            results = await bridge.discover_similar_prompts(
                query=query,
                limit=limit,
                min_quality=min_quality,
                analytics_engine=engine
            )

            return [TextContent(type="text", text=json.dumps({
                "query": query,
                "results": results,
                "total_matches": len(results)
            }, indent=2))]

        # ========== Prompt Management ==========
        elif name == "promptly_get":
            prompt_name = arguments["name"]
            version = arguments.get("version")

            prompt_data = p.get(prompt_name, version=version)

            return [TextContent(type="text", text=json.dumps({
                "name": prompt_data["name"],
                "content": prompt_data["content"],
                "version": prompt_data["version"],
                "branch": prompt_data["branch"],
                "commit": prompt_data["commit_hash"][:12],
                "metadata": prompt_data.get("metadata", {}),
                "created_at": prompt_data.get("created_at", "")
            }, indent=2))]

        elif name == "promptly_list":
            pattern = arguments.get("pattern")
            tag = arguments.get("tag")
            limit = arguments.get("limit", 50)

            prompts = p.list()

            # Filter by pattern
            if pattern:
                import fnmatch
                prompts = [pr for pr in prompts if fnmatch.fnmatch(pr["name"], pattern)]

            # Filter by tag
            if tag:
                prompts = [pr for pr in prompts if tag in pr.get("metadata", {}).get("tags", [])]

            # Apply limit
            prompts = prompts[:limit]

            return [TextContent(type="text", text=json.dumps({
                "prompts": prompts,
                "count": len(prompts)
            }, indent=2))]

        elif name == "promptly_create":
            prompt_name = arguments["name"]
            content = arguments["content"]
            metadata = arguments.get("metadata", {})

            commit = p.add(prompt_name, content, metadata=metadata)

            return [TextContent(type="text", text=json.dumps({
                "created": True,
                "name": prompt_name,
                "commit": commit[:12],
                "message": f"Prompt '{prompt_name}' created successfully"
            }, indent=2))]

        elif name == "promptly_record_execution":
            engine = get_analytics()
            if not engine:
                return [TextContent(
                    type="text",
                    text="Analytics engine not available. Check installation."
                )]

            prompt_name = arguments["prompt_name"]
            quality_score = arguments["quality_score"]
            version = arguments.get("version", 1)
            task_type = arguments.get("task_type")
            input_data = arguments.get("input_data")
            output = arguments.get("output")
            latency_ms = arguments.get("latency_ms")
            llm_provider = arguments.get("llm_provider")
            llm_model = arguments.get("llm_model")
            token_count = arguments.get("token_count")

            # Record execution
            execution_id = engine.record_execution(
                prompt_name=prompt_name,
                version=version,
                quality_score=quality_score,
                task_type=task_type,
                input_data=input_data,
                output=output,
                latency_ms=latency_ms,
                llm_provider=llm_provider,
                llm_model=llm_model,
                token_count=token_count
            )

            # Get updated stats
            is_success = quality_score >= 0.7

            return [TextContent(type="text", text=json.dumps({
                "recorded": True,
                "execution_id": execution_id,
                "prompt_name": prompt_name,
                "version": version,
                "quality_score": quality_score,
                "is_success": is_success,
                "thompson_prior_updated": True
            }, indent=2))]

        # ====================================================================
        # MRF Tools
        # ====================================================================

        elif name == "promptly_mrf_enhance":
            mrf = get_mrf()
            if mrf is None:
                return [TextContent(type="text", text=json.dumps({
                    "error": "MRF integration not available"
                }, indent=2))]

            prompt_name = arguments.get("prompt_name")
            prompt_content = arguments.get("prompt_content")
            task_type = arguments.get("task_type")
            model_provider = arguments.get("model_provider", "claude")

            if not prompt_name or not prompt_content:
                return [TextContent(type="text", text=json.dumps({
                    "error": "prompt_name and prompt_content are required"
                }, indent=2))]

            # Convert model_provider string to ModelProvider enum
            try:
                model = ModelProvider(model_provider) if model_provider else None
            except ValueError:
                model = None

            result = await mrf.enhance_prompt(
                prompt_name=prompt_name,
                prompt_content=prompt_content,
                task_type=task_type,
                model=model
            )

            return [TextContent(type="text", text=json.dumps({
                "original_prompt": result.original_prompt,
                "enhanced_prompt": result.enhanced_prompt,
                "framework_used": result.framework_used,
                "model_optimized_for": result.model_optimized_for,
                "components_applied": result.components_applied,
                "quality_score": result.quality_score,
                "metadata": result.metadata
            }, indent=2))]

        elif name == "promptly_mrf_refine":
            mrf = get_mrf()
            if mrf is None:
                return [TextContent(type="text", text=json.dumps({
                    "error": "MRF integration not available"
                }, indent=2))]

            prompt_name = arguments.get("prompt_name")
            query = arguments.get("query")
            response = arguments.get("response")
            strategy = arguments.get("strategy", "auto")
            query_type = arguments.get("query_type")
            max_iterations = arguments.get("max_iterations", 3)
            quality_threshold = arguments.get("quality_threshold", 0.85)

            if not prompt_name or not query or not response:
                return [TextContent(type="text", text=json.dumps({
                    "error": "prompt_name, query, and response are required"
                }, indent=2))]

            # Convert strategy string to RefinementStrategy enum
            try:
                strat = RefinementStrategy(strategy.upper()) if strategy else RefinementStrategy.AUTO
            except ValueError:
                strat = RefinementStrategy.AUTO

            result = await mrf.refine_response(
                prompt_name=prompt_name,
                query=query,
                response=response,
                strategy=strat,
                query_type=query_type,
                max_iterations=max_iterations,
                quality_threshold=quality_threshold
            )

            return [TextContent(type="text", text=json.dumps({
                "original_response": result.original_response,
                "refined_response": result.refined_response,
                "strategy_used": result.strategy_used,
                "iterations": result.iterations,
                "quality_trajectory": result.quality_trajectory,
                "final_quality": result.final_quality,
                "improvements": result.improvements,
                "metadata": result.metadata
            }, indent=2))]

        elif name == "promptly_mrf_recommend":
            mrf = get_mrf()
            if mrf is None:
                return [TextContent(type="text", text=json.dumps({
                    "error": "MRF integration not available"
                }, indent=2))]

            query_type = arguments.get("query_type")

            if not query_type:
                return [TextContent(type="text", text=json.dumps({
                    "error": "query_type is required"
                }, indent=2))]

            result = mrf.get_strategy_recommendation(query_type=query_type)

            return [TextContent(type="text", text=json.dumps({
                "recommended_strategy": result.recommended_strategy,
                "confidence": result.confidence,
                "expected_quality": result.expected_quality,
                "reasoning": result.reasoning,
                "alternatives": [
                    {"strategy": alt.strategy, "expected_quality": alt.expected_quality}
                    for alt in result.alternatives
                ] if result.alternatives else []
            }, indent=2))]

        elif name == "promptly_mrf_analytics":
            mrf = get_mrf()
            if mrf is None:
                return [TextContent(type="text", text=json.dumps({
                    "error": "MRF integration not available"
                }, indent=2))]

            result = mrf.get_learning_statistics()

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "promptly_mrf_history":
            mrf = get_mrf()
            if mrf is None:
                return [TextContent(type="text", text=json.dumps({
                    "error": "MRF integration not available"
                }, indent=2))]

            # Get available task templates as a proxy for history
            templates = mrf.get_task_templates()

            return [TextContent(type="text", text=json.dumps({
                "available_templates": templates,
                "count": len(templates)
            }, indent=2))]

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
    """Run the MCP analytics server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
