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
- Complete reasoning provenance

MCP Tool Categories (17 Tools)
===============================

CORE MEMORY (3 tools):
  - hololoom_experience: Form memories from any input
  - hololoom_recall: Retrieve memories with semantic search
  - hololoom_weave: Recursive reasoning with strategy selection

ANALYTICS & MONITORING (1 tool):
  - hololoom_analytics_summary: Performance metrics and recommendations

CODE QUALITY & REVIEW (7 tools):
  - skill_code_reviewer: Multi-pass code quality analysis
  - skill_bug_detective: Root cause debugging analysis
  - skill_test_generator: Comprehensive test suite generation
  - skill_performance_profiler: Bottleneck identification
  - skill_refactoring_expert: Code structure improvement
  - skill_security_auditor: Vulnerability assessment (OWASP)
  - skill_sql_optimizer: Query performance tuning

DOCUMENTATION & LEARNING (3 tools):
  - skill_documentation_writer: API/tutorial documentation
  - skill_code_explainer: Code breakdown and explanation
  - skill_naming_consultant: Variable/function naming

ARCHITECTURE & DEVELOPMENT (3 tools):
  - skill_api_designer: REST/GraphQL API design
  - skill_architecture_advisor: System architecture strategy
  - skill_migration_planner: Technology transition planning

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
Updated: 2025-11-16 - Added categories and complexity levels
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
            description="Form a memory from any input (text, code, notes, etc.) and store in knowledge graph with extracted entities and motifs. Supports all modalities.\n\n**Categories**: Memory, Core, Knowledge-Building\n**Complexity**: Simple\n**Strategy**: Direct storage with semantic indexing\n\n**When to use**: Learning new concepts, storing code snippets, persisting project notes, building knowledge base\n\n**Example**: content=\"Thompson Sampling balances exploration vs exploitation using Bayesian priors.\"\n\n**Returns**: Memory object with extracted entities, confidence score, and timestamp",
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
            description="Retrieve memories using semantic search + knowledge graph traversal. Returns relevant context with confidence scores.\n\n**Categories**: Memory, Core, Retrieval\n**Complexity**: Simple\n**Strategy**: Hybrid BM25 + semantic search with graph expansion\n\n**When to use**: Finding relevant context, building question-answer chains, connecting ideas, research synthesis\n\n**Example**: query=\"What did I learn about exploration strategies?\", limit=5\n\n**Returns**: List of memory shards with content, similarity score, and source metadata",
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
            description="Complete reasoning cycle with recursive refinement. Auto-selects optimal strategy (refine/critique/decompose/verify/explore) and iterates until quality threshold met.\n\n**Categories**: Reasoning, Core, Analysis\n**Complexity**: Complex\n**Strategy**: Multi-pass refinement with adaptive strategy selection (6 strategies)\n\n**When to use**: Complex analysis, verification-critical answers, research papers, architectural decisions, multi-step reasoning\n\n**Strategies available**:\n  - refine: Iterative improvement through multiple passes\n  - critique: Self-critique analysis\n  - decompose: Break problem into steps\n  - explore: Multi-angle exploration\n  - verify: Answer verification + consistency checks\n  - hofstadter: Meta-reasoning (thinks about thinking)\n  - adaptive: Auto-selects best strategy\n\n**Example**: query=\"Explain Thompson Sampling tradeoffs\", strategy=\"verify\", max_iterations=3, quality_threshold=0.85\n\n**Returns**: Final response with confidence score, reasoning journal, iterations used, and strategy applied",
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
            description="Get comprehensive analytics dashboard: strategy performance metrics, quality trajectory, cost breakdown, and optimization recommendations.\n\n**Categories**: Analytics, Monitoring\n**Complexity**: Simple\n**Strategy**: Aggregation and trend analysis\n\n**When to use**: Session review, performance monitoring, strategy effectiveness analysis, cost optimization, system tuning\n\n**Metrics provided**:\n  - Success rate by strategy\n  - Average quality and latency\n  - Cost breakdown\n  - Optimization recommendations\n  - Trend analysis (7-day, 30-day moving averages)\n\n**Returns**: Performance statistics (success rate by strategy, avg quality/latency), recommendations, and cost summary",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
    ])

    # Professional skill tools (13 skills)
    skill_tools = [
        Tool(
            name="skill_code_reviewer",
            description="Thorough code review focusing on best practices, potential bugs, and quality improvements. Uses CRITIQUE strategy for multi-pass analysis.\n\n**Categories**: Code-Quality, Review\n**Complexity**: Moderate\n**Strategy**: CRITIQUE (multi-pass analysis)\n\n**When to use**: Code quality checks, pre-commit review, pull request analysis, mentoring, finding bugs\n\n**Review focuses**:\n  - Best practices and code style\n  - Potential bugs and edge cases\n  - Performance optimizations\n  - Readability and maintainability\n  - Security concerns\n\n**Example**: code=\"def calculate(x): return x*2\", language=\"python\", focus_areas=\"efficiency, readability\"\n\n**Returns**: Detailed critique with improvement suggestions, risk areas, and quality score",
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
            description="Root cause analysis and systematic debugging. Decomposes problem into steps, traces execution path, identifies failure point. Uses DECOMPOSE strategy.\n\n**Categories**: Debugging, Code-Quality\n**Complexity**: Moderate\n**Strategy**: DECOMPOSE (systematic breakdown)\n\n**When to use**: Production debugging, crash investigation, test failure analysis, performance regressions, debugging stubborn bugs\n\n**Analysis includes**:\n  - Execution path tracing\n  - Failure point identification\n  - Root cause analysis\n  - Reproduction steps\n  - Fix recommendations\n\n**Example**: code=\"for i in range(n): arr[i] = i*2\", bug_description=\"Off-by-one error on last iteration\", language=\"python\"\n\n**Returns**: Root cause identification, execution trace, reproduction steps, and fix recommendation",
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
            description="Generate comprehensive test suites with high coverage. Explores edge cases, boundary conditions, error paths. Uses EXPLORE strategy.\n\n**Categories**: Code-Quality, Testing\n**Complexity**: Moderate\n**Strategy**: EXPLORE (multi-angle edge case discovery)\n\n**When to use**: Test-driven development, legacy code testing, coverage gap filling, regression prevention, TDD workflows\n\n**Test types generated**:\n  - Happy path tests\n  - Edge case tests\n  - Boundary condition tests\n  - Error handling tests\n  - Performance tests\n  - Security tests\n\n**Example**: code=\"def divide(a,b): return a/b\", language=\"python\", framework=\"pytest\", edge_cases=true\n\n**Returns**: Complete test suite with happy path tests, edge cases, error handling, and coverage estimate",
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
            description="Design REST/GraphQL APIs following industry best practices. Refines iteratively for optimal resource modeling, versioning, auth. Uses REFINE strategy.\n\n**Categories**: Development, Architecture\n**Complexity**: Moderate\n**Strategy**: REFINE (iterative API refinement)\n\n**When to use**: API architecture planning, greenfield API development, protocol selection, schema design, technology selection\n\n**Design covers**:\n  - REST vs GraphQL comparison\n  - Resource modeling and endpoints\n  - Authentication and authorization\n  - Versioning strategy (URI, header, etc.)\n  - Error handling and status codes\n  - Rate limiting and pagination\n  - Documentation standards\n\n**Example**: requirements=\"User management CRUD with OAuth2 login\", auth_type=\"oauth2\", version_strategy=\"uri\"\n\n**Returns**: Complete API specification with endpoints, schemas, auth flow, versioning strategy, and error handling",
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
            description="Generate comprehensive documentation (README, API docs, tutorials, reference). Refines iteratively for clarity and completeness. Uses REFINE strategy.\n\n**Categories**: Documentation, Learning\n**Complexity**: Simple\n**Strategy**: REFINE (iterative clarity improvement)\n\n**When to use**: Project onboarding, API documentation, library publishing, knowledge transfer, open source projects\n\n**Documentation types**:\n  - README with quick start\n  - API reference with examples\n  - Tutorial guides\n  - Troubleshooting guides\n  - Best practices\n  - Contributing guidelines\n\n**Example**: code=\"class DataProcessor: def process(self, data): ...\", language=\"python\", doc_type=\"api\"\n\n**Returns**: Formatted documentation with examples, parameter descriptions, use cases, and best practices",
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
            description="Analyze performance bottlenecks and suggest optimizations. Decomposes hot paths, identifies algorithmic inefficiencies, estimates impact. Uses DECOMPOSE strategy.\n\n**Categories**: Code-Quality, Performance\n**Complexity**: Moderate\n**Strategy**: DECOMPOSE (hot path analysis)\n\n**When to use**: Performance optimization, scaling issues, slow endpoints, resource usage reduction, cost optimization\n\n**Analysis includes**:\n  - Hot path identification\n  - Algorithmic inefficiency detection\n  - Memory usage analysis\n  - I/O optimization opportunities\n  - Caching opportunities\n  - Parallelization potential\n  - Estimated speedup calculation\n\n**Example**: code=\"def search(arr): return max(arr)\", language=\"python\", profile_data=\"{calls: 1M, time: 500ms}\"\n\n**Returns**: Performance analysis with bottleneck identification, optimization suggestions, estimated speedup",
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
            description="Strategic system architecture guidance considering scale, tradeoffs, constraints. Meta-reasoning through recursive design analysis. Uses HOFSTADTER meta-reasoning.\n\n**Categories**: Development, Architecture\n**Complexity**: Complex\n**Strategy**: HOFSTADTER (meta-reasoning about design decisions)\n\n**When to use**: System redesign, scaling strategy, technology selection, architectural patterns, infrastructure decisions, high-stakes choices\n\n**Design considerations**:\n  - Scalability patterns (horizontal/vertical)\n  - High availability and disaster recovery\n  - Microservices vs monolith tradeoffs\n  - Data consistency strategies\n  - Technology stack recommendations\n  - Cost optimization\n  - Security and compliance\n\n**Example**: requirements=\"Handle 1M users, 99.99% uptime\", scale=\"large\", constraints=\"$500k/month cloud budget\"\n\n**Returns**: Architecture recommendation with component diagram, tradeoff analysis, implementation roadmap, and risk assessment",
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
            description="Create detailed migration strategies for technology transitions. Step-by-step planning with risk mitigation. Uses DECOMPOSE for task breakdown.\n\n**Categories**: Development, Migration\n**Complexity**: Moderate\n**Strategy**: DECOMPOSE (phase breakdown)\n\n**When to use**: Framework upgrades, database migrations, language transitions, cloud migrations, dependency updates, system modernization\n\n**Planning includes**:\n  - Phase-by-phase breakdown\n  - Timeline estimation\n  - Rollback strategy\n  - Testing plan\n  - Data migration approach\n  - Downtime minimization\n  - Risk mitigation\n  - Success metrics\n\n**Example**: source=\"MySQL 5.7\", target=\"PostgreSQL 14\", context=\"ecommerce platform, 500GB data\"\n\n**Returns**: Phase-by-phase migration plan with timeline, rollback strategy, testing checklist, and risk mitigation",
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
            description="Break down complex code into simple explanations. Tailors level (beginner/intermediate/advanced), refines for maximum clarity. Uses REFINE strategy.\n\n**Categories**: Documentation, Learning\n**Complexity**: Simple\n**Strategy**: REFINE (iterative clarity)\n\n**When to use**: Code learning, documentation, onboarding, teaching, understanding third-party libraries, code comprehension\n\n**Explanation includes**:\n  - Step-by-step breakdown\n  - Real-world analogies\n  - Conceptual examples\n  - Related concepts and connections\n  - Common pitfalls\n  - Further learning resources\n\n**Audience levels**:\n  - beginner: No prior knowledge assumed\n  - intermediate: Programming basics known\n  - advanced: Looking for deep technical details\n\n**Example**: code=\"@decorator\\ndef func(): ...\", language=\"python\", level=\"beginner\"\n\n**Returns**: Explanation with step-by-step breakdown, analogies, examples, and related concepts",
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
            description="Improve code readability through better naming. Suggests alternatives for variables, functions, classes. Uses CRITIQUE strategy.\n\n**Categories**: Naming, Code-Quality\n**Complexity**: Simple\n**Strategy**: CRITIQUE (naming analysis)\n\n**When to use**: Code refactoring, readability improvement, self-documenting code, naming conventions, style consistency\n\n**Naming review covers**:\n  - Variable naming clarity\n  - Function naming conventions\n  - Class naming patterns\n  - Constant naming standards\n  - Boolean variable naming\n  - Domain-specific terminology\n  - Consistency across codebase\n\n**Example**: code=\"def calc(x, y): return x + y\", language=\"python\", context=\"financial calculations\"\n\n**Returns**: Naming suggestions with reasoning, before/after comparison, and impact on readability",
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
            description="Optimize SQL queries for performance and readability. Analyzes execution plans, suggests indexes, improves clarity. Uses REFINE strategy.\n\n**Categories**: Code-Quality, Performance\n**Complexity**: Moderate\n**Strategy**: REFINE (query optimization)\n\n**When to use**: Query performance tuning, N+1 problem fixing, slow query optimization, query refactoring, schema optimization\n\n**Optimization includes**:\n  - Execution plan analysis\n  - Index recommendations\n  - Join optimization\n  - Subquery to JOIN conversion\n  - N+1 problem detection\n  - Query readability improvements\n  - Pagination optimization\n  - Estimated performance improvement\n\n**Example**: query=\"SELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE region='US')\", database_type=\"postgresql\"\n\n**Returns**: Optimized query with explain plan, index recommendations, estimated performance improvement",
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
            description="Refactor code for maintainability, performance, or testability. Applies design patterns, improves structure, reduces complexity. Uses CRITIQUE strategy.\n\n**Categories**: Code-Quality, Refactoring\n**Complexity**: Moderate\n**Strategy**: CRITIQUE (multi-pass refactoring)\n\n**When to use**: Legacy code cleanup, technical debt reduction, complexity reduction, design pattern application, code modernization\n\n**Refactoring focuses**:\n  - Nested complexity reduction\n  - Design pattern application\n  - DRY principle application\n  - SOLID principles\n  - Cyclomatic complexity reduction\n  - Testability improvements\n  - Performance optimizations\n  - Before/after metrics\n\n**Example**: code=\"if cond1: if cond2: if cond3: do_work()...\", language=\"python\", focus=\"maintainability\"\n\n**Returns**: Refactored code with changes explained, before/after metrics, applied patterns, and improvement summary",
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
            description="Comprehensive security audit finding OWASP vulnerabilities, injection risks, auth issues. Verifies fixes. Uses VERIFY strategy.\n\n**Categories**: Security, Code-Quality\n**Complexity**: Complex\n**Strategy**: VERIFY (multi-pass security analysis)\n\n**When to use**: Security code review, vulnerability assessment, pre-deployment audit, compliance verification, threat modeling\n\n**Security checks cover**:\n  - OWASP Top 10 vulnerabilities\n  - SQL injection and injection attacks\n  - Cross-site scripting (XSS)\n  - Authentication/authorization issues\n  - Cryptography and encryption\n  - Secrets and credentials handling\n  - Input validation\n  - Error handling and logging\n  - Dependency vulnerabilities\n  - Severity levels and CVSS scoring\n\n**Example**: code=\"user = request.args['user']; query = f'SELECT * FROM users WHERE id={user}'\", language=\"python\"\n\n**Returns**: Security findings with severity levels, OWASP mapping, exploitation scenarios, and remediation steps",
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

    try:
        # Validate required parameters
        if "content" not in args:
            raise KeyError("content")

        content = args["content"]
        context = args.get("context", "")

        # Validate parameter types
        if not isinstance(content, str):
            raise TypeError(f"content must be str, got {type(content).__name__}")
        if not isinstance(context, str):
            raise TypeError(f"context must be str, got {type(context).__name__}")

        # Validate content is not empty
        if not content.strip():
            raise ValueError("content cannot be empty")

        if len(content) > 100000:
            raise ValueError(f"content too long: {len(content)} chars (max: 100000)")

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

    except KeyError as e:
        logger.error(f"Missing required parameter in handle_experience: {str(e)}")
        return [TextContent(
            type="text",
            text=f"❌ Missing required parameter: {str(e)}\n\n"
                 f"Expected parameters:\n"
                 f"  - content: str (required) - Text, code, or notes to remember\n"
                 f"  - context: str (optional) - Additional metadata or context\n\n"
                 f"Example:\n"
                 f"  {{'content': 'Thompson Sampling balances exploration vs exploitation', "
                 f"'context': 'Bayesian methods'}}"
        )]

    except TypeError as e:
        logger.error(f"Invalid parameter type in handle_experience: {str(e)}")
        return [TextContent(
            type="text",
            text=f"❌ Invalid parameter type: {str(e)}\n\n"
                 f"Parameter requirements:\n"
                 f"  - content: string (required)\n"
                 f"  - context: string (optional)\n\n"
                 f"Example:\n"
                 f"  {{'content': 'Learning text here', 'context': 'Domain or category'}}"
        )]

    except ValueError as e:
        logger.error(f"Invalid parameter value in handle_experience: {str(e)}")
        return [TextContent(
            type="text",
            text=f"❌ Invalid parameter value: {str(e)}\n\n"
                 f"Constraints:\n"
                 f"  - content must not be empty\n"
                 f"  - content must be ≤ 100,000 characters\n\n"
                 f"Example:\n"
                 f"  {{'content': 'Meaningful content to store in memory', 'context': 'optional context'}}"
        )]

    except Exception as e:
        logger.error(f"Unexpected error in handle_experience: {str(e)}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"❌ Unexpected error storing memory: {str(e)}\n\n"
                 f"Possible causes:\n"
                 f"  1. HoloLoom initialization failed\n"
                 f"  2. Memory backend unavailable\n"
                 f"  3. Storage error\n\n"
                 f"Check server logs for details."
        )]


async def handle_recall(args: Dict[str, Any]) -> List[TextContent]:
    """Handle hololoom_recall tool."""
    from HoloLoom.hololoom import HoloLoom

    try:
        # Validate required parameters
        if "query" not in args:
            raise KeyError("query")

        query = args["query"]
        limit = args.get("limit", 5)

        # Validate parameter types
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        if not isinstance(limit, int):
            raise TypeError(f"limit must be int, got {type(limit).__name__}")

        # Validate parameter values
        if not query.strip():
            raise ValueError("query cannot be empty")
        if limit < 1:
            raise ValueError(f"limit must be ≥ 1, got {limit}")
        if limit > 100:
            raise ValueError(f"limit must be ≤ 100, got {limit}")

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

    except KeyError as e:
        logger.error(f"Missing required parameter in handle_recall: {str(e)}")
        return [TextContent(
            type="text",
            text=f"❌ Missing required parameter: {str(e)}\n\n"
                 f"Expected parameters:\n"
                 f"  - query: str (required) - Search query for memory retrieval\n"
                 f"  - limit: int (optional) - Max results to return (default: 5, max: 100)\n\n"
                 f"Example:\n"
                 f"  {{'query': 'What did I learn about Thompson Sampling?', 'limit': 5}}"
        )]

    except TypeError as e:
        logger.error(f"Invalid parameter type in handle_recall: {str(e)}")
        return [TextContent(
            type="text",
            text=f"❌ Invalid parameter type: {str(e)}\n\n"
                 f"Parameter requirements:\n"
                 f"  - query: string (required)\n"
                 f"  - limit: integer (optional, default: 5)\n\n"
                 f"Example:\n"
                 f"  {{'query': 'Search for memories', 'limit': 10}}"
        )]

    except ValueError as e:
        logger.error(f"Invalid parameter value in handle_recall: {str(e)}")
        return [TextContent(
            type="text",
            text=f"❌ Invalid parameter value: {str(e)}\n\n"
                 f"Constraints:\n"
                 f"  - query must not be empty\n"
                 f"  - limit must be between 1 and 100 (default: 5)\n\n"
                 f"Example:\n"
                 f"  {{'query': 'Meaningful search query', 'limit': 5}}"
        )]

    except Exception as e:
        logger.error(f"Unexpected error in handle_recall: {str(e)}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"❌ Unexpected error retrieving memories: {str(e)}\n\n"
                 f"Possible causes:\n"
                 f"  1. HoloLoom initialization failed\n"
                 f"  2. Memory backend unavailable\n"
                 f"  3. Search engine error\n"
                 f"  4. No memories stored yet\n\n"
                 f"Check server logs for details."
        )]


async def handle_weave(args: Dict[str, Any]) -> List[TextContent]:
    """Handle hololoom_weave tool with recursive reasoning."""
    try:
        # Validate required parameters
        if "query" not in args:
            raise KeyError("query")

        query_text = args["query"]
        strategy_name = args.get("strategy", "adaptive")
        max_iterations = args.get("max_iterations", 3)
        quality_threshold = args.get("quality_threshold", 0.85)

        # Validate parameter types
        if not isinstance(query_text, str):
            raise TypeError(f"query must be str, got {type(query_text).__name__}")
        if not isinstance(strategy_name, str):
            raise TypeError(f"strategy must be str, got {type(strategy_name).__name__}")
        if not isinstance(max_iterations, int):
            raise TypeError(f"max_iterations must be int, got {type(max_iterations).__name__}")
        if not isinstance(quality_threshold, (int, float)):
            raise TypeError(f"quality_threshold must be number, got {type(quality_threshold).__name__}")

        # Validate parameter values
        if not query_text.strip():
            raise ValueError("query cannot be empty")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be ≥ 1, got {max_iterations}")
        if max_iterations > 10:
            raise ValueError(f"max_iterations must be ≤ 10, got {max_iterations}")
        if quality_threshold < 0.0 or quality_threshold > 1.0:
            raise ValueError(f"quality_threshold must be between 0.0 and 1.0, got {quality_threshold}")

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

        strategy_lower = strategy_name.lower()
        if strategy_lower not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy = strategy_map[strategy_lower]

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

    except KeyError as e:
        logger.error(f"Missing required parameter in handle_weave: {str(e)}")
        return [TextContent(
            type="text",
            text=f"❌ Missing required parameter: {str(e)}\n\n"
                 f"Expected parameters:\n"
                 f"  - query: str (required) - Query to weave with reasoning\n"
                 f"  - strategy: str (optional) - Reasoning strategy (default: adaptive)\n"
                 f"  - max_iterations: int (optional) - Max refinement iterations (default: 3)\n"
                 f"  - quality_threshold: float (optional) - Quality target 0-1 (default: 0.85)\n\n"
                 f"Example:\n"
                 f"  {{'query': 'Explain Thompson Sampling tradeoffs', "
                 f"'strategy': 'verify', 'max_iterations': 3, 'quality_threshold': 0.85}}"
        )]

    except TypeError as e:
        logger.error(f"Invalid parameter type in handle_weave: {str(e)}")
        return [TextContent(
            type="text",
            text=f"❌ Invalid parameter type: {str(e)}\n\n"
                 f"Parameter requirements:\n"
                 f"  - query: string (required)\n"
                 f"  - strategy: string (optional, default: 'adaptive')\n"
                 f"  - max_iterations: integer (optional, default: 3)\n"
                 f"  - quality_threshold: number (optional, default: 0.85)\n\n"
                 f"Example:\n"
                 f"  {{'query': 'Your question here', 'strategy': 'verify'}}"
        )]

    except ValueError as e:
        logger.error(f"Invalid parameter value in handle_weave: {str(e)}")
        valid_strategies = ', '.join(['refine', 'critique', 'decompose', 'explore', 'verify', 'hofstadter', 'adaptive'])
        return [TextContent(
            type="text",
            text=f"❌ Invalid parameter value: {str(e)}\n\n"
                 f"Constraints:\n"
                 f"  - query must not be empty\n"
                 f"  - strategy must be one of: {valid_strategies}\n"
                 f"  - max_iterations must be 1-10 (default: 3)\n"
                 f"  - quality_threshold must be 0.0-1.0 (default: 0.85)\n\n"
                 f"Strategy guide:\n"
                 f"  - refine: Iterative improvement\n"
                 f"  - critique: Self-critique analysis\n"
                 f"  - decompose: Break into steps\n"
                 f"  - explore: Multi-angle exploration\n"
                 f"  - verify: Verify + consistency check\n"
                 f"  - hofstadter: Meta-reasoning\n"
                 f"  - adaptive: Auto-select best strategy\n\n"
                 f"Example:\n"
                 f"  {{'query': 'Complex question', 'strategy': 'verify', 'max_iterations': 3}}"
        )]

    except Exception as e:
        logger.error(f"Unexpected error in handle_weave: {str(e)}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"❌ Unexpected error during weaving: {str(e)}\n\n"
                 f"Possible causes:\n"
                 f"  1. Orchestrator initialization failed\n"
                 f"  2. Weaving timeout (try reducing max_iterations)\n"
                 f"  3. Memory backend unavailable\n"
                 f"  4. Strategy execution error\n\n"
                 f"Try:\n"
                 f"  - Reduce max_iterations (lower = faster)\n"
                 f"  - Use 'adaptive' strategy for auto-selection\n"
                 f"  - Check server logs for details"
        )]


async def handle_analytics_summary(args: Dict[str, Any]) -> List[TextContent]:
    """Handle hololoom_analytics_summary tool."""
    try:
        # Validate no unexpected parameters
        allowed_params = set()
        unexpected_params = set(args.keys()) - allowed_params
        if unexpected_params:
            logger.warning(f"Unexpected parameters in handle_analytics_summary: {unexpected_params}")

        # Create temporary orchestrator to access analytics
        try:
            async with RecursiveWeavingOrchestrator(
                cfg=config,
                shards=[],
                enable_analytics=True
            ) as orch:

                summary = orch.get_analytics_summary()

                # Handle empty analytics
                if not summary or not any(summary.values()):
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "status": "success",
                            "message": "No analytics data available yet",
                            "info": "Analytics will populate after executing queries with hololoom_weave"
                        }, indent=2)
                    )]

                return [TextContent(
                    type="text",
                    text=json.dumps(summary, indent=2)
                )]

        except AttributeError as e:
            logger.error(f"Orchestrator method not found in handle_analytics_summary: {str(e)}")
            return [TextContent(
                type="text",
                text=f"❌ Analytics unavailable: {str(e)}\n\n"
                     f"Possible causes:\n"
                     f"  1. Orchestrator initialization failed\n"
                     f"  2. Analytics module not loaded\n"
                     f"  3. Method 'get_analytics_summary' not available\n\n"
                     f"To use analytics:\n"
                     f"  1. Execute queries using hololoom_weave\n"
                     f"  2. Wait a moment for analytics to populate\n"
                     f"  3. Call analytics_summary again"
            )]

    except Exception as e:
        logger.error(f"Unexpected error in handle_analytics_summary: {str(e)}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"❌ Unexpected error retrieving analytics: {str(e)}\n\n"
                 f"Possible causes:\n"
                 f"  1. Analytics database unavailable\n"
                 f"  2. No queries executed yet\n"
                 f"  3. Analytics data corruption\n\n"
                 f"Troubleshooting:\n"
                 f"  - Execute some queries first using hololoom_weave\n"
                 f"  - Check that analytics is enabled in config\n"
                 f"  - See server logs for details\n\n"
                 f"Note: Analytics require at least one completed query"
        )]


async def handle_skill_execution(skill_name: str, args: Dict[str, Any]) -> List[TextContent]:
    """Handle professional skill execution."""
    from HoloLoom.agentic.skill_agents import execute_skill

    try:
        # Validate skill name
        if not skill_name or not isinstance(skill_name, str):
            raise ValueError(f"Invalid skill name: {skill_name}")

        # Check if skill is available
        available_skills = [
            "code-reviewer", "bug-detective", "test-generator", "api-designer",
            "documentation-writer", "performance-profiler", "architecture-advisor",
            "migration-planner", "code-explainer", "naming-consultant",
            "sql-optimizer", "refactoring-expert", "security-auditor"
        ]

        if skill_name not in available_skills:
            return [TextContent(
                type="text",
                text=f"❌ Unknown skill: {skill_name}\n\n"
                     f"Available skills ({len(available_skills)}):\n"
                     f"  - code-reviewer: Multi-pass code quality analysis\n"
                     f"  - bug-detective: Root cause analysis and debugging\n"
                     f"  - test-generator: Generate comprehensive test suites\n"
                     f"  - api-designer: Design REST/GraphQL APIs\n"
                     f"  - documentation-writer: Generate documentation\n"
                     f"  - performance-profiler: Identify bottlenecks\n"
                     f"  - architecture-advisor: Strategic system design\n"
                     f"  - migration-planner: Plan technology migrations\n"
                     f"  - code-explainer: Break down complex code\n"
                     f"  - naming-consultant: Improve variable/function names\n"
                     f"  - sql-optimizer: Optimize SQL queries\n"
                     f"  - refactoring-expert: Refactor for quality\n"
                     f"  - security-auditor: Security vulnerability assessment\n\n"
                     f"Example:\n"
                     f"  skill_name='code-reviewer' with code and language params"
            )]

        # Validate parameters are provided
        if not args:
            logger.warning(f"No parameters provided for skill: {skill_name}")
            return [TextContent(
                type="text",
                text=f"❌ Missing required parameters for skill: {skill_name}\n\n"
                     f"Check the skill documentation for required parameters.\n\n"
                     f"Common parameters:\n"
                     f"  - code: str (required for most skills)\n"
                     f"  - language: str (required for code-based skills)\n"
                     f"  - other skill-specific parameters\n\n"
                     f"Example for code-reviewer:\n"
                     f"  {{'code': 'def foo(): pass', 'language': 'python'}}"
            )]

        # Validate parameter types are dict/serializable
        for key, value in args.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                logger.warning(f"Non-serializable parameter {key} for skill {skill_name}")
                return [TextContent(
                    type="text",
                    text=f"❌ Invalid parameter type: {key} = {type(value).__name__}\n\n"
                         f"Parameter '{key}' has unsupported type: {type(value).__name__}\n\n"
                         f"Supported types:\n"
                         f"  - str (strings)\n"
                         f"  - int/float (numbers)\n"
                         f"  - bool (true/false)\n"
                         f"  - list (arrays)\n"
                         f"  - dict (objects)\n\n"
                         f"Example:\n"
                         f"  {{'code': 'code string', 'language': 'python', 'line_count': 42}}"
                )]

        # Execute skill
        try:
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

        except KeyError as e:
            logger.error(f"Missing parameter for skill {skill_name}: {str(e)}")
            return [TextContent(
                type="text",
                text=f"❌ Missing required parameter: {str(e)}\n\n"
                     f"Skill '{skill_name}' requires this parameter.\n\n"
                     f"Check the skill's parameter requirements:\n"
                     f"  - Most code-based skills need: code, language\n"
                     f"  - Some need additional parameters\n\n"
                     f"Example:\n"
                     f"  {{'code': 'source code', 'language': 'python', "
                     f"'other_param': 'value'}}"
            )]

        except ValueError as e:
            logger.error(f"Invalid parameter value for skill {skill_name}: {str(e)}")
            return [TextContent(
                type="text",
                text=f"❌ Invalid parameter value: {str(e)}\n\n"
                     f"One or more parameters have invalid values.\n\n"
                     f"Common issues:\n"
                     f"  - Empty string values\n"
                     f"  - Out-of-range values\n"
                     f"  - Invalid enum choices\n"
                     f"  - Incompatible parameter combinations\n\n"
                     f"Example of valid parameters:\n"
                     f"  {{'code': 'non-empty code', 'language': 'python'}}"
            )]

    except Exception as e:
        logger.error(f"Unexpected error in handle_skill_execution({skill_name}): {str(e)}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"❌ Unexpected error executing skill '{skill_name}': {str(e)}\n\n"
                 f"Possible causes:\n"
                 f"  1. Skill module not loaded\n"
                 f"  2. Orchestrator initialization failed\n"
                 f"  3. Skill executor error\n"
                 f"  4. Parameter validation failed\n\n"
                 f"Troubleshooting:\n"
                 f"  - Verify skill name is correct\n"
                 f"  - Check parameter types and values\n"
                 f"  - Review server logs for details\n"
                 f"  - Try a simpler skill first to verify system works"
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
