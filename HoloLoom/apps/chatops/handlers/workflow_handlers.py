"""
Workflow Handlers for Matrix ChatOps
=====================================

Matrix command handlers for HoloLoom Visual Workflow Builder integration.

Commands:
- !workflow list - List saved workflow versions
- !workflow run <name/JSON> - Execute a workflow
- !workflow status - Show executor status and active workflows
- !workflow validate <JSON> - Validate a workflow without executing
- !workflow generate <description> - Generate workflow from natural language
- !workflow agents - List available agent types
- !workflow optimize <workflow_id> - Get optimization suggestions
- !workflow help - Show help

Usage:
    from HoloLoom.apps.chatops.handlers.workflow_handlers import register_workflow_handlers

    # In run_chatops.py:
    register_workflow_handlers(registry)

Created: 2025-12-11
"""

import json
import logging
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass

try:
    from nio import MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False

# HTTP client for calling workflow executor API
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Workflow generator for NLP synthesis
try:
    from HoloLoom.apps.workflow_builder.workflow_generator import WorkflowGenerator
    GENERATOR_AVAILABLE = True
except ImportError:
    GENERATOR_AVAILABLE = False
    WorkflowGenerator = None

# Handler registry
try:
    from HoloLoom.apps.chatops.handlers.handler_registry import (
        HandlerRegistry, HandlerCategory, chatops_handler
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Default workflow executor URL (can be overridden)
WORKFLOW_EXECUTOR_URL = "http://localhost:8001"

_executor_url: str = WORKFLOW_EXECUTOR_URL
_generator: Optional['WorkflowGenerator'] = None


def get_executor_url() -> str:
    """Get the workflow executor URL."""
    return _executor_url


def set_executor_url(url: str) -> None:
    """Set the workflow executor URL."""
    global _executor_url
    _executor_url = url


def get_generator() -> Optional['WorkflowGenerator']:
    """Get the WorkflowGenerator instance."""
    global _generator
    if _generator is None and GENERATOR_AVAILABLE:
        _generator = WorkflowGenerator()
    return _generator


def set_generator(generator: 'WorkflowGenerator') -> None:
    """Set the WorkflowGenerator instance."""
    global _generator
    _generator = generator


# ============================================================================
# HTTP Client Helpers
# ============================================================================

async def _call_executor_api(
    method: str,
    endpoint: str,
    data: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Call the workflow executor REST API.

    Args:
        method: HTTP method (GET, POST)
        endpoint: API endpoint (e.g., /api/agents)
        data: Optional JSON payload for POST requests
        timeout: Request timeout in seconds

    Returns:
        JSON response from the API

    Raises:
        Exception: If API call fails
    """
    if not AIOHTTP_AVAILABLE:
        raise ImportError("aiohttp not available. Install with: pip install aiohttp")

    url = f"{get_executor_url()}{endpoint}"

    async with aiohttp.ClientSession() as session:
        try:
            if method.upper() == "GET":
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise Exception(f"API error {resp.status}: {text}")
                    return await resp.json()
            elif method.upper() == "POST":
                async with session.post(
                    url,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise Exception(f"API error {resp.status}: {text}")
                    return await resp.json()
            else:
                raise ValueError(f"Unsupported method: {method}")
        except aiohttp.ClientError as e:
            raise Exception(f"Connection error: {e}")


# ============================================================================
# Command Handlers
# ============================================================================

async def handle_workflow_list(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !workflow list command - List saved workflow versions.

    Args:
        room: Matrix room
        event: Message event
        args: Optional branch name

    Returns:
        List of saved workflows
    """
    branch = args.strip() if args.strip() else "main"

    try:
        result = await _call_executor_api("GET", f"/api/workflow/versions?branch={branch}")

        if not result or not result.get('versions'):
            return f"📋 **Workflow Versions** (branch: {branch})\n\nNo saved workflows found."

        response = f"📋 **Workflow Versions** (branch: {branch})\n\n"

        versions = result.get('versions', [])
        for v in versions[:10]:  # Limit to 10
            version_num = v.get('version', '?')
            name = v.get('name', 'Unnamed')
            message = v.get('message', 'No message')
            timestamp = v.get('timestamp', '')[:19] if v.get('timestamp') else ''

            response += f"**v{version_num}**: {name}\n"
            response += f"  _{message}_\n"
            if timestamp:
                response += f"  📅 {timestamp}\n"
            response += "\n"

        if len(versions) > 10:
            response += f"_...and {len(versions) - 10} more_\n"

        return response

    except ImportError as e:
        return f"❌ {str(e)}"
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        return f"❌ Failed to list workflows: {str(e)}"


async def handle_workflow_run(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !workflow run command - Execute a workflow.

    Args:
        room: Matrix room
        event: Message event
        args: Workflow JSON or version reference

    Returns:
        Execution result
    """
    if not args.strip():
        return "❌ Usage: !workflow run <workflow_json>\nExample: !workflow run {\"version\":\"1.0\",\"name\":\"test\",\"nodes\":[],\"connections\":[]}"

    try:
        # Try to parse as JSON workflow
        workflow_data = json.loads(args.strip())

        # Build execution request
        request_data = {
            "workflow": workflow_data,
            "input_data": workflow_data.get("input_data", {})
        }

        result = await _call_executor_api("POST", "/api/workflow/execute", request_data, timeout=60.0)

        # Format response
        response = "🚀 **Workflow Executed**\n\n"

        if result.get('success', True):
            response += f"**Status**: ✅ Success\n"
        else:
            response += f"**Status**: ⚠️ Partial\n"

        if result.get('outputs'):
            response += f"\n**Outputs**:\n"
            for key, value in list(result.get('outputs', {}).items())[:5]:
                val_str = str(value)[:200]
                response += f"  • {key}: {val_str}\n"

        if result.get('execution_time'):
            response += f"\n⏱️ Duration: {result['execution_time']:.2f}s\n"

        if result.get('nodes_executed'):
            response += f"🔗 Nodes executed: {result['nodes_executed']}\n"

        return response

    except json.JSONDecodeError:
        return "❌ Invalid JSON. Provide a valid workflow JSON.\nExample: !workflow run {\"version\":\"1.0\",\"name\":\"test\",\"nodes\":[],\"connections\":[]}"
    except ImportError as e:
        return f"❌ {str(e)}"
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        return f"❌ Workflow execution failed: {str(e)}"


async def handle_workflow_status(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !workflow status command - Show executor status.

    Args:
        room: Matrix room
        event: Message event
        args: Not used

    Returns:
        Executor health status
    """
    try:
        result = await _call_executor_api("GET", "/health")

        response = "📊 **Workflow Executor Status**\n\n"

        status = result.get('status', 'unknown')
        if status == 'healthy':
            response += f"**Status**: ✅ Healthy\n"
        else:
            response += f"**Status**: ⚠️ {status}\n"

        response += f"**Active Workflows**: {result.get('active_workflows', 0)}\n"
        response += f"**WebSocket Connections**: {result.get('ws_connections', 0)}\n"
        response += f"\n**Executor URL**: {get_executor_url()}\n"

        return response

    except ImportError as e:
        return f"❌ {str(e)}"
    except Exception as e:
        logger.error(f"Error checking executor status: {e}")
        return f"⚠️ **Workflow Executor Offline**\n\nCannot reach executor at {get_executor_url()}\n\nStart with: `python HoloLoom/web_dashboard/workflow_executor.py`\n\nError: {str(e)}"


async def handle_workflow_validate(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !workflow validate command - Validate a workflow.

    Args:
        room: Matrix room
        event: Message event
        args: Workflow JSON

    Returns:
        Validation result
    """
    if not args.strip():
        return "❌ Usage: !workflow validate <workflow_json>"

    try:
        workflow_data = json.loads(args.strip())

        result = await _call_executor_api("POST", "/api/workflow/validate", workflow_data)

        if result.get('valid'):
            response = "✅ **Workflow Valid**\n\n"
            response += f"**Nodes**: {result.get('nodes', 0)}\n"
            response += f"**Connections**: {result.get('connections', 0)}\n"

            start_nodes = result.get('start_nodes', [])
            if start_nodes:
                response += f"**Start Nodes**: {', '.join(start_nodes)}\n"

            return response
        else:
            return f"❌ **Workflow Invalid**\n\nError: {result.get('error', 'Unknown error')}"

    except json.JSONDecodeError:
        return "❌ Invalid JSON. Provide a valid workflow JSON."
    except ImportError as e:
        return f"❌ {str(e)}"
    except Exception as e:
        logger.error(f"Error validating workflow: {e}")
        return f"❌ Validation failed: {str(e)}"


async def handle_workflow_generate(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !workflow generate command - Generate workflow from NLP.

    Args:
        room: Matrix room
        event: Message event
        args: Natural language description

    Returns:
        Generated workflow JSON
    """
    if not args.strip():
        return "❌ Usage: !workflow generate <description>\nExample: !workflow generate research Thompson Sampling with multiple perspectives"

    if not GENERATOR_AVAILABLE:
        return "❌ WorkflowGenerator not available. Check HoloLoom installation."

    description = args.strip()

    try:
        generator = get_generator()
        if not generator:
            return "❌ Failed to initialize WorkflowGenerator"

        workflow = generator.generate(description)

        if not workflow:
            return f"⚠️ **Could not generate workflow**\n\nDescription: {description}\n\nTry being more specific about what you want to do."

        response = "🎨 **Generated Workflow**\n\n"
        response += f"**Name**: {workflow.get('name', 'Generated Workflow')}\n"
        response += f"**Nodes**: {len(workflow.get('nodes', []))}\n"
        response += f"**Connections**: {len(workflow.get('connections', []))}\n\n"

        # List nodes
        nodes = workflow.get('nodes', [])
        if nodes:
            response += "**Agents**:\n"
            for node in nodes[:8]:
                agent_type = node.get('agentType', 'unknown')
                node_id = node.get('id', '?')
                response += f"  • [{node_id}] {agent_type}\n"
            if len(nodes) > 8:
                response += f"  _...and {len(nodes) - 8} more_\n"

        response += "\n```json\n"
        response += json.dumps(workflow, indent=2)[:1500]  # Limit JSON size
        if len(json.dumps(workflow)) > 1500:
            response += "\n... (truncated)"
        response += "\n```"

        return response

    except Exception as e:
        logger.error(f"Error generating workflow: {e}")
        return f"❌ Generation failed: {str(e)}"


async def handle_workflow_agents(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !workflow agents command - List available agent types.

    Args:
        room: Matrix room
        event: Message event
        args: Not used

    Returns:
        List of available agents
    """
    try:
        result = await _call_executor_api("GET", "/api/agents")

        response = "🤖 **Available Workflow Agents**\n\n"

        categories = {
            'query': '🔍 Query',
            'process': '⚙️ Process',
            'memory': '💾 Memory',
            'decision': '🎯 Decision',
            'output': '📤 Output',
            'control': '🔄 Control'
        }

        for category, agents in result.items():
            cat_name = categories.get(category, category.title())
            response += f"**{cat_name}**:\n"
            for agent in agents:
                response += f"  • {agent}\n"
            response += "\n"

        response += "_Use these agents in the Visual Workflow Builder_"

        return response

    except ImportError as e:
        return f"❌ {str(e)}"
    except Exception as e:
        logger.error(f"Error fetching agents: {e}")
        # Return hardcoded list as fallback
        return """🤖 **Available Workflow Agents**

**🔍 Query**:
  • hololoom - Full weaving cycle
  • search - Memory search
  • multiquery - Break into sub-questions

**⚙️ Process**:
  • embedder - Multi-scale embeddings
  • synthesizer - Entity/motif extraction
  • refiner - Quality refinement

**💾 Memory**:
  • store - Persist to graph+vector
  • retrieve - Retrieve context
  • fusion - Multi-hop traversal

**🎯 Decision**:
  • thompson - Bayesian exploration
  • convergence - Decision collapse
  • safety - Risk gating

**📤 Output**:
  • response - Generate response
  • format - JSON/Markdown/HTML

**🔄 Control**:
  • conditional - If/else logic
  • loop - Repeat until condition
  • parallel - Concurrent execution

_Use these agents in the Visual Workflow Builder_"""


async def handle_workflow_optimize(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !workflow optimize command - Get optimization suggestions.

    Args:
        room: Matrix room
        event: Message event
        args: Workflow ID

    Returns:
        Optimization suggestions
    """
    if not args.strip():
        return "❌ Usage: !workflow optimize <workflow_id>"

    workflow_id = args.strip()

    try:
        result = await _call_executor_api("POST", f"/api/workflow/{workflow_id}/optimize", {})

        if not result.get('suggestions'):
            return f"✅ **No optimizations suggested** for workflow `{workflow_id}`\n\nWorkflow appears to be well-optimized."

        response = f"💡 **Optimization Suggestions** for `{workflow_id}`\n\n"

        suggestions = result.get('suggestions', [])
        for i, s in enumerate(suggestions[:5], 1):
            sug_type = s.get('suggestion_type', 'unknown')
            desc = s.get('description', 'No description')
            improvement = s.get('expected_improvement', 0)
            confidence = s.get('confidence', 0)

            type_icons = {
                'parallelize': '⚡',
                'cache': '💾',
                'reorder': '🔀',
                'substitute': '🔄'
            }
            icon = type_icons.get(sug_type, '💡')

            response += f"**{i}. {icon} {sug_type.title()}**\n"
            response += f"   {desc}\n"
            response += f"   📈 Expected improvement: {improvement:.1f}%\n"
            response += f"   🎯 Confidence: {confidence:.1%}\n\n"

        if len(suggestions) > 5:
            response += f"_...and {len(suggestions) - 5} more suggestions_\n"

        return response

    except ImportError as e:
        return f"❌ {str(e)}"
    except Exception as e:
        logger.error(f"Error getting optimizations: {e}")
        return f"❌ Optimization failed: {str(e)}"


async def handle_workflow_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !workflow help command.

    Returns:
        Help message
    """
    return """🔧 **Workflow Commands**

**Discovery:**
• `!workflow agents` - List available agent types
• `!workflow list [branch]` - List saved workflows
• `!workflow status` - Show executor status

**Execution:**
• `!workflow run <json>` - Execute a workflow
• `!workflow validate <json>` - Validate without executing

**Generation:**
• `!workflow generate <description>` - Generate from natural language
• `!workflow optimize <workflow_id>` - Get optimization suggestions

**Examples:**
```
!workflow agents
!workflow list main
!workflow status
!workflow generate research Thompson Sampling with multiple perspectives
!workflow optimize wf-12345
```

**Workflow Builder:**
Open `HoloLoom/web_dashboard/workflow_builder.html` in a browser for the visual editor.

**Start Executor:**
```bash
cd HoloLoom/web_dashboard
python workflow_executor.py
```

**Agent Types:**
- **Query**: hololoom, search, multiquery
- **Process**: embedder, synthesizer, refiner
- **Memory**: store, retrieve, fusion
- **Decision**: thompson, convergence, safety
- **Output**: response, format
- **Control**: conditional, loop, parallel
"""


# ============================================================================
# Registration
# ============================================================================

def register_workflow_handlers(
    registry=None
) -> 'WorkflowHandlers':
    """
    Register all Workflow command handlers.

    Args:
        registry: Optional HandlerRegistry instance

    Returns:
        WorkflowHandlers instance
    """
    logger.info("Registering Workflow command handlers")

    handlers = WorkflowHandlers()

    if registry and REGISTRY_AVAILABLE:
        # Register with decorator-based registry
        registry.register_handler(
            command="workflow list",
            handler=handlers.handle_list,
            description="List saved workflow versions",
            usage="!workflow list [branch]",
            category=HandlerCategory.SYSTEM
        )
        registry.register_handler(
            command="workflow run",
            handler=handlers.handle_run,
            description="Execute a workflow",
            usage="!workflow run <workflow_json>",
            category=HandlerCategory.QUERY
        )
        registry.register_handler(
            command="workflow status",
            handler=handlers.handle_status,
            description="Show executor status",
            usage="!workflow status",
            category=HandlerCategory.SYSTEM
        )
        registry.register_handler(
            command="workflow validate",
            handler=handlers.handle_validate,
            description="Validate a workflow",
            usage="!workflow validate <workflow_json>",
            category=HandlerCategory.SYSTEM
        )
        registry.register_handler(
            command="workflow generate",
            handler=handlers.handle_generate,
            description="Generate workflow from natural language",
            usage="!workflow generate <description>",
            category=HandlerCategory.QUERY
        )
        registry.register_handler(
            command="workflow agents",
            handler=handlers.handle_agents,
            description="List available agent types",
            usage="!workflow agents",
            category=HandlerCategory.SYSTEM
        )
        registry.register_handler(
            command="workflow optimize",
            handler=handlers.handle_optimize,
            description="Get optimization suggestions",
            usage="!workflow optimize <workflow_id>",
            category=HandlerCategory.SYSTEM
        )
        registry.register_handler(
            command="workflow help",
            handler=handlers.handle_help,
            description="Show workflow help",
            usage="!workflow help",
            category=HandlerCategory.SYSTEM,
            hidden=True
        )

    logger.info(f"Registered 8 workflow commands")
    return handlers


# ============================================================================
# Decorator-Based Handler Class
# ============================================================================

class WorkflowHandlers:
    """
    Container for Workflow handlers.

    Usage:
        from HoloLoom.apps.chatops.handlers.workflow_handlers import WorkflowHandlers

        handlers = WorkflowHandlers()
        registry.register_instance(handlers)
    """

    def __init__(self, executor_url: str = None):
        """
        Initialize WorkflowHandlers.

        Args:
            executor_url: Optional custom executor URL
        """
        if executor_url:
            set_executor_url(executor_url)

    async def handle_list(self, room, event, args: str) -> str:
        """Handle !workflow list command."""
        return await handle_workflow_list(room, event, args)

    async def handle_run(self, room, event, args: str) -> str:
        """Handle !workflow run command."""
        return await handle_workflow_run(room, event, args)

    async def handle_status(self, room, event, args: str) -> str:
        """Handle !workflow status command."""
        return await handle_workflow_status(room, event, args)

    async def handle_validate(self, room, event, args: str) -> str:
        """Handle !workflow validate command."""
        return await handle_workflow_validate(room, event, args)

    async def handle_generate(self, room, event, args: str) -> str:
        """Handle !workflow generate command."""
        return await handle_workflow_generate(room, event, args)

    async def handle_agents(self, room, event, args: str) -> str:
        """Handle !workflow agents command."""
        return await handle_workflow_agents(room, event, args)

    async def handle_optimize(self, room, event, args: str) -> str:
        """Handle !workflow optimize command."""
        return await handle_workflow_optimize(room, event, args)

    async def handle_help(self, room, event, args: str) -> str:
        """Handle !workflow help command."""
        return await handle_workflow_help(room, event, args)
