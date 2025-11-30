import asyncio
import logging
import sys
import json
from typing import Dict, Any, Optional
from mcp.server.fastmcp import FastMCP, Context

# Redirect stdout to stderr to avoid breaking MCP protocol
# (Since EggrollIntegration uses print statements)
class StderrRedirect:
    def write(self, message):
        sys.stderr.write(message)
    def flush(self):
        sys.stderr.flush()

sys.stdout = StderrRedirect()

from HoloLoom.eggroll.integration import EggrollIntegration

# Initialize FastMCP Server
mcp = FastMCP("EGGROLL Service")

# Service State
class EggrollService:
    def __init__(self):
        self.integration = EggrollIntegration(num_workers=5)
        self.is_running = False
        self.task = None
        self.status = {
            "state": "IDLE",
            "epoch": 0,
            "avg_reward": 0.0,
            "current_pattern": None,
            "best_adapter_version": 0
        }

    async def run_loop(self, epochs: int, pattern: Optional[str] = None):
        self.is_running = True
        self.status["state"] = "DREAMING"
        
        try:
            # We need to modify the integration loop to be controllable or just run it
            # For this MVP, we'll run the existing loop but update status
            # We'll monkey-patch the integration's print or hook into it if possible
            # But for now, let's just run it.
            
            # To support "pattern" override, we might need to inject it
            # The current integration selects it via MCTS.
            # We can force it by mocking the shuttle if needed, 
            # but let's trust the MCTS for now unless specified.
            
            await self.integration.run_evolution_loop(num_epochs=epochs)
            
            self.status["state"] = "COMPLETED"
            self.status["best_adapter_version"] = self.integration.mirror_core.version
            
        except Exception as e:
            self.status["state"] = "ERROR"
            self.status["error"] = str(e)
            logging.error(f"Dream loop failed: {e}")
        finally:
            self.is_running = False

service = EggrollService()

@mcp.tool()
async def start_dreaming(epochs: int = 3, workers: int = 5, pattern: str = "auto") -> str:
    """
    Start the EGGROLL evolutionary training loop.
    
    Args:
        epochs: Number of evolutionary epochs to run.
        workers: Number of parallel workers (simulated).
        pattern: Pattern card strategy ('auto' uses MCTS, or specific name).
    """
    if service.is_running:
        return "EGGROLL is already dreaming."
    
    service.integration.num_workers = workers
    
    # Start background task
    # Note: In FastMCP, we can just fire and forget if we want async,
    # but we need to keep a reference.
    service.task = asyncio.create_task(service.run_loop(epochs, pattern if pattern != "auto" else None))
    
    return f"Started EGGROLL Dream Cycle (Epochs={epochs}, Workers={workers})"

@mcp.tool()
async def stop_dreaming() -> str:
    """Stop the current dream cycle."""
    if not service.is_running or not service.task:
        return "EGGROLL is not dreaming."
    
    service.task.cancel()
    service.is_running = False
    service.status["state"] = "STOPPED"
    return "Dream cycle stopped."

@mcp.resource("eggroll://status")
def get_status() -> str:
    """Get the current status of the EGGROLL service."""
    # Update status from integration state if running
    if service.is_running:
        # This is a bit hacky since integration doesn't expose live state easily
        # In a real impl, integration would update a shared state object
        pass
        
    return json.dumps(service.status, indent=2)

@mcp.resource("eggroll://lineage")
def get_lineage() -> str:
    """Get the evolutionary lineage graph."""
    # Convert networkx graph to JSON-friendly format
    graph_data = {
        "nodes": list(service.integration.yarn.graph.nodes(data=True)),
        "edges": list(service.integration.yarn.graph.edges())
    }
    return json.dumps(graph_data, indent=2)

if __name__ == "__main__":
    mcp.run()
