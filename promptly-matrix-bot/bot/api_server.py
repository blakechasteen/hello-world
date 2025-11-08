#!/usr/bin/env python3
"""
FastAPI Server for Promptly Bot Testing

Simple REST API to test HoloLoom integration without Matrix/Synapse.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from bot.promptly_core import get_promptly_core

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Promptly Bot API",
    description="Test API for HoloLoom + DSPy integration",
    version="0.1.0"
)

# Initialize Promptly Core
core = get_promptly_core()


# Request models
class OptimizeRequest(BaseModel):
    task: str
    examples: List[Dict[str, Any]]
    inputs: Optional[List[str]] = None
    outputs: Optional[List[str]] = None


class WorkflowRequest(BaseModel):
    workflow_name: str
    input_data: str
    context: Optional[Dict[str, Any]] = None


# Routes
@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Promptly Bot API",
        "version": "0.1.0",
        "status": "running",
        "hololoom_available": core.initialized,
        "endpoints": {
            "health": "/health",
            "optimize": "/optimize (POST)",
            "workflow": "/workflow (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "hololoom_initialized": core.initialized,
        "mode": "stub" if not core.initialized else "production"
    }


@app.post("/optimize")
async def optimize_prompt(request: OptimizeRequest):
    """
    Optimize a prompt using DSPy + HoloLoom

    Example request:
    ```json
    {
      "task": "Answer customer support questions",
      "examples": [
        {"input": "How to reset password?", "output": "Click Settings > Reset Password"},
        {"input": "Where is my order?", "output": "Check Orders page"}
      ]
    }
    ```
    """
    try:
        logger.info(f"Optimizing prompt for task: {request.task}")

        result = await core.optimize_prompt(
            task=request.task,
            examples=request.examples,
            inputs=request.inputs,
            outputs=request.outputs
        )

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflow")
async def run_workflow(request: WorkflowRequest):
    """
    Run a workflow

    Example request:
    ```json
    {
      "workflow_name": "qa_basic",
      "input_data": "What is Thompson Sampling?"
    }
    ```
    """
    try:
        logger.info(f"Running workflow: {request.workflow_name}")

        result = await core.run_workflow(
            workflow_name=request.workflow_name,
            input_data=request.input_data,
            context=request.context
        )

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflows")
async def list_workflows():
    """List available workflows"""
    return {
        "workflows": [
            {
                "name": "qa_basic",
                "description": "Simple Q&A workflow",
                "inputs": ["question"],
                "outputs": ["answer"]
            }
        ]
    }
