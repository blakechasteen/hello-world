"""
Quick start script for HoloLoom Agentic API Server
Uses port 8001 to avoid conflicts with other services.
"""

import uvicorn
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting HoloLoom Agentic API on port 8001...")
    logger.info("Features:")
    logger.info("  ✓ Real LLM calls (Ollama/Anthropic)")
    logger.info("  ✓ Persistent memory (Neo4j + Qdrant)")
    logger.info("  ✓ Agentic reasoning (4 modes)")
    logger.info("  ✓ WebSocket streaming for UI")
    logger.info("")
    logger.info("API available at: http://localhost:8001")
    logger.info("Docs available at: http://localhost:8001/docs")
    logger.info("")

    uvicorn.run(
        "HoloLoom.apps.server.agentic_api_integrated:app",
        host="0.0.0.0",
        port=8001,  # Using 8001 instead of 8000
        reload=True,
        log_level="info"
    )
