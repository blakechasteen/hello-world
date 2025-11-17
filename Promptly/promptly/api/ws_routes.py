"""
WebSocket Routes
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import logging

from .websocket import (
    manager,
    StreamingEvaluationHandler,
    StreamingChainHandler
)

router = APIRouter()
logger = logging.getLogger("promptly.ws_routes")


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    api_key: str = Query(..., description="API key for authentication")
):
    """
    Main WebSocket endpoint

    Connect to receive real-time updates for evaluations, chains, and branch changes.
    """
    # Verify API key (simplified for demo)
    # In production, use proper authentication
    from .middleware.auth import verify_api_key

    if not verify_api_key(api_key):
        await websocket.close(code=1008)
        return

    # Accept connection
    await manager.connect(websocket, client_id, {"api_key": api_key})

    try:
        # Send welcome message
        await manager.send_personal_message(
            {
                "type": "connected",
                "client_id": client_id,
                "message": "Connected to Promptly WebSocket"
            },
            websocket
        )

        # Listen for messages
        while True:
            data = await websocket.receive_json()

            # Handle different message types
            message_type = data.get("type")

            if message_type == "ping":
                await manager.send_personal_message({"type": "pong"}, websocket)

            elif message_type == "subscribe":
                # Handle subscription to specific events
                await manager.send_personal_message(
                    {
                        "type": "subscribed",
                        "events": data.get("events", [])
                    },
                    websocket
                )

            else:
                # Echo unknown messages
                await manager.send_personal_message(
                    {
                        "type": "echo",
                        "original": data
                    },
                    websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"Client {client_id} disconnected")

    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(websocket)


@router.websocket("/ws/evaluations/{evaluation_id}")
async def evaluation_stream(
    websocket: WebSocket,
    evaluation_id: str,
    api_key: str = Query(..., description="API key for authentication")
):
    """
    Stream evaluation results in real-time

    Connect to receive live updates as evaluation progresses.
    """
    from .middleware.auth import verify_api_key

    if not verify_api_key(api_key):
        await websocket.close(code=1008)
        return

    await manager.connect(websocket, f"eval_{evaluation_id}")

    try:
        handler = StreamingEvaluationHandler(websocket)

        # Send initial message
        await handler.send_progress("started", 0.0, {"evaluation_id": evaluation_id})

        # Listen for control messages
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "cancel":
                await handler.send_error("Evaluation cancelled by user")
                break

    except WebSocketDisconnect:
        manager.disconnect(websocket)

    except Exception as e:
        logger.error(f"Evaluation stream error: {e}")
        manager.disconnect(websocket)


@router.websocket("/ws/chains/{execution_id}")
async def chain_stream(
    websocket: WebSocket,
    execution_id: str,
    api_key: str = Query(..., description="API key for authentication")
):
    """
    Stream chain execution in real-time

    Connect to receive live updates as chain executes.
    """
    from .middleware.auth import verify_api_key

    if not verify_api_key(api_key):
        await websocket.close(code=1008)
        return

    await manager.connect(websocket, f"chain_{execution_id}")

    try:
        handler = StreamingChainHandler(websocket)

        # Listen for control messages
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "cancel":
                await handler.send_error("Chain execution cancelled by user")
                break

    except WebSocketDisconnect:
        manager.disconnect(websocket)

    except Exception as e:
        logger.error(f"Chain stream error: {e}")
        manager.disconnect(websocket)


@router.get("/ws/stats")
async def get_websocket_stats():
    """
    Get WebSocket connection statistics

    Returns current connection stats.
    """
    return manager.get_stats()
