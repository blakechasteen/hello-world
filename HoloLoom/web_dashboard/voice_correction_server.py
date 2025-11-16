"""
Voice Correction WebSocket Server
==================================

WebSocket server for real-time voice correction processing.

Integrates with:
- Voice Correction System
- Self-Tuning Engine
- Schema-Aware Spinners

Usage:
    python voice_correction_server.py

Then open: voice_correction_ui.html

Author: Claude Code
Date: January 2025
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Set
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# HoloLoom imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HoloLoom.spinningWheel.voice_correction import (
    VoiceCorrector,
    SelfTuningEngine,
    IntentParser
)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# FastAPI app
app = FastAPI(title="Voice Correction Server")


# Global state
voice_corrector: VoiceCorrector = None
tuning_engine: SelfTuningEngine = None
active_connections: Set[WebSocket] = set()
transformation_cache: Dict[str, Dict] = {}


@app.on_event("startup")
async def startup():
    """Initialize voice correction system."""
    global voice_corrector, tuning_engine

    logger.info("Initializing voice correction system...")

    # Create tuning engine with persistence
    tuning_engine = SelfTuningEngine(
        storage_path=Path("./learned_patterns.json"),
        min_confidence=0.7
    )

    # Create intent parser
    intent_parser = IntentParser(use_llm=False)

    # Create voice corrector
    voice_corrector = VoiceCorrector(
        tuning_engine=tuning_engine,
        intent_parser=intent_parser
    )

    logger.info(f"Voice correction system ready. Loaded {len(tuning_engine.patterns)} patterns.")


@app.get("/")
async def get():
    """Serve the voice UI."""
    with open(Path(__file__).parent / "voice_correction_ui.html") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice commands."""
    await websocket.accept()
    active_connections.add(websocket)

    logger.info(f"Client connected. Total connections: {len(active_connections)}")

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)

            logger.info(f"Received: {message['type']}")

            # Process message
            if message['type'] == 'voice_command':
                await handle_voice_command(websocket, message)

            elif message['type'] == 'apply_correction':
                await handle_apply_correction(websocket, message)

            elif message['type'] == 'get_stats':
                await handle_get_stats(websocket)

            elif message['type'] == 'get_patterns':
                await handle_get_patterns(websocket)

    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(active_connections)}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        active_connections.discard(websocket)


async def handle_voice_command(websocket: WebSocket, message: Dict):
    """Handle voice command processing."""
    command = message['command']
    transformation_id = message.get('transformation_id')

    logger.info(f"Processing voice command: {command}")

    # Parse intent
    intent = await voice_corrector.intent_parser.parse(command)

    # Send intent back
    await websocket.send_json({
        'type': 'intent',
        'intent': {
            'type': intent.type.value,
            'field_name': intent.field_name,
            'field_value': intent.field_value,
            'source_field': intent.source_field,
            'target_field': intent.target_field,
            'category_name': intent.category_name,
            'confidence': intent.confidence
        }
    })

    # If we have transformation data, apply correction
    if transformation_id and transformation_id in transformation_cache:
        original_data = transformation_cache[transformation_id]

        correction = await voice_corrector.apply_correction(
            transformation_id=transformation_id,
            voice_command=command,
            original_data=original_data,
            schema_name='expenses'  # TODO: get from context
        )

        # Send pattern learned (if any)
        if correction.pattern_learned:
            pattern = correction.pattern_learned

            await websocket.send_json({
                'type': 'pattern_learned',
                'pattern': {
                    'source_pattern': pattern.source_pattern,
                    'target_action': pattern.target_action,
                    'confidence': pattern.confidence,
                    'pattern_type': pattern.pattern_type.value
                },
                'stats': get_stats()
            })

        # Get suggestions for corrected data
        suggestions = voice_corrector.get_suggestions(
            correction.corrected_data,
            schema_name='expenses'
        )

        await websocket.send_json({
            'type': 'suggestions',
            'suggestions': suggestions
        })


async def handle_apply_correction(websocket: WebSocket, message: Dict):
    """Handle applying a correction."""
    command = message['command']
    transformation_id = message.get('transformation_id')

    logger.info(f"Applying correction: {command}")

    if transformation_id and transformation_id in transformation_cache:
        # Apply correction
        await handle_voice_command(websocket, {
            'type': 'voice_command',
            'command': command,
            'transformation_id': transformation_id
        })

        # Broadcast to all clients
        await broadcast({
            'type': 'correction_applied',
            'transformation_id': transformation_id,
            'command': command
        })


async def handle_get_stats(websocket: WebSocket):
    """Send current statistics."""
    stats = get_stats()

    await websocket.send_json({
        'type': 'stats',
        'stats': stats
    })


async def handle_get_patterns(websocket: WebSocket):
    """Send all learned patterns."""
    patterns = [
        {
            'pattern_id': p.pattern_id,
            'pattern_type': p.pattern_type.value,
            'source_pattern': p.source_pattern,
            'target_action': p.target_action,
            'confidence': p.confidence,
            'usage_count': p.usage_count,
            'success_rate': p.success_rate
        }
        for p in tuning_engine.patterns.values()
    ]

    await websocket.send_json({
        'type': 'patterns',
        'patterns': patterns
    })


def get_stats() -> Dict:
    """Get current statistics."""
    total_corrections = len(voice_corrector.corrections)
    total_patterns = len(tuning_engine.patterns)

    # Calculate accuracy (patterns with >80% success rate)
    high_conf_patterns = [
        p for p in tuning_engine.patterns.values()
        if p.success_rate > 0.8
    ]

    accuracy = len(high_conf_patterns) / total_patterns if total_patterns > 0 else 0.0

    return {
        'corrections': total_corrections,
        'patterns': total_patterns,
        'accuracy': accuracy
    }


async def broadcast(message: Dict):
    """Broadcast message to all connected clients."""
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.error(f"Broadcast error: {e}")


@app.post("/api/transformation")
async def register_transformation(data: Dict):
    """Register a transformation for voice correction."""
    transformation_id = data['transformation_id']
    extracted_data = data['extracted_data']

    transformation_cache[transformation_id] = extracted_data

    logger.info(f"Registered transformation: {transformation_id}")

    return {'status': 'ok', 'transformation_id': transformation_id}


@app.get("/api/stats")
async def get_api_stats():
    """Get statistics via REST API."""
    return get_stats()


@app.get("/api/patterns")
async def get_api_patterns():
    """Get patterns via REST API."""
    patterns = [
        {
            'pattern_id': p.pattern_id,
            'pattern_type': p.pattern_type.value,
            'source_pattern': p.source_pattern,
            'target_action': p.target_action,
            'confidence': p.confidence,
            'usage_count': p.usage_count,
            'success_rate': p.success_rate
        }
        for p in tuning_engine.patterns.values()
    ]

    return {'patterns': patterns}


if __name__ == "__main__":
    print("=" * 70)
    print("VOICE CORRECTION SERVER")
    print("=" * 70)
    print("\nStarting WebSocket server...")
    print("  URL: http://localhost:8001")
    print("  WebSocket: ws://localhost:8001/ws/voice")
    print("\nOpen http://localhost:8001 in your browser")
    print("=" * 70)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
