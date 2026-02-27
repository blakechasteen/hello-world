#!/usr/bin/env python3
"""
Voice-Enhanced Scratchpad API
==============================

FastAPI server with TTS feedback for conversational scratchpad.

New endpoints:
- POST /api/voice/speak - TTS synthesis
- POST /api/voice/listen - Wait for voice response
- GET /api/voice/pending - Get pending voice prompt
- WebSocket /ws/voice - Real-time voice interaction

Author: Claude Code
Date: November 2025
"""

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from pathlib import Path
import tempfile
import asyncio
import json

from HoloLoom.spinningWheel.voice_scratchpad import (
    VoiceScratchpad,
    VoicePrompt,
    VoiceResponse,
    VoiceInteractionManager
)

# Try to import Whisper and TTS
try:
    from HoloLoom.spinningWheel.whisper_spinner import WhisperSpinner
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from openai import OpenAI
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Voice Scratchpad API",
    description="Conversational scratchpad with TTS feedback",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
scratchpad = VoiceScratchpad(enable_voice=True)
voice_manager = VoiceInteractionManager()
whisper = WhisperSpinner(model_size="base") if WHISPER_AVAILABLE else None
openai_client = OpenAI() if TTS_AVAILABLE else None


# ============================================================================
# Request/Response Models
# ============================================================================

class SpeakRequest(BaseModel):
    text: str
    voice: str = "nova"  # nova, alloy, echo, fable, onyx, shimmer


class VoiceResponseModel(BaseModel):
    transcription: str
    understood: bool
    extracted_value: Any


# ============================================================================
# Voice Interaction Endpoints
# ============================================================================

@app.post("/api/voice/speak")
async def speak_text(request: SpeakRequest):
    """
    Synthesize speech from text using OpenAI TTS.

    Returns audio stream (MP3).
    """
    if not TTS_AVAILABLE or not openai_client:
        raise HTTPException(
            status_code=503,
            detail="TTS not available. Install openai package."
        )

    try:
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice=request.voice,
            input=request.text
        )

        # Stream audio
        def audio_stream():
            for chunk in response.iter_bytes():
                yield chunk

        return StreamingResponse(
            audio_stream(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@app.get("/api/voice/pending")
async def get_pending_prompt():
    """
    Get current pending voice prompt.

    Returns None if no prompt pending.
    """
    if voice_manager.pending_prompt:
        return {
            "has_prompt": True,
            "text": voice_manager.pending_prompt.text,
            "type": voice_manager.pending_prompt.prompt_type,
            "field_name": voice_manager.pending_prompt.field_name,
            "expected_response": voice_manager.pending_prompt.expected_response,
            "choices": voice_manager.pending_prompt.choices
        }
    else:
        return {"has_prompt": False}


@app.post("/api/voice/respond")
async def submit_voice_response(response: VoiceResponseModel):
    """
    Submit voice response to pending prompt.

    Used to answer TTS questions.
    """
    if not voice_manager.pending_prompt:
        raise HTTPException(status_code=400, detail="No pending prompt")

    voice_response = VoiceResponse(
        transcription=response.transcription,
        understood=response.understood,
        extracted_value=response.extracted_value,
        confidence=0.9  # TODO: Calculate from Whisper
    )

    # Queue response
    await voice_manager.response_queue.put(voice_response)

    return {"success": True}


# ============================================================================
# WebSocket for Real-Time Voice Interaction
# ============================================================================

@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """
    WebSocket for real-time voice interaction.

    Client sends audio chunks, receives TTS prompts and confirmations.
    """
    await websocket.accept()

    try:
        while True:
            # Wait for message from client
            message = await websocket.receive_json()

            msg_type = message.get("type")

            if msg_type == "transcription":
                # Client sent transcription
                text = message.get("text")
                await scratchpad.update_from_transcription(text)

                # Send updated state
                state = scratchpad.get_current_state()
                await websocket.send_json({
                    "type": "state_update",
                    "state": state
                })

            elif msg_type == "voice_response":
                # Client answered voice prompt
                response = VoiceResponse(
                    transcription=message.get("transcription", ""),
                    understood=message.get("understood", False),
                    extracted_value=message.get("value"),
                    confidence=message.get("confidence", 0.0)
                )
                await voice_manager.response_queue.put(response)

            elif msg_type == "ping":
                # Keep-alive
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")


# ============================================================================
# Enhanced Scratchpad Endpoints (Voice-Aware)
# ============================================================================

@app.post("/api/scratchpad/start")
async def start_recording_voice(domain: str):
    """Start recording with voice guidance"""
    await scratchpad.start_recording(domain)

    # Voice greeting
    await scratchpad._speak(
        f"Starting {domain.replace('_', ' ')} entry. Please speak naturally.",
        prompt_type='guidance'
    )

    return {
        "success": True,
        "domain": scratchpad.active_domain,
        "state": scratchpad.get_current_state()
    }


@app.post("/api/scratchpad/transcribe_voice")
async def transcribe_with_voice(audio: UploadFile = File(...)):
    """
    Transcribe audio with voice feedback.

    Adds conversational prompts for clarity and confirmation.
    """
    if not WHISPER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Whisper not available"
        )

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
        content = await audio.read()
        temp_file.write(content)
        temp_path = Path(temp_file.name)

    try:
        # Transcribe
        result = await whisper.spin(temp_path)
        transcription = result.shards[0].text if result.success else ""

        # Update with voice feedback
        await scratchpad.update_from_transcription(transcription)

        # Get state (includes any voice prompts)
        state = scratchpad.get_current_state()

        # Check for pending voice prompt
        pending_prompt = None
        if voice_manager.pending_prompt:
            pending_prompt = {
                "text": voice_manager.pending_prompt.text,
                "type": voice_manager.pending_prompt.prompt_type,
                "expected_response": voice_manager.pending_prompt.expected_response,
                "choices": voice_manager.pending_prompt.choices
            }

        return {
            "success": True,
            "transcription": transcription,
            "state": state,
            "voice_prompt": pending_prompt
        }

    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/api/scratchpad/finalize_voice")
async def finalize_with_voice(data: Dict[str, Any], skip_readback: bool = False):
    """
    Finalize entry with voice read-back and confirmation.

    TTS reads back entry and asks "Is this correct?"
    """
    # Update data
    scratchpad.current_data = data.get('data', {})

    try:
        # Finalize with voice confirmation
        shard = await scratchpad.finalize(skip_readback=skip_readback)

        return {
            "success": True,
            "shard_id": shard.id,
            "message": "Entry saved successfully!"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Health & Info
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Voice Scratchpad API",
        "version": "2.0.0",
        "whisper_available": WHISPER_AVAILABLE,
        "tts_available": TTS_AVAILABLE,
        "features": [
            "TTS feedback",
            "Voice confirmations",
            "Clarity requests",
            "Read-back verification",
            "Safety confirmations"
        ]
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "whisper": WHISPER_AVAILABLE,
        "tts": TTS_AVAILABLE,
        "active_domain": scratchpad.active_domain,
        "pending_prompt": voice_manager.pending_prompt is not None
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Register voice callbacks
    def on_voice_prompt(prompt: VoicePrompt):
        """Handle voice prompt (TTS)"""
        print(f"[TTS] {prompt.text}")
        voice_manager.pending_prompt = prompt

    def on_voice_response(response: VoiceResponse):
        """Handle voice response"""
        print(f"[Voice Response] {response.transcription}")

    scratchpad.register_voice_callbacks(on_voice_prompt, on_voice_response)

    print("[VoiceScratchpadAPI] Starting server on http://localhost:8003")
    print(f"[VoiceScratchpadAPI] Whisper available: {WHISPER_AVAILABLE}")
    print(f"[VoiceScratchpadAPI] TTS available: {TTS_AVAILABLE}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )
