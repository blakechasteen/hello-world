#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice & Multimodal API
======================

Endpoints for voice chat, speech-to-text, and multimodal ingestion.

Features:
- Speech-to-text transcription (Whisper)
- Conversational voice chat with session memory
- Audio file ingestion into HoloLoom memory
- Multimodal content processing

Created: 2025-01-20
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import base64
import tempfile
import os
import logging

router = APIRouter(prefix="/voice", tags=["voice"])
logger = logging.getLogger(__name__)

# Try to import Whisper for speech-to-text
try:
    import whisper
    WHISPER_MODEL = whisper.load_model("base")
    WHISPER_AVAILABLE = True
    logger.info("✓ Whisper model loaded successfully")
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("⚠ Whisper not available. Install: pip install openai-whisper")

# Import HoloLoom components
try:
    from hololoom.spinningWheel.modalities.audio import AudioSpinner
    from hololoom.unified_api import hololoom
    HOLOLOOM_AVAILABLE = True
except ImportError as e:
    HOLOLOOM_AVAILABLE = False
    logger.warning(f"⚠ HoloLoom components not available: {e}")

# Global HoloLoom instance
hololoom_instance: Optional[Any] = None

# Session storage (in production, use Redis)
voice_sessions: Dict[str, List[Dict]] = {}


async def get_hololoom():
    """Get or create HoloLoom instance."""
    global hololoom_instance
    if hololoom_instance is None and HOLOLOOM_AVAILABLE:
        hololoom_instance = await HoloLoom.create(
            pattern="fast",
            memory_backend="simple"
        )
    return hololoom_instance


# ============== 1. SPEECH-TO-TEXT ==============

class TranscribeResponse(BaseModel):
    text: str
    language: str
    confidence: float
    segments: List[Dict[str, Any]]


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Convert speech to text using Whisper.

    Supports: WAV, MP3, OGG, M4A
    Returns: Transcribed text with language detection
    """

    if not WHISPER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Whisper not available. Install: pip install openai-whisper"
        )

    try:
        # Read audio file
        audio_bytes = await audio.read()

        # Save temporarily (Whisper needs file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            # Transcribe
            logger.info(f"Transcribing audio file: {audio.filename}")
            result = WHISPER_MODEL.transcribe(tmp_path)

            return TranscribeResponse(
                text=result["text"],
                language=result["language"],
                confidence=0.95,  # Whisper doesn't provide confidence
                segments=result["segments"]
            )
        finally:
            # Cleanup
            os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


# ============== 2. CONVERSATIONAL VOICE CHAT ==============

class VoiceChatRequest(BaseModel):
    audio_base64: Optional[str] = None  # Base64 encoded audio
    text: Optional[str] = None          # Or direct text input
    session_id: str
    context: Optional[Dict[str, Any]] = None


class VoiceChatResponse(BaseModel):
    text_response: str
    suggestions: List[str]
    context_updated: Dict[str, Any]


@router.post("/chat", response_model=VoiceChatResponse)
async def voice_chat(request: VoiceChatRequest):
    """
    Conversational voice interface with session memory.

    Supports both audio and text input.
    Maintains conversation context across requests.
    """

    # Get or create session
    if request.session_id not in voice_sessions:
        voice_sessions[request.session_id] = []

    session = voice_sessions[request.session_id]

    try:
        # 1. Convert audio to text (if audio provided)
        user_text = request.text

        if request.audio_base64 and not user_text:
            if not WHISPER_AVAILABLE:
                raise HTTPException(
                    status_code=503,
                    detail="Audio transcription not available. Provide text instead."
                )

            # Decode base64 audio
            audio_bytes = base64.b64decode(request.audio_base64)

            # Transcribe
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                result = WHISPER_MODEL.transcribe(tmp_path)
                user_text = result["text"]
            finally:
                os.unlink(tmp_path)

        if not user_text:
            raise HTTPException(
                status_code=400,
                detail="Either 'text' or 'audio_base64' must be provided"
            )

        # 2. Add to session history
        session.append({
            "role": "user",
            "text": user_text,
            "context": request.context
        })

        # 3. Query HoloLoom with conversation context
        if HOLOLOOM_AVAILABLE:
            loom = await get_hololoom()

            # Build context from session (last 5 messages)
            conversation_history = "\n".join([
                f"{msg['role']}: {msg['text']}"
                for msg in session[-5:]
            ])

            # Query with context
            memories = await loom.recall(f"{conversation_history}\n\nuser: {user_text}")

            # Generate response (simplified - in production use full orchestrator)
            response_text = (
                f"Based on our conversation: {memories[0].text}"
                if memories else
                "I don't have enough context to answer that."
            )
        else:
            # Fallback response
            response_text = f"Echo: {user_text}"

        # 4. Add response to session
        session.append({
            "role": "assistant",
            "text": response_text
        })

        # 5. Generate suggestions
        suggestions = [
            "Tell me more",
            "Show me an example",
            "Explain that differently"
        ]

        return VoiceChatResponse(
            text_response=response_text,
            suggestions=suggestions,
            context_updated={"session_length": len(session)}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# ============== 3. MULTIMODAL INGESTION ==============

@router.post("/ingest/audio")
async def ingest_audio_file(
    audio: UploadFile = File(...),
    episode: str = "audio_session"
):
    """
    Ingest audio file into HoloLoom memory.

    Transcribes audio and stores as memory shard.
    """

    if not WHISPER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Audio ingestion requires Whisper. Install: pip install openai-whisper"
        )

    try:
        # Transcribe audio
        audio_bytes = await audio.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = WHISPER_MODEL.transcribe(tmp_path)
            transcript = result["text"]
        finally:
            os.unlink(tmp_path)

        # Create memory shard
        if HOLOLOOM_AVAILABLE:
            spinner = AudioSpinner()
            shards = await spinner.spin({
                'transcript': transcript,
                'episode': episode,
                'metadata': {
                    'filename': audio.filename,
                    'content_type': audio.content_type
                }
            })

            # Store in HoloLoom
            loom = await get_hololoom()
            for shard in shards:
                await loom.experience(shard.text)

            return {
                "success": True,
                "shards_created": len(shards),
                "transcript": transcript
            }
        else:
            return {
                "success": True,
                "shards_created": 0,
                "transcript": transcript,
                "message": "HoloLoom not available - transcript only"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ============== 4. SESSION MANAGEMENT ==============

@router.get("/sessions")
async def list_sessions():
    """List all active voice chat sessions."""
    return {
        "sessions": [
            {
                "session_id": sid,
                "message_count": len(messages)
            }
            for sid, messages in voice_sessions.items()
        ],
        "total": len(voice_sessions)
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a voice chat session."""
    if session_id in voice_sessions:
        del voice_sessions[session_id]
        return {"success": True, "message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    if session_id not in voice_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "messages": voice_sessions[session_id],
        "total": len(voice_sessions[session_id])
    }
