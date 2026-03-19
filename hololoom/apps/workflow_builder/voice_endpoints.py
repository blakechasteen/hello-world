"""
Voice Endpoints for Agentic Dashboard
======================================

Add these endpoints to agentic_server.py for voice capabilities.

To integrate:
    from hololoom.apps.workflow_builder.voice_endpoints import add_voice_endpoints
    add_voice_endpoints(app, voice_integration)
"""

import logging

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


def add_voice_endpoints(app: FastAPI, voice_integration):
    """
    Add voice endpoints to FastAPI app.

    Args:
        app: FastAPI application instance
        voice_integration: VoiceIntegration instance
    """

    @app.post("/api/voice/transcribe")
    async def voice_transcribe(audio: UploadFile = File(...)):
        """
        Transcribe audio file to text using Whisper.

        Request:
            - audio: Audio file (wav, mp3, m4a, etc.)

        Returns:
            {
                "success": true,
                "transcript": "transcribed text...",
                "filename": "recording.wav"
            }
        """
        if not voice_integration or not voice_integration.transcription_available:
            return JSONResponse(
                status_code=503,
                content={'error': 'Whisper transcription not available. Install: pip install openai-whisper'}
            )

        try:
            # Read audio bytes
            audio_bytes = await audio.read()

            logger.info(f"Transcribing audio: {audio.filename} ({len(audio_bytes)} bytes)")

            # Transcribe
            transcript = await voice_integration.transcribe_audio(audio_bytes)

            if transcript:
                return {
                    'success': True,
                    'transcript': transcript,
                    'filename': audio.filename
                }
            else:
                return JSONResponse(
                    status_code=400,
                    content={'error': 'Transcription failed'}
                )

        except Exception as e:
            logger.error(f"Voice transcription error: {e}")
            return JSONResponse(
                status_code=500,
                content={'error': str(e)}
            )

    @app.post("/api/voice/speak")
    async def voice_speak(
        text: str = Form(...),
        metric_type: str = Form(None),
        emotion: str = Form(None)
    ):
        """
        Generate speech from text using TTS.

        Request:
            - text: Text to speak
            - metric_type: Optional business metric type ("positive", "negative", "neutral", "warning")
            - emotion: Optional manual emotion ("happy", "concerned", "patient", etc.)

        Returns:
            Audio bytes (audio/wav)
        """
        if not voice_integration or not voice_integration.tts_available:
            return JSONResponse(
                status_code=503,
                content={'error': 'TTS not available. Install BARK: pip install bark'}
            )

        try:
            logger.info(f"Generating speech: {text[:50]}...")

            # Generate speech
            audio_bytes = await voice_integration.speak(
                text=text,
                metric_type=metric_type if metric_type else None,
                emotion=emotion if emotion else None
            )

            if audio_bytes:
                return Response(
                    content=audio_bytes,
                    media_type="audio/wav",
                    headers={
                        'Content-Disposition': 'attachment; filename="response.wav"'
                    }
                )
            else:
                # pyttsx3 returns None (speaks directly)
                return JSONResponse(
                    status_code=200,
                    content={'success': True, 'message': 'Speech generated (pyttsx3 - no audio bytes)'}
                )

        except Exception as e:
            logger.error(f"Voice generation error: {e}")
            return JSONResponse(
                status_code=500,
                content={'error': str(e)}
            )

    @app.post("/api/voice/speak_response")
    async def voice_speak_response(
        response: str = Form(...),
        confidence: float = Form(0.5),
        mode: str = Form("direct")
    ):
        """
        Speak an agentic response with appropriate vocal delivery.

        Automatically determines vocal delivery based on confidence and mode.

        Request:
            - response: Response text
            - confidence: Confidence score (0.0-1.0)
            - mode: Reasoning mode ("direct", "verify", "research", "plan_execute")

        Returns:
            Audio bytes (audio/wav)
        """
        if not voice_integration or not voice_integration.tts_available:
            return JSONResponse(
                status_code=503,
                content={'error': 'TTS not available'}
            )

        try:
            response_data = {
                'response': response,
                'confidence': confidence,
                'mode': mode
            }

            logger.info(f"Speaking response (confidence: {confidence}, mode: {mode})")

            audio_bytes = await voice_integration.speak_response(response_data)

            if audio_bytes:
                return Response(
                    content=audio_bytes,
                    media_type="audio/wav",
                    headers={
                        'Content-Disposition': 'attachment; filename="response.wav"'
                    }
                )
            else:
                return JSONResponse(
                    status_code=200,
                    content={'success': True, 'message': 'Speech generated (pyttsx3)'}
                )

        except Exception as e:
            logger.error(f"Response speech error: {e}")
            return JSONResponse(
                status_code=500,
                content={'error': str(e)}
            )

    @app.get("/api/voice/status")
    async def voice_status():
        """
        Get voice integration status.

        Returns:
            {
                "tts_available": true,
                "tts_backend": "bark",
                "transcription_available": true,
                "whisper_model": "base",
                "auto_speak": false
            }
        """
        if not voice_integration:
            return {
                'tts_available': False,
                'tts_backend': None,
                'transcription_available': False,
                'whisper_model': None,
                'auto_speak': False
            }

        return voice_integration.get_status()

    logger.info("Voice endpoints added to FastAPI app")
