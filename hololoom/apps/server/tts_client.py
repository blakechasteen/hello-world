"""
TTS Client — Chatterbox voice synthesis for Promptly.

Calls the Chatterbox TTS API (OpenAI-compatible) on the inference rig.
Returns raw audio bytes (WAV) that NanoClaw can deliver as a Matrix attachment.

Graceful degradation: if Chatterbox is unavailable, returns None.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

# Chatterbox runs on the inference rig, port 8004
_RIG_IP = os.environ.get("MINING_RIG_IP", "127.0.0.1")
CHATTERBOX_URL = os.environ.get("CHATTERBOX_URL", f"http://{_RIG_IP}:8004")
CHATTERBOX_TIMEOUT = int(os.environ.get("CHATTERBOX_TIMEOUT", "30"))

# Default voice — can be overridden per-request
DEFAULT_VOICE = os.environ.get("CHATTERBOX_VOICE", "predefined/male_1")

# Max text length to synthesize (avoid monster responses eating GPU time)
MAX_TTS_CHARS = 1500


async def synthesize(
    text: str,
    voice: str | None = None,
) -> bytes | None:
    """Synthesize text to speech via Chatterbox.

    Returns WAV audio bytes, or None if TTS is unavailable/fails.
    """
    if not text or not text.strip():
        return None

    # Truncate long text
    if len(text) > MAX_TTS_CHARS:
        text = text[:MAX_TTS_CHARS] + "..."
        logger.debug("Truncated TTS text to %d chars", MAX_TTS_CHARS)

    # Strip markdown formatting that sounds bad spoken aloud
    text = _strip_markdown(text)

    try:
        async with httpx.AsyncClient(timeout=CHATTERBOX_TIMEOUT) as client:
            # OpenAI-compatible endpoint
            resp = await client.post(
                f"{CHATTERBOX_URL}/v1/audio/speech",
                json={
                    "input": text,
                    "voice": voice or DEFAULT_VOICE,
                    "response_format": "wav",
                },
            )
            resp.raise_for_status()
            audio = resp.content

            logger.info(
                "TTS synthesized: %d chars → %d bytes audio",
                len(text), len(audio),
            )
            return audio

    except httpx.ConnectError:
        logger.debug("Chatterbox unavailable at %s", CHATTERBOX_URL)
        return None
    except httpx.TimeoutException:
        logger.warning("Chatterbox timeout after %ds", CHATTERBOX_TIMEOUT)
        return None
    except Exception as e:
        logger.warning("TTS synthesis failed: %s", e)
        return None


async def is_available() -> bool:
    """Check if Chatterbox is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{CHATTERBOX_URL}/health")
            return resp.status_code == 200
    except Exception:
        return False


def _strip_markdown(text: str) -> str:
    """Strip markdown formatting that sounds bad in TTS."""
    import re
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove bold/italic markers
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    # Remove headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove links, keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove bullet markers
    text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
