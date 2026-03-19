"""
GPU Peer — lightweight relay peer that announces Ollama GPU capability.

Connects to the HoloLoom relay using the proper wire protocol (DID auth,
hex-encoded message envelopes), announces this machine's GPU/models,
and handles inference_request messages by calling local Ollama.

This lets the Pi daemon route /infer requests to this dev machine
without running the full HoloLoom daemon.

Wire protocol (matches handoff/transport.py WebSocketTransport):
    1. register:  {"type": "register", "device_id": "...", "did": "did:key:z...", "timestamp": ...}
    2. challenge: {"type": "challenge", "nonce": "<hex>"}  (relay -> client)
    3. auth:      {"type": "auth", "signature": "<hex>"}   (client -> relay)
    4. registered: {"type": "registered"}                   (relay -> client)
    5. message:   {"type": "message", "from": "...", "to": "...", "data": "<hex>", "timestamp": ...}
    6. heartbeat: {"type": "heartbeat", "device_id": "...", "timestamp": ...}

Usage:
    peer = GpuPeer(relay_url="ws://10.0.0.58:8765")
    await peer.start()       # connect + register + announce
    await peer.run_forever() # handle inference requests
    await peer.stop()

Or as context manager:
    async with GpuPeer("ws://10.0.0.58:8765") as peer:
        await peer.run_forever()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger("hololoom.gpu_peer")

DEFAULT_RELAY = "ws://10.0.0.58:8765"
DEFAULT_OLLAMA = "http://localhost:11434"
DEFAULT_IDENTITY_PATH = "~/.hololoom/identity"


def _load_identity(identity_path: str) -> tuple:
    """Load identity key and metadata from disk.

    Returns (device_id, did, private_key_bytes).
    """
    path = Path(identity_path).expanduser()
    json_path = path / "identity.json"
    key_path = path / "identity.key"

    if not json_path.exists():
        raise FileNotFoundError(f"Identity JSON not found: {json_path}")
    if not key_path.exists():
        raise FileNotFoundError(f"Identity key not found: {key_path}")

    with open(json_path) as f:
        meta = json.load(f)

    device_id = meta.get("current_device_id", "")
    did = meta.get("did", "")

    with open(key_path) as f:
        hex_str = f.read().strip()

    # Key file is 128 hex chars: 64 private + 64 public
    if len(hex_str) == 128:
        private_key = bytes.fromhex(hex_str[:64])
    elif len(hex_str) == 64:
        private_key = bytes.fromhex(hex_str)
    else:
        raw = key_path.read_bytes()
        if len(raw) == 32:
            private_key = raw
        elif len(raw) == 64:
            private_key = raw[:32]
        else:
            raise ValueError(f"Unexpected key file size: {len(raw)} bytes")

    return device_id, did, private_key


def _sign_nonce(private_key: bytes, nonce_hex: str) -> str:
    """Sign relay challenge nonce with Ed25519 private key."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    nonce_bytes = bytes.fromhex(nonce_hex)
    priv = Ed25519PrivateKey.from_private_bytes(private_key)
    signature = priv.sign(nonce_bytes)
    return signature.hex()


class GpuPeer:

    def __init__(
        self,
        relay_url: str = DEFAULT_RELAY,
        ollama_url: str = DEFAULT_OLLAMA,
        identity_path: str = DEFAULT_IDENTITY_PATH,
        device_id: str | None = None,
    ):
        self.relay_url = relay_url
        self.ollama_url = ollama_url

        # Load identity for DID auth
        loaded_id, self._did, self._private_key = _load_identity(identity_path)
        self.device_id = device_id or loaded_id

        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._running = False
        self._registered = False
        self._models: list[str] = []
        self._heartbeat_task: asyncio.Task | None = None

    async def start(self) -> bool:
        """Connect to relay, authenticate, detect GPU, announce capability."""
        self._session = aiohttp.ClientSession()

        # Detect local Ollama models
        self._models = await self._detect_models()
        if not self._models:
            logger.warning("No Ollama models found — peer will announce but can't serve")

        # Connect to relay
        try:
            self._ws = await self._session.ws_connect(self.relay_url)
            logger.info(f"Connected to relay at {self.relay_url}")
        except Exception as e:
            logger.error(f"Failed to connect to relay: {e}")
            return False

        # Register with DID auth
        if not await self._register():
            logger.error("Relay registration failed")
            return False

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Announce GPU capability
        await self._announce()
        self._running = True
        return True

    async def stop(self) -> None:
        """Disconnect from relay."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("GPU peer stopped")

    async def run_forever(self) -> None:
        """Listen for inference requests and respond."""
        if not self._ws:
            raise RuntimeError("Not connected. Call start() first.")

        logger.info(f"GPU peer listening (device={self.device_id}, models={self._models})")

        while self._running:
            try:
                msg = await asyncio.wait_for(self._ws.receive(), timeout=30.0)

                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_relay_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await self._handle_relay_message(msg.data.decode("utf-8"))
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning("WebSocket closed/error, reconnecting...")
                    await self._reconnect()

            except asyncio.TimeoutError:
                # Re-announce periodically to stay visible
                await self._announce()
            except Exception as e:
                logger.error(f"Error in message loop: {e}")
                await asyncio.sleep(1)

    # ── Relay wire protocol ────────────────────────────────────────────

    async def _register(self) -> bool:
        """Register with relay using DID + Ed25519 challenge/response."""
        # Send register with DID
        register_msg = {
            "type": "register",
            "device_id": self.device_id,
            "did": self._did,
            "timestamp": time.time(),
        }
        await self._ws.send_str(json.dumps(register_msg))
        logger.debug(f"Sent register (device={self.device_id}, did={self._did[:30]}...)")

        # Wait for challenge or registered
        try:
            resp = await asyncio.wait_for(self._ws.receive(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for relay registration response")
            return False

        if resp.type != aiohttp.WSMsgType.TEXT:
            logger.error(f"Unexpected response type: {resp.type}")
            return False

        data = json.loads(resp.data)

        if data.get("type") == "registered":
            # No challenge (DID not recognized or relay skipped auth)
            self._registered = True
            logger.info("Registered with relay (no challenge)")
            return True

        if data.get("type") == "challenge":
            nonce = data["nonce"]
            signature = _sign_nonce(self._private_key, nonce)
            auth_msg = {"type": "auth", "signature": signature}
            await self._ws.send_str(json.dumps(auth_msg))
            logger.debug("Sent auth response to challenge")

            # Wait for registered/auth_failed
            try:
                resp2 = await asyncio.wait_for(self._ws.receive(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for auth result")
                return False

            data2 = json.loads(resp2.data)
            if data2.get("type") == "registered":
                self._registered = True
                logger.info("Registered with relay (DID authenticated)")
                return True
            else:
                logger.error(f"Auth failed: {data2}")
                return False

        logger.error(f"Unexpected register response: {data}")
        return False

    async def _send_relay_message(self, target: str, payload: dict) -> None:
        """Wrap payload in relay wire envelope and send."""
        inner = json.dumps(payload).encode("utf-8")
        envelope = {
            "type": "message",
            "from": self.device_id,
            "to": target,
            "data": inner.hex(),
            "timestamp": time.time(),
        }
        await self._ws.send_str(json.dumps(envelope))

    async def _handle_relay_message(self, raw: str) -> None:
        """Parse relay envelope, extract inner payload, route."""
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = envelope.get("type")

        if msg_type == "message":
            # Extract hex-encoded inner payload
            hex_data = envelope.get("data", "")
            try:
                inner_bytes = bytes.fromhex(hex_data)
                payload = json.loads(inner_bytes.decode("utf-8"))
            except (ValueError, json.JSONDecodeError) as e:
                logger.debug(f"Failed to decode relay message data: {e}")
                return

            source = envelope.get("from", "unknown")
            await self._handle_payload(payload, source)

        elif msg_type in ("registered", "challenge", "peers"):
            # Protocol messages during registration — handled elsewhere
            logger.debug(f"Relay protocol message: {msg_type}")

    async def _handle_payload(self, payload: dict, source: str) -> None:
        """Route decoded inner payloads."""
        msg_type = payload.get("type")

        if msg_type == "inference_request":
            await self._handle_inference_request(payload, source)
        elif msg_type == "device_announce":
            peer = payload.get("device", {}).get("device_id", "unknown")
            logger.debug(f"Peer announced: {peer}")
        elif msg_type == "sync_request":
            logger.debug(f"Ignoring sync_request from {source}")
        else:
            logger.debug(f"Unhandled message type: {msg_type} from {source}")

    async def _heartbeat_loop(self) -> None:
        """Send heartbeats to relay every 30s to stay connected."""
        while self._running:
            try:
                await asyncio.sleep(30)
                if self._ws and not self._ws.closed:
                    hb = {
                        "type": "heartbeat",
                        "device_id": self.device_id,
                        "timestamp": time.time(),
                    }
                    await self._ws.send_str(json.dumps(hb))
                    logger.debug("Heartbeat sent")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")

    # ── Inference handling ─────────────────────────────────────────────

    async def _handle_inference_request(self, payload: dict[str, Any], source: str) -> None:
        """Handle inference request by calling local Ollama."""
        request_id = payload.get("request_id", "")
        prompt = payload.get("prompt", "")
        model = payload.get("model")
        max_tokens = payload.get("max_tokens", 2048)
        temperature = payload.get("temperature", 0.7)
        source_device = payload.get("source_device", source)

        logger.info(
            f"Inference request {request_id[:8]} from {source_device[:8]}: "
            f"{len(prompt)} chars, model={model}"
        )

        if not self._models:
            await self._send_inference_response(
                request_id, source_device, error="No models available"
            )
            return

        resolved_model = model or self._models[0]

        try:
            result = await self._call_ollama(prompt, resolved_model, max_tokens, temperature)
            await self._send_inference_response(
                request_id, source_device,
                response=result.get("response", ""),
                tokens_used=result.get("eval_count", 0) + result.get("prompt_eval_count", 0),
                model=resolved_model,
            )
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            await self._send_inference_response(request_id, source_device, error=str(e))

    async def _call_ollama(
        self, prompt: str, model: str, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        """Call local Ollama /api/chat with think=false."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        async with self._session.post(
            f"{self.ollama_url}/api/chat", json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            data = await resp.json()
            message = data.get("message", {})
            return {
                "response": message.get("content", ""),
                "eval_count": data.get("eval_count", 0),
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "model": data.get("model", model),
            }

    async def _send_inference_response(
        self,
        request_id: str,
        target_device: str,
        response: str = "",
        tokens_used: int = 0,
        model: str = "",
        error: str = "",
    ) -> None:
        """Send inference_response back via relay envelope."""
        msg = {
            "type": "inference_response",
            "request_id": request_id,
            "target_device": target_device,
        }
        if error:
            msg["ok"] = False
            msg["error"] = error
        else:
            msg["ok"] = True
            msg["response"] = response
            msg["tokens_used"] = tokens_used
            msg["model"] = model

        # Send via relay wire protocol — targeted to the requesting device
        await self._send_relay_message(target_device, msg)
        logger.info(
            f"Response sent for {request_id[:8]}: "
            f"{'error=' + error if error else f'{tokens_used} tokens'}"
        )

    # ── Announce & discover ────────────────────────────────────────────

    async def _announce(self) -> None:
        """Announce GPU capability to all peers via relay broadcast."""
        from datetime import datetime

        now = datetime.utcnow().isoformat()
        announce = {
            "type": "device_announce",
            "device": {
                "device_id": self.device_id,
                "device_name": "desktop-gpu",
                "public_key": self._did[9:] if self._did.startswith("did:key:z") else "",
                "api_key_id": f"apikey_{self.device_id}",
                "capabilities": ["weave", "sync", "recall", "experience", "reflect"],
                "status": "ACTIVE",
                "paired_at": now,
                "last_seen": now,
                "last_sync": None,
                "metadata": {},
            },
            "did": self._did,
            "capabilities": {
                "has_gpu": bool(self._models),
                "models": self._models,
            },
            "clock": 0,
        }
        if self._ws and not self._ws.closed and self._registered:
            await self._send_relay_message("*", announce)
            logger.debug(f"Announced: device={self.device_id}, models={self._models}")

    # ── Utilities ──────────────────────────────────────────────────────

    async def _detect_models(self) -> list[str]:
        """Probe local Ollama for available models."""
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"{self.ollama_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Ollama detection failed: {e}")
        return []

    async def _reconnect(self) -> None:
        """Reconnect to relay after disconnect."""
        self._registered = False
        for attempt in range(5):
            try:
                await asyncio.sleep(2 ** attempt)
                self._ws = await self._session.ws_connect(self.relay_url)
                if await self._register():
                    await self._announce()
                    logger.info("Reconnected to relay")
                    return
            except Exception:
                continue
        logger.error("Failed to reconnect after 5 attempts")
        self._running = False

    # ── Context manager ────────────────────────────────────────────────

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()
