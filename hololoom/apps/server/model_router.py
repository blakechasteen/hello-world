"""
Model Router — Smart query-to-model routing with Thompson Sampling
==================================================================

A model registry + query classifier + bandit that learns which model
works best for which kind of query. Adding a model is adding a dict entry.

Design:
- Registry: flat list of ModelSpec entries (capabilities, hardware, endpoint)
- Classifier: heuristic query → intent tags (chat, code, reasoning, factual, creative)
- Bandit: Thompson Sampling per (intent, model) pair — learns from outcomes
- Health: async probe before routing to catch sleeping rigs / busy GPUs
- Fallback: always has a local model that works (qwen3.5:9b on desktop)

Created: 2026-03-08
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

# GPU rig IP — set MINING_RIG_IP env var or default to localhost
_RIG_IP = os.environ.get("MINING_RIG_IP", "127.0.0.1")

logger = logging.getLogger(__name__)


# ============================================================================
# Intent Tags (canonical definition in routing/intent_classifier.py)
# ============================================================================

from hololoom.routing.intent_classifier import Intent

# ============================================================================
# Model Registry
# ============================================================================

class Backend(str, Enum):
    OLLAMA = "ollama"       # Local Ollama HTTP API
    AIRLLM = "airllm"       # AirLLM OpenAI-compatible API (GPU rig)
    OPENAI = "openai"       # Any OpenAI-compatible endpoint
    LLAMACPP = "llamacpp"   # llama-server (OpenAI-compatible + GBNF grammar)


@dataclass
class ModelSpec:
    """A model available for routing."""
    id: str                          # Unique key: "qwen3.5:9b"
    backend: Backend                 # How to call it
    base_url: str                    # Endpoint base URL
    strengths: set[Intent]           # What it's good at
    context_window: int = 131072     # Max context tokens
    max_tokens: int = 8192           # Max output tokens
    vram_gb: float = 0               # VRAM footprint (for scheduling)
    speed: float = 1.0               # Relative speed (1.0 = baseline)
    quality: float = 1.0             # Relative quality (1.0 = baseline)
    always_available: bool = False   # Skip health check (local, always up)
    api_key: str = ""                # For authenticated endpoints
    tags: set[str] = field(default_factory=set)  # Freeform: "local", "rig", "api"


# Default registry — extend via PROMPTLY_MODELS env or API
DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec(
        id="qwen3.5:9b",
        backend=Backend.OLLAMA,
        base_url="http://127.0.0.1:11434",
        strengths={Intent.CHAT, Intent.FACTUAL, Intent.CODE},
        vram_gb=6.6,
        speed=3.0,     # Fast — small model
        quality=0.6,
        always_available=True,
        tags={"local", "default", "fast"},
    ),
    ModelSpec(
        id="qwen3.5:30b",
        backend=Backend.OLLAMA,
        base_url="http://127.0.0.1:11434",
        strengths={Intent.CODE, Intent.REASONING, Intent.PLANNING},
        vram_gb=10.5,
        speed=1.5,
        quality=0.8,
        always_available=True,
        tags={"local", "mid"},
    ),
    ModelSpec(
        id="qwen3:30b-155k",
        backend=Backend.OLLAMA,
        base_url=f"http://{_RIG_IP}:11434",
        strengths={Intent.CHAT, Intent.CODE, Intent.REASONING, Intent.FACTUAL},
        context_window=155648,
        vram_gb=12.0,
        speed=1.0,
        quality=0.85,
        tags={"rig", "mid"},
    ),
    ModelSpec(
        id="qwen3.5:108b-planner",
        backend=Backend.AIRLLM,
        base_url=f"http://{_RIG_IP}:8001",
        strengths={Intent.PLANNING, Intent.REASONING, Intent.CREATIVE},
        vram_gb=12.0,
        speed=0.3,     # Slow — AirLLM sequential
        quality=1.0,
        tags={"rig", "deep", "planner"},
    ),
    ModelSpec(
        id="qwen3.5:108b-critic",
        backend=Backend.AIRLLM,
        base_url=f"http://{_RIG_IP}:8002",
        strengths={Intent.REASONING, Intent.CODE, Intent.FACTUAL},
        vram_gb=12.0,
        speed=0.3,
        quality=1.0,
        tags={"rig", "deep", "critic"},
    ),
    ModelSpec(
        id="qwen3.5:108b-synthesizer",
        backend=Backend.AIRLLM,
        base_url=f"http://{_RIG_IP}:8003",
        strengths={Intent.CREATIVE, Intent.PLANNING, Intent.CHAT},
        vram_gb=12.0,
        speed=0.3,
        quality=1.0,
        tags={"rig", "deep", "synthesizer"},
    ),
]


# ============================================================================
# Query Classifier (canonical definition in routing/intent_classifier.py)
# ============================================================================

from hololoom.routing.intent_classifier import classify_intent  # noqa: F811

# ============================================================================
# Thompson Sampling Bandit
# ============================================================================
from hololoom.ts_core.arm import BanditArm as _SharedBanditArm


@dataclass
class BanditArm:
    """Beta distribution parameters for one (intent, model) pair.

    Thin wrapper around ts_core.arm.BanditArm for backward compatibility
    with save/load (uses successes/failures field names).
    """
    successes: float = 1.0   # Alpha prior
    failures: float = 1.0    # Beta prior

    def sample(self) -> float:
        """Draw from Beta(alpha, beta)."""
        return _SharedBanditArm(alpha=self.successes, beta=self.failures).sample()

    def update(self, success: bool) -> None:
        if success:
            self.successes += 1.0
        else:
            self.failures += 1.0

    def geodesic_certainty(self) -> float:
        """Fisher-Rao distance from uniform prior. See ts_core.arm for details."""
        return _SharedBanditArm(alpha=self.successes, beta=self.failures).geodesic_certainty()

    @property
    def mean(self) -> float:
        return self.successes / (self.successes + self.failures)


class ModelBandit:
    """Thompson Sampling over models, keyed by intent."""

    def __init__(self):
        self._arms: dict[tuple[Intent, str], BanditArm] = {}

    def _key(self, intent: Intent, model_id: str) -> tuple[Intent, str]:
        return (intent, model_id)

    def _get_arm(self, intent: Intent, model_id: str) -> BanditArm:
        key = self._key(intent, model_id)
        if key not in self._arms:
            self._arms[key] = BanditArm()
        return self._arms[key]

    def select(self, intent: Intent, candidates: list[ModelSpec],
               speed_weight: float = 0.5) -> ModelSpec:
        """
        Thompson Sample the best model for this intent.

        speed_weight: 0.0 = pure quality, 1.0 = pure speed.
        Blends bandit score with model speed/quality priors.
        """
        if not candidates:
            raise ValueError("No candidate models")

        best_score = -1.0
        best_model = candidates[0]

        for model in candidates:
            arm = self._get_arm(intent, model.id)
            bandit_score = arm.sample()

            # Blend: bandit learns quality, model.speed is prior for latency
            blended = (
                (1 - speed_weight) * bandit_score * model.quality
                + speed_weight * (model.speed / 5.0)  # Normalize speed to ~0-1
            )

            # Exploration bonus for uncertain arms (low geodesic certainty).
            # Arms we know little about get a small boost to encourage
            # exploration, measured via Fisher-Rao distance from uniform.
            certainty = arm.geodesic_certainty()
            if certainty < 1.0:  # Low-evidence arm
                blended += 0.05 * (1.0 - certainty)  # Up to 5% bonus

            # Bonus for strength match
            if intent in model.strengths:
                blended *= 1.2

            if blended > best_score:
                best_score = blended
                best_model = model

        return best_model

    def update(self, intent: Intent, model_id: str, success: bool) -> None:
        self._get_arm(intent, model_id).update(success)

    def stats(self) -> dict[str, dict]:
        """Return bandit statistics for observability."""
        result = {}
        for (intent, model_id), arm in self._arms.items():
            key = f"{intent.value}:{model_id}"
            result[key] = {
                "successes": arm.successes,
                "failures": arm.failures,
                "mean": round(arm.mean, 3),
                "certainty": round(arm.geodesic_certainty(), 3),
            }
        return result

    def save(self, path: Path) -> None:
        """Persist bandit state to JSON."""
        data = {}
        for (intent, model_id), arm in self._arms.items():
            key = f"{intent.value}:{model_id}"
            data[key] = {"s": arm.successes, "f": arm.failures}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)  # Atomic on POSIX, near-atomic on Windows
        logger.debug("Bandit state saved: %d arms → %s", len(data), path)

    def load(self, path: Path) -> None:
        """Load bandit state from JSON. Missing/corrupt file = fresh start."""
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            loaded = 0
            for key, vals in data.items():
                parts = key.split(":", 1)
                if len(parts) != 2:
                    continue
                try:
                    intent = Intent(parts[0])
                except ValueError:
                    continue
                model_id = parts[1]
                arm = BanditArm(successes=vals["s"], failures=vals["f"])
                self._arms[(intent, model_id)] = arm
                loaded += 1
            logger.info("Bandit state loaded: %d arms from %s", loaded, path)
        except Exception as e:
            logger.warning("Failed to load bandit state from %s: %s", path, e)


# ============================================================================
# Health Checker
# ============================================================================

class HealthCache:
    """Cached health status for models. Avoids hammering endpoints."""

    def __init__(self, ttl: float = 30.0):
        self._cache: dict[str, tuple[bool, float]] = {}
        self._ttl = ttl

    async def is_healthy(self, model: ModelSpec) -> bool:
        if model.always_available:
            return True

        cached = self._cache.get(model.id)
        if cached and (time.time() - cached[1]) < self._ttl:
            return cached[0]

        healthy = await self._probe(model)
        self._cache[model.id] = (healthy, time.time())
        return healthy

    async def _probe(self, model: ModelSpec) -> bool:
        """Quick health check — different per backend."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                if model.backend == Backend.OLLAMA:
                    resp = await client.get(f"{model.base_url}/api/tags")
                    return resp.status_code == 200
                elif model.backend in (Backend.AIRLLM, Backend.OPENAI):
                    resp = await client.get(f"{model.base_url}/health")
                    if resp.status_code == 200:
                        data = resp.json()
                        return data.get("status") in ("ready", "idle", "ok")
                    return False
                elif model.backend == Backend.LLAMACPP:
                    # llama-server: GET /health returns {"status": "ok"} when ready
                    resp = await client.get(f"{model.base_url}/health")
                    if resp.status_code == 200:
                        data = resp.json()
                        return data.get("status") in ("ok", "no slot available")
                    return False
        except Exception:
            return False


# ============================================================================
# Router
# ============================================================================

@dataclass
class RoutingDecision:
    """What the router decided."""
    model: ModelSpec
    intent: Intent
    confidence: float
    reason: str
    fallback: ModelSpec | None = None  # If primary fails, try this
    health_status: dict[str, bool] = field(default_factory=dict)


class ModelRouter:
    """
    Smart model selection.

    Usage:
        router = ModelRouter()
        decision = await router.route("How do I implement a B-tree?")
        # decision.model is the best ModelSpec for this query
        # Later:
        router.feedback(decision, success=True)
    """

    def __init__(self, models: list[ModelSpec] | None = None,
                 persist_path: Path | None = None):
        self.models = models or list(DEFAULT_MODELS)
        self.bandit = ModelBandit()
        self.health = HealthCache(ttl=30.0)
        self._persist_path = persist_path
        self._feedback_count = 0
        self._save_every = 10  # Save after every N feedbacks
        if self._persist_path:
            self.bandit.load(self._persist_path)

    def add_model(self, spec: ModelSpec) -> None:
        """Add a model to the registry at runtime."""
        # Replace if same ID exists
        self.models = [m for m in self.models if m.id != spec.id] + [spec]
        logger.info("Model registered: %s (%s)", spec.id, spec.backend.value)

    def remove_model(self, model_id: str) -> None:
        self.models = [m for m in self.models if m.id != model_id]

    async def route(
        self,
        query: str,
        speed_weight: float = 0.5,
        require_tags: set[str] | None = None,
        exclude_tags: set[str] | None = None,
    ) -> RoutingDecision:
        """
        Route a query to the best available model.

        Args:
            query: The user's message
            speed_weight: 0.0 = optimize quality, 1.0 = optimize speed
            require_tags: Only consider models with ALL these tags
            exclude_tags: Skip models with ANY of these tags
        """
        intent, confidence = classify_intent(query)

        # Filter candidates
        candidates = []
        for m in self.models:
            if require_tags and not require_tags.issubset(m.tags):
                continue
            if exclude_tags and exclude_tags.intersection(m.tags):
                continue
            candidates.append(m)

        if not candidates:
            candidates = [m for m in self.models if m.always_available]

        # Health-check candidates (parallel)
        health_results = await asyncio.gather(
            *[self.health.is_healthy(m) for m in candidates]
        )
        health_map = {m.id: ok for m, ok in zip(candidates, health_results)}
        healthy = [m for m, ok in zip(candidates, health_results) if ok]

        if not healthy:
            # Everything is down — fall back to any always_available model
            healthy = [m for m in self.models if m.always_available]
            if not healthy:
                raise RuntimeError("No models available")

        # Thompson Sample from healthy candidates
        selected = self.bandit.select(intent, healthy, speed_weight)

        # Pick a fallback (different from selected, prefer always_available)
        fallback = None
        fallback_candidates = [m for m in healthy if m.id != selected.id]
        if fallback_candidates:
            fallback = next(
                (m for m in fallback_candidates if m.always_available),
                fallback_candidates[0],
            )

        reason = (
            f"intent={intent.value} conf={confidence:.2f} "
            f"model={selected.id} speed_w={speed_weight}"
        )
        logger.info("Routed: %s", reason)

        return RoutingDecision(
            model=selected,
            intent=intent,
            confidence=confidence,
            reason=reason,
            fallback=fallback,
            health_status=health_map,
        )

    def feedback(self, decision: RoutingDecision, success: bool) -> None:
        """Update bandit with outcome. Call after each response."""
        self.bandit.update(decision.intent, decision.model.id, success)
        self._feedback_count += 1
        if self._persist_path and self._feedback_count % self._save_every == 0:
            self.bandit.save(self._persist_path)

    def list_models(self) -> list[dict]:
        """Return model registry for status endpoint."""
        return [
            {
                "id": m.id,
                "backend": m.backend.value,
                "strengths": [s.value for s in m.strengths],
                "speed": m.speed,
                "quality": m.quality,
                "tags": sorted(m.tags),
                "always_available": m.always_available,
            }
            for m in self.models
        ]

    def bandit_stats(self) -> dict[str, dict]:
        return self.bandit.stats()


# ============================================================================
# Module-level singleton
# ============================================================================

# Default persist path: ~/.cache/hololoom/bandit_state.json
# Override with PROMPTLY_BANDIT_PATH env var
_DEFAULT_BANDIT_PATH = Path(
    os.environ.get(
        "PROMPTLY_BANDIT_PATH",
        Path.home() / ".cache" / "hololoom" / "bandit_state.json",
    )
)

_router: ModelRouter | None = None


def get_router() -> ModelRouter:
    """Get or create the global router instance."""
    global _router
    if _router is None:
        _router = ModelRouter(persist_path=_DEFAULT_BANDIT_PATH)
    return _router
