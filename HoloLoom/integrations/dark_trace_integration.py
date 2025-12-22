"""
Dark Trace Integration with HoloLoom Orchestrator

Phase 10: Production integration of interpretability system with HoloLoom's
weaving orchestrator. Enables automatic activation analysis, steering,
and cross-model feature comparison during the weaving cycle.

Key Features:
- Automatic activation capture during weaving
- Real-time interpretability analysis
- Steering vector injection into policy decisions
- Cross-model fingerprinting support
- Safety-aware feature monitoring

Usage:
    from HoloLoom.integrations.dark_trace_integration import (
        DarkTraceIntegration,
        create_integration,
        enable_dark_trace,
    )

    # Create integration
    integration = create_integration(orchestrator, config=TraceConfig.standard())

    # Enable on orchestrator
    enable_dark_trace(orchestrator, integration)

    # Weave with interpretability
    spacetime = await orchestrator.weave(query)

    # Access interpretability results
    trace_result = integration.get_last_trace()
    print(trace_result.explain(verbosity=2))

    # Apply steering
    integration.set_steering({"semantic.Warmth": 1.5, "sae.42": -0.5})
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
import asyncio
import logging
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

import numpy as np

# Dark Trace imports
from HoloLoom.dark_trace.engine import DarkTraceEngine, create_engine
from HoloLoom.dark_trace.trace_config import TraceConfig, TraceMode
from HoloLoom.dark_trace.result import TraceResult, SteeringResult
from HoloLoom.dark_trace.protocol import (
    Feature,
    FeatureActivation,
    SafetyFlag,
    LensType,
)
from HoloLoom.dark_trace.models.policy_adapter import (
    PolicyAdapter,
    create_policy_adapter,
)
from HoloLoom.dark_trace.models.fingerprint import (
    ModelFingerprinter,
    FeatureFingerprint,
    FingerprintConfig,
    compare_fingerprints,
    compare_models,
    ModelComparisonReport,
)


logger = logging.getLogger(__name__)


class IntegrationMode(Enum):
    """Modes for Dark Trace integration."""
    DISABLED = "disabled"      # No interpretability
    PASSIVE = "passive"        # Analysis only, no steering
    ACTIVE = "active"          # Analysis + steering enabled
    FULL = "full"              # All features including safety override


@dataclass
class IntegrationConfig:
    """Configuration for Dark Trace integration."""

    mode: IntegrationMode = IntegrationMode.PASSIVE
    trace_config: Optional[TraceConfig] = None

    # Automatic analysis settings
    auto_analyze: bool = True  # Analyze every weave
    analyze_layers: List[str] = field(default_factory=lambda: [
        "block.0.mha",
        "block.1.mha",
        "readout",
    ])

    # Steering settings
    enable_steering: bool = False
    steering_goals: Dict[str, float] = field(default_factory=dict)
    steering_scale: float = 1.0

    # Safety settings
    safety_monitoring: bool = True
    block_on_safety_concern: bool = False  # Block weave if safety issues
    safety_callback: Optional[Callable[[TraceResult], None]] = None

    # Fingerprinting
    enable_fingerprinting: bool = False
    fingerprint_every_n: int = 100  # Fingerprint every N weaves

    # Performance
    async_analysis: bool = True  # Run analysis in background
    max_trace_history: int = 100  # Keep last N traces

    @classmethod
    def passive(cls) -> "IntegrationConfig":
        """Create passive (analysis-only) config."""
        return cls(
            mode=IntegrationMode.PASSIVE,
            auto_analyze=True,
            enable_steering=False,
        )

    @classmethod
    def active(cls) -> "IntegrationConfig":
        """Create active (analysis + steering) config."""
        return cls(
            mode=IntegrationMode.ACTIVE,
            auto_analyze=True,
            enable_steering=True,
        )

    @classmethod
    def full(cls) -> "IntegrationConfig":
        """Create full-featured config."""
        return cls(
            mode=IntegrationMode.FULL,
            auto_analyze=True,
            enable_steering=True,
            safety_monitoring=True,
            enable_fingerprinting=True,
        )


@dataclass
class IntegrationResult:
    """Result of integrated weave with interpretability."""

    spacetime: Any  # Original spacetime result
    trace: Optional[TraceResult] = None
    steering_applied: bool = False
    safety_blocked: bool = False
    analysis_time_ms: float = 0.0

    def explain(self, verbosity: int = 1) -> str:
        """Generate explanation including interpretability."""
        lines = []

        if self.trace:
            lines.append(self.trace.explain(verbosity))

        if self.steering_applied:
            lines.append("\n[Steering was applied]")

        if self.safety_blocked:
            lines.append("\n[BLOCKED: Safety concern detected]")

        return "\n".join(lines)


class DarkTraceIntegration:
    """
    Integration layer between Dark Trace and HoloLoom orchestrator.

    Provides:
    - Automatic activation capture during weaving
    - Real-time interpretability analysis
    - Steering vector injection
    - Safety monitoring
    - Cross-model fingerprinting
    """

    def __init__(
        self,
        config: Optional[IntegrationConfig] = None,
        trace_config: Optional[TraceConfig] = None,
    ):
        """
        Initialize Dark Trace integration.

        Args:
            config: Integration configuration
            trace_config: Dark Trace engine configuration
        """
        self.config = config or IntegrationConfig.passive()
        self._trace_config = trace_config or self.config.trace_config or TraceConfig.standard()

        # Initialize Dark Trace engine
        self._engine = create_engine(config=self._trace_config)

        # Policy adapter (set when attached to orchestrator)
        self._policy_adapter: Optional[PolicyAdapter] = None

        # State tracking
        self._weave_count = 0
        self._trace_history: List[TraceResult] = []
        self._last_trace: Optional[TraceResult] = None
        self._fingerprints: Dict[str, Dict[str, FeatureFingerprint]] = {}

        # Steering state
        self._active_steering: Dict[str, float] = {}

        # Callbacks
        self._pre_weave_callbacks: List[Callable] = []
        self._post_weave_callbacks: List[Callable] = []

        logger.info(f"DarkTraceIntegration initialized (mode={self.config.mode.name})")

    @property
    def engine(self) -> DarkTraceEngine:
        """Access underlying Dark Trace engine."""
        return self._engine

    @property
    def adapter(self) -> Optional[PolicyAdapter]:
        """Access policy adapter."""
        return self._policy_adapter

    def attach_to_orchestrator(self, orchestrator: Any) -> None:
        """
        Attach integration to a WeavingOrchestrator.

        Args:
            orchestrator: HoloLoom WeavingOrchestrator instance
        """
        # Extract policy from orchestrator
        policy = None
        if hasattr(orchestrator, 'policy'):
            policy = orchestrator.policy
        elif hasattr(orchestrator, '_policy'):
            policy = orchestrator._policy

        if policy is None:
            logger.warning("Could not find policy in orchestrator")
            return

        # Create policy adapter
        self._policy_adapter = create_policy_adapter(policy)

        # Store orchestrator reference
        self._orchestrator = orchestrator

        # Hook into weaving if orchestrator supports it
        if hasattr(orchestrator, 'add_weave_callback'):
            orchestrator.add_weave_callback(self._on_weave)
        elif hasattr(orchestrator, '_post_weave_hooks'):
            orchestrator._post_weave_hooks.append(self._on_weave)

        logger.info("Dark Trace attached to orchestrator")

    def detach(self) -> None:
        """Detach from orchestrator."""
        self._policy_adapter = None
        self._orchestrator = None
        logger.info("Dark Trace detached from orchestrator")

    # =========================================================================
    # Analysis
    # =========================================================================

    def analyze(
        self,
        activations: Union[np.ndarray, "torch.Tensor"],
        layer_name: Optional[str] = None,
    ) -> TraceResult:
        """
        Analyze activations through Dark Trace.

        Args:
            activations: Model activations
            layer_name: Optional layer name for provenance

        Returns:
            TraceResult with interpretability analysis
        """
        if TORCH_AVAILABLE and isinstance(activations, np.ndarray):
            activations = torch.tensor(activations, dtype=torch.float32)

        result = self._engine.analyze(
            activations,
            layer_name=layer_name,
            include_safety=self.config.safety_monitoring,
        )

        # Store in history
        self._last_trace = result
        self._trace_history.append(result)
        if len(self._trace_history) > self.config.max_trace_history:
            self._trace_history.pop(0)

        # Safety callback
        if (self.config.safety_monitoring and
            result.has_safety_concerns and
            self.config.safety_callback):
            self.config.safety_callback(result)

        return result

    def analyze_policy_decision(
        self,
        mem: Union[np.ndarray, "torch.Tensor"],
        ctrl: Union[np.ndarray, "torch.Tensor"],
        adapter_idx: int = 0,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, TraceResult]:
        """
        Analyze policy network activations across multiple layers.

        Args:
            mem: Context memory [B, M, D]
            ctrl: Motif control [B, n_motifs]
            adapter_idx: Adapter index
            layers: Layers to analyze (default: configured analyze_layers)

        Returns:
            Dict mapping layer name to TraceResult
        """
        if self._policy_adapter is None:
            raise RuntimeError("Not attached to orchestrator")

        layers = layers or self.config.analyze_layers
        results = {}

        inputs = {
            "mem": mem,
            "ctrl": ctrl,
            "adapter_idx": adapter_idx,
        }

        for layer in layers:
            try:
                acts = self._policy_adapter.get_activations(inputs, layer)
                if TORCH_AVAILABLE:
                    acts_tensor = torch.tensor(acts, dtype=torch.float32)
                else:
                    acts_tensor = acts

                result = self._engine.analyze(acts_tensor, layer_name=layer)
                results[layer] = result

            except Exception as e:
                logger.warning(f"Failed to analyze layer {layer}: {e}")

        return results

    def get_last_trace(self) -> Optional[TraceResult]:
        """Get the most recent trace result."""
        return self._last_trace

    def get_trace_history(self, limit: int = 10) -> List[TraceResult]:
        """Get recent trace history."""
        return self._trace_history[-limit:]

    # =========================================================================
    # Steering
    # =========================================================================

    def set_steering(
        self,
        goals: Dict[str, float],
        validate_safety: bool = True,
    ) -> SteeringResult:
        """
        Set steering goals for policy decisions.

        Args:
            goals: {feature_id: target_strength}
            validate_safety: Check safety before allowing

        Returns:
            SteeringResult with validation info
        """
        if not self.config.enable_steering:
            logger.warning("Steering not enabled in config")
            return SteeringResult(
                steering_vector=None,
                target_features=goals,
                features_blocked=list(goals.keys()),
            )

        result = self._engine.steer(goals, validate_safety=validate_safety)

        if result.steering_vector:
            self._active_steering = goals

            # Apply to policy adapter if attached
            if self._policy_adapter and result.features_applied:
                for feature_id in result.features_applied:
                    strength = goals[feature_id]
                    # Map feature to layer (simplified - full impl would use registry)
                    layer = self._feature_to_layer(feature_id)
                    if layer:
                        self._policy_adapter.inject_steering(
                            vector=result.steering_vector.vector.numpy(),
                            layer=layer,
                            scale=self.config.steering_scale * strength,
                        )

        return result

    def clear_steering(self) -> int:
        """Clear all active steering. Returns count cleared."""
        self._active_steering.clear()

        if self._policy_adapter:
            return self._policy_adapter.clear_steering()

        return 0

    def _feature_to_layer(self, feature_id: str) -> Optional[str]:
        """Map feature ID to policy layer (heuristic)."""
        if feature_id.startswith("sae."):
            return "block.1.mha"  # Default to later attention
        elif feature_id.startswith("semantic."):
            return "readout"  # Semantic features at output
        return None

    # =========================================================================
    # Fingerprinting
    # =========================================================================

    def fingerprint_policy(
        self,
        probe_inputs: List[Dict[str, Any]],
        model_id: str = "policy",
        layers: Optional[List[str]] = None,
    ) -> Dict[str, FeatureFingerprint]:
        """
        Generate fingerprints for policy network.

        Args:
            probe_inputs: List of input dicts for probing
            model_id: Identifier for this model
            layers: Layers to fingerprint

        Returns:
            Dict mapping feature_id to fingerprint
        """
        if self._policy_adapter is None:
            raise RuntimeError("Not attached to orchestrator")

        layers = layers or self.config.analyze_layers

        fingerprinter = ModelFingerprinter(
            adapter=self._policy_adapter,
            model_id=model_id,
            config=FingerprintConfig(),
        )

        # Collect activations for each input
        all_fingerprints = {}
        for layer in layers:
            layer_fps = fingerprinter.fingerprint_layer(
                layer=layer,
                inputs=probe_inputs,
            )
            all_fingerprints.update(layer_fps)

        # Cache fingerprints
        self._fingerprints[model_id] = all_fingerprints

        return all_fingerprints

    def compare_with_model(
        self,
        other_fingerprints: Dict[str, FeatureFingerprint],
        other_model_id: str,
    ) -> ModelComparisonReport:
        """
        Compare current policy fingerprints with another model.

        Args:
            other_fingerprints: Fingerprints from another model
            other_model_id: ID of other model

        Returns:
            ModelComparisonReport with comparison results
        """
        if "policy" not in self._fingerprints:
            raise RuntimeError("No policy fingerprints available")

        fingerprints_by_model = {
            "policy": self._fingerprints["policy"],
            other_model_id: other_fingerprints,
        }

        return compare_models(fingerprints_by_model)

    # =========================================================================
    # Weaving Integration
    # =========================================================================

    async def wrap_weave(
        self,
        weave_fn: Callable,
        *args,
        **kwargs,
    ) -> IntegrationResult:
        """
        Wrap a weave call with interpretability.

        Args:
            weave_fn: Original weave function
            *args, **kwargs: Arguments for weave

        Returns:
            IntegrationResult with spacetime and trace
        """
        start_time = time.time()
        self._weave_count += 1

        # Pre-weave analysis if needed
        trace = None
        safety_blocked = False

        try:
            # Run original weave
            spacetime = await weave_fn(*args, **kwargs)

            # Post-weave analysis
            if (self.config.auto_analyze and
                self.config.mode != IntegrationMode.DISABLED):

                # Extract activations from spacetime if available
                if hasattr(spacetime, 'features') and spacetime.features is not None:
                    if TORCH_AVAILABLE:
                        acts = torch.tensor(spacetime.features, dtype=torch.float32)
                    else:
                        acts = spacetime.features

                    if self.config.async_analysis:
                        # Run analysis in background
                        loop = asyncio.get_event_loop()
                        trace = await loop.run_in_executor(
                            None,
                            lambda: self._engine.analyze(acts)
                        )
                    else:
                        trace = self._engine.analyze(acts)

                    # Store trace
                    self._last_trace = trace
                    self._trace_history.append(trace)

                    # Safety check
                    if (self.config.safety_monitoring and
                        trace.has_safety_concerns and
                        self.config.block_on_safety_concern):
                        safety_blocked = True
                        logger.warning(f"Safety concern: {trace.safety_summary}")

            # Fingerprint periodically
            if (self.config.enable_fingerprinting and
                self._weave_count % self.config.fingerprint_every_n == 0):
                # Would need probe inputs here - skip for now
                pass

        except Exception as e:
            logger.error(f"Error in wrapped weave: {e}")
            raise

        analysis_time = (time.time() - start_time) * 1000

        return IntegrationResult(
            spacetime=spacetime,
            trace=trace,
            steering_applied=bool(self._active_steering),
            safety_blocked=safety_blocked,
            analysis_time_ms=analysis_time,
        )

    def _on_weave(self, spacetime: Any) -> None:
        """Callback for orchestrator weave completion."""
        if self.config.mode == IntegrationMode.DISABLED:
            return

        # Post-weave analysis would go here
        for callback in self._post_weave_callbacks:
            try:
                callback(spacetime)
            except Exception as e:
                logger.error(f"Post-weave callback error: {e}")

    def add_post_weave_callback(self, callback: Callable) -> None:
        """Add callback for post-weave events."""
        self._post_weave_callbacks.append(callback)

    # =========================================================================
    # Explanation & Reporting
    # =========================================================================

    def explain_decision(
        self,
        spacetime: Any,
        verbosity: int = 2,
    ) -> str:
        """
        Generate explanation for a weaving decision.

        Args:
            spacetime: Spacetime result from weaving
            verbosity: 1=brief, 2=moderate, 3=detailed

        Returns:
            Human-readable explanation
        """
        lines = []

        # Basic decision info
        if hasattr(spacetime, 'response'):
            lines.append(f"Response: {spacetime.response[:100]}...")
        if hasattr(spacetime, 'confidence'):
            lines.append(f"Confidence: {spacetime.confidence:.3f}")
        if hasattr(spacetime, 'tool'):
            lines.append(f"Tool: {spacetime.tool}")

        # Trace explanation
        if self._last_trace:
            lines.append("\n--- Interpretability Analysis ---")
            lines.append(self._last_trace.explain(verbosity))

        # Steering info
        if self._active_steering:
            lines.append("\n--- Active Steering ---")
            for fid, strength in self._active_steering.items():
                lines.append(f"  {fid}: {strength:+.2f}")

        return "\n".join(lines)

    def get_feature_explanation(self, feature_id: str) -> str:
        """Get explanation for a specific feature."""
        return self._engine.explain_feature(feature_id)

    # =========================================================================
    # Statistics & State
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        stats = self._engine.get_statistics()
        stats.update({
            "weave_count": self._weave_count,
            "trace_history_size": len(self._trace_history),
            "active_steering_count": len(self._active_steering),
            "fingerprints_count": len(self._fingerprints),
            "integration_mode": self.config.mode.name,
        })
        return stats

    def summary(self) -> str:
        """Generate integration summary."""
        lines = [
            "=" * 60,
            "Dark Trace Integration",
            "=" * 60,
            f"Mode: {self.config.mode.name}",
            f"Weaves analyzed: {self._weave_count}",
            f"Active steering: {len(self._active_steering)} features",
            f"Trace history: {len(self._trace_history)} entries",
            "",
            self._engine.summary(),
        ]
        return "\n".join(lines)

    def save_state(self, path: str) -> None:
        """Save integration state."""
        import os
        os.makedirs(path, exist_ok=True)

        # Save engine state
        self._engine.save_state(os.path.join(path, "engine"))

        # Save fingerprints
        import json
        fp_path = os.path.join(path, "fingerprints.json")
        fp_data = {
            model_id: {
                fid: fp.to_dict()
                for fid, fp in fingerprints.items()
            }
            for model_id, fingerprints in self._fingerprints.items()
        }
        with open(fp_path, 'w') as f:
            json.dump(fp_data, f, indent=2)

        logger.info(f"Integration state saved to {path}")

    def load_state(self, path: str) -> None:
        """Load integration state."""
        import os

        # Load engine state
        engine_path = os.path.join(path, "engine")
        if os.path.exists(engine_path):
            self._engine.load_state(engine_path)

        # Load fingerprints
        import json
        fp_path = os.path.join(path, "fingerprints.json")
        if os.path.exists(fp_path):
            with open(fp_path, 'r') as f:
                fp_data = json.load(f)

            self._fingerprints = {
                model_id: {
                    fid: FeatureFingerprint.from_dict(fp_dict)
                    for fid, fp_dict in fingerprints.items()
                }
                for model_id, fingerprints in fp_data.items()
            }

        logger.info(f"Integration state loaded from {path}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_integration(
    orchestrator: Optional[Any] = None,
    config: Optional[IntegrationConfig] = None,
    trace_config: Optional[TraceConfig] = None,
) -> DarkTraceIntegration:
    """
    Create Dark Trace integration with optional orchestrator attachment.

    Args:
        orchestrator: HoloLoom WeavingOrchestrator (optional)
        config: Integration configuration
        trace_config: Dark Trace engine configuration

    Returns:
        Configured DarkTraceIntegration
    """
    integration = DarkTraceIntegration(
        config=config,
        trace_config=trace_config,
    )

    if orchestrator is not None:
        integration.attach_to_orchestrator(orchestrator)

    return integration


def enable_dark_trace(
    orchestrator: Any,
    integration: Optional[DarkTraceIntegration] = None,
    mode: IntegrationMode = IntegrationMode.PASSIVE,
) -> DarkTraceIntegration:
    """
    Enable Dark Trace on an orchestrator.

    Args:
        orchestrator: HoloLoom WeavingOrchestrator
        integration: Existing integration (creates new if None)
        mode: Integration mode

    Returns:
        DarkTraceIntegration attached to orchestrator
    """
    if integration is None:
        config = IntegrationConfig(mode=mode)
        integration = DarkTraceIntegration(config=config)

    integration.attach_to_orchestrator(orchestrator)

    # Store reference on orchestrator
    orchestrator._dark_trace = integration

    return integration


def get_dark_trace(orchestrator: Any) -> Optional[DarkTraceIntegration]:
    """Get Dark Trace integration from orchestrator if attached."""
    return getattr(orchestrator, '_dark_trace', None)


# =============================================================================
# Convenience Decorators
# =============================================================================

def with_interpretability(
    mode: IntegrationMode = IntegrationMode.PASSIVE,
):
    """
    Decorator to add interpretability to weave method.

    Usage:
        @with_interpretability(mode=IntegrationMode.ACTIVE)
        async def weave(self, query):
            ...
    """
    def decorator(weave_method):
        async def wrapper(self, *args, **kwargs):
            # Get or create integration
            integration = get_dark_trace(self)
            if integration is None:
                integration = enable_dark_trace(self, mode=mode)

            # Wrap weave with interpretability
            return await integration.wrap_weave(
                lambda *a, **k: weave_method(self, *a, **k),
                *args,
                **kwargs,
            )
        return wrapper
    return decorator
