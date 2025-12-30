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

# Phase 2: Adversarial Probing imports
from HoloLoom.dark_trace.research.adversarial_prober import (
    AdversarialProber,
    ProbeConfig,
    VulnerabilityLevel,
    VulnerabilityReport,
    create_prober,
)
from HoloLoom.dark_trace.research.adversarial_catalog import (
    AdversarialCatalog,
    CatalogedVulnerability,
    VulnerabilityStatus,
    create_catalog,
)

# Phase 2: Visualization Dashboard imports
from HoloLoom.dark_trace.visualization import (
    DarkTraceStreamingServer,
    StreamingConfig,
    create_streaming_server,
    create_streaming_config,
    DarkTraceDashboardAPI,
    DashboardAPIConfig,
    create_dashboard_api,
    DarkTraceOrchestratorHook,
    HookConfig,
    create_orchestrator_hook,
    create_hook_config,
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

    # Phase 2: Adversarial Safety Probing
    enable_safety_probing: bool = False
    probe_config: Optional[ProbeConfig] = None
    vulnerability_threshold: VulnerabilityLevel = VulnerabilityLevel.MEDIUM
    probe_on_safety_concern: bool = True  # Run probes when safety flagged
    probe_every_n: int = 50  # Periodic probing every N weaves
    catalog_path: Optional[str] = None  # Path for vulnerability catalog
    block_on_vulnerability: bool = False  # Block if HIGH/CRITICAL found

    # Phase 2: Real-time Visualization Dashboard
    enable_visualization: bool = False
    visualization_streaming: bool = False  # Enable WebSocket streaming
    visualization_host: str = "localhost"
    visualization_port: int = 8765
    visualization_api_port: int = 8766
    visualization_throttle_ms: float = 50.0  # Broadcast throttle
    visualization_monitored_layers: Optional[List[int]] = None  # None = all
    visualization_activation_threshold: float = 0.01

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

    @classmethod
    def with_safety_probing(cls, catalog_path: Optional[str] = None) -> "IntegrationConfig":
        """Create config with adversarial safety probing enabled."""
        return cls(
            mode=IntegrationMode.FULL,
            auto_analyze=True,
            enable_steering=True,
            safety_monitoring=True,
            enable_safety_probing=True,
            probe_config=ProbeConfig.balanced(),
            catalog_path=catalog_path,
            block_on_vulnerability=True,
        )

    @classmethod
    def with_visualization(
        cls,
        host: str = "localhost",
        port: int = 8765,
        api_port: int = 8766,
        streaming: bool = True,
    ) -> "IntegrationConfig":
        """
        Create config with real-time visualization dashboard enabled.

        Args:
            host: WebSocket server host
            port: WebSocket server port
            api_port: REST API port
            streaming: Enable WebSocket streaming

        Returns:
            IntegrationConfig with visualization enabled
        """
        return cls(
            mode=IntegrationMode.ACTIVE,
            auto_analyze=True,
            enable_steering=True,
            safety_monitoring=True,
            enable_visualization=True,
            visualization_streaming=streaming,
            visualization_host=host,
            visualization_port=port,
            visualization_api_port=api_port,
        )


@dataclass
class IntegrationResult:
    """Result of integrated weave with interpretability."""

    spacetime: Any  # Original spacetime result
    trace: Optional[TraceResult] = None
    steering_applied: bool = False
    safety_blocked: bool = False
    analysis_time_ms: float = 0.0

    # Phase 2: Vulnerability probing results
    vulnerability_report: Optional[VulnerabilityReport] = None
    vulnerability_blocked: bool = False
    probing_performed: bool = False

    def explain(self, verbosity: int = 1) -> str:
        """Generate explanation including interpretability."""
        lines = []

        if self.trace:
            lines.append(self.trace.explain(verbosity))

        if self.steering_applied:
            lines.append("\n[Steering was applied]")

        if self.safety_blocked:
            lines.append("\n[BLOCKED: Safety concern detected]")

        if self.vulnerability_blocked:
            lines.append("\n[BLOCKED: High vulnerability detected]")

        if self.probing_performed and self.vulnerability_report:
            lines.append(f"\n[Probing: {self.vulnerability_report.vulnerability_level.name} vulnerability]")

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

        # Phase 2: Adversarial Probing
        self._prober: Optional[AdversarialProber] = None
        self._catalog: Optional[AdversarialCatalog] = None
        self._probe_count = 0
        self._vulnerability_history: List[VulnerabilityReport] = []

        if self.config.enable_safety_probing:
            self._initialize_probing()

        # Phase 2: Real-time Visualization Dashboard
        self._streaming_server: Optional[DarkTraceStreamingServer] = None
        self._dashboard_api: Optional[DarkTraceDashboardAPI] = None
        self._orchestrator_hook: Optional[DarkTraceOrchestratorHook] = None
        self._visualization_running: bool = False

        if self.config.enable_visualization:
            self._initialize_visualization()

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

        # Phase 2: Probing state
        vulnerability_report = None
        vulnerability_blocked = False
        probing_performed = False

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
                    has_safety_concern = False
                    if self.config.safety_monitoring and trace.has_safety_concerns:
                        has_safety_concern = True
                        if self.config.block_on_safety_concern:
                            safety_blocked = True
                            logger.warning(f"Safety concern: {trace.safety_summary}")

                    # Phase 2: Adversarial probing
                    if self._should_probe(has_safety_concern):
                        try:
                            probing_performed = True
                            acts_np = acts.numpy() if TORCH_AVAILABLE else acts
                            vulnerability_report = self.probe_for_vulnerabilities(
                                activations=acts_np,
                            )

                            # Check if vulnerability should block
                            if self._is_vulnerability_blocking(vulnerability_report):
                                vulnerability_blocked = True
                                logger.warning(
                                    f"Vulnerability blocked: {vulnerability_report.vulnerability_level.name} "
                                    f"(score: {vulnerability_report.max_vulnerability_score:.3f})"
                                )
                        except Exception as probe_error:
                            logger.error(f"Probing error: {probe_error}")

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
            vulnerability_report=vulnerability_report,
            vulnerability_blocked=vulnerability_blocked,
            probing_performed=probing_performed,
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
    # Phase 2: Adversarial Safety Probing
    # =========================================================================

    def _initialize_probing(self) -> None:
        """Initialize adversarial probing components."""
        # Create prober with config
        probe_config = self.config.probe_config or ProbeConfig.balanced()
        self._prober = create_prober(config=probe_config)

        # Create catalog for persistence
        if self.config.catalog_path:
            self._catalog = create_catalog(self.config.catalog_path)
        else:
            self._catalog = create_catalog()  # In-memory catalog

        logger.info("Adversarial safety probing initialized")

    def probe_for_vulnerabilities(
        self,
        activations: Union[np.ndarray, "torch.Tensor"],
        feature_indices: Optional[List[int]] = None,
        model_version: str = "current",
    ) -> VulnerabilityReport:
        """
        Probe activations for adversarial vulnerabilities.

        Args:
            activations: Model activations to probe
            feature_indices: Specific feature indices to probe (None = all)
            model_version: Version tag for cataloging

        Returns:
            VulnerabilityReport with vulnerability assessment
        """
        if self._prober is None:
            raise RuntimeError("Safety probing not enabled. Set enable_safety_probing=True")

        # Convert to numpy if needed
        if TORCH_AVAILABLE and hasattr(activations, 'detach'):
            activations = activations.detach().cpu().numpy()

        # Probe the activations
        report = self._prober.probe(
            activations=activations,
            feature_indices=feature_indices,
        )

        self._probe_count += 1
        self._vulnerability_history.append(report)

        # Catalog if significant vulnerability found
        if (self._catalog is not None and
            report.vulnerability_level.value >= self.config.vulnerability_threshold.value):

            for idx, score in report.vulnerability_scores.items():
                self._catalog.add_vulnerability(
                    feature_idx=idx,
                    model_version=model_version,
                    vulnerability_score=score,
                    vulnerability_level=report.vulnerability_level.name,
                    attack_method=report.attack_method,
                    epsilon=report.epsilon,
                    best_perturbation=report.best_perturbations.get(idx),
                    activation_before=report.activations_before.get(idx, 0.0),
                    activation_after=report.activations_after.get(idx, 0.0),
                )

        return report

    def get_vulnerability_report(
        self,
        model_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get vulnerability report from catalog.

        Args:
            model_version: Filter by model version (None = all)

        Returns:
            Dict with vulnerability summary and details
        """
        if self._catalog is None:
            return {"error": "No catalog available"}

        summary = self._catalog.get_summary()

        vulnerabilities = self._catalog.get_all()
        if model_version:
            vulnerabilities = self._catalog.get_by_version(model_version)

        # Get active critical vulnerabilities
        active_critical = self._catalog.get_active_critical()

        return {
            "summary": {
                "total": summary.total_vulnerabilities,
                "active": summary.active_count,
                "mitigated": summary.mitigated_count,
                "regressed": summary.regressed_count,
                "avg_score": summary.avg_vulnerability_score,
            },
            "by_level": summary.by_level,
            "by_method": summary.by_method,
            "most_vulnerable_features": summary.most_vulnerable_features,
            "active_critical_count": len(active_critical),
            "vulnerabilities_count": len(vulnerabilities),
            "last_updated": summary.last_updated,
        }

    def compare_vulnerability_versions(
        self,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """
        Compare vulnerabilities between two model versions.

        Args:
            version1: First version
            version2: Second version

        Returns:
            Comparison report with regressions and improvements
        """
        if self._catalog is None:
            return {"error": "No catalog available"}

        report = self._catalog.compare_versions(version1, version2)

        return {
            "from_version": report.from_version,
            "to_version": report.to_version,
            "summary": report.summary,
            "new_vulnerabilities": len(report.new_vulnerabilities),
            "fixed_vulnerabilities": len(report.fixed_vulnerabilities),
            "regressed_vulnerabilities": len(report.regressed_vulnerabilities),
            "regression_detected": report.summary.get("regression_detected", False),
        }

    def _should_probe(self, has_safety_concern: bool) -> bool:
        """Determine if probing should be performed."""
        if not self.config.enable_safety_probing:
            return False

        # Probe on safety concern
        if has_safety_concern and self.config.probe_on_safety_concern:
            return True

        # Periodic probing
        if self._weave_count % self.config.probe_every_n == 0:
            return True

        return False

    def _is_vulnerability_blocking(self, report: VulnerabilityReport) -> bool:
        """Check if vulnerability should block the weave."""
        if not self.config.block_on_vulnerability:
            return False

        # Block on HIGH or CRITICAL
        return report.vulnerability_level in (
            VulnerabilityLevel.HIGH,
            VulnerabilityLevel.CRITICAL,
        )

    # =========================================================================
    # Phase 2: Real-time Visualization Dashboard
    # =========================================================================

    def _initialize_visualization(self) -> None:
        """Initialize visualization components."""
        # Create streaming config from integration config
        streaming_config = create_streaming_config(
            preset="standard",
            host=self.config.visualization_host,
            port=self.config.visualization_port,
            activation_throttle_ms=self.config.visualization_throttle_ms,
        )

        # Create streaming server
        self._streaming_server = create_streaming_server(
            config=streaming_config,
            engine=self._engine,
        )

        # Create dashboard API
        api_config = DashboardAPIConfig(
            host=self.config.visualization_host,
            port=self.config.visualization_api_port,
        )
        self._dashboard_api = create_dashboard_api(
            config=api_config,
            engine=self._engine,
            streaming_server=self._streaming_server,
        )

        # Create orchestrator hook
        hook_config = create_hook_config(
            preset="standard",
            monitored_layers=self.config.visualization_monitored_layers,
            activation_threshold=self.config.visualization_activation_threshold,
            broadcast_throttle_ms=self.config.visualization_throttle_ms,
        )
        self._orchestrator_hook = create_orchestrator_hook(
            engine=self._engine,
            streaming_server=self._streaming_server,
            enable_streaming=self.config.visualization_streaming,
            **{k: v for k, v in hook_config.__dict__.items() if k not in ['enable_streaming']}
        )

        logger.info(
            f"Visualization dashboard initialized "
            f"(host={self.config.visualization_host}, "
            f"ws_port={self.config.visualization_port}, "
            f"api_port={self.config.visualization_api_port})"
        )

    async def start_visualization(self) -> None:
        """
        Start the visualization dashboard servers.

        Starts:
        - WebSocket server for real-time streaming
        - REST API for dashboard data

        Call this before weaving to enable real-time visualization.
        """
        if not self.config.enable_visualization:
            raise RuntimeError("Visualization not enabled. Use IntegrationConfig.with_visualization()")

        if self._visualization_running:
            logger.warning("Visualization already running")
            return

        # Start streaming server
        if self._streaming_server and self.config.visualization_streaming:
            await self._streaming_server.start()
            logger.info(
                f"WebSocket streaming server started on "
                f"ws://{self.config.visualization_host}:{self.config.visualization_port}"
            )

        # Start dashboard API
        if self._dashboard_api:
            await self._dashboard_api.start()
            logger.info(
                f"Dashboard API started on "
                f"http://{self.config.visualization_host}:{self.config.visualization_api_port}"
            )

        self._visualization_running = True
        logger.info("Visualization dashboard started")

    async def stop_visualization(self) -> None:
        """Stop the visualization dashboard servers."""
        if not self._visualization_running:
            return

        # Stop streaming server
        if self._streaming_server:
            await self._streaming_server.stop()

        # Stop dashboard API
        if self._dashboard_api:
            await self._dashboard_api.stop()

        self._visualization_running = False
        logger.info("Visualization dashboard stopped")

    @property
    def orchestrator_hook(self) -> Optional[DarkTraceOrchestratorHook]:
        """Access the orchestrator hook for weaving integration."""
        return self._orchestrator_hook

    @property
    def streaming_server(self) -> Optional[DarkTraceStreamingServer]:
        """Access the streaming server."""
        return self._streaming_server

    @property
    def dashboard_api(self) -> Optional[DarkTraceDashboardAPI]:
        """Access the dashboard API."""
        return self._dashboard_api

    @property
    def visualization_running(self) -> bool:
        """Check if visualization is currently running."""
        return self._visualization_running

    async def broadcast_activations(
        self,
        layer: int,
        feature_indices: List[int],
        activation_values: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Broadcast activations to connected visualization clients.

        Args:
            layer: Layer index
            feature_indices: Indices of activated features
            activation_values: Activation values
            metadata: Optional metadata
        """
        if not self._visualization_running or not self._streaming_server:
            return

        if self._orchestrator_hook:
            await self._orchestrator_hook.on_activations(
                layer=layer,
                feature_indices=feature_indices,
                activation_values=activation_values,
                metadata=metadata,
            )

    def get_visualization_statistics(self) -> Dict[str, Any]:
        """Get visualization system statistics."""
        stats = {
            "enabled": self.config.enable_visualization,
            "running": self._visualization_running,
            "streaming_enabled": self.config.visualization_streaming,
            "host": self.config.visualization_host,
            "websocket_port": self.config.visualization_port,
            "api_port": self.config.visualization_api_port,
        }

        if self._orchestrator_hook:
            hook_stats = self._orchestrator_hook.get_statistics()
            stats["hook"] = hook_stats

        if self._streaming_server:
            stats["server"] = {
                "connected_clients": getattr(self._streaming_server, 'connected_clients', 0),
            }

        return stats

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

        # Phase 2: Probing statistics
        if self.config.enable_safety_probing:
            stats["probing"] = {
                "probe_count": self._probe_count,
                "vulnerability_history_size": len(self._vulnerability_history),
                "catalog_available": self._catalog is not None,
            }

            if self._catalog:
                catalog_summary = self._catalog.get_summary()
                stats["probing"]["catalog"] = {
                    "total_vulnerabilities": catalog_summary.total_vulnerabilities,
                    "active_count": catalog_summary.active_count,
                    "mitigated_count": catalog_summary.mitigated_count,
                    "regressed_count": catalog_summary.regressed_count,
                    "avg_score": catalog_summary.avg_vulnerability_score,
                }

            # Recent vulnerability level distribution
            if self._vulnerability_history:
                recent = self._vulnerability_history[-10:]
                level_counts = {}
                for report in recent:
                    level = report.vulnerability_level.name
                    level_counts[level] = level_counts.get(level, 0) + 1
                stats["probing"]["recent_levels"] = level_counts

        # Phase 2: Visualization statistics
        if self.config.enable_visualization:
            stats["visualization"] = self.get_visualization_statistics()

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
        ]

        # Phase 2: Probing summary
        if self.config.enable_safety_probing:
            lines.append("")
            lines.append("--- Adversarial Safety Probing ---")
            lines.append(f"Probes performed: {self._probe_count}")
            lines.append(f"Vulnerability history: {len(self._vulnerability_history)} entries")

            if self._catalog:
                catalog_summary = self._catalog.get_summary()
                lines.append(f"Catalog total: {catalog_summary.total_vulnerabilities}")
                lines.append(f"  Active: {catalog_summary.active_count}")
                lines.append(f"  Mitigated: {catalog_summary.mitigated_count}")
                lines.append(f"  Regressed: {catalog_summary.regressed_count}")
                if catalog_summary.avg_vulnerability_score > 0:
                    lines.append(f"  Avg score: {catalog_summary.avg_vulnerability_score:.3f}")
                if catalog_summary.most_vulnerable_features:
                    lines.append(f"  Most vulnerable: {catalog_summary.most_vulnerable_features[:5]}")

        # Phase 2: Visualization summary
        if self.config.enable_visualization:
            lines.append("")
            lines.append("--- Real-time Visualization ---")
            lines.append(f"Enabled: {self.config.enable_visualization}")
            lines.append(f"Running: {self._visualization_running}")
            if self._visualization_running:
                lines.append(f"Streaming: ws://{self.config.visualization_host}:{self.config.visualization_port}")
                lines.append(f"Dashboard API: http://{self.config.visualization_host}:{self.config.visualization_api_port}")

            if self._orchestrator_hook:
                hook_stats = self._orchestrator_hook.get_statistics()
                lines.append(f"Total broadcasts: {hook_stats.get('total_broadcasts', 0)}")
                lines.append(f"Activations recorded: {hook_stats.get('total_activations_recorded', 0)}")

        lines.append("")
        lines.append(self._engine.summary())

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
