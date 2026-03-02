"""
Dark Trace Engine: Main API for Interpretability and Responsibility

The DarkTraceEngine orchestrates multiple interpretability lenses
(SAE, Semantic, Domain) to provide unified analysis, steering, and explanation.

Core Operations:
- analyze(activations) → TraceResult
- explain(activations, verbosity) → str
- steer(goals) → SteeringVector
- ablate(feature_ids) → AblationResult
- discover_circuits(input, output) → CircuitTrace

Plugin System (Phase 11):
- load_plugins() → Load external plugins with safety validation
- get_plugin() → Access registered plugins
- list_plugins() → List all registered plugins

This is the FOUNDATION of HoloLoom's interpretability system.

"Dark Trace = Responsibility + Interpretability"

Date: 2025-12-22
Phase: 11.8 - Engine Integration
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import time
import logging

import torch

from hololoom.dark_trace.protocol import (
    TraceLens,
    BaseLens,
    Feature,
    FeatureActivation,
    SteeringVector,
    LensType,
    FeatureSource,
    SafetyFlag,
)
from hololoom.dark_trace.result import (
    TraceResult,
    LensResult,
    SteeringResult,
    AblationResult,
    InjectionResult,
    PatchResult,
    CircuitTrace,
    AnalysisStatus,
    SteeringOutcome,
)
from hololoom.dark_trace.registry import FeatureRegistry, create_sae_features
from hololoom.dark_trace.trace_config import TraceConfig, TraceMode

# Plugin system imports (Phase 11)
from hololoom.dark_trace.plugins.safety_gate import (
    PluginSafetyGate,
    TrustLevel,
    PluginCapability,
)
from hololoom.dark_trace.plugins.alignment_bridge import (
    PluginAlignmentBridge,
)
from hololoom.dark_trace.plugins.registry import PluginRegistry
from hololoom.dark_trace.plugins.loader import (
    PluginLoader,
    LoaderConfig,
    BulkLoadResult,
)
from hololoom.dark_trace.plugins.interface import (
    DarkTracePlugin,
    PluginMetadata,
    MonitorPlugin,
)


logger = logging.getLogger(__name__)


class DarkTraceEngine:
    """
    Main API for Dark Trace interpretability system.

    Orchestrates multiple lenses for unified analysis:
    - SAE Lens: Learned sparse features
    - Semantic Lens: 244 interpretable axes
    - Domain Lens: Domain-specific features

    Usage:
        engine = DarkTraceEngine(config=TraceConfig.standard())
        engine.register_lens(sae_lens)
        engine.register_lens(semantic_lens)

        result = engine.analyze(activations)
        explanation = engine.explain(activations, verbosity=2)
        steering = engine.steer({"semantic.Warmth": 1.5, "sae.42": -0.5})
    """

    def __init__(
        self,
        config: Optional[TraceConfig] = None,
        enable_plugins: bool = True,
        auto_load_builtins: bool = True,
    ):
        """
        Initialize DarkTraceEngine.

        Args:
            config: Configuration (defaults to TraceConfig.standard())
            enable_plugins: Enable plugin system (Phase 11)
            auto_load_builtins: Auto-load built-in plugins
        """
        self.config = config or TraceConfig.standard()

        # Lens registry: {LensType: TraceLens}
        self._lenses: Dict[LensType, TraceLens] = {}

        # Feature registry (unified namespace)
        self.registry = FeatureRegistry(
            enable_history=self.config.enable_history,
            history_size=self.config.history_size,
        )

        # Analysis counter for correlation updates
        self._analysis_count = 0

        # Plugin system (Phase 11)
        self._plugins_enabled = enable_plugins
        self._alignment_bridge: Optional[PluginAlignmentBridge] = None
        self._safety_gate: Optional[PluginSafetyGate] = None
        self._plugin_registry: Optional[PluginRegistry] = None
        self._plugin_loader: Optional[PluginLoader] = None

        if enable_plugins:
            self._init_plugin_system(auto_load_builtins)

        # Logging
        self._setup_logging()

        logger.info(f"DarkTraceEngine initialized (mode={self.config.mode.name}, plugins={enable_plugins})")

    def _init_plugin_system(self, auto_load_builtins: bool = True) -> None:
        """Initialize the plugin system with safety-first architecture."""
        # 1. Create alignment bridge (auditing layer)
        self._alignment_bridge = PluginAlignmentBridge()

        # 2. Create safety gate (validation layer)
        # Note: SafetyGate uses guardrails (optional), not alignment_bridge
        # The alignment_bridge is used by PluginRegistry for auditing
        self._safety_gate = PluginSafetyGate()

        # 3. Create plugin registry
        self._plugin_registry = PluginRegistry(
            safety_gate=self._safety_gate,
            alignment_bridge=self._alignment_bridge,
        )

        # 4. Create plugin loader
        # Note: PluginLoader requires engine reference for plugin initialization context
        self._plugin_loader = PluginLoader(
            engine=self,  # Pass self as the engine reference
            safety_gate=self._safety_gate,
            alignment_bridge=self._alignment_bridge,
            registry=self._plugin_registry,
            config=LoaderConfig(
                sandbox_init=True,
                init_timeout_seconds=5.0,
                verify_behavior=True,
            ),
        )

        # 5. Auto-load built-in plugins if requested
        if auto_load_builtins:
            self._load_builtin_plugins()

        logger.info("Plugin system initialized with safety-first architecture")

    def _load_builtin_plugins(self) -> None:
        """Load built-in CORE and TRUSTED plugins."""
        from hololoom.dark_trace.plugins.builtin import (
            SafetyMonitorPlugin,
            AlignmentValidatorPlugin,
            MetricsExporterPlugin,
            get_auto_load_plugins,
        )

        if not self._plugin_registry:
            return

        # Load auto-load plugins (CORE level)
        for name, plugin_class in get_auto_load_plugins():
            try:
                plugin = plugin_class()

                # Built-in plugins bypass normal safety gate
                # They're registered directly with CORE trust
                asyncio.create_task(
                    self._register_builtin_plugin(plugin, TrustLevel.CORE)
                )

                logger.info(f"Auto-loaded built-in plugin: {name}")

            except Exception as e:
                logger.error(f"Failed to load built-in plugin {name}: {e}")

    async def _register_builtin_plugin(
        self,
        plugin: DarkTracePlugin,
        trust_level: TrustLevel,
    ) -> None:
        """Register a built-in plugin with specified trust level."""
        if not self._plugin_registry or not self._safety_gate:
            return

        try:
            # Initialize plugin
            await plugin.initialize(self, self._safety_gate)

            # Register with specified trust level
            await self._plugin_registry.register(
                plugin,
                trust_level=trust_level,
                skip_safety_check=True,  # Built-ins are trusted
            )

            # Audit registration
            if self._alignment_bridge:
                await self._alignment_bridge.audit_plugin_registration(
                    plugin=plugin,
                    trust_level=trust_level,
                )

        except Exception as e:
            logger.error(f"Failed to register built-in plugin {plugin.metadata.name}: {e}")

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        if not self.config.logging.enabled:
            return

        level = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
            3: logging.DEBUG,
        }.get(self.config.logging.verbosity, logging.INFO)

        logging.basicConfig(level=level)

        if self.config.logging.log_to_file and self.config.logging.log_path:
            handler = logging.FileHandler(self.config.logging.log_path)
            handler.setLevel(level)
            logger.addHandler(handler)

    # =========================================================================
    # Lens Management
    # =========================================================================

    def register_lens(self, lens: TraceLens) -> None:
        """
        Register an interpretability lens.

        Args:
            lens: TraceLens implementation
        """
        lens_type = lens.lens_type
        self._lenses[lens_type] = lens

        # Register lens features in unified registry
        if isinstance(lens, BaseLens):
            for feature in lens.list_features():
                self.registry.register(feature)

        logger.info(f"Registered lens: {lens_type.name} ({lens.feature_dim} features)")

    def get_lens(self, lens_type: LensType) -> Optional[TraceLens]:
        """Get a registered lens by type."""
        return self._lenses.get(lens_type)

    def list_lenses(self) -> List[LensType]:
        """List all registered lens types."""
        return list(self._lenses.keys())

    def _get_active_lenses(self) -> Dict[LensType, TraceLens]:
        """Get lenses that are enabled in config."""
        active = {}

        if self.config.sae.enabled and LensType.SAE in self._lenses:
            active[LensType.SAE] = self._lenses[LensType.SAE]

        if self.config.semantic.enabled and LensType.SEMANTIC in self._lenses:
            active[LensType.SEMANTIC] = self._lenses[LensType.SEMANTIC]

        if self.config.domain.enabled and LensType.DOMAIN in self._lenses:
            active[LensType.DOMAIN] = self._lenses[LensType.DOMAIN]

        return active

    # =========================================================================
    # Core Analysis
    # =========================================================================

    def analyze(
        self,
        activations: torch.Tensor,
        layer_name: Optional[str] = None,
        include_safety: bool = True,
    ) -> TraceResult:
        """
        Analyze activations through all active lenses.

        This is the primary analysis entry point.

        Args:
            activations: Model activations [batch, seq, dim] or [batch, dim] or [dim]
            layer_name: Optional layer name for provenance
            include_safety: Include safety analysis

        Returns:
            TraceResult with complete analysis from all lenses
        """
        trace_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        logger.debug(f"Starting analysis {trace_id} (shape={activations.shape})")

        # Ensure tensor on CPU for analysis
        if activations.is_cuda:
            activations = activations.cpu()

        # Get active lenses
        active_lenses = self._get_active_lenses()

        if not active_lenses:
            logger.warning("No active lenses registered")
            return TraceResult.create_failed(trace_id, "No active lenses")

        # Run analysis through each lens
        lens_results: Dict[LensType, LensResult] = {}
        all_features: List[FeatureActivation] = []
        sae_activations: Dict[str, float] = {}
        semantic_activations: Dict[str, float] = {}

        for lens_type, lens in active_lenses.items():
            lens_start = time.time()

            try:
                # Project activations
                projections = lens.project(activations)

                # Get active features
                active_features = lens.get_active(
                    activations,
                    threshold=self._get_threshold(lens_type),
                    top_k=self._get_top_k(lens_type),
                )

                # Generate explanation
                explanation = lens.explain(activations, verbosity=self.config.logging.verbosity)

                # Compute lens-specific metrics
                metrics = self._compute_lens_metrics(lens_type, projections)

                lens_result = LensResult(
                    lens_type=lens_type,
                    lens_name=lens.name if hasattr(lens, 'name') else lens_type.name,
                    active_features=active_features,
                    feature_projections=projections,
                    explanation=explanation,
                    reconstruction_loss=metrics.get("reconstruction_loss"),
                    sparsity=metrics.get("sparsity"),
                    coherence_score=metrics.get("coherence"),
                    compute_time_ms=(time.time() - lens_start) * 1000,
                    status=AnalysisStatus.SUCCESS,
                )

                lens_results[lens_type] = lens_result
                all_features.extend(active_features)

                # Collect for correlation tracking
                for feat in active_features:
                    if feat.feature.source == FeatureSource.SAE:
                        sae_activations[feat.feature.id] = feat.activation
                    elif feat.feature.source == FeatureSource.SEMANTIC:
                        semantic_activations[feat.feature.id] = feat.activation

                    # Record in registry
                    self.registry.record_activation(feat.feature.id, feat.activation)

            except Exception as e:
                logger.error(f"Lens {lens_type.name} failed: {e}")
                lens_results[lens_type] = LensResult(
                    lens_type=lens_type,
                    lens_name=lens_type.name,
                    active_features=[],
                    status=AnalysisStatus.FAILED,
                    error_message=str(e),
                )

        # Update correlations
        self._analysis_count += 1
        if (self.config.correlation.enabled and
            sae_activations and semantic_activations and
            self._analysis_count % self.config.correlation.update_every_n_analyses == 0):
            self.registry.update_correlations(sae_activations, semantic_activations)

        # Sort all features by activation
        all_features.sort(reverse=True)
        dominant_features = all_features[:10]

        # Safety analysis
        safety_flags = {}
        has_safety_concerns = False
        safety_summary = None

        if include_safety and self.config.safety.enabled:
            safety_flags, has_safety_concerns, safety_summary = self._safety_analysis(all_features)

        # Build SAE ↔ Semantic correlations for result
        sae_semantic_correlations = {}
        if self.config.correlation.enabled:
            for sae_id in sae_activations.keys():
                meanings = self.registry.correlations.get_semantic_meanings(sae_id, k=3)
                if meanings:
                    sae_semantic_correlations[sae_id] = meanings

        # Determine overall status
        successful = [r for r in lens_results.values() if r.status == AnalysisStatus.SUCCESS]
        if len(successful) == len(lens_results):
            status = AnalysisStatus.SUCCESS
        elif successful:
            status = AnalysisStatus.PARTIAL
        else:
            status = AnalysisStatus.FAILED

        result = TraceResult(
            trace_id=trace_id,
            timestamp=datetime.now(),
            input_shape=tuple(activations.shape),
            input_dtype=str(activations.dtype),
            layer_name=layer_name,
            lens_results=lens_results,
            all_features=all_features,
            dominant_features=dominant_features,
            safety_flags=safety_flags,
            has_safety_concerns=has_safety_concerns,
            safety_summary=safety_summary,
            sae_semantic_correlations=sae_semantic_correlations,
            status=status,
            total_time_ms=(time.time() - start_time) * 1000,
            config_used={"mode": self.config.mode.name},
            lenses_used=[lt.name for lt in lens_results.keys()],
        )

        logger.info(
            f"Analysis {trace_id} complete: {len(all_features)} features, "
            f"{result.total_time_ms:.2f}ms, safety={has_safety_concerns}"
        )

        return result

    def _get_threshold(self, lens_type: LensType) -> float:
        """Get activation threshold for lens type."""
        if lens_type == LensType.SAE:
            return self.config.sae.activation_threshold
        return 0.1  # Default

    def _get_top_k(self, lens_type: LensType) -> int:
        """Get top-k features for lens type."""
        if lens_type == LensType.SAE:
            return self.config.sae.top_k_features
        return 20  # Default

    def _compute_lens_metrics(
        self,
        lens_type: LensType,
        projections: torch.Tensor
    ) -> Dict[str, float]:
        """Compute lens-specific metrics."""
        metrics = {}

        # Sparsity (fraction of non-zero activations)
        if projections is not None:
            nonzero = (projections.abs() > 0.01).float().mean().item()
            metrics["sparsity"] = 1.0 - nonzero  # Higher = more sparse

        return metrics

    def _safety_analysis(
        self,
        features: List[FeatureActivation]
    ) -> tuple:
        """
        Perform safety analysis on features.

        Returns:
            (safety_flags, has_concerns, summary)
        """
        safety_flags = {}
        concerns = []

        for feat in features:
            feature = feat.feature
            activation = feat.activation

            # Check forbidden features
            if feature.id in self.config.safety.forbidden_features:
                safety_flags[feature.id] = SafetyFlag.BLOCKED
                concerns.append(f"FORBIDDEN: {feature.id} active at {activation:.3f}")
                continue

            # Check dangerous threshold
            if activation > self.config.safety.dangerous_activation_threshold:
                if feature.safety_flag in [SafetyFlag.DANGEROUS, SafetyFlag.SENSITIVE]:
                    safety_flags[feature.id] = feature.safety_flag
                    concerns.append(f"{feature.safety_flag.name}: {feature.id} at {activation:.3f}")

            # Check existing safety flag
            if feature.safety_flag != SafetyFlag.SAFE:
                safety_flags[feature.id] = feature.safety_flag

        # Check required features
        active_ids = {f.feature.id for f in features if f.activation > 0.1}
        missing_required = self.config.safety.required_features - active_ids
        for required_id in missing_required:
            concerns.append(f"MISSING REQUIRED: {required_id}")

        has_concerns = len(concerns) > 0
        summary = "; ".join(concerns) if concerns else None

        return safety_flags, has_concerns, summary

    # =========================================================================
    # Explanation
    # =========================================================================

    def explain(
        self,
        activations: torch.Tensor,
        verbosity: int = 1,
        focus_features: Optional[List[str]] = None,
    ) -> str:
        """
        Generate human-readable explanation of activations.

        Args:
            activations: Model activations
            verbosity: 1=brief, 2=moderate, 3=detailed
            focus_features: Only explain these features (optional)

        Returns:
            Human-readable explanation string
        """
        result = self.analyze(activations)

        if focus_features:
            # Filter to focus features
            focused_result = TraceResult(
                trace_id=result.trace_id,
                dominant_features=[
                    f for f in result.all_features
                    if f.feature.id in focus_features
                ],
                **{k: v for k, v in result.__dict__.items()
                   if k not in ['trace_id', 'dominant_features']}
            )
            return focused_result.explain(verbosity)

        return result.explain(verbosity)

    def explain_feature(self, feature_id: str) -> str:
        """
        Generate explanation for a specific feature.

        Uses SAE ↔ Semantic correlations to provide meaning.
        """
        feature = self.registry.get(feature_id)
        if not feature:
            return f"Unknown feature: {feature_id}"

        lines = [f"Feature: {feature_id}"]

        if feature.label:
            lines.append(f"Label: {feature.label}")

        if feature.description:
            lines.append(f"Description: {feature.description}")

        # Statistics
        stats = self.registry.get_activation_stats(feature_id)
        if stats["count"] > 0:
            lines.append(f"Activations: {stats['count']} (mean={stats['mean']:.3f}, std={stats['std']:.3f})")

        # Safety
        lines.append(f"Safety: {feature.safety_flag.name}")

        # Semantic meanings (for SAE features)
        if feature.source == FeatureSource.SAE:
            meanings = self.registry.correlations.get_semantic_meanings(feature_id, k=3)
            if meanings:
                lines.append("Semantic meanings:")
                for name, corr in meanings:
                    lines.append(f"  → {name}: {corr:+.2f}")

        return "\n".join(lines)

    # =========================================================================
    # Steering
    # =========================================================================

    def steer(
        self,
        goals: Dict[str, float],
        validate_safety: bool = True,
    ) -> SteeringResult:
        """
        Generate steering vector for goals.

        Args:
            goals: {feature_id: target_strength}
            validate_safety: Check safety before allowing steering

        Returns:
            SteeringResult with steering vector and validation info
        """
        features_applied = []
        features_blocked = []
        vectors = []

        for feature_id, strength in goals.items():
            # Safety check
            if validate_safety:
                feature = self.registry.get(feature_id)
                if feature and feature.safety_flag in [SafetyFlag.BLOCKED, SafetyFlag.DANGEROUS]:
                    features_blocked.append(feature_id)
                    logger.warning(f"Steering blocked for {feature_id} (safety={feature.safety_flag.name})")
                    continue

            # Find lens that owns this feature
            lens = self._find_lens_for_feature(feature_id)
            if not lens:
                logger.warning(f"No lens found for feature {feature_id}")
                continue

            try:
                steering = lens.steer({feature_id: strength})
                vectors.append(steering.vector)
                features_applied.append(feature_id)
            except Exception as e:
                logger.error(f"Steering failed for {feature_id}: {e}")

        if not vectors:
            return SteeringResult(
                steering_vector=SteeringVector(
                    vector=torch.zeros(self.config.input_dim),
                    features={},
                    source=LensType.SAE,
                ),
                target_features=goals,
                outcome=SteeringOutcome.REJECTED if features_blocked else SteeringOutcome.PARTIAL,
                features_blocked=features_blocked,
            )

        # Combine vectors
        combined = torch.stack(vectors).sum(dim=0)

        steering_vector = SteeringVector(
            vector=combined,
            features={fid: goals[fid] for fid in features_applied},
            source=LensType.SAE,  # Primary source
            target_description=f"Steering toward {len(features_applied)} features",
            safety_validated=validate_safety,
        )

        outcome = SteeringOutcome.VALIDATED if validate_safety else SteeringOutcome.APPLIED
        if features_blocked:
            outcome = SteeringOutcome.PARTIAL

        return SteeringResult(
            steering_vector=steering_vector,
            target_features=goals,
            outcome=outcome,
            features_applied=features_applied,
            features_blocked=features_blocked,
            safety_validated=validate_safety,
        )

    def _find_lens_for_feature(self, feature_id: str) -> Optional[TraceLens]:
        """Find the lens that owns a feature."""
        feature = self.registry.get(feature_id)
        if not feature:
            return None

        source_to_lens = {
            FeatureSource.SAE: LensType.SAE,
            FeatureSource.SEMANTIC: LensType.SEMANTIC,
            FeatureSource.DOMAIN: LensType.DOMAIN,
        }

        lens_type = source_to_lens.get(feature.source)
        return self._lenses.get(lens_type)

    # =========================================================================
    # Causal Validation
    # =========================================================================

    def ablate(
        self,
        activations: torch.Tensor,
        feature_ids: List[str],
    ) -> AblationResult:
        """
        Ablate (remove) features from activations.

        Args:
            activations: Original activations
            feature_ids: Features to ablate

        Returns:
            AblationResult with impact measurements
        """
        if not self.config.causal.enabled:
            logger.warning("Causal validation disabled")
            return AblationResult(ablated_features=feature_ids)

        modified = activations.clone()

        for feature_id in feature_ids:
            lens = self._find_lens_for_feature(feature_id)
            if not lens or not hasattr(lens, 'ablate'):
                continue

            # Project to find feature direction, then remove
            # This is a simplified ablation - full implementation would use proper CausalValidator
            projections = lens.project(activations)
            feature = self.registry.get(feature_id)
            if feature:
                # Zero out the feature direction (simplified)
                modified = modified  # Placeholder - full implementation needed

        return AblationResult(
            ablated_features=feature_ids,
            original_activations=activations,
            modified_activations=modified,
        )

    # =========================================================================
    # Circuit Discovery
    # =========================================================================

    def discover_circuits(
        self,
        input_activations: torch.Tensor,
        output_activations: torch.Tensor,
    ) -> CircuitTrace:
        """
        Discover circuits from input to output.

        This is a placeholder for Phase 6 Circuit Discovery implementation.

        Args:
            input_activations: Activations at input
            output_activations: Activations at output

        Returns:
            CircuitTrace (placeholder)
        """
        logger.info("Circuit discovery not yet implemented (Phase 6)")
        return CircuitTrace(
            input_description="input",
            output_description="output",
        )

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_state(self, path: str) -> None:
        """
        Save engine state (registry, correlations).

        Args:
            path: Directory path for state files
        """
        import os
        os.makedirs(path, exist_ok=True)

        # Save registry
        self.registry.save(os.path.join(path, "registry.json"))

        # Compute and save final correlations
        if self.config.correlation.enabled:
            self.registry.compute_correlations(self.config.correlation.min_samples)

        logger.info(f"Engine state saved to {path}")

    def load_state(self, path: str) -> None:
        """
        Load engine state.

        Args:
            path: Directory path containing state files
        """
        import os

        registry_path = os.path.join(path, "registry.json")
        if os.path.exists(registry_path):
            self.registry = FeatureRegistry.load(registry_path)
            logger.info(f"Loaded registry from {registry_path}")

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "analyses_performed": self._analysis_count,
            "lenses_registered": [lt.name for lt in self._lenses.keys()],
            "features_registered": len(self.registry),
            "config_mode": self.config.mode.name,
            "correlations_computed": self._analysis_count > 0,
            "plugins_enabled": self._plugins_enabled,
        }

        # Add plugin statistics if enabled
        if self._plugins_enabled and self._plugin_registry:
            stats["plugins_registered"] = len(self._plugin_registry.list_plugins())
            stats["plugin_trust_levels"] = {
                name: info.get("trust_level", "unknown")
                for name, info in self._plugin_registry.get_all_plugin_info().items()
            }

        return stats

    def summary(self) -> str:
        """Generate engine summary."""
        lines = [
            "Dark Trace Engine",
            f"Mode: {self.config.mode.name}",
            f"Analyses: {self._analysis_count}",
            f"Plugins: {'enabled' if self._plugins_enabled else 'disabled'}",
            "",
            "Registered Lenses:",
        ]

        for lens_type, lens in self._lenses.items():
            lines.append(f"  {lens_type.name}: {lens.feature_dim} features")

        # Add plugin info
        if self._plugins_enabled and self._plugin_registry:
            lines.append("")
            lines.append("Registered Plugins:")
            for name in self._plugin_registry.get_plugin_names():
                info = self._plugin_registry.get_plugin_info(name)
                trust = info.get("trust_level", "unknown") if info else "unknown"
                lines.append(f"  {name}: {trust}")

        lines.append("")
        lines.append(self.registry.summary())

        return "\n".join(lines)

    # =========================================================================
    # Plugin System (Phase 11)
    # =========================================================================

    async def load_plugins(
        self,
        plugin_dirs: Optional[List[str]] = None,
        auto_discover: bool = True,
        min_trust_level: TrustLevel = TrustLevel.SANDBOXED,
    ) -> BulkLoadResult:
        """
        Load plugins from directories with safety validation.

        All plugins go through the safety gate before loading.

        Args:
            plugin_dirs: Directories to search for plugins
            auto_discover: Use auto-discovery (entry points, standard paths)
            min_trust_level: Minimum trust level to accept

        Returns:
            BulkLoadResult with loaded, blocked, and failed plugins
        """
        if not self._plugins_enabled or not self._plugin_loader:
            logger.warning("Plugin system not enabled")
            return BulkLoadResult(
                loaded=[],
                blocked=[],
                failed=[],
                total_time_seconds=0.0,
            )

        return await self._plugin_loader.load_plugins(
            plugin_dirs=plugin_dirs,
            auto_discover=auto_discover,
            min_trust_level=min_trust_level,
            engine=self,
        )

    async def load_plugin(
        self,
        plugin_path: str,
        trust_level: Optional[TrustLevel] = None,
    ):
        """
        Load a single plugin from path.

        Args:
            plugin_path: Path to plugin module or package
            trust_level: Trust level to assign (None for auto-detect)

        Returns:
            PluginLoadResult
        """
        if not self._plugins_enabled or not self._plugin_loader:
            logger.warning("Plugin system not enabled")
            return None

        return await self._plugin_loader.load_plugin(
            plugin_path,
            engine=self,
            trust_level=trust_level,
        )

    def get_plugin(self, name: str) -> Optional[DarkTracePlugin]:
        """
        Get a registered plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None if not found
        """
        if not self._plugin_registry:
            return None
        return self._plugin_registry.get_plugin(name)

    def list_plugins(self) -> List[str]:
        """
        List all registered plugin names.

        Returns:
            List of plugin names
        """
        if not self._plugin_registry:
            return []
        return self._plugin_registry.get_plugin_names()

    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a plugin.

        Args:
            name: Plugin name

        Returns:
            Plugin info dict or None
        """
        if not self._plugin_registry:
            return None
        return self._plugin_registry.get_plugin_info(name)

    async def unload_plugin(self, name: str) -> bool:
        """
        Unload a plugin.

        Args:
            name: Plugin name to unload

        Returns:
            True if successfully unloaded
        """
        if not self._plugin_registry:
            return False

        plugin = self._plugin_registry.get_plugin(name)
        if plugin:
            await plugin.shutdown()
            await self._plugin_registry.unregister(name)

            if self._alignment_bridge:
                await self._alignment_bridge.audit_plugin_unregistration(name)

            logger.info(f"Unloaded plugin: {name}")
            return True

        return False

    async def _notify_plugins_analysis_complete(
        self,
        trace_result: TraceResult,
    ) -> None:
        """Notify all monitor plugins of analysis completion."""
        if not self._plugin_registry:
            return

        for name in self._plugin_registry.get_plugin_names():
            plugin = self._plugin_registry.get_plugin(name)
            if plugin and isinstance(plugin, MonitorPlugin):
                try:
                    await plugin.on_analysis_complete(trace_result)
                except Exception as e:
                    logger.warning(f"Plugin {name} failed on analysis notification: {e}")

    @property
    def safety_gate(self) -> Optional[PluginSafetyGate]:
        """Access the plugin safety gate."""
        return self._safety_gate

    @property
    def alignment_bridge(self) -> Optional[PluginAlignmentBridge]:
        """Access the alignment bridge."""
        return self._alignment_bridge

    @property
    def plugin_registry(self) -> Optional[PluginRegistry]:
        """Access the plugin registry."""
        return self._plugin_registry

    # =========================================================================
    # Async Context Manager (Phase 11)
    # =========================================================================

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - clean shutdown of plugins."""
        if self._plugin_registry:
            # Shutdown all plugins
            for name in self._plugin_registry.list_plugins():
                try:
                    plugin = self._plugin_registry.get_plugin(name)
                    if plugin:
                        await plugin.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down plugin {name}: {e}")

        logger.info("DarkTraceEngine shutdown complete")
        return False


# =============================================================================
# Factory Functions
# =============================================================================

def create_engine(
    config: Optional[TraceConfig] = None,
    auto_register_sae: bool = False,
    sae_dim: int = 384,
) -> DarkTraceEngine:
    """
    Create a DarkTraceEngine with optional auto-registration.

    Args:
        config: Configuration (defaults to standard)
        auto_register_sae: Auto-register SAE features
        sae_dim: Input dimension for SAE

    Returns:
        Configured DarkTraceEngine
    """
    engine = DarkTraceEngine(config=config or TraceConfig.standard())

    if auto_register_sae:
        n_features = sae_dim * (config or TraceConfig.standard()).sae.expansion_factor
        features = create_sae_features(n_features)
        engine.registry.register_batch(features)

    return engine
