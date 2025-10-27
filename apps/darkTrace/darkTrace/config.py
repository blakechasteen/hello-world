"""
darkTrace Configuration
=======================
Configuration management for darkTrace semantic analysis.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class AnalysisLayer(Enum):
    """darkTrace analysis layers."""
    OBSERVATION = "observation"  # Real-time semantic monitoring
    ANALYSIS = "analysis"        # Trajectory prediction, fingerprinting
    CONTROL = "control"          # Embedding manipulation (future)
    EXPLOITATION = "exploitation"  # Security research (future)


class DomainType(Enum):
    """Analysis domain types for semantic calculus."""
    NARRATIVE = "narrative"
    DIALOGUE = "dialogue"
    TECHNICAL = "technical"
    GENERAL = "general"


@dataclass
class ObserverConfig:
    """Configuration for Layer 1: Observation."""

    # Semantic calculus config
    dimensions: int = 36
    selection_strategy: str = "hybrid"
    domain: str = "general"

    # Recording
    enable_recording: bool = True
    max_trajectory_length: int = 1000

    # Performance
    compute_flow: bool = True
    compute_curvature: bool = True
    compute_ethics: bool = False  # Optional, expensive


@dataclass
class AnalyzerConfig:
    """Configuration for Layer 2: Analysis."""

    # Trajectory prediction
    enable_trajectory_prediction: bool = True
    prediction_horizon: int = 10  # Steps ahead

    # Fingerprinting
    enable_fingerprinting: bool = True
    fingerprint_dimensions: int = 128

    # Pattern recognition
    enable_pattern_recognition: bool = True
    min_pattern_length: int = 3

    # Attractor detection
    enable_attractor_detection: bool = True
    attractor_threshold: float = 0.7


@dataclass
class DarkTraceConfig:
    """Main darkTrace configuration."""

    # Active layers
    layers: list[AnalysisLayer] = field(default_factory=lambda: [
        AnalysisLayer.OBSERVATION,
        AnalysisLayer.ANALYSIS
    ])

    # Component configs
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)

    # Storage
    persist_path: Optional[str] = None
    auto_save: bool = True
    save_interval_seconds: float = 300.0  # 5 minutes

    # Performance
    max_memory_mb: int = 1024
    enable_profiling: bool = False

    @classmethod
    def bare(cls) -> 'DarkTraceConfig':
        """Minimal configuration for fast prototyping."""
        return cls(
            layers=[AnalysisLayer.OBSERVATION],
            observer=ObserverConfig(
                dimensions=16,
                selection_strategy="balanced",
                compute_flow=False,
                compute_curvature=False,
                compute_ethics=False,
            ),
            analyzer=AnalyzerConfig(
                enable_trajectory_prediction=False,
                enable_fingerprinting=False,
                enable_pattern_recognition=False,
                enable_attractor_detection=False,
            )
        )

    @classmethod
    def fast(cls) -> 'DarkTraceConfig':
        """Balanced configuration for general use."""
        return cls(
            layers=[AnalysisLayer.OBSERVATION, AnalysisLayer.ANALYSIS],
            observer=ObserverConfig(
                dimensions=36,
                selection_strategy="hybrid",
                compute_flow=True,
                compute_curvature=False,
                compute_ethics=False,
            ),
            analyzer=AnalyzerConfig(
                enable_trajectory_prediction=True,
                enable_fingerprinting=True,
                enable_pattern_recognition=False,
                enable_attractor_detection=False,
            )
        )

    @classmethod
    def fused(cls) -> 'DarkTraceConfig':
        """Full configuration for deep analysis."""
        return cls(
            layers=[AnalysisLayer.OBSERVATION, AnalysisLayer.ANALYSIS],
            observer=ObserverConfig(
                dimensions=36,
                selection_strategy="hybrid",
                domain="general",
                compute_flow=True,
                compute_curvature=True,
                compute_ethics=True,
            ),
            analyzer=AnalyzerConfig(
                enable_trajectory_prediction=True,
                enable_fingerprinting=True,
                enable_pattern_recognition=True,
                enable_attractor_detection=True,
            )
        )

    @classmethod
    def narrative(cls) -> 'DarkTraceConfig':
        """Optimized for narrative/literary analysis."""
        return cls(
            layers=[AnalysisLayer.OBSERVATION, AnalysisLayer.ANALYSIS],
            observer=ObserverConfig(
                dimensions=36,
                selection_strategy="narrative",
                domain="narrative",
                compute_flow=True,
                compute_curvature=True,
                compute_ethics=True,
            ),
            analyzer=AnalyzerConfig(
                enable_trajectory_prediction=True,
                enable_fingerprinting=True,
                enable_pattern_recognition=True,
                enable_attractor_detection=True,
                prediction_horizon=20,  # Longer for narrative arcs
            )
        )

    @classmethod
    def dialogue(cls) -> 'DarkTraceConfig':
        """Optimized for conversation/dialogue analysis."""
        return cls(
            layers=[AnalysisLayer.OBSERVATION, AnalysisLayer.ANALYSIS],
            observer=ObserverConfig(
                dimensions=36,
                selection_strategy="dialogue",
                domain="dialogue",
                compute_flow=True,
                compute_curvature=False,  # Less important for dialogue
                compute_ethics=False,  # Faster for real-time
            ),
            analyzer=AnalyzerConfig(
                enable_trajectory_prediction=True,
                enable_fingerprinting=False,  # Less relevant for dialogue
                enable_pattern_recognition=True,
                enable_attractor_detection=False,
                prediction_horizon=5,  # Shorter for real-time
            )
        )

    @classmethod
    def technical(cls) -> 'DarkTraceConfig':
        """Optimized for technical/documentation analysis."""
        return cls(
            layers=[AnalysisLayer.OBSERVATION, AnalysisLayer.ANALYSIS],
            observer=ObserverConfig(
                dimensions=36,
                selection_strategy="balanced",
                domain="technical",
                compute_flow=True,
                compute_curvature=False,
                compute_ethics=False,
            ),
            analyzer=AnalyzerConfig(
                enable_trajectory_prediction=True,
                enable_fingerprinting=True,
                enable_pattern_recognition=True,
                enable_attractor_detection=True,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layers": [layer.value for layer in self.layers],
            "observer": {
                "dimensions": self.observer.dimensions,
                "selection_strategy": self.observer.selection_strategy,
                "domain": self.observer.domain,
                "enable_recording": self.observer.enable_recording,
                "compute_flow": self.observer.compute_flow,
                "compute_curvature": self.observer.compute_curvature,
                "compute_ethics": self.observer.compute_ethics,
            },
            "analyzer": {
                "enable_trajectory_prediction": self.analyzer.enable_trajectory_prediction,
                "enable_fingerprinting": self.analyzer.enable_fingerprinting,
                "enable_pattern_recognition": self.analyzer.enable_pattern_recognition,
                "enable_attractor_detection": self.analyzer.enable_attractor_detection,
                "prediction_horizon": self.analyzer.prediction_horizon,
            },
            "persist_path": self.persist_path,
            "auto_save": self.auto_save,
        }
