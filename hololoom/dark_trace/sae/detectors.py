"""
SAE-backed governance detectors.

Implements FeatureDetector protocol using Sparse Autoencoder feature decomposition.
Replaces heuristic detection (Jaccard, keyword grep, regex) with learned features.

Usage:
    sae = ResearchSAE.load("path/to/trained.pt")
    embedder = MatryoshkaEmbeddings(...)

    deception = SAEDeceptionDetector(sae, embedder, feature_map=loaded_map)
    result = deception.detect("some text", context={...})
    # result.score = 0.73, result.activated_features = ["intent_drift", "goal_misalign"]
"""

import logging
from dataclasses import dataclass, field

import torch

from hololoom.alignment.feature_detector import DetectionResult

logger = logging.getLogger(__name__)


@dataclass
class FeatureMap:
    """Maps SAE feature indices to governance-relevant categories.

    Built by the autolabeler after training. Serializable to/from JSON.
    """
    deception_features: dict[int, str] = field(default_factory=dict)     # idx -> label
    power_seeking_features: dict[int, str] = field(default_factory=dict)
    self_modification_features: dict[int, str] = field(default_factory=dict)
    adversarial_features: dict[int, str] = field(default_factory=dict)

    def save(self, path: str) -> None:
        import json
        import os
        import tempfile
        data = {
            "deception": {str(k): v for k, v in self.deception_features.items()},
            "power_seeking": {str(k): v for k, v in self.power_seeking_features.items()},
            "self_modification": {str(k): v for k, v in self.self_modification_features.items()},
            "adversarial": {str(k): v for k, v in self.adversarial_features.items()},
        }
        dir_name = os.path.dirname(path) or "."
        with tempfile.NamedTemporaryFile(
            mode="w", dir=dir_name, suffix=".tmp", delete=False
        ) as f:
            json.dump(data, f, indent=2)
            tmp = f.name
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "FeatureMap":
        import json
        with open(path) as f:
            data = json.load(f)
        return cls(
            deception_features={int(k): v for k, v in data.get("deception", {}).items()},
            power_seeking_features={
                int(k): v for k, v in data.get("power_seeking", {}).items()
            },
            self_modification_features={
                int(k): v for k, v in data.get("self_modification", {}).items()
            },
            adversarial_features={
                int(k): v for k, v in data.get("adversarial", {}).items()
            },
        )


class SAEDetectorBase:
    """Base class for SAE-backed detectors."""

    def __init__(self, sae, embedder, relevant_features: dict[int, str]):
        self.sae = sae  # ResearchSAE
        self.embedder = embedder  # MatryoshkaEmbeddings
        self.relevant_features = relevant_features  # {idx: label}
        self._last_active: list[str] = []

    def _encode_text(self, text: str) -> torch.Tensor:
        """Text -> Matryoshka 384d embedding -> torch tensor."""
        embedding = self.embedder.encode([text])  # [1, 384]
        return torch.tensor(embedding, dtype=torch.float32)

    def _get_relevant_activations(self, text: str) -> tuple:
        """Returns (all_active_features, relevant_activations, relevant_labels)."""
        x = self._encode_text(text)
        active = self.sae.get_active_features(x)  # {idx: activation_value}

        relevant_activations = {}
        relevant_labels = []
        for idx, value in active.items():
            if idx in self.relevant_features:
                label = self.relevant_features[idx]
                relevant_activations[idx] = value
                relevant_labels.append(label)

        self._last_active = relevant_labels
        return active, relevant_activations, relevant_labels

    def get_active_features(self) -> list[str]:
        return list(self._last_active)


class SAEDeceptionDetector(SAEDetectorBase):
    """Replaces Jaccard word overlap in deception_detection.py."""

    def __init__(self, sae, embedder, feature_map: FeatureMap):
        super().__init__(sae, embedder, feature_map.deception_features)

    def detect(self, text: str, context: dict | None = None) -> DetectionResult:
        all_active, relevant, labels = self._get_relevant_activations(text)

        if not relevant:
            return DetectionResult(
                score=0.0, activated_features=[], metadata={"method": "sae"}
            )

        # Score = mean activation of deception-relevant features, normalized
        activations = list(relevant.values())
        raw_score = sum(activations) / len(activations)
        score = float(min(1.0, raw_score))  # clamp

        return DetectionResult(
            score=score,
            activated_features=labels,
            metadata={
                "method": "sae",
                "n_total_active": len(all_active),
                "n_deception_active": len(relevant),
                "raw_activations": {str(k): float(v) for k, v in relevant.items()},
            },
        )


class SAEPowerSeekingDetector(SAEDetectorBase):
    """Replaces keyword grep in instrumental_convergence.py detect_power_seeking()."""

    def __init__(self, sae, embedder, feature_map: FeatureMap):
        super().__init__(sae, embedder, feature_map.power_seeking_features)

    def detect(self, text: str, context: dict | None = None) -> DetectionResult:
        all_active, relevant, labels = self._get_relevant_activations(text)

        if not relevant:
            return DetectionResult(
                score=0.0, activated_features=[], metadata={"method": "sae"}
            )

        activations = list(relevant.values())
        score = float(min(1.0, max(activations)))  # max activation = strongest signal

        return DetectionResult(
            score=score,
            activated_features=labels,
            metadata={
                "method": "sae",
                "n_power_features": len(relevant),
                "max_activation": float(max(activations)),
            },
        )


class SAESelfModificationDetector(SAEDetectorBase):
    """Replaces substring match in instrumental_convergence.py detect_self_modification()."""

    def __init__(self, sae, embedder, feature_map: FeatureMap):
        super().__init__(sae, embedder, feature_map.self_modification_features)

    def detect(self, text: str, context: dict | None = None) -> DetectionResult:
        all_active, relevant, labels = self._get_relevant_activations(text)
        score = 1.0 if relevant else 0.0  # binary: any self-mod feature fires

        return DetectionResult(
            score=score,
            activated_features=labels,
            metadata={"method": "sae", "n_features": len(relevant)},
        )


class SAEAdversarialDetector(SAEDetectorBase):
    """Replaces regex patterns in safety_guardrails.py AdversarialDetector.detect()."""

    def __init__(self, sae, embedder, feature_map: FeatureMap):
        super().__init__(sae, embedder, feature_map.adversarial_features)

    def detect(self, text: str, context: dict | None = None) -> DetectionResult:
        all_active, relevant, labels = self._get_relevant_activations(text)

        if not relevant:
            return DetectionResult(
                score=0.0, activated_features=[], metadata={"method": "sae"}
            )

        activations = list(relevant.values())
        score = float(
            min(1.0, sum(activations) / max(1, len(self.relevant_features)))
        )

        return DetectionResult(
            score=score,
            activated_features=labels,
            metadata={
                "method": "sae",
                "attack_types": labels,
                "n_adversarial_features": len(relevant),
            },
        )
