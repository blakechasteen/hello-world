"""
Confidence Calibration
======================

🐷 Phase 4: Confidence Calibration Tracking 🐷

Tracks how well confidence scores match actual outcomes. A well-calibrated system
should have:
- Fixes with 90% confidence succeed 90% of the time
- Fixes with 70% confidence succeed 70% of the time
- etc.

Calibration Metrics:
- ECE (Expected Calibration Error): Average difference between confidence and accuracy
- Overconfidence: Confidence > actual success rate
- Underconfidence: Confidence < actual success rate
- Calibration curve: Plot of predicted vs actual

This enables:
- Confidence score adjustments
- Better fix decisions
- Trust in the system
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


@dataclass
class CalibrationBin:
    """
    One bin in confidence calibration histogram.

    Groups fixes by confidence range (e.g., 0.80-0.90).
    """
    bin_start: float  # e.g., 0.80
    bin_end: float    # e.g., 0.90
    bin_center: float # e.g., 0.85

    # Samples in this bin
    total_samples: int = 0
    successful_samples: int = 0

    # Confidence values
    avg_confidence: float = 0.0
    sum_confidence: float = 0.0

    def add_sample(self, confidence: float, success: bool):
        """Add a sample to this bin"""
        self.total_samples += 1
        self.sum_confidence += confidence

        if success:
            self.successful_samples += 1

        # Update average
        self.avg_confidence = self.sum_confidence / self.total_samples

    @property
    def accuracy(self) -> float:
        """Actual success rate in this bin"""
        if self.total_samples == 0:
            return 0.0
        return self.successful_samples / self.total_samples

    @property
    def calibration_error(self) -> float:
        """Difference between predicted confidence and actual accuracy"""
        return abs(self.avg_confidence - self.accuracy)

    @property
    def is_overconfident(self) -> bool:
        """Is this bin overconfident? (confidence > accuracy)"""
        if self.total_samples == 0:
            return False
        return self.avg_confidence > self.accuracy + 0.05  # 5% threshold

    @property
    def is_underconfident(self) -> bool:
        """Is this bin underconfident? (confidence < accuracy)"""
        if self.total_samples == 0:
            return False
        return self.avg_confidence < self.accuracy - 0.05  # 5% threshold

    def get_stats(self) -> Dict:
        """Get bin statistics"""
        return {
            'bin_range': f'{self.bin_start:.2f}-{self.bin_end:.2f}',
            'bin_center': self.bin_center,
            'total_samples': self.total_samples,
            'successful_samples': self.successful_samples,
            'avg_confidence': self.avg_confidence,
            'accuracy': self.accuracy,
            'calibration_error': self.calibration_error,
            'is_overconfident': self.is_overconfident,
            'is_underconfident': self.is_underconfident
        }


@dataclass
class ConfidenceCalibrator:
    """
    Tracks confidence calibration across all fix attempts.

    Bins fixes by confidence range and computes calibration metrics.
    """

    # Binning configuration
    num_bins: int = 10  # 10 bins: [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]
    bins: List[CalibrationBin] = field(default_factory=list)

    # Raw samples (for ECE computation)
    samples: List[Tuple[float, bool]] = field(default_factory=list)

    # Persistence
    state_path: Optional[Path] = None

    def __post_init__(self):
        """Initialize bins"""
        if not self.bins:
            bin_width = 1.0 / self.num_bins
            for i in range(self.num_bins):
                bin_start = i * bin_width
                bin_end = (i + 1) * bin_width
                bin_center = (bin_start + bin_end) / 2
                self.bins.append(CalibrationBin(
                    bin_start=bin_start,
                    bin_end=bin_end,
                    bin_center=bin_center
                ))

    def add_sample(self, confidence: float, success: bool):
        """
        Add a fix attempt to calibration tracking.

        Args:
            confidence: Predicted confidence (0.0-1.0)
            success: Did the fix succeed?
        """
        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        # Add to samples
        self.samples.append((confidence, success))

        # Find appropriate bin
        bin_idx = min(int(confidence * self.num_bins), self.num_bins - 1)
        self.bins[bin_idx].add_sample(confidence, success)

        # Persist if path set
        if self.state_path:
            self.save_state(self.state_path)

    def get_expected_calibration_error(self) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE = Σ (n_b / n) * |acc(b) - conf(b)|

        Where:
        - n_b = number of samples in bin b
        - n = total samples
        - acc(b) = accuracy in bin b
        - conf(b) = average confidence in bin b

        Returns:
            ECE value (0.0 = perfect calibration, 1.0 = worst)
        """
        if len(self.samples) == 0:
            return 0.0

        total_samples = len(self.samples)
        ece = 0.0

        for bin in self.bins:
            if bin.total_samples > 0:
                weight = bin.total_samples / total_samples
                ece += weight * bin.calibration_error

        return ece

    def get_calibration_curve(self) -> List[Dict]:
        """
        Get calibration curve data for plotting.

        Returns:
            List of points: [{confidence: 0.85, accuracy: 0.82, count: 42}, ...]
        """
        curve = []
        for bin in self.bins:
            if bin.total_samples > 0:
                curve.append({
                    'confidence': bin.avg_confidence,
                    'accuracy': bin.accuracy,
                    'count': bin.total_samples,
                    'bin_range': f'{bin.bin_start:.2f}-{bin.bin_end:.2f}',
                    'calibration_error': bin.calibration_error
                })
        return curve

    def get_overconfident_bins(self) -> List[CalibrationBin]:
        """Get bins where system is overconfident"""
        return [bin for bin in self.bins if bin.is_overconfident and bin.total_samples > 0]

    def get_underconfident_bins(self) -> List[CalibrationBin]:
        """Get bins where system is underconfident"""
        return [bin for bin in self.bins if bin.is_underconfident and bin.total_samples > 0]

    def get_calibration_summary(self) -> Dict:
        """
        Get comprehensive calibration summary.

        Returns:
            Dictionary with:
            - ece: Expected Calibration Error
            - total_samples: Number of samples
            - overconfident_bins: Bins where overconfident
            - underconfident_bins: Bins where underconfident
            - calibration_curve: Data for plotting
            - recommendations: Suggested improvements
        """
        ece = self.get_expected_calibration_error()
        overconfident = self.get_overconfident_bins()
        underconfident = self.get_underconfident_bins()

        # Generate recommendations
        recommendations = []
        if ece > 0.10:
            recommendations.append("ECE > 0.10: Significant calibration issues detected")

        if overconfident:
            ranges = [f"{b.bin_start:.1f}-{b.bin_end:.1f}" for b in overconfident]
            recommendations.append(f"Overconfident in ranges: {', '.join(ranges)}")
            recommendations.append("Consider lowering confidence scores in these ranges")

        if underconfident:
            ranges = [f"{b.bin_start:.1f}-{b.bin_end:.1f}" for b in underconfident]
            recommendations.append(f"Underconfident in ranges: {', '.join(ranges)}")
            recommendations.append("Consider raising confidence scores in these ranges")

        if ece < 0.05:
            recommendations.append("Excellent calibration (ECE < 0.05)")

        return {
            'ece': ece,
            'total_samples': len(self.samples),
            'num_bins_with_data': sum(1 for b in self.bins if b.total_samples > 0),
            'overconfident_bins': [b.get_stats() for b in overconfident],
            'underconfident_bins': [b.get_stats() for b in underconfident],
            'calibration_curve': self.get_calibration_curve(),
            'recommendations': recommendations,
            'calibration_quality': self._get_calibration_quality(ece)
        }

    def _get_calibration_quality(self, ece: float) -> str:
        """Get human-readable calibration quality"""
        if ece < 0.05:
            return "Excellent"
        elif ece < 0.10:
            return "Good"
        elif ece < 0.15:
            return "Fair"
        else:
            return "Poor"

    def suggest_confidence_adjustment(self, confidence: float) -> Tuple[float, str]:
        """
        Suggest adjusted confidence based on calibration data.

        Args:
            confidence: Predicted confidence

        Returns:
            (adjusted_confidence, reason)
        """
        # Find bin
        bin_idx = min(int(confidence * self.num_bins), self.num_bins - 1)
        bin = self.bins[bin_idx]

        if bin.total_samples < 10:
            # Not enough data
            return confidence, "Insufficient calibration data in this range"

        # Adjust toward accuracy
        if bin.is_overconfident:
            # Lower confidence
            adjusted = confidence * 0.9  # 10% reduction
            reason = f"Bin is overconfident ({bin.avg_confidence:.2f} vs {bin.accuracy:.2f})"
            return adjusted, reason

        elif bin.is_underconfident:
            # Raise confidence
            adjusted = min(1.0, confidence * 1.1)  # 10% increase
            reason = f"Bin is underconfident ({bin.avg_confidence:.2f} vs {bin.accuracy:.2f})"
            return adjusted, reason

        else:
            # Well calibrated
            return confidence, f"Well calibrated ({bin.avg_confidence:.2f} vs {bin.accuracy:.2f})"

    def save_state(self, path: Path):
        """Save calibration state to JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'version': '1.0',
            'num_bins': self.num_bins,
            'bins': [
                {
                    'bin_start': bin.bin_start,
                    'bin_end': bin.bin_end,
                    'bin_center': bin.bin_center,
                    'total_samples': bin.total_samples,
                    'successful_samples': bin.successful_samples,
                    'sum_confidence': bin.sum_confidence,
                    'avg_confidence': bin.avg_confidence
                }
                for bin in self.bins
            ],
            'total_samples': len(self.samples)
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, path: Path) -> 'ConfidenceCalibrator':
        """Load calibration state from JSON"""
        with open(path) as f:
            state = json.load(f)

        calibrator = cls(
            num_bins=state['num_bins'],
            state_path=path
        )

        # Restore bins
        calibrator.bins = []
        for bin_data in state['bins']:
            bin = CalibrationBin(
                bin_start=bin_data['bin_start'],
                bin_end=bin_data['bin_end'],
                bin_center=bin_data['bin_center'],
                total_samples=bin_data['total_samples'],
                successful_samples=bin_data['successful_samples'],
                sum_confidence=bin_data['sum_confidence'],
                avg_confidence=bin_data['avg_confidence']
            )
            calibrator.bins.append(bin)

        return calibrator


def create_confidence_calibrator(
    num_bins: int = 10,
    state_path: Optional[Path] = None
) -> ConfidenceCalibrator:
    """
    Factory function to create ConfidenceCalibrator.

    Args:
        num_bins: Number of calibration bins
        state_path: Path to load/save state

    Returns:
        ConfidenceCalibrator instance
    """
    # Load from state if exists
    if state_path and Path(state_path).exists():
        return ConfidenceCalibrator.load_state(state_path)

    # Create new calibrator
    return ConfidenceCalibrator(
        num_bins=num_bins,
        state_path=state_path
    )
