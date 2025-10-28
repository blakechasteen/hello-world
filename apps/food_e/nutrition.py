"""
food-e Nutritional Spectrum
============================
Spectral analysis of nutrition - treating macros as frequency harmonics.

PHILOSOPHY:
Your body processes nutrients at different timescales (frequencies):
- Protein: High frequency (fast turnover, 4-6 hours, building/repair)
- Carbs: Mid frequency (energy cycles, 2-6 hours)
- Fat: Low frequency (sustained energy, storage, days)
- Fiber: Modulates absorption (affects all frequencies)

A balanced diet has harmonic resonance.
Deficiencies are missing frequencies.
"""

from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np

from .models import NutritionalProfile


class NutritionalSpectrum:
    """
    Maps nutrition to frequency domain for elegant analysis.

    This is more intuitive than raw numbers:
    - "You need building blocks" vs "you need 30g protein"
    - "Your diet lacks foundational energy" vs "low fat"
    - "Good harmonic resonance" vs "macros are balanced"

    Inspired by HoloLoom's spectral graph features.
    """

    # Frequency bands (inspired by audio processing)
    BANDS = {
        "sub_bass": (0, 8),       # Base metabolic functions
        "bass": (8, 16),          # Fat metabolism (low freq, sustained)
        "low_mid": (16, 24),      # Complex carbs
        "mid": (24, 40),          # Simple carbs, quick energy
        "high_mid": (40, 56),     # Protein synthesis
        "presence": (56, 64),     # Micronutrients, cofactors
    }

    def __init__(self, nutrition: NutritionalProfile):
        self.nutrition = nutrition
        self._spectrum = self._compute_spectrum()

    def _compute_spectrum(self) -> np.ndarray:
        """
        Convert nutrition to 64-dimensional frequency spectrum.

        Each band represents a metabolic timescale:
        - Low frequencies: slow metabolism (fat storage/use)
        - Mid frequencies: energy cycles (glucose metabolism)
        - High frequencies: fast turnover (protein synthesis)
        """
        spectrum = np.zeros(64)
        total_cals = max(self.nutrition.calories, 1)

        # === Low Frequency: Fat Metabolism ===
        # Fat: 9 cal/g, slow oxidation, sustained energy
        fat_energy = (self.nutrition.fat_g * 9) / total_cals

        # Fill bass band (8-16) with fat signal
        spectrum[8:16] = fat_energy

        # === Mid Frequency: Carb Metabolism ===
        # Carbs: 4 cal/g, faster oxidation, cyclical energy
        carb_energy = (self.nutrition.carbs_g * 4) / total_cals

        # Complex carbs (with fiber) = lower mid (16-24)
        # Simple carbs (sugar) = mid (24-40)
        if self.nutrition.fiber_g > 5:
            # High fiber slows absorption, shifts to lower frequencies
            complex_ratio = min(self.nutrition.fiber_g / 30, 0.7)
            spectrum[16:24] = carb_energy * complex_ratio
            spectrum[24:40] = carb_energy * (1 - complex_ratio)
        else:
            # Low fiber = more simple carbs, higher frequency
            spectrum[24:40] = carb_energy * 0.8
            spectrum[16:24] = carb_energy * 0.2

        # === High Frequency: Protein Metabolism ===
        # Protein: 4 cal/g, fast turnover, building/repair
        protein_energy = (self.nutrition.protein_g * 4) / total_cals

        # Protein fills high-mid band (40-56)
        spectrum[40:56] = protein_energy

        # Essential amino acids create presence band (56-64)
        # Assume complete proteins for now
        spectrum[56:64] = protein_energy * 0.3

        # === Harmonics: Micronutrients ===
        # Vitamins/minerals modulate the fundamental frequencies

        # Sodium affects water retention, modulates bass
        if self.nutrition.sodium_mg > 2300:
            spectrum[8:16] *= 1.1  # Excess sodium amplifies low freq

        # Iron affects oxygen transport, modulates all frequencies
        if hasattr(self.nutrition, 'iron_mg') and self.nutrition.iron_mg:
            iron_factor = min(self.nutrition.iron_mg / 18, 1.2)
            spectrum *= iron_factor

        # Normalize to unit energy
        if np.sum(spectrum) > 0:
            spectrum /= np.sum(spectrum)

        return spectrum

    def harmonic_resonance(self, target: 'NutritionalSpectrum') -> float:
        """
        How well does this nutrition resonate with target?

        Uses spectral similarity (like audio fingerprinting):
        - 1.0 = perfect alignment
        - 0.8-1.0 = excellent (harmonious)
        - 0.6-0.8 = good (balanced)
        - 0.4-0.6 = fair (some imbalance)
        - 0.0-0.4 = poor (dissonant)

        This captures:
        - Overall balance (macro ratios)
        - Timescale alignment (energy availability matches needs)
        - Quality (fiber modulating carbs, complete proteins)
        """
        self_spec = self._spectrum
        target_spec = target._spectrum

        # Spectral correlation (like audio matching)
        dot_product = np.dot(self_spec, target_spec)
        norm_product = np.linalg.norm(self_spec) * np.linalg.norm(target_spec)

        if norm_product == 0:
            return 0.0

        # Cosine similarity in frequency domain
        resonance = dot_product / norm_product

        return float(resonance)

    def missing_frequencies(self, target: 'NutritionalSpectrum') -> Dict[str, float]:
        """
        Which nutritional frequencies are you missing?

        Returns poetic descriptions of deficits:
        - "foundational_energy": Need more sustained energy (fat)
        - "sustaining_fuel": Need more carbs
        - "building_blocks": Need more protein
        - "metabolic_harmony": Overall balance score
        """
        deficit_spectrum = target._spectrum - self._spectrum

        # Analyze each frequency band
        bass_deficit = np.sum(deficit_spectrum[8:16])     # Fat
        mid_deficit = np.sum(deficit_spectrum[16:40])     # Carbs
        high_deficit = np.sum(deficit_spectrum[40:56])    # Protein
        presence_deficit = np.sum(deficit_spectrum[56:64])  # Micros

        # Calculate overall harmonic distortion
        mse = np.mean((target._spectrum - self._spectrum) ** 2)
        harmonic_distortion = np.sqrt(mse)

        return {
            "foundational_energy": float(bass_deficit),
            "sustaining_fuel": float(mid_deficit),
            "building_blocks": float(high_deficit),
            "metabolic_harmony": 1.0 - float(harmonic_distortion),
            "micronutrient_presence": float(presence_deficit)
        }

    def band_energy(self, band: str) -> float:
        """Get total energy in a frequency band"""
        start, end = self.BANDS[band]
        return float(np.sum(self._spectrum[start:end]))

    def dominant_frequency(self) -> str:
        """
        What's the dominant metabolic frequency?

        Useful for understanding meal character:
        - "bass" heavy = high fat, slow energy
        - "mid" heavy = carb-loaded, quick energy
        - "high_mid" heavy = protein-rich, building
        """
        band_energies = {
            band: self.band_energy(band)
            for band in self.BANDS.keys()
        }

        return max(band_energies, key=band_energies.get)

    def visualize(self) -> str:
        """
        ASCII visualization of spectrum (for CLI).

        Like an audio equalizer display.
        """
        lines = []
        lines.append("Nutritional Spectrum:")
        lines.append("-" * 50)

        for band, (start, end) in self.BANDS.items():
            energy = self.band_energy(band)
            bar_length = int(energy * 40)
            bar = "#" * bar_length + "." * (40 - bar_length)
            lines.append(f"{band:12} |{bar}| {energy:.2f}")

        lines.append("-" * 50)
        lines.append(f"Dominant: {self.dominant_frequency()}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization"""
        return {
            'spectrum': self._spectrum.tolist(),
            'dominant_frequency': self.dominant_frequency(),
            'band_energies': {
                band: self.band_energy(band)
                for band in self.BANDS.keys()
            }
        }