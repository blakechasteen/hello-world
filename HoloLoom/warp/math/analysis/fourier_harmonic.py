"""
Fourier and Harmonic Analysis for HoloLoom Warp Drive
=====================================================

Signal processing, frequency analysis, and time-frequency decompositions.

Core Concepts:
- Fourier Transform: Time → Frequency domain
- Fourier Series: Periodic function decomposition
- Wavelets: Localized time-frequency analysis
- Gabor Transform: Short-time Fourier transform
- Harmonic Analysis: Study of decomposing functions into harmonics

Mathematical Foundation:
Fourier Transform: F(ω) = ∫ f(t) e^(-iωt) dt
Inverse: f(t) = (1/2π) ∫ F(ω) e^(iωt) dω

Wavelet Transform: W_ψ f(a,b) = ∫ f(t) ψ*((t-b)/a) dt
where ψ is the mother wavelet, a is scale, b is translation

Applications to Warp Space:
- Frequency domain embeddings
- Multi-scale wavelet decompositions for knowledge graphs
- Time-frequency analysis of temporal patterns
- Spectral feature extraction

Author: HoloLoom Team
Date: 2025-10-26
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# FOURIER TRANSFORM
# ============================================================================

class FourierTransform:
    """
    Fourier transform and inverse transform.

    Continuous FT: F(ω) = ∫ f(t) e^(-iωt) dt
    Discrete FT: X[k] = Σₙ x[n] e^(-2πikn/N)
    """

    @staticmethod
    def fft(signal: np.ndarray) -> np.ndarray:
        """
        Fast Fourier Transform.

        Compute DFT using FFT algorithm (O(N log N)).
        """
        return np.fft.fft(signal)

    @staticmethod
    def ifft(frequency_coeffs: np.ndarray) -> np.ndarray:
        """
        Inverse Fast Fourier Transform.

        Recover signal from frequency coefficients.
        """
        return np.fft.ifft(frequency_coeffs)

    @staticmethod
    def fft_frequencies(n: int, sample_rate: float = 1.0) -> np.ndarray:
        """
        Get frequency bins for FFT output.

        Args:
            n: Number of samples
            sample_rate: Sampling frequency (Hz)

        Returns:
            Array of frequencies corresponding to FFT bins
        """
        return np.fft.fftfreq(n, d=1.0/sample_rate)

    @staticmethod
    def power_spectrum(signal: np.ndarray) -> np.ndarray:
        """
        Compute power spectrum |F(ω)|².

        Power at frequency ω.
        """
        freq_coeffs = FourierTransform.fft(signal)
        return np.abs(freq_coeffs) ** 2

    @staticmethod
    def magnitude_spectrum(signal: np.ndarray) -> np.ndarray:
        """
        Compute magnitude spectrum |F(ω)|.
        """
        freq_coeffs = FourierTransform.fft(signal)
        return np.abs(freq_coeffs)

    @staticmethod
    def phase_spectrum(signal: np.ndarray) -> np.ndarray:
        """
        Compute phase spectrum arg(F(ω)).
        """
        freq_coeffs = FourierTransform.fft(signal)
        return np.angle(freq_coeffs)

    @staticmethod
    def apply_bandpass_filter(
        signal: np.ndarray,
        low_freq: float,
        high_freq: float,
        sample_rate: float = 1.0
    ) -> np.ndarray:
        """
        Apply bandpass filter in frequency domain.

        Keep frequencies in [low_freq, high_freq], zero others.
        """
        # FFT
        freq_coeffs = FourierTransform.fft(signal)
        frequencies = FourierTransform.fft_frequencies(len(signal), sample_rate)

        # Bandpass filter
        mask = (np.abs(frequencies) >= low_freq) & (np.abs(frequencies) <= high_freq)
        filtered_coeffs = freq_coeffs * mask

        # Inverse FFT
        filtered_signal = FourierTransform.ifft(filtered_coeffs)

        return np.real(filtered_signal)


# ============================================================================
# FOURIER SERIES
# ============================================================================

class FourierSeries:
    """
    Fourier series for periodic functions.

    f(t) = a₀/2 + Σₙ [aₙ cos(nωt) + bₙ sin(nωt)]

    Or in complex form:
    f(t) = Σₙ cₙ e^(inωt)
    """

    @staticmethod
    def compute_coefficients(
        signal: np.ndarray,
        period: float,
        n_harmonics: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute Fourier series coefficients.

        Returns:
            (a_coeffs, b_coeffs, a0) where:
            - a_coeffs[n] = aₙ (cosine coefficients)
            - b_coeffs[n] = bₙ (sine coefficients)
            - a0 = DC component
        """
        N = len(signal)
        t = np.linspace(0, period, N, endpoint=False)
        omega = 2 * np.pi / period

        # DC component
        a0 = 2 * np.mean(signal)

        # Harmonic coefficients
        a_coeffs = np.zeros(n_harmonics)
        b_coeffs = np.zeros(n_harmonics)

        for n in range(1, n_harmonics + 1):
            # aₙ = (2/T) ∫ f(t) cos(nωt) dt
            a_coeffs[n-1] = 2 * np.mean(signal * np.cos(n * omega * t))

            # bₙ = (2/T) ∫ f(t) sin(nωt) dt
            b_coeffs[n-1] = 2 * np.mean(signal * np.sin(n * omega * t))

        return a_coeffs, b_coeffs, a0

    @staticmethod
    def reconstruct(
        a_coeffs: np.ndarray,
        b_coeffs: np.ndarray,
        a0: float,
        t: np.ndarray,
        period: float
    ) -> np.ndarray:
        """
        Reconstruct signal from Fourier coefficients.

        f(t) = a₀/2 + Σₙ [aₙ cos(nωt) + bₙ sin(nωt)]
        """
        omega = 2 * np.pi / period
        signal = a0 / 2 * np.ones_like(t)

        for n, (a_n, b_n) in enumerate(zip(a_coeffs, b_coeffs), start=1):
            signal += a_n * np.cos(n * omega * t)
            signal += b_n * np.sin(n * omega * t)

        return signal

    @staticmethod
    def complex_coefficients(signal: np.ndarray, period: float, n_harmonics: int = 10) -> np.ndarray:
        """
        Compute complex Fourier coefficients.

        cₙ = (1/T) ∫ f(t) e^(-inωt) dt
        """
        N = len(signal)
        t = np.linspace(0, period, N, endpoint=False)
        omega = 2 * np.pi / period

        coeffs = np.zeros(2*n_harmonics + 1, dtype=complex)

        for n in range(-n_harmonics, n_harmonics + 1):
            # cₙ = (1/T) ∫ f(t) e^(-inωt) dt
            integrand = signal * np.exp(-1j * n * omega * t)
            coeffs[n + n_harmonics] = np.mean(integrand)

        return coeffs


# ============================================================================
# WAVELETS
# ============================================================================

class WaveletTransform:
    """
    Wavelet transform for time-frequency localization.

    Wavelet transform: W_ψ f(a,b) = (1/√a) ∫ f(t) ψ*((t-b)/a) dt
    where:
    - ψ is the mother wavelet
    - a is scale parameter
    - b is translation parameter
    """

    @staticmethod
    def haar_wavelet(t: np.ndarray) -> np.ndarray:
        """
        Haar wavelet (simplest wavelet).

        ψ(t) = { 1  if 0 ≤ t < 0.5
               { -1 if 0.5 ≤ t < 1
               { 0  otherwise
        """
        wavelet = np.zeros_like(t)
        wavelet[(t >= 0) & (t < 0.5)] = 1
        wavelet[(t >= 0.5) & (t < 1)] = -1
        return wavelet

    @staticmethod
    def mexican_hat_wavelet(t: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Mexican hat (Ricker) wavelet.

        ψ(t) = (1 - t²/σ²) exp(-t²/(2σ²))

        Second derivative of Gaussian.
        """
        normalized_t = t / sigma
        wavelet = (1 - normalized_t**2) * np.exp(-normalized_t**2 / 2)
        return wavelet / (np.sqrt(2/3 * sigma) * np.pi**0.25)

    @staticmethod
    def morlet_wavelet(t: np.ndarray, omega0: float = 5.0) -> np.ndarray:
        """
        Morlet wavelet (complex).

        ψ(t) = π^(-1/4) exp(iω₀t) exp(-t²/2)

        Gaussian-windowed complex exponential.
        """
        wavelet = np.pi**(-0.25) * np.exp(1j * omega0 * t) * np.exp(-t**2 / 2)
        return wavelet

    @staticmethod
    def continuous_wavelet_transform(
        signal: np.ndarray,
        wavelet: Callable[[np.ndarray], np.ndarray],
        scales: np.ndarray,
        sample_rate: float = 1.0
    ) -> np.ndarray:
        """
        Continuous Wavelet Transform (CWT).

        Args:
            signal: Input signal
            wavelet: Mother wavelet function
            scales: Array of scale parameters
            sample_rate: Sampling rate

        Returns:
            2D array: CWT coefficients (scales × time)
        """
        n = len(signal)
        cwt_matrix = np.zeros((len(scales), n), dtype=complex)

        for i, scale in enumerate(scales):
            # Generate scaled and translated wavelets
            for tau in range(n):
                t_scaled = (np.arange(n) - tau) / scale
                psi = wavelet(t_scaled) / np.sqrt(scale)
                cwt_matrix[i, tau] = np.sum(signal * np.conj(psi)) / sample_rate

        return cwt_matrix

    @staticmethod
    def discrete_wavelet_transform(signal: np.ndarray, wavelet: str = 'haar', level: int = 2) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Discrete Wavelet Transform (DWT).

        Decomposes signal into approximation and detail coefficients.

        Returns:
            (approximation, [detail1, detail2, ...])
        """
        # Simple Haar DWT implementation
        if wavelet != 'haar':
            logger.warning(f"Only Haar wavelet supported, using Haar")

        approx = signal.copy()
        details = []

        for _ in range(level):
            n = len(approx)
            if n < 2:
                break

            # Ensure even length
            if n % 2 == 1:
                approx = np.append(approx, approx[-1])
                n += 1

            # Low-pass filter (approximation)
            new_approx = (approx[::2] + approx[1::2]) / np.sqrt(2)

            # High-pass filter (detail)
            detail = (approx[::2] - approx[1::2]) / np.sqrt(2)

            details.append(detail)
            approx = new_approx

        return approx, details

    @staticmethod
    def inverse_discrete_wavelet_transform(
        approx: np.ndarray,
        details: List[np.ndarray]
    ) -> np.ndarray:
        """
        Inverse Discrete Wavelet Transform.

        Reconstruct signal from approximation and detail coefficients.
        """
        signal = approx.copy()

        for detail in reversed(details):
            n_approx = len(signal)
            n_detail = len(detail)
            n = min(n_approx, n_detail)

            # Upsample and filter
            reconstructed = np.zeros(2 * n)
            reconstructed[::2] = (signal[:n] + detail[:n]) / np.sqrt(2)
            reconstructed[1::2] = (signal[:n] - detail[:n]) / np.sqrt(2)

            signal = reconstructed

        return signal


# ============================================================================
# TIME-FREQUENCY ANALYSIS
# ============================================================================

class TimeFrequencyAnalysis:
    """
    Time-frequency representations.

    - Short-Time Fourier Transform (STFT)
    - Spectrogram
    - Gabor Transform
    """

    @staticmethod
    def stft(
        signal: np.ndarray,
        window_size: int,
        hop_size: int,
        window_type: str = 'hann'
    ) -> np.ndarray:
        """
        Short-Time Fourier Transform.

        Divide signal into windows and apply FFT to each.

        Args:
            signal: Input signal
            window_size: Size of analysis window
            hop_size: Step size between windows
            window_type: Window function ('hann', 'hamming', 'rectangular')

        Returns:
            2D array: STFT coefficients (frequency × time)
        """
        # Create window
        if window_type == 'hann':
            window = np.hanning(window_size)
        elif window_type == 'hamming':
            window = np.hamming(window_size)
        else:
            window = np.ones(window_size)

        # Number of windows
        n_windows = (len(signal) - window_size) // hop_size + 1

        # STFT matrix
        stft_matrix = np.zeros((window_size // 2 + 1, n_windows), dtype=complex)

        for i in range(n_windows):
            start = i * hop_size
            end = start + window_size

            if end > len(signal):
                break

            # Windowed signal
            windowed = signal[start:end] * window

            # FFT (take only positive frequencies)
            fft_result = np.fft.rfft(windowed)
            stft_matrix[:, i] = fft_result

        return stft_matrix

    @staticmethod
    def istft(
        stft_matrix: np.ndarray,
        hop_size: int,
        window_type: str = 'hann'
    ) -> np.ndarray:
        """
        Inverse Short-Time Fourier Transform.

        Reconstruct signal from STFT.
        """
        window_size = (stft_matrix.shape[0] - 1) * 2
        n_windows = stft_matrix.shape[1]

        # Create window
        if window_type == 'hann':
            window = np.hanning(window_size)
        elif window_type == 'hamming':
            window = np.hamming(window_size)
        else:
            window = np.ones(window_size)

        # Output signal length
        signal_length = window_size + (n_windows - 1) * hop_size
        signal = np.zeros(signal_length)
        window_sum = np.zeros(signal_length)

        for i in range(n_windows):
            start = i * hop_size
            end = start + window_size

            # Inverse FFT
            windowed = np.fft.irfft(stft_matrix[:, i])

            # Overlap-add
            signal[start:end] += windowed * window
            window_sum[start:end] += window ** 2

        # Normalize by window sum
        signal = np.divide(signal, window_sum, where=window_sum > 1e-10)

        return signal

    @staticmethod
    def spectrogram(
        signal: np.ndarray,
        window_size: int,
        hop_size: int,
        window_type: str = 'hann'
    ) -> np.ndarray:
        """
        Compute spectrogram |STFT|².

        Energy density in time-frequency plane.
        """
        stft_matrix = TimeFrequencyAnalysis.stft(signal, window_size, hop_size, window_type)
        return np.abs(stft_matrix) ** 2

    @staticmethod
    def gabor_transform(
        signal: np.ndarray,
        sigma: float,
        frequencies: np.ndarray,
        sample_rate: float = 1.0
    ) -> np.ndarray:
        """
        Gabor transform (windowed Fourier transform).

        G(t,ω) = ∫ f(τ) g(τ-t) e^(-iωτ) dτ

        where g(t) = exp(-t²/(2σ²)) is Gaussian window.

        Args:
            signal: Input signal
            sigma: Gaussian window width
            frequencies: Array of frequencies to analyze
            sample_rate: Sampling rate

        Returns:
            2D array: Gabor coefficients (frequency × time)
        """
        n = len(signal)
        t = np.arange(n) / sample_rate

        gabor_matrix = np.zeros((len(frequencies), n), dtype=complex)

        for i, freq in enumerate(frequencies):
            for tau_idx in range(n):
                # Gaussian window centered at tau
                tau = t[tau_idx]
                window = np.exp(-(t - tau)**2 / (2 * sigma**2))

                # Windowed signal
                windowed = signal * window

                # Fourier coefficient at frequency
                integrand = windowed * np.exp(-1j * 2 * np.pi * freq * t)
                gabor_matrix[i, tau_idx] = np.sum(integrand) / sample_rate

        return gabor_matrix


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'FourierTransform',
    'FourierSeries',
    'WaveletTransform',
    'TimeFrequencyAnalysis'
]
