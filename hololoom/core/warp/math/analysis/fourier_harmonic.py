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

import logging
from collections.abc import Callable

import numpy as np

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
    ) -> tuple[np.ndarray, np.ndarray, float]:
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
    def discrete_wavelet_transform(signal: np.ndarray, wavelet: str = 'haar', level: int = 2) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Discrete Wavelet Transform (DWT).

        Decomposes signal into approximation and detail coefficients.

        Returns:
            (approximation, [detail1, detail2, ...])
        """
        # Simple Haar DWT implementation
        if wavelet != 'haar':
            logger.warning("Only Haar wavelet supported, using Haar")

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
        details: list[np.ndarray]
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
    'TimeFrequencyAnalysis',
    'SpectralTimeSeries',
    'MultitaperSpectral',
]


# ============================================================================
# TIME SERIES SPECTRAL ANALYSIS
# ============================================================================

class SpectralTimeSeries:
    """Spectral methods specifically for time series analysis.

    Extends the Fourier/wavelet tools above with time-series-specific
    methods: periodogram, Welch PSD estimation, dominant frequency detection.
    """

    @staticmethod
    def periodogram(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw periodogram: |FFT(x)|^2 / N.

        The periodogram is the simplest spectral estimator but has high
        variance — each frequency bin is estimated from a single sample.

        Args:
            series: Input time series (real-valued).

        Returns:
            (frequencies, power) where frequencies are normalized [0, 0.5]
            and power is |X(f)|^2 / N.
        """
        N = len(series)
        # Remove mean to focus on oscillatory components
        centered = series - np.mean(series)

        # One-sided FFT
        freqs = np.fft.rfftfreq(N)
        fft_vals = np.fft.rfft(centered)

        # Periodogram: |X(f)|^2 / N
        power = (np.abs(fft_vals) ** 2) / N

        # Double the power for one-sided spectrum (except DC and Nyquist)
        if N % 2 == 0:
            power[1:-1] *= 2
        else:
            power[1:] *= 2

        return freqs, power

    @staticmethod
    def welch(series: np.ndarray, segment_length: int = 256,
              overlap: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """Welch's method: averaged modified periodograms.

        Reduces variance at cost of frequency resolution by:
        1. Dividing data into overlapping segments
        2. Windowing each segment (Hann)
        3. Computing periodogram of each
        4. Averaging across segments

        Args:
            series: Input time series.
            segment_length: Length of each segment. Clamped to len(series).
            overlap: Fraction of overlap between segments, in [0, 1).

        Returns:
            (frequencies, psd_estimate) — averaged power spectral density.
        """
        N = len(series)
        segment_length = min(segment_length, N)
        hop = max(1, int(segment_length * (1.0 - overlap)))

        # Hann window and its power normalization
        window = np.hanning(segment_length)
        win_power = np.mean(window ** 2)

        freqs = np.fft.rfftfreq(segment_length)
        n_freq_bins = len(freqs)
        accumulated = np.zeros(n_freq_bins)
        n_segments = 0

        start = 0
        while start + segment_length <= N:
            segment = series[start:start + segment_length]
            segment = (segment - np.mean(segment)) * window

            fft_vals = np.fft.rfft(segment)
            power = (np.abs(fft_vals) ** 2) / (segment_length * win_power)

            accumulated += power
            n_segments += 1
            start += hop

        if n_segments == 0:
            # Fallback: use entire series as single segment
            return SpectralTimeSeries.periodogram(series)

        psd = accumulated / n_segments

        # One-sided scaling
        if segment_length % 2 == 0:
            psd[1:-1] *= 2
        else:
            psd[1:] *= 2

        return freqs, psd

    @staticmethod
    def dominant_frequencies(series: np.ndarray, top_k: int = 3) -> list[tuple[float, float]]:
        """Find the top_k dominant frequencies.

        Uses the periodogram and finds peaks by selecting the k frequency
        bins with highest power, excluding DC (f=0).

        Args:
            series: Input time series.
            top_k: Number of dominant frequencies to return.

        Returns:
            [(frequency, power)] sorted by power descending.
        """
        freqs, power = SpectralTimeSeries.periodogram(series)

        # Exclude DC component (index 0)
        if len(freqs) > 1:
            freqs_no_dc = freqs[1:]
            power_no_dc = power[1:]
        else:
            return [(freqs[0], power[0])]

        # Get indices of top_k peaks
        k = min(top_k, len(power_no_dc))
        top_indices = np.argsort(power_no_dc)[-k:][::-1]

        return [(float(freqs_no_dc[i]), float(power_no_dc[i])) for i in top_indices]

    @staticmethod
    def spectral_density(series: np.ndarray, method: str = 'welch') -> np.ndarray:
        """Estimate power spectral density.

        Args:
            series: Input time series.
            method: 'welch' (default, lower variance) or 'periodogram' (higher resolution).

        Returns:
            Power spectral density array.
        """
        if method == 'welch':
            _, psd = SpectralTimeSeries.welch(series)
        elif method == 'periodogram':
            _, psd = SpectralTimeSeries.periodogram(series)
        else:
            logger.warning(f"Unknown method '{method}', falling back to Welch")
            _, psd = SpectralTimeSeries.welch(series)
        return psd


class MultitaperSpectral:
    """Multitaper spectral estimation using Slepian sequences.

    Better bias-variance tradeoff than single-taper methods. Uses multiple
    orthogonal tapers (DPSS / Slepian sequences) to produce independent
    spectral estimates, then averages them.

    The bandwidth parameter W controls the tradeoff:
    - Larger W = more bias, less variance (smoother estimate)
    - Smaller W = less bias, more variance (sharper peaks)
    """

    @staticmethod
    def _dpss(N: int, n_tapers: int, bandwidth: float) -> np.ndarray:
        """Compute Discrete Prolate Spheroidal Sequences (Slepian tapers).

        Approximated via the tridiagonal matrix method:
        The DPSS are eigenvectors of a tridiagonal matrix with entries:
          diagonal: ((N-1)/2 - n)^2 * cos(2*pi*W)
          off-diagonal: n*(N-n)/2

        Args:
            N: Length of the data.
            n_tapers: Number of tapers to compute.
            bandwidth: Half-bandwidth parameter W (in normalized frequency).

        Returns:
            Array of shape (n_tapers, N) containing the taper sequences.
        """
        W = bandwidth / N

        # Build the tridiagonal matrix
        n = np.arange(N, dtype=np.float64)
        diagonal = ((N - 1.0) / 2.0 - n) ** 2 * np.cos(2.0 * np.pi * W)

        off_diag = np.zeros(N - 1)
        for i in range(N - 1):
            off_diag[i] = (i + 1.0) * (N - 1.0 - i) / 2.0

        # Construct symmetric tridiagonal matrix
        T = np.diag(diagonal) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

        # Eigendecomposition — largest eigenvalues correspond to best tapers
        eigenvalues, eigenvectors = np.linalg.eigh(T)

        # Take the n_tapers with largest eigenvalues (they are sorted ascending)
        indices = np.argsort(eigenvalues)[-n_tapers:][::-1]
        tapers = eigenvectors[:, indices].T

        # Normalize each taper to unit energy
        for i in range(n_tapers):
            tapers[i] /= np.linalg.norm(tapers[i])

        return tapers

    @staticmethod
    def estimate(series: np.ndarray, n_tapers: int = 4,
                 bandwidth: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
        """Multitaper PSD estimate.

        Uses DPSS (discrete prolate spheroidal sequences) as tapers.
        Each taper produces an independent spectral estimate; averaging
        across tapers reduces variance while controlling spectral leakage.

        Args:
            series: Input time series (real-valued).
            n_tapers: Number of Slepian tapers (typically 2*bandwidth - 1).
            bandwidth: Half-bandwidth parameter controlling resolution.

        Returns:
            (frequencies, psd) — multitaper power spectral density estimate.
        """
        N = len(series)
        centered = series - np.mean(series)

        n_tapers = min(n_tapers, N - 1)  # Can't have more tapers than data - 1

        # Compute DPSS tapers
        tapers = MultitaperSpectral._dpss(N, n_tapers, bandwidth)

        freqs = np.fft.rfftfreq(N)
        n_freq_bins = len(freqs)
        psd_sum = np.zeros(n_freq_bins)

        for k in range(n_tapers):
            tapered = centered * tapers[k]
            fft_vals = np.fft.rfft(tapered)
            psd_sum += np.abs(fft_vals) ** 2

        # Average across tapers and normalize
        psd = psd_sum / (n_tapers * N)

        # One-sided scaling
        if N % 2 == 0:
            psd[1:-1] *= 2
        else:
            psd[1:] *= 2

        return freqs, psd
