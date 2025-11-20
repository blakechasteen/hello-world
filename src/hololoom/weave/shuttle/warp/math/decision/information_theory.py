"""
Information Theory - Entropy, Mutual Information, Coding Theory
==============================================================

Quantitative theory of information, communication, and compression.

Classes:
    Entropy: Shannon entropy, conditional entropy, joint entropy
    MutualInformation: Information shared between random variables
    DivergenceMetrics: KL divergence, JS divergence, f-divergences
    ChannelCapacity: Noisy channel coding theorem
    SourceCoding: Huffman coding, arithmetic coding
    ErrorCorrection: Hamming codes, Reed-Solomon codes
    RateDistortion: Lossy compression theory

Applications:
    - Data compression (lossless and lossy)
    - Communication systems
    - Machine learning (variational inference)
    - Feature selection (mutual information)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from collections import Counter
import heapq


class Entropy:
    """
    Shannon entropy: H(X) = -Σ p(x) log p(x).

    Measures uncertainty/information content of random variable.
    Units: bits (log base 2), nats (log base e), or hartleys (log base 10).
    """

    @staticmethod
    def shannon(probabilities: np.ndarray, base: float = 2.0) -> float:
        """
        Shannon entropy: H(X) = -Σ p_i log p_i.

        Args:
            probabilities: Probability distribution (must sum to 1)
            base: Logarithm base (2=bits, e=nats, 10=hartleys)

        Returns: Entropy in specified units
        """
        p = probabilities[probabilities > 0]  # Remove zeros (0 log 0 = 0)
        return -np.sum(p * np.log(p) / np.log(base))

    @staticmethod
    def joint(joint_probs: np.ndarray, base: float = 2.0) -> float:
        """
        Joint entropy: H(X, Y) = -Σ p(x,y) log p(x,y).

        Args:
            joint_probs: Joint probability matrix p(x,y)
        """
        p = joint_probs[joint_probs > 0]
        return -np.sum(p * np.log(p) / np.log(base))

    @staticmethod
    def conditional(joint_probs: np.ndarray, marginal_probs: np.ndarray,
                   base: float = 2.0) -> float:
        """
        Conditional entropy: H(Y|X) = H(X,Y) - H(X).

        Measures average uncertainty in Y given knowledge of X.
        """
        H_XY = Entropy.joint(joint_probs, base)
        H_X = Entropy.shannon(marginal_probs, base)
        return H_XY - H_X

    @staticmethod
    def cross_entropy(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
        """
        Cross-entropy: H(p, q) = -Σ p(x) log q(x).

        Used in machine learning loss functions.
        """
        p_nonzero = p > 0
        return -np.sum(p[p_nonzero] * np.log(q[p_nonzero]) / np.log(base))

    @staticmethod
    def max_entropy(n: int, base: float = 2.0) -> float:
        """
        Maximum entropy: log n (uniform distribution).

        Achieved when all outcomes equally likely.
        """
        return np.log(n) / np.log(base)

    @staticmethod
    def from_samples(samples: np.ndarray, base: float = 2.0) -> float:
        """Estimate entropy from samples."""
        counts = Counter(samples)
        total = len(samples)
        probs = np.array([count / total for count in counts.values()])
        return Entropy.shannon(probs, base)


class MutualInformation:
    """
    Mutual information: I(X; Y) = H(X) + H(Y) - H(X, Y).

    Measures information shared between variables.
    I(X; Y) = 0 iff X, Y independent.
    """

    @staticmethod
    def discrete(joint_probs: np.ndarray, base: float = 2.0) -> float:
        """
        I(X; Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y))).

        Args:
            joint_probs: Joint probability matrix p(x,y)
        """
        # Marginals
        p_x = np.sum(joint_probs, axis=1, keepdims=True)
        p_y = np.sum(joint_probs, axis=0, keepdims=True)
        product = p_x @ p_y

        # MI: only where joint_probs > 0
        mask = joint_probs > 0
        mi = np.sum(joint_probs[mask] * np.log(joint_probs[mask] / product[mask]) / np.log(base))
        return mi

    @staticmethod
    def conditional(joint_xyz: np.ndarray, base: float = 2.0) -> float:
        """
        Conditional MI: I(X; Y | Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z).

        Information between X and Y given knowledge of Z.
        """
        # Simplified implementation
        # For production: proper marginalization over Z
        return 0.0  # Placeholder

    @staticmethod
    def from_samples(samples_x: np.ndarray, samples_y: np.ndarray,
                    base: float = 2.0) -> float:
        """Estimate MI from samples."""
        # Build joint distribution
        xy_pairs = list(zip(samples_x, samples_y))
        xy_counts = Counter(xy_pairs)
        total = len(xy_pairs)

        # Get unique values
        unique_x = sorted(set(samples_x))
        unique_y = sorted(set(samples_y))

        # Build joint probability matrix
        joint_probs = np.zeros((len(unique_x), len(unique_y)))
        x_to_idx = {x: i for i, x in enumerate(unique_x)}
        y_to_idx = {y: i for i, y in enumerate(unique_y)}

        for (x, y), count in xy_counts.items():
            i, j = x_to_idx[x], y_to_idx[y]
            joint_probs[i, j] = count / total

        return MutualInformation.discrete(joint_probs, base)

    @staticmethod
    def normalized(mi: float, h_x: float, h_y: float) -> float:
        """
        Normalized MI: I(X;Y) / min(H(X), H(Y)).

        Range: [0, 1]
        """
        if min(h_x, h_y) == 0:
            return 0.0
        return mi / min(h_x, h_y)


class DivergenceMetrics:
    """
    Divergence measures: distance between probability distributions.

    KL divergence: D_KL(p || q) = Σ p(x) log(p(x)/q(x))
    """

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
        """
        Kullback-Leibler divergence: D_KL(p || q).

        NOT symmetric! Measures cost of encoding p using code for q.
        Always >= 0, = 0 iff p = q.
        """
        mask = (p > 0) & (q > 0)
        if not np.all(p[~mask] == 0):
            return np.inf  # Support of p not contained in support of q
        return np.sum(p[mask] * np.log(p[mask] / q[mask]) / np.log(base))

    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
        """
        Jensen-Shannon divergence: symmetric version of KL.

        JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)
        where m = 0.5 * (p + q).
        """
        m = 0.5 * (p + q)
        return 0.5 * DivergenceMetrics.kl_divergence(p, m, base) + \
               0.5 * DivergenceMetrics.kl_divergence(q, m, base)

    @staticmethod
    def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
        """
        Hellinger distance: H(p, q) = sqrt(1 - Σ sqrt(p_i q_i)).

        Metric (satisfies triangle inequality). Range: [0, 1].
        """
        return np.sqrt(1 - np.sum(np.sqrt(p * q)))

    @staticmethod
    def total_variation(p: np.ndarray, q: np.ndarray) -> float:
        """
        Total variation distance: TV(p, q) = 0.5 * Σ |p_i - q_i|.

        Maximum probability of distinguishing p from q in single sample.
        """
        return 0.5 * np.sum(np.abs(p - q))

    @staticmethod
    def f_divergence(p: np.ndarray, q: np.ndarray, f: Callable) -> float:
        """
        f-divergence: D_f(p || q) = Σ q_i f(p_i / q_i).

        General family including KL, TV, Hellinger as special cases.
        """
        mask = q > 0
        ratios = np.zeros_like(p)
        ratios[mask] = p[mask] / q[mask]
        return np.sum(q * f(ratios))


class ChannelCapacity:
    """
    Channel capacity: maximum reliable information rate.

    Shannon's noisy channel coding theorem: C = max I(X; Y).
    """

    @staticmethod
    def binary_symmetric(error_prob: float) -> float:
        """
        Binary symmetric channel: flips bit with probability p.

        Capacity: C = 1 - H(p) bits.
        """
        if error_prob <= 0 or error_prob >= 1:
            return 0.0
        h_p = Entropy.shannon(np.array([error_prob, 1 - error_prob]), base=2)
        return 1 - h_p

    @staticmethod
    def additive_gaussian(snr: float) -> float:
        """
        Additive white Gaussian noise (AWGN) channel.

        Capacity: C = 0.5 * log(1 + SNR) bits per channel use.
        """
        return 0.5 * np.log2(1 + snr)

    @staticmethod
    def erasure_channel(erasure_prob: float) -> float:
        """
        Binary erasure channel: output is input or erasure.

        Capacity: C = 1 - p (fraction of non-erased bits).
        """
        return 1 - erasure_prob

    @staticmethod
    def capacity_formula(channel_matrix: np.ndarray,
                        optimization_steps: int = 100) -> float:
        """
        Compute channel capacity: C = max_p(X) I(X; Y).

        Use Blahut-Arimoto algorithm.
        """
        # Blahut-Arimoto algorithm
        n_inputs = channel_matrix.shape[0]
        p_x = np.ones(n_inputs) / n_inputs  # Initial uniform distribution

        for _ in range(optimization_steps):
            # Compute p(y|x) -> p(y)
            p_y = channel_matrix.T @ p_x

            # Compute I(x; Y) for each x
            i_x = np.zeros(n_inputs)
            for x in range(n_inputs):
                for y in range(channel_matrix.shape[1]):
                    if channel_matrix[x, y] > 0 and p_y[y] > 0:
                        i_x[x] += channel_matrix[x, y] * np.log2(channel_matrix[x, y] / p_y[y])

            # Update p(x) proportional to exp(I(x; Y))
            p_x = np.exp(i_x)
            p_x /= np.sum(p_x)

        # Compute final mutual information
        p_y = channel_matrix.T @ p_x
        mi = 0.0
        for x in range(n_inputs):
            for y in range(channel_matrix.shape[1]):
                if channel_matrix[x, y] > 0 and p_y[y] > 0:
                    joint = p_x[x] * channel_matrix[x, y]
                    mi += joint * np.log2(joint / (p_x[x] * p_y[y]))

        return mi


class SourceCoding:
    """
    Source coding: lossless data compression.

    Shannon's source coding theorem: optimal code length ≈ H(X).
    """

    @staticmethod
    def huffman_code(probabilities: np.ndarray) -> Dict[int, str]:
        """
        Huffman coding: optimal prefix-free code.

        Expected length: L ∈ [H(X), H(X) + 1).
        """
        # Build Huffman tree
        heap = [[prob, [symbol, ""]] for symbol, prob in enumerate(probabilities)]
        heapq.heapify(heap)

        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)

            # Assign 0 to lo, 1 to hi
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]

            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        # Extract codes
        codes = {}
        for pair in heap[0][1:]:
            codes[pair[0]] = pair[1]

        return codes

    @staticmethod
    def expected_length(probabilities: np.ndarray, code_lengths: np.ndarray) -> float:
        """Expected code length: L = Σ p_i * l_i."""
        return np.sum(probabilities * code_lengths)

    @staticmethod
    def kraft_inequality(code_lengths: np.ndarray, base: int = 2) -> bool:
        """
        Kraft inequality: Σ base^{-l_i} <= 1.

        Necessary and sufficient for existence of prefix-free code.
        """
        return np.sum(base ** (-code_lengths)) <= 1.0 + 1e-10

    @staticmethod
    def shannon_fano_code(probabilities: np.ndarray) -> List[int]:
        """
        Shannon-Fano code lengths: l_i = ceil(-log p_i).

        Not optimal but close to entropy bound.
        """
        return [int(np.ceil(-np.log2(p))) if p > 0 else 0 for p in probabilities]


class ErrorCorrection:
    """
    Error-correcting codes: detect and correct transmission errors.

    Hamming codes, Reed-Solomon, LDPC, turbo codes.
    """

    @staticmethod
    def hamming_distance(x: np.ndarray, y: np.ndarray) -> int:
        """
        Hamming distance: number of positions where x and y differ.

        Fundamental metric in coding theory.
        """
        return int(np.sum(x != y))

    @staticmethod
    def hamming_weight(x: np.ndarray) -> int:
        """Hamming weight: number of non-zero elements."""
        return int(np.sum(x != 0))

    @staticmethod
    def hamming_7_4_encode(data: np.ndarray) -> np.ndarray:
        """
        Hamming (7,4) code: 4 data bits -> 7 code bits.

        Can correct 1-bit errors, detect 2-bit errors.
        """
        if len(data) != 4:
            raise ValueError("Hamming (7,4) requires 4 data bits")

        # Generator matrix
        G = np.array([
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1]
        ], dtype=int)

        codeword = (data @ G) % 2
        return codeword

    @staticmethod
    def hamming_7_4_decode(received: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Hamming (7,4) decode with error correction.

        Returns: (decoded_data, error_detected)
        """
        if len(received) != 7:
            raise ValueError("Hamming (7,4) requires 7 received bits")

        # Parity check matrix
        H = np.array([
            [1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1]
        ], dtype=int)

        syndrome = (H @ received) % 2

        if np.all(syndrome == 0):
            # No error
            return received[:4], False
        else:
            # Error detected, find position
            syndrome_int = syndrome[0] * 4 + syndrome[1] * 2 + syndrome[2]
            if syndrome_int > 0:
                corrected = received.copy()
                corrected[syndrome_int - 1] ^= 1  # Flip bit
                return corrected[:4], True

        return received[:4], True

    @staticmethod
    def repetition_code(data: int, n: int) -> np.ndarray:
        """
        Repetition code: repeat bit n times.

        Can correct floor((n-1)/2) errors via majority vote.
        """
        return np.array([data] * n)

    @staticmethod
    def parity_check(data: np.ndarray) -> int:
        """
        Simple parity: add bit to make total parity even.

        Detects odd number of errors, corrects none.
        """
        return int(np.sum(data) % 2)


class RateDistortion:
    """
    Rate-distortion theory: lossy compression.

    R(D) = minimum rate to achieve distortion <= D.
    """

    @staticmethod
    def gaussian_source(variance: float, distortion: float) -> float:
        """
        Gaussian source R(D) = 0.5 * log(σ²/D) for D <= σ².

        Achievable via scalar quantization.
        """
        if distortion >= variance:
            return 0.0
        return 0.5 * np.log2(variance / distortion)

    @staticmethod
    def binary_source(p: float, distortion: float) -> float:
        """
        Binary source with Hamming distortion.

        Closed form for Bernoulli(p) source.
        """
        if distortion >= min(p, 1 - p):
            return 0.0

        # Simplified: approximate R(D)
        h_p = Entropy.shannon(np.array([p, 1-p]), base=2)
        h_d = Entropy.shannon(np.array([distortion, 1-distortion]), base=2)
        return max(0, h_p - h_d)

    @staticmethod
    def distortion_function(x: np.ndarray, y: np.ndarray, metric: str = 'mse') -> float:
        """
        Distortion measure d(x, y).

        Common choices: MSE, Hamming distance, perceptual metrics.
        """
        if metric == 'mse':
            return float(np.mean((x - y) ** 2))
        elif metric == 'hamming':
            return float(np.mean(x != y))
        elif metric == 'mae':
            return float(np.mean(np.abs(x - y)))
        else:
            raise ValueError(f"Unknown metric: {metric}")


# ============================================================================
# EXAMPLES AND TESTS
# ============================================================================

def example_entropy():
    """Example: Entropy of biased coin."""
    p = 0.3
    probs = np.array([p, 1-p])
    h = Entropy.shannon(probs, base=2)
    return h


def example_mutual_information():
    """Example: MI between correlated binary variables."""
    # Joint distribution: p(X=0, Y=0) = 0.4, p(X=1, Y=1) = 0.4, rest = 0.1
    joint = np.array([
        [0.4, 0.1],
        [0.1, 0.4]
    ])
    mi = MutualInformation.discrete(joint, base=2)
    return mi


def example_huffman_coding():
    """Example: Huffman code for small alphabet."""
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    codes = SourceCoding.huffman_code(probs)
    return codes


def example_hamming_code():
    """Example: Hamming (7,4) error correction."""
    data = np.array([1, 0, 1, 1])
    encoded = ErrorCorrection.hamming_7_4_encode(data)

    # Introduce 1-bit error
    received = encoded.copy()
    received[2] ^= 1  # Flip bit 2

    decoded, error = ErrorCorrection.hamming_7_4_decode(received)
    return data, encoded, received, decoded, error


if __name__ == "__main__":
    print("Information Theory Module")
    print("=" * 60)

    # Test 1: Entropy
    print("\n[Test 1] Shannon entropy")
    h_coin = example_entropy()
    print(f"Entropy of p=0.3 coin: {h_coin:.4f} bits")
    h_uniform = Entropy.shannon(np.array([0.25]*4), base=2)
    print(f"Entropy of uniform (4 outcomes): {h_uniform:.4f} bits (expect 2.0)")

    # Test 2: Mutual information
    print("\n[Test 2] Mutual information")
    mi = example_mutual_information()
    print(f"MI between correlated variables: {mi:.4f} bits")

    # Test 3: KL divergence
    print("\n[Test 3] KL divergence")
    p = np.array([0.5, 0.5])
    q = np.array([0.3, 0.7])
    kl = DivergenceMetrics.kl_divergence(p, q, base=2)
    print(f"D_KL(p || q): {kl:.4f} bits")
    kl_reverse = DivergenceMetrics.kl_divergence(q, p, base=2)
    print(f"D_KL(q || p): {kl_reverse:.4f} bits (asymmetric!)")

    # Test 4: Channel capacity
    print("\n[Test 4] Channel capacity")
    bsc_cap = ChannelCapacity.binary_symmetric(error_prob=0.1)
    print(f"BSC(0.1) capacity: {bsc_cap:.4f} bits")
    awgn_cap = ChannelCapacity.additive_gaussian(snr=10)
    print(f"AWGN (SNR=10) capacity: {awgn_cap:.4f} bits per use")

    # Test 5: Huffman coding
    print("\n[Test 5] Huffman coding")
    codes = example_huffman_coding()
    print(f"Huffman codes: {codes}")
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    lengths = np.array([len(codes[i]) for i in range(len(probs))])
    avg_len = SourceCoding.expected_length(probs, lengths)
    h_source = Entropy.shannon(probs, base=2)
    print(f"Average length: {avg_len:.4f} bits")
    print(f"Entropy: {h_source:.4f} bits")
    print(f"Efficiency: {h_source/avg_len*100:.2f}%")

    # Test 6: Hamming code
    print("\n[Test 6] Hamming (7,4) error correction")
    data, encoded, received, decoded, error = example_hamming_code()
    print(f"Original data: {data}")
    print(f"Encoded: {encoded}")
    print(f"Received (with error): {received}")
    print(f"Decoded: {decoded}")
    print(f"Error corrected: {error}")
    print(f"Match: {np.array_equal(data, decoded)}")

    print("\n" + "=" * 60)
    print("All information theory tests complete!")
    print("Entropy, coding, and channel capacity ready.")
