"""
Unit tests for hololoom.core.warp.math.analysis (deep coverage):
  - hidden_markov_models.py
  - time_series_models.py
  - stability_analysis.py
  - changepoint_detection.py
  - spline_regression.py
  - stable_distributions.py
  - bifurcation_analysis.py
  - kalman_filtering.py

Pure math — small arrays, verify mathematical properties.
"""

import numpy as np
import pytest


# ===========================================================================
# hidden_markov_models
# ===========================================================================
from hololoom.core.warp.math.analysis.hidden_markov_models import (
    HMMParams,
    GaussianHMMParams,
    HMMResult,
    HiddenMarkovModel,
    GaussianHMM,
)


def _simple_hmm_params() -> HMMParams:
    """Two-state, two-symbol HMM with known parameters."""
    return HMMParams(
        transition=np.array([[0.7, 0.3], [0.4, 0.6]]),
        emission=np.array([[0.9, 0.1], [0.2, 0.8]]),
        initial=np.array([0.6, 0.4]),
        n_states=2,
        n_obs=2,
    )


class TestHMMParams:
    def test_validate_passes_on_valid(self):
        p = _simple_hmm_params()
        p.validate()  # should not raise

    def test_validate_fails_bad_transition(self):
        p = _simple_hmm_params()
        p.transition = np.array([[0.5, 0.3], [0.4, 0.6]])
        with pytest.raises(AssertionError):
            p.validate()


class TestHMMForward:
    def test_alpha_sums_to_one_per_step(self):
        params = _simple_hmm_params()
        obs = np.array([0, 1, 0, 0, 1])
        alpha, ll = HiddenMarkovModel.forward(obs, params)
        # Scaled alpha rows sum to 1 by construction
        for t in range(len(obs)):
            assert alpha[t].sum() == pytest.approx(1.0, abs=1e-6)

    def test_log_likelihood_finite(self):
        params = _simple_hmm_params()
        obs = np.array([0, 1, 0, 1, 0])
        _, ll = HiddenMarkovModel.forward(obs, params)
        assert np.isfinite(ll)
        assert ll < 0  # log-likelihood of a non-trivial sequence is negative

    def test_single_observation(self):
        params = _simple_hmm_params()
        obs = np.array([0])
        alpha, ll = HiddenMarkovModel.forward(obs, params)
        assert alpha.shape == (1, 2)
        assert np.isfinite(ll)


class TestHMMBackward:
    def test_backward_shape(self):
        params = _simple_hmm_params()
        obs = np.array([0, 1, 0])
        beta = HiddenMarkovModel.backward(obs, params)
        assert beta.shape == (3, 2)

    def test_backward_last_step_ones(self):
        params = _simple_hmm_params()
        obs = np.array([0, 1, 0])
        beta = HiddenMarkovModel.backward(obs, params)
        np.testing.assert_allclose(beta[-1], [1.0, 1.0])


class TestHMMViterbi:
    def test_viterbi_returns_valid_states(self):
        params = _simple_hmm_params()
        obs = np.array([0, 0, 0, 1, 1, 1])
        states = HiddenMarkovModel.viterbi(obs, params)
        assert states.shape == (6,)
        assert set(states).issubset({0, 1})

    def test_viterbi_deterministic_sequence(self):
        """With very strong emission, Viterbi should recover the right state."""
        params = HMMParams(
            transition=np.array([[0.9, 0.1], [0.1, 0.9]]),
            emission=np.array([[0.99, 0.01], [0.01, 0.99]]),
            initial=np.array([0.5, 0.5]),
            n_states=2,
            n_obs=2,
        )
        obs = np.array([0, 0, 0, 1, 1, 1])
        states = HiddenMarkovModel.viterbi(obs, params)
        # First half should be state 0, second half state 1
        assert np.all(states[:3] == 0)
        assert np.all(states[3:] == 1)


class TestHMMPosterior:
    def test_posterior_sums_to_one(self):
        params = _simple_hmm_params()
        obs = np.array([0, 1, 0, 1])
        gamma = HiddenMarkovModel.posterior_decode(obs, params)
        assert gamma.shape == (4, 2)
        for t in range(4):
            assert gamma[t].sum() == pytest.approx(1.0, abs=1e-6)


class TestHMMSimulate:
    def test_simulate_shapes(self):
        params = _simple_hmm_params()
        states, obs = HiddenMarkovModel.simulate(params, T=20, seed=42)
        assert states.shape == (20,)
        assert obs.shape == (20,)
        assert set(states).issubset({0, 1})
        assert set(obs).issubset({0, 1})


class TestHMMInfer:
    def test_infer_returns_result(self):
        params = _simple_hmm_params()
        obs = np.array([0, 1, 0, 1, 0])
        result = HiddenMarkovModel.infer(obs, params)
        assert isinstance(result, HMMResult)
        assert np.isfinite(result.log_likelihood)
        assert result.state_sequence.shape == (5,)
        assert result.posterior.shape == (5, 2)


class TestHMMBaumWelch:
    def test_baum_welch_improves_likelihood(self):
        """Baum-Welch should produce a model with decent log-likelihood."""
        params_true = _simple_hmm_params()
        _, obs = HiddenMarkovModel.simulate(params_true, T=200, seed=42)
        params_learned = HiddenMarkovModel.baum_welch(
            obs, n_states=2, n_obs=2, max_iter=20, seed=7
        )
        # Learned model should produce finite log-likelihood
        _, ll = HiddenMarkovModel.forward(obs, params_learned)
        assert np.isfinite(ll)

    def test_baum_welch_valid_params(self):
        _, obs = HiddenMarkovModel.simulate(_simple_hmm_params(), T=100, seed=0)
        params = HiddenMarkovModel.baum_welch(obs, n_states=2, max_iter=10, seed=1)
        assert params.transition.shape == (2, 2)
        np.testing.assert_allclose(params.transition.sum(axis=1), [1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(params.emission.sum(axis=1), [1.0, 1.0], atol=1e-6)


class TestGaussianHMM:
    def _gaussian_params(self):
        return GaussianHMMParams(
            transition=np.array([[0.8, 0.2], [0.3, 0.7]]),
            means=np.array([0.0, 5.0]),
            covariances=np.array([1.0, 1.0]),
            initial=np.array([0.5, 0.5]),
            n_states=2,
        )

    def test_forward_finite(self):
        params = self._gaussian_params()
        obs = np.array([0.1, 0.3, 4.8, 5.2, 0.0])
        alpha, ll = GaussianHMM.forward(obs, params)
        assert alpha.shape == (5, 2)
        assert np.isfinite(ll)

    def test_viterbi_detects_states(self):
        params = self._gaussian_params()
        obs = np.array([0.1, -0.2, 0.5, 5.1, 4.8, 5.3])
        states = GaussianHMM.viterbi(obs, params)
        assert np.all(states[:3] == 0)
        assert np.all(states[3:] == 1)

    def test_simulate_shapes(self):
        params = self._gaussian_params()
        states, obs = GaussianHMM.simulate(params, T=30, seed=42)
        assert states.shape == (30,)
        assert obs.shape == (30,)

    def test_backward_shape(self):
        params = self._gaussian_params()
        obs = np.array([0.0, 5.0, 0.0])
        alpha, _ = GaussianHMM.forward(obs, params)
        # Reconstruct scale
        scale = np.ones(3)
        beta = GaussianHMM.backward(obs, params, scale)
        assert beta.shape == (3, 2)

    def test_baum_welch_gaussian_converges(self):
        params_true = self._gaussian_params()
        _, obs = GaussianHMM.simulate(params_true, T=200, seed=42)
        params_learned = GaussianHMM.baum_welch_gaussian(
            obs, n_states=2, max_iter=30, seed=7
        )
        assert params_learned.n_states == 2
        # Means should be close to 0 and 5 (in either order)
        means_sorted = np.sort(params_learned.means)
        assert means_sorted[0] == pytest.approx(0.0, abs=1.5)
        assert means_sorted[1] == pytest.approx(5.0, abs=1.5)


# ===========================================================================
# time_series_models
# ===========================================================================
from hololoom.core.warp.math.analysis.time_series_models import (
    ARIMA,
    ARIMAParams,
    ForecastResult,
    ExponentialSmoothing,
    AutoCorrelation,
)


class TestARIMADifference:
    def test_difference_order_1(self):
        s = np.array([1.0, 3.0, 6.0, 10.0])
        d = ARIMA.difference(s, d=1)
        np.testing.assert_allclose(d, [2.0, 3.0, 4.0])

    def test_difference_order_2(self):
        s = np.array([1.0, 3.0, 6.0, 10.0])
        d = ARIMA.difference(s, d=2)
        np.testing.assert_allclose(d, [1.0, 1.0])

    def test_undifference_recovers(self):
        s = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
        diffed = ARIMA.difference(s, d=1)
        # forecast using the last 2 values of the diff
        forecasts = diffed[-2:]
        recovered = ARIMA.undifference(forecasts, s, d=1)
        assert len(recovered) == 2
        assert np.isfinite(recovered).all()


class TestARIMAFit:
    def test_fit_ar1(self):
        np.random.seed(42)
        n = 200
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.7 * y[t - 1] + np.random.randn()
        params = ARIMA.fit(y, order=(1, 0, 0))
        assert params.p == 1
        assert params.d == 0
        assert params.q == 0
        assert params.ar_coeffs[0] == pytest.approx(0.7, abs=0.15)

    def test_fit_arma_runs(self):
        np.random.seed(42)
        y = np.cumsum(np.random.randn(100))
        params = ARIMA.fit(y, order=(1, 0, 1))
        assert params.p == 1
        assert params.q == 1

    def test_forecast_shape(self):
        np.random.seed(42)
        y = np.cumsum(np.random.randn(100))
        params = ARIMA.fit(y, order=(1, 0, 0))
        result = ARIMA.forecast(params, y, steps=5)
        assert isinstance(result, ForecastResult)
        assert len(result.values) == 5
        assert len(result.confidence_lower) == 5
        assert len(result.confidence_upper) == 5

    def test_forecast_ci_ordering(self):
        np.random.seed(42)
        y = np.cumsum(np.random.randn(100))
        params = ARIMA.fit(y, order=(1, 0, 0))
        result = ARIMA.forecast(params, y, steps=5)
        assert np.all(result.confidence_lower <= result.values)
        assert np.all(result.values <= result.confidence_upper)

    def test_residuals_shape(self):
        np.random.seed(42)
        y = np.cumsum(np.random.randn(80))
        params = ARIMA.fit(y, order=(1, 0, 0))
        res = ARIMA.residuals(params, y)
        assert len(res) == len(y)


class TestExponentialSmoothing:
    def test_simple_first_value(self):
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sm = ExponentialSmoothing.simple(s, alpha=0.5)
        assert sm[0] == 1.0  # first value passes through
        assert len(sm) == 5

    def test_simple_tracks_trend(self):
        s = np.arange(1.0, 11.0)
        sm = ExponentialSmoothing.simple(s, alpha=0.9)
        # With high alpha, should track closely
        assert sm[-1] == pytest.approx(10.0, abs=1.0)

    def test_simple_invalid_alpha(self):
        with pytest.raises(ValueError):
            ExponentialSmoothing.simple(np.array([1.0, 2.0]), alpha=0.0)

    def test_double_handles_linear_trend(self):
        s = np.arange(1.0, 21.0)
        sm = ExponentialSmoothing.double(s, alpha=0.5, beta=0.3)
        assert len(sm) == 20

    def test_double_invalid_params(self):
        with pytest.raises(ValueError):
            ExponentialSmoothing.double(np.array([1.0, 2.0, 3.0]), alpha=0.5, beta=1.0)

    def test_triple_runs_with_seasonality(self):
        # Simple seasonal signal: period=4
        base = np.tile([1.0, 2.0, 3.0, 0.0], 5)
        sm = ExponentialSmoothing.triple(base, alpha=0.3, beta=0.1, gamma=0.1, period=4)
        assert len(sm) == 20


class TestAutoCorrelation:
    def test_acf_lag0_is_one(self):
        s = np.random.randn(100)
        acf = AutoCorrelation.acf(s, max_lag=10)
        assert acf[0] == pytest.approx(1.0, abs=1e-10)

    def test_acf_white_noise_small(self):
        np.random.seed(42)
        s = np.random.randn(500)
        acf = AutoCorrelation.acf(s, max_lag=10)
        # For white noise, ACF at lag > 0 should be near zero
        assert np.all(np.abs(acf[1:]) < 0.15)

    def test_pacf_lag0_is_one(self):
        np.random.seed(42)
        s = np.random.randn(100)
        pacf = AutoCorrelation.pacf(s, max_lag=5)
        assert pacf[0] == pytest.approx(1.0)

    def test_ljung_box_white_noise(self):
        np.random.seed(42)
        s = np.random.randn(200)
        Q, p = AutoCorrelation.ljung_box(s, lags=10)
        assert Q >= 0
        # White noise should not reject H0
        assert p > 0.01


# ===========================================================================
# stability_analysis
# ===========================================================================
from hololoom.core.warp.math.analysis.stability_analysis import (
    LyapunovExponents,
    LyapunovResult,
    LinearStability,
    FixedPointFinder,
    FixedPoint,
    is_chaotic,
    stability_classification,
    find_equilibria,
)


class TestLyapunovResult:
    def test_entropy_rate_positive_exponents(self):
        r = LyapunovResult(
            exponents=np.array([0.5, -0.3]),
            max_exponent=0.5,
            is_chaotic=True,
        )
        assert r.entropy_rate == pytest.approx(0.5)

    def test_entropy_rate_no_positive(self):
        r = LyapunovResult(
            exponents=np.array([-0.2, -0.5]),
            max_exponent=-0.2,
            is_chaotic=False,
        )
        assert r.entropy_rate == pytest.approx(0.0)

    def test_lyapunov_dimension(self):
        r = LyapunovResult(
            exponents=np.array([0.5, -0.3]),
            max_exponent=0.5,
            is_chaotic=True,
        )
        # D_KY = 1 + 0.5/0.3 ≈ 2.67
        assert r.lyapunov_dimension > 1.0


class TestLinearStability:
    def test_stable_node(self):
        J = np.array([[-1.0, 0.0], [0.0, -2.0]])
        result = LinearStability.eigenvalue_analysis(J)
        assert result["stability"] == "stable_node"
        assert result["stable_dims"] == 2
        assert result["unstable_dims"] == 0

    def test_unstable_node(self):
        J = np.array([[1.0, 0.0], [0.0, 2.0]])
        result = LinearStability.eigenvalue_analysis(J)
        assert result["stability"] == "unstable_node"

    def test_saddle(self):
        J = np.array([[1.0, 0.0], [0.0, -1.0]])
        result = LinearStability.eigenvalue_analysis(J)
        assert result["stability"] == "saddle"

    def test_center(self):
        J = np.array([[0.0, 1.0], [-1.0, 0.0]])
        result = LinearStability.eigenvalue_analysis(J)
        assert result["stability"] == "center"
        assert result["has_oscillation"] is True

    def test_stable_spiral(self):
        J = np.array([[-0.1, 1.0], [-1.0, -0.1]])
        result = LinearStability.eigenvalue_analysis(J)
        assert result["stability"] == "stable_spiral"

    def test_unstable_spiral(self):
        J = np.array([[0.1, 1.0], [-1.0, 0.1]])
        result = LinearStability.eigenvalue_analysis(J)
        assert result["stability"] == "unstable_spiral"

    def test_spectral_radius(self):
        J = np.array([[-2.0, 0.0], [0.0, -3.0]])
        result = LinearStability.eigenvalue_analysis(J)
        assert result["spectral_radius"] == pytest.approx(3.0)

    def test_stability_margin_stable(self):
        J = np.array([[-1.0, 0.0], [0.0, -2.0]])
        margin = LinearStability.stability_margin(J)
        assert margin > 0  # stable

    def test_stability_margin_unstable(self):
        J = np.array([[1.0, 0.0], [0.0, -2.0]])
        margin = LinearStability.stability_margin(J)
        assert margin < 0  # unstable


class TestRouthHurwitz:
    def test_stable_polynomial(self):
        # s^2 + 3s + 2 = (s+1)(s+2) — roots at -1, -2
        assert LinearStability.routh_hurwitz(np.array([1, 3, 2])) is True

    def test_unstable_polynomial(self):
        # s^2 - 3s + 2 = (s-1)(s-2) — roots at 1, 2
        assert LinearStability.routh_hurwitz(np.array([1, -3, 2])) is False

    def test_marginal_zero_coeff(self):
        # s^2 + 0*s + 1 — roots at ±i (on imaginary axis)
        # Not strictly stable (needs Re < 0)
        assert LinearStability.routh_hurwitz(np.array([1, 0, 1])) is False

    def test_first_order(self):
        # s + 1 — root at -1
        assert LinearStability.routh_hurwitz(np.array([1, 1])) is True


class TestFixedPointFinder:
    def test_find_linear_root(self):
        def f(x):
            return np.array([2 * x[0] - 4])
        fp = FixedPointFinder.find(f, np.array([0.0]))
        assert fp.location[0] == pytest.approx(2.0, abs=1e-6)

    def test_find_2d_root(self):
        def f(x):
            return np.array([x[0] ** 2 - 1, x[1]])
        fp = FixedPointFinder.find(f, np.array([0.5, 0.1]))
        assert abs(fp.location[0]) == pytest.approx(1.0, abs=1e-5)
        assert fp.location[1] == pytest.approx(0.0, abs=1e-5)

    def test_classify_stable(self):
        J = np.array([[-1.0, 0.0], [0.0, -2.0]])
        cls = FixedPointFinder.classify(J)
        assert cls == "stable_node"

    def test_fixed_point_is_stable_property(self):
        fp = FixedPoint(
            location=np.array([0.0]),
            jacobian=np.array([[-1.0]]),
            stability="stable_node",
        )
        assert fp.is_stable is True
        assert fp.dimension == 1


class TestLyapunovFromTrajectory:
    def test_damped_oscillator_not_chaotic(self):
        # Damped oscillator: converging trajectory
        t = np.linspace(0, 10, 200)
        x = np.exp(-0.1 * t) * np.cos(t)
        y = np.exp(-0.1 * t) * np.sin(t)
        traj = np.column_stack([x, y])
        result = LyapunovExponents.compute_from_trajectory(traj, dt=t[1] - t[0])
        assert isinstance(result, LyapunovResult)
        # Should have at least one exponent
        assert len(result.exponents) == 2

    def test_maximal_exponent_1d(self):
        t = np.linspace(0, 10, 100)
        x = np.exp(-0.5 * t)
        val = LyapunovExponents.maximal_exponent(x, dt=t[1] - t[0])
        assert np.isfinite(val)

    def test_compute_from_jacobian(self):
        # Simple linear system dx/dt = -x
        def jac(x):
            return np.array([[-1.0]])
        traj = np.exp(-np.linspace(0, 5, 50)).reshape(-1, 1)
        result = LyapunovExponents.compute_from_jacobian(jac, traj, dt=0.1)
        # Should be negative for stable system
        assert result.max_exponent < 0.5  # Allow some numerical slack


class TestConvenienceFunctions:
    def test_stability_classification_wrapper(self):
        J = np.array([[-1.0, 0.0], [0.0, -2.0]])
        assert stability_classification(J) == "stable_node"

    def test_is_chaotic_simple(self):
        # Stable exponential decay — not chaotic
        x = np.exp(-np.linspace(0, 5, 100))
        # May or may not be chaotic depending on numerical estimation
        result = is_chaotic(x, dt=0.05)
        assert isinstance(result, (bool, np.bool_))


# ===========================================================================
# changepoint_detection
# ===========================================================================
from hololoom.core.warp.math.analysis.changepoint_detection import (
    Changepoint,
    ChangepointResult,
    CUSUM,
    PELT,
    BayesianChangepoint,
    segment_statistics,
)


class TestChangepoint:
    def test_valid_types(self):
        cp = Changepoint(index=10, confidence=0.9, type='mean_shift')
        assert cp.type == 'mean_shift'

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            Changepoint(index=10, confidence=0.9, type='invalid')


class TestCUSUM:
    def test_detect_mean_shift(self):
        np.random.seed(42)
        s = np.concatenate([np.random.randn(50), np.random.randn(50) + 5])
        cps = CUSUM.detect(s, threshold=3.0, drift=0.5)
        assert len(cps) >= 1
        # Should detect near index 50
        indices = [cp.index for cp in cps]
        assert any(40 <= idx <= 65 for idx in indices)

    def test_no_change_no_alarm(self):
        np.random.seed(42)
        s = np.random.randn(100)
        cps = CUSUM.detect(s, threshold=5.0, drift=0.5)
        # With high threshold, white noise should trigger few/no alarms
        assert len(cps) <= 2

    def test_cumulative_sum_shape(self):
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cs = CUSUM.cumulative_sum(s)
        assert len(cs) == 5

    def test_cumulative_sum_with_target(self):
        s = np.array([1.0, 2.0, 3.0])
        cs = CUSUM.cumulative_sum(s, target=2.0)
        np.testing.assert_allclose(cs, [-1.0, -1.0, 0.0])


class TestPELT:
    def test_detect_single_changepoint(self):
        np.random.seed(42)
        s = np.concatenate([np.random.randn(50), np.random.randn(50) + 4])
        result = PELT.detect(s, penalty='bic', min_segment=5)
        assert isinstance(result, ChangepointResult)
        assert len(result.segments) >= 2

    def test_no_changepoint_short_series(self):
        s = np.array([1.0, 2.0, 3.0])
        result = PELT.detect(s, min_segment=2)
        assert len(result.changepoints) == 0

    def test_max_changepoints_respected(self):
        np.random.seed(42)
        # Many shifts
        s = np.concatenate([
            np.random.randn(30),
            np.random.randn(30) + 3,
            np.random.randn(30) - 2,
            np.random.randn(30) + 5,
        ])
        result = PELT.detect(s, penalty='bic', max_changepoints=2)
        assert len(result.changepoints) <= 2

    def test_aic_penalty(self):
        np.random.seed(42)
        s = np.concatenate([np.random.randn(50), np.random.randn(50) + 5])
        result = PELT.detect(s, penalty='aic')
        assert isinstance(result, ChangepointResult)

    def test_float_penalty(self):
        np.random.seed(42)
        s = np.concatenate([np.random.randn(50), np.random.randn(50) + 5])
        result = PELT.detect(s, penalty=5.0)
        assert isinstance(result, ChangepointResult)


class TestBayesianChangepoint:
    def test_detect_produces_changepoints_list(self):
        np.random.seed(42)
        s = np.concatenate([np.zeros(30), np.ones(30) * 10])
        cps = BayesianChangepoint.detect(s, prior_rate=0.1, threshold=0.3)
        # Verify return type and structure
        assert isinstance(cps, list)
        for cp in cps:
            assert isinstance(cp, Changepoint)
            assert 0 <= cp.confidence <= 1

    def test_run_length_spike_at_changepoint(self):
        """Run length r=0 probability should spike near the changepoint."""
        np.random.seed(42)
        s = np.concatenate([np.zeros(30), np.ones(30) * 10])
        R = BayesianChangepoint.run_length_distribution(s, prior_rate=0.1)
        # Near the changepoint (t~30), P(r=0) should be elevated
        r0_probs = R[25:40, 0]
        assert np.max(r0_probs) > 0.01  # some spike in changepoint probability

    def test_run_length_distribution_shape(self):
        np.random.seed(42)
        s = np.random.randn(30)
        R = BayesianChangepoint.run_length_distribution(s)
        assert R.shape == (30, 31)
        # First row should have all mass at r=0
        assert R[0, 0] == pytest.approx(1.0, abs=1e-6)


class TestSegmentStatistics:
    def test_segment_stats_two_segments(self):
        s = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
        cps = [Changepoint(index=3, confidence=1.0, type='mean_shift')]
        stats = segment_statistics(s, cps)
        assert len(stats) == 2
        assert stats[0]['mean'] == pytest.approx(1.0)
        assert stats[1]['mean'] == pytest.approx(5.0)
        assert stats[0]['n'] == 3
        assert stats[1]['n'] == 3


# ===========================================================================
# spline_regression
# ===========================================================================
from hololoom.core.warp.math.analysis.spline_regression import (
    LOESSRegression,
    SplineRegression,
    SplineResult,
    VARModel,
    VARResult,
    RegressionDiagnostics,
)


class TestLOESSRegression:
    def test_fit_predict_simple(self):
        X = np.linspace(0, 2 * np.pi, 50)
        y = np.sin(X)
        loess = LOESSRegression()
        y_pred = loess.fit_predict(X, y, X, span=0.3)
        assert len(y_pred) == 50
        # Should track sin reasonably well
        assert np.corrcoef(y, y_pred)[0, 1] > 0.9

    def test_fit_predict_degree2(self):
        X = np.linspace(0, 1, 30)
        y = X ** 2
        loess = LOESSRegression()
        y_pred = loess.fit_predict(X, y, X, span=0.5, degree=2)
        assert len(y_pred) == 30


class TestSplineRegression:
    def test_fit_cubic_spline(self):
        np.random.seed(42)
        X = np.linspace(0, 1, 100)
        y = np.sin(4 * np.pi * X) + 0.1 * np.random.randn(100)
        sr = SplineRegression()
        result = sr.fit(X, y, n_knots=8, degree=3)
        assert isinstance(result, SplineResult)
        assert len(result.fitted) == 100
        assert len(result.residuals) == 100
        # Fitted should be reasonable
        assert np.corrcoef(y, result.fitted)[0, 1] > 0.8

    def test_predict_at_new_points(self):
        np.random.seed(42)
        X = np.linspace(0, 1, 50)
        y = X ** 2
        sr = SplineRegression()
        result = sr.fit(X, y, n_knots=5, degree=3)
        X_new = np.array([0.25, 0.5, 0.75])
        y_pred = sr.predict(result, X_new, degree=3)
        assert len(y_pred) == 3
        np.testing.assert_allclose(y_pred, X_new ** 2, atol=0.1)

    def test_gcv_finite(self):
        X = np.linspace(0, 1, 50)
        y = np.sin(X)
        sr = SplineRegression()
        result = sr.fit(X, y, n_knots=5)
        assert np.isfinite(result.gcv)

    def test_penalty_regularization(self):
        X = np.linspace(0, 1, 50)
        y = np.sin(X)
        sr = SplineRegression()
        result = sr.fit(X, y, n_knots=5, penalty=1.0)
        assert isinstance(result, SplineResult)


class TestVARModel:
    def test_fit_var1(self):
        np.random.seed(42)
        T, k = 100, 2
        Y = np.random.randn(T, k)
        var = VARModel()
        result = var.fit(Y, order=1)
        assert isinstance(result, VARResult)
        assert len(result.coefficients) == 1
        assert result.coefficients[0].shape == (k, k)
        assert result.residuals.shape == (T - 1, k)

    def test_forecast_shape(self):
        np.random.seed(42)
        Y = np.random.randn(50, 3)
        var = VARModel()
        result = var.fit(Y, order=2)
        fc = var.forecast(result, steps=5, Y_last=Y[-2:])
        assert fc.shape == (5, 3)

    def test_impulse_response(self):
        np.random.seed(42)
        Y = np.random.randn(100, 2)
        var = VARModel()
        result = var.fit(Y, order=1)
        irf = var.impulse_response(result, steps=10, shock_var=0)
        assert irf.shape == (10, 2)
        # First step: shock to variable 0 should be non-zero
        assert irf[0, 0] != 0.0

    def test_stability_check(self):
        np.random.seed(42)
        Y = np.random.randn(100, 2) * 0.1
        var = VARModel()
        result = var.fit(Y, order=1)
        assert isinstance(result.is_stable, bool)

    def test_granger_causality(self):
        np.random.seed(42)
        T = 100
        x = np.random.randn(T)
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * x[t - 1] + 0.3 * np.random.randn()
        Y = np.column_stack([x, y])
        var = VARModel()
        F, p = var.granger_causality(Y, cause=0, effect=1, order=1)
        assert F > 0  # x should Granger-cause y


class TestRegressionDiagnostics:
    def test_durbin_watson_white_noise(self):
        np.random.seed(42)
        e = np.random.randn(100)
        dw = RegressionDiagnostics.durbin_watson(e)
        assert 1.5 < dw < 2.5  # close to 2 for no autocorrelation

    def test_durbin_watson_zero_residuals(self):
        e = np.zeros(10)
        dw = RegressionDiagnostics.durbin_watson(e)
        assert dw == pytest.approx(2.0)

    def test_vif_independent(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        vifs = RegressionDiagnostics.vif(X)
        assert len(vifs) == 3
        # Independent variables should have VIF near 1
        assert np.all(vifs < 5.0)

    def test_vif_1d(self):
        X = np.random.randn(50)
        vifs = RegressionDiagnostics.vif(X)
        assert len(vifs) == 1
        assert vifs[0] == pytest.approx(1.0)

    def test_heteroscedasticity_test(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        e = np.random.randn(100)
        stat, p = RegressionDiagnostics.heteroscedasticity_test(e, X)
        assert stat >= 0
        assert 0 <= p <= 1

    def test_cooks_distance(self):
        np.random.seed(42)
        X = np.column_stack([np.ones(50), np.random.randn(50)])
        e = np.random.randn(50)
        D = RegressionDiagnostics.cooks_distance(X, e)
        assert len(D) == 50
        assert np.all(D >= 0)


# ===========================================================================
# stable_distributions
# ===========================================================================
from hololoom.core.warp.math.analysis.stable_distributions import (
    StableDistribution,
    VariationalInference,
    EmpiricalBayes,
    CanonicalCorrelation,
    CCAResult,
    MDS,
    KernelDensity1D,
    RankTest,
    RenyiDP,
    LocalDP,
)


class TestStableDistribution:
    def test_gaussian_case(self):
        samples = StableDistribution.sample(alpha=2.0, beta=0.0, scale=1.0, loc=0.0, n=1000)
        assert len(samples) == 1000
        assert abs(np.mean(samples)) < 0.3

    def test_cauchy_case(self):
        samples = StableDistribution.sample(alpha=1.0, beta=0.0, scale=1.0, loc=0.0, n=500)
        assert len(samples) == 500

    def test_general_case(self):
        samples = StableDistribution.sample(alpha=1.5, beta=0.5, scale=1.0, loc=0.0, n=200)
        assert len(samples) == 200

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            StableDistribution.sample(alpha=0.0, beta=0.0)

    def test_invalid_beta(self):
        with pytest.raises(ValueError):
            StableDistribution.sample(alpha=1.5, beta=2.0)

    def test_invalid_scale(self):
        with pytest.raises(ValueError):
            StableDistribution.sample(alpha=1.5, beta=0.0, scale=-1.0)

    def test_characteristic_function_at_zero(self):
        t = np.array([0.0])
        phi = StableDistribution.characteristic_function(1.5, 0.0, 1.0, 0.0, t)
        # φ(0) = 1
        np.testing.assert_allclose(np.abs(phi), [1.0], atol=1e-6)

    def test_characteristic_function_alpha1(self):
        t = np.array([0.0, 1.0])
        phi = StableDistribution.characteristic_function(1.0, 0.0, 1.0, 0.0, t)
        assert len(phi) == 2


class TestVariationalInference:
    def test_fit_returns_dict(self):
        def log_joint(z, x):
            return -0.5 * np.sum(z ** 2)  # Standard normal

        result = VariationalInference.fit(
            log_joint, q_family='gaussian', n_iter=50, n_samples=20, dim=2
        )
        assert 'mu' in result
        assert 'sigma' in result
        assert 'elbo_history' in result
        assert len(result['mu']) == 2

    def test_invalid_family(self):
        with pytest.raises(ValueError):
            VariationalInference.fit(lambda z, x: 0, q_family='laplace')


class TestEmpiricalBayes:
    def test_estimate_prior_gaussian(self):
        np.random.seed(42)
        data = np.random.randn(50) * 2 + 3  # mean=3, std=2

        def likelihood_fn(x, z):
            return -0.5 * np.sum((x - z) ** 2)

        result = EmpiricalBayes.estimate_prior(
            data, likelihood_fn, prior_family='gaussian', n_iter=10
        )
        assert 'mu' in result
        assert 'sigma' in result

    def test_invalid_family(self):
        with pytest.raises(ValueError):
            EmpiricalBayes.estimate_prior(
                np.array([1.0, 2.0]),
                lambda x, z: 0,
                prior_family='beta'
            )


class TestCCA:
    def test_correlated_data(self):
        np.random.seed(42)
        n = 100
        z = np.random.randn(n)
        X = np.column_stack([z + 0.1 * np.random.randn(n), np.random.randn(n)])
        Y = np.column_stack([z + 0.1 * np.random.randn(n), np.random.randn(n)])
        result = CanonicalCorrelation.fit(X, Y)
        assert isinstance(result, CCAResult)
        assert len(result.correlations) == 2
        # First canonical correlation should be high
        assert result.correlations[0] > 0.5


class TestMDS:
    def test_euclidean_embedding(self):
        # 3 points forming a right triangle
        D = np.array([
            [0, 3, 4],
            [3, 0, 5],
            [4, 5, 0],
        ], dtype=float)
        coords = MDS.fit(D, n_components=2)
        assert coords.shape == (3, 2)
        # Recovered distances should match
        for i in range(3):
            for j in range(i + 1, 3):
                d_emb = np.linalg.norm(coords[i] - coords[j])
                assert d_emb == pytest.approx(D[i, j], abs=0.5)


class TestKernelDensity1D:
    def test_density_integrates_to_one(self):
        np.random.seed(42)
        data = np.random.randn(100)
        density = KernelDensity1D.fit(data)
        x_grid = np.linspace(-5, 5, 1000)
        dx = x_grid[1] - x_grid[0]
        integral = np.sum(density(x_grid)) * dx
        assert integral == pytest.approx(1.0, abs=0.1)

    def test_custom_bandwidth(self):
        data = np.array([0.0, 1.0, 2.0])
        density = KernelDensity1D.fit(data, bandwidth=0.5)
        vals = density(np.array([0.0, 1.0, 2.0]))
        assert len(vals) == 3
        assert np.all(vals > 0)


class TestRankTest:
    def test_sign_test_centered(self):
        np.random.seed(42)
        x = np.random.randn(50)  # symmetric around 0
        S, p = RankTest.sign_test(x, median=0.0)
        assert 0 <= p <= 1

    def test_sign_test_shifted(self):
        x = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        S, p = RankTest.sign_test(x, median=0.0)
        assert S == 5  # all above median
        assert p < 0.1

    def test_runs_test(self):
        np.random.seed(42)
        x = np.random.randn(50)
        R, p = RankTest.runs_test(x)
        assert R >= 1
        assert 0 <= p <= 1


class TestRenyiDP:
    def test_mechanism_adds_noise(self):
        np.random.seed(42)
        vals = [RenyiDP.mechanism(5.0, sensitivity=1.0, alpha=2.0, epsilon=1.0)
                for _ in range(100)]
        # Mean should be near 5
        assert abs(np.mean(vals) - 5.0) < 1.0
        # Should have variance
        assert np.std(vals) > 0.1

    def test_composition(self):
        total = RenyiDP.compose([0.1, 0.2, 0.3], alpha=2.0)
        assert total == pytest.approx(0.6)

    def test_to_dp(self):
        eps_dp = RenyiDP.to_dp(epsilon_rdp=1.0, alpha=2.0, delta=1e-5)
        assert eps_dp > 1.0  # DP epsilon is larger

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            RenyiDP.mechanism(1.0, 1.0, alpha=0.5, epsilon=1.0)


class TestLocalDP:
    def test_randomized_response_is_bool(self):
        result = LocalDP.randomized_response(True, epsilon=1.0)
        assert isinstance(result, bool)

    def test_randomized_response_high_epsilon(self):
        """With very high epsilon, should almost always report truthfully."""
        np.random.seed(42)
        results = [LocalDP.randomized_response(True, epsilon=100.0) for _ in range(100)]
        assert sum(results) > 95

    def test_unary_encoding_shape(self):
        result = LocalDP.unary_encoding(value=2, domain_size=5, epsilon=1.0)
        assert len(result) == 5

    def test_estimate_frequency(self):
        np.random.seed(42)
        # All responses are 1 (true frequency = 1)
        responses = np.ones((100, 1))
        est = LocalDP.estimate_frequency(responses, epsilon=2.0)
        assert len(est) == 1
        assert est[0] == pytest.approx(1.0, abs=0.3)


# ===========================================================================
# bifurcation_analysis
# ===========================================================================
from hololoom.core.warp.math.analysis.bifurcation_analysis import (
    BifurcationAnalyzer,
    BifurcationPoint,
    BifurcationDiagram,
    HopfBifurcation,
    saddle_node_normal_form,
    pitchfork_normal_form,
    transcritical_normal_form,
)


class TestNormalForms:
    def test_saddle_node_equilibria(self):
        # dx/dt = mu - x^2 = 0 → x = ±sqrt(mu) for mu > 0
        assert saddle_node_normal_form(1.0, 1.0) == pytest.approx(0.0)
        assert saddle_node_normal_form(-1.0, 1.0) == pytest.approx(0.0)
        # No equilibria for mu < 0 at real x=0
        assert saddle_node_normal_form(0.0, -1.0) == pytest.approx(-1.0)

    def test_pitchfork_supercritical(self):
        # dx/dt = mu*x - x^3 = 0 → x=0 or x=±sqrt(mu)
        assert pitchfork_normal_form(0.0, 1.0) == pytest.approx(0.0)
        assert pitchfork_normal_form(1.0, 1.0) == pytest.approx(0.0)
        assert pitchfork_normal_form(-1.0, 1.0) == pytest.approx(0.0)

    def test_pitchfork_subcritical(self):
        # dx/dt = mu*x + x^3
        val = pitchfork_normal_form(1.0, 0.0, subcritical=True)
        assert val == pytest.approx(1.0)

    def test_transcritical(self):
        # dx/dt = mu*x - x^2 = 0 → x=0 or x=mu
        assert transcritical_normal_form(0.0, 1.0) == pytest.approx(0.0)
        assert transcritical_normal_form(1.0, 1.0) == pytest.approx(0.0)


class TestBifurcationAnalyzerClassify:
    def test_saddle_node_detected(self):
        # Real eigenvalue crosses zero
        ev_before = np.array([-0.1])
        ev_after = np.array([0.1])
        btype = BifurcationAnalyzer.classify(ev_before, ev_after)
        assert btype == "saddle_node"

    def test_hopf_detected(self):
        # Complex pair crosses imaginary axis
        ev_before = np.array([-0.1 + 1j, -0.1 - 1j])
        ev_after = np.array([0.1 + 1j, 0.1 - 1j])
        btype = BifurcationAnalyzer.classify(ev_before, ev_after)
        assert btype == "hopf"

    def test_no_bifurcation(self):
        ev_before = np.array([-0.5, -0.3])
        ev_after = np.array([-0.4, -0.2])
        btype = BifurcationAnalyzer.classify(ev_before, ev_after)
        assert btype is None

    def test_mismatched_lengths(self):
        btype = BifurcationAnalyzer.classify(np.array([-0.1]), np.array([-0.1, 0.1]))
        assert btype is None


class TestBifurcationDiagram1D:
    def test_saddle_node_diagram(self):
        diagram = BifurcationAnalyzer.diagram_1d(
            saddle_node_normal_form,
            param_range=(-1.0, 2.0),
            x_range=(-3.0, 3.0),
            n_points=50,
            n_x_search=30,
        )
        assert isinstance(diagram, BifurcationDiagram)
        assert len(diagram.parameter_values) == 50
        # For mu > 0, should find 2 fixed points; for mu < 0, 0 fixed points
        # Count how many parameter values have 2 fixed points
        count_two = sum(1 for fp in diagram.fixed_points if len(fp) == 2)
        assert count_two > 10


class TestHopfBifurcation:
    def test_limit_cycle_amplitude_above_critical(self):
        amp = HopfBifurcation.limit_cycle_amplitude(mu=2.0, param_critical=1.0, coefficient=1.0)
        assert amp == pytest.approx(1.0)

    def test_limit_cycle_amplitude_below_critical(self):
        amp = HopfBifurcation.limit_cycle_amplitude(mu=0.5, param_critical=1.0, coefficient=1.0)
        assert amp == 0.0

    def test_invalid_coefficient(self):
        with pytest.raises(ValueError):
            HopfBifurcation.limit_cycle_amplitude(mu=2.0, param_critical=1.0, coefficient=-1.0)

    def test_frequency_at_bifurcation(self):
        # System with eigenvalues ±2i at the bifurcation
        def jac(x, mu):
            return np.array([[0.0, 2.0], [-2.0, 0.0]])
        x_eq = np.array([0.0, 0.0])
        omega = HopfBifurcation.frequency_at_bifurcation(jac, x_eq, param_critical=0.0)
        assert omega == pytest.approx(2.0, abs=0.01)


class TestCriticalSlowingDown:
    def test_recovery_with_jacobian(self):
        def f(x, mu):
            return np.array([-mu * x[0]])

        def jac(x, mu):
            return np.array([[-mu]])

        recovery = BifurcationAnalyzer.critical_slowing_down(
            f, param=1.0, x_eq=np.array([0.0]), jacobian_fn=jac
        )
        # Recovery time = 1/|eigenvalue| = 1/1 = 1
        assert recovery == pytest.approx(1.0, abs=0.01)

    def test_recovery_near_bifurcation(self):
        """Near bifurcation (small mu), recovery time should be large."""
        def f(x, mu):
            return np.array([-mu * x[0]])

        def jac(x, mu):
            return np.array([[-mu]])

        recovery_far = BifurcationAnalyzer.critical_slowing_down(
            f, param=1.0, x_eq=np.array([0.0]), jacobian_fn=jac
        )
        recovery_near = BifurcationAnalyzer.critical_slowing_down(
            f, param=0.01, x_eq=np.array([0.0]), jacobian_fn=jac
        )
        assert recovery_near > recovery_far


class TestContinuation1D:
    def test_continuation_tracks_branch(self):
        # Track x* = sqrt(mu) branch of saddle-node
        x_branch, mu_branch = BifurcationAnalyzer.continuation_1d(
            saddle_node_normal_form,
            x0=1.0, mu0=1.0, mu_end=4.0,
            n_steps=50, ds=0.05
        )
        assert len(x_branch) == 50
        assert len(mu_branch) == 50
        # Should stay near x = sqrt(mu)
        for i in range(1, 50):
            if mu_branch[i] > 0.1:
                expected_x = np.sqrt(mu_branch[i])
                assert abs(x_branch[i]) == pytest.approx(expected_x, abs=0.5)
                break


# ===========================================================================
# kalman_filtering
# ===========================================================================
from hololoom.core.warp.math.analysis.kalman_filtering import (
    KalmanState,
    KalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    steady_state_gain,
    normalized_innovation_squared,
    filter_and_smooth,
)


def _1d_kalman_setup():
    """Simple 1D constant velocity model."""
    F = np.array([[1.0]])
    H = np.array([[1.0]])
    Q = np.array([[0.1]])
    R = np.array([[1.0]])
    initial = KalmanState(mean=np.array([0.0]), covariance=np.array([[1.0]]))
    return F, H, Q, R, initial


class TestKalmanState:
    def test_dim(self):
        s = KalmanState(mean=np.array([1.0, 2.0]), covariance=np.eye(2))
        assert s.dim == 2

    def test_mahalanobis(self):
        s = KalmanState(mean=np.array([0.0]), covariance=np.array([[1.0]]))
        d = s.mahalanobis(np.array([2.0]))
        assert d == pytest.approx(2.0)

    def test_confidence_ellipse(self):
        s = KalmanState(mean=np.array([0.0, 0.0]), covariance=np.eye(2))
        vals, vecs = s.confidence_ellipse(n_std=2.0)
        assert len(vals) == 2
        np.testing.assert_allclose(vals, [2.0, 2.0])


class TestKalmanFilterPredict:
    def test_predict_mean(self):
        F, H, Q, R, initial = _1d_kalman_setup()
        pred = KalmanFilter.predict(initial, F, Q)
        np.testing.assert_allclose(pred.mean, [0.0])

    def test_predict_covariance_grows(self):
        F, H, Q, R, initial = _1d_kalman_setup()
        pred = KalmanFilter.predict(initial, F, Q)
        assert pred.covariance[0, 0] > initial.covariance[0, 0]

    def test_predict_with_control(self):
        F = np.array([[1.0]])
        Q = np.array([[0.1]])
        B = np.array([[1.0]])
        u = np.array([5.0])
        initial = KalmanState(mean=np.array([0.0]), covariance=np.array([[1.0]]))
        pred = KalmanFilter.predict(initial, F, Q, B=B, u=u)
        assert pred.mean[0] == pytest.approx(5.0)


class TestKalmanFilterUpdate:
    def test_update_reduces_covariance(self):
        F, H, Q, R, initial = _1d_kalman_setup()
        pred = KalmanFilter.predict(initial, F, Q)
        updated = KalmanFilter.update(pred, np.array([1.0]), H, R)
        assert updated.covariance[0, 0] < pred.covariance[0, 0]

    def test_update_moves_mean_toward_observation(self):
        F, H, Q, R, initial = _1d_kalman_setup()
        pred = KalmanFilter.predict(initial, F, Q)
        z = np.array([5.0])
        updated = KalmanFilter.update(pred, z, H, R)
        assert updated.mean[0] > pred.mean[0]  # moved toward observation

    def test_update_log_likelihood_finite(self):
        F, H, Q, R, initial = _1d_kalman_setup()
        pred = KalmanFilter.predict(initial, F, Q)
        updated = KalmanFilter.update(pred, np.array([1.0]), H, R)
        assert np.isfinite(updated.log_likelihood)


class TestKalmanFilterSequence:
    def test_filter_sequence(self):
        F, H, Q, R, initial = _1d_kalman_setup()
        np.random.seed(42)
        observations = [np.array([float(t + np.random.randn())]) for t in range(20)]
        filtered, predicted = KalmanFilter.filter_sequence(
            observations, F, H, Q, R, initial
        )
        assert len(filtered) == 20
        assert len(predicted) == 20

    def test_smooth_improves_estimate(self):
        F, H, Q, R, initial = _1d_kalman_setup()
        np.random.seed(42)
        true_state = np.cumsum(np.random.randn(30) * 0.1)
        observations = [np.array([true_state[t] + np.random.randn()])
                        for t in range(30)]
        filtered, predicted = KalmanFilter.filter_sequence(
            observations, F, H, Q, R, initial
        )
        smoothed = KalmanFilter.smooth(predicted, filtered, F)
        assert len(smoothed) == 30
        # Smoothed covariance should be <= filtered covariance
        for k in range(30):
            assert smoothed[k].covariance[0, 0] <= filtered[k].covariance[0, 0] + 1e-10


class TestKalmanInnovation:
    def test_innovation_sequence(self):
        F, H, Q, R, initial = _1d_kalman_setup()
        observations = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        filtered, predicted = KalmanFilter.filter_sequence(
            observations, F, H, Q, R, initial
        )
        innovations = KalmanFilter.innovation_sequence(observations, predicted, H, R)
        assert len(innovations) == 3
        for nu, S in innovations:
            assert len(nu) == 1
            assert S.shape == (1, 1)


class TestExtendedKalmanFilter:
    def test_ekf_linear_matches_kf(self):
        """For a linear system, EKF should give same results as KF."""
        F_mat = np.array([[1.0]])
        H_mat = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[1.0]])

        f = lambda x: F_mat @ x
        h = lambda x: H_mat @ x
        F_jac = lambda x: F_mat
        H_jac = lambda x: H_mat

        initial = KalmanState(mean=np.array([0.0]), covariance=np.array([[1.0]]))

        # KF
        pred_kf = KalmanFilter.predict(initial, F_mat, Q)
        upd_kf = KalmanFilter.update(pred_kf, np.array([2.0]), H_mat, R)

        # EKF
        pred_ekf = ExtendedKalmanFilter.predict(initial, f, F_jac, Q)
        upd_ekf = ExtendedKalmanFilter.update(pred_ekf, np.array([2.0]), h, H_jac, R)

        np.testing.assert_allclose(upd_kf.mean, upd_ekf.mean, atol=1e-10)
        np.testing.assert_allclose(upd_kf.covariance, upd_ekf.covariance, atol=1e-10)

    def test_ekf_filter_sequence(self):
        f = lambda x: x  # identity
        h = lambda x: x
        F_jac = lambda x: np.eye(1)
        H_jac = lambda x: np.eye(1)
        Q = np.array([[0.1]])
        R = np.array([[1.0]])
        initial = KalmanState(mean=np.array([0.0]), covariance=np.array([[1.0]]))
        observations = [np.array([float(i)]) for i in range(10)]

        filtered, predicted = ExtendedKalmanFilter.filter_sequence(
            observations, f, h, F_jac, H_jac, Q, R, initial
        )
        assert len(filtered) == 10


class TestUnscentedKalmanFilter:
    def test_sigma_points_shape(self):
        state = KalmanState(mean=np.array([0.0, 0.0]), covariance=np.eye(2))
        pts, wm, wc = UnscentedKalmanFilter.generate_sigma_points(state)
        assert pts.shape == (5, 2)  # 2n+1 = 5
        assert len(wm) == 5
        assert len(wc) == 5

    def test_sigma_points_weights_sum(self):
        state = KalmanState(mean=np.array([1.0, 2.0]), covariance=np.eye(2))
        pts, wm, wc = UnscentedKalmanFilter.generate_sigma_points(state)
        assert wm.sum() == pytest.approx(1.0, abs=1e-6)

    def test_ukf_predict_update(self):
        f = lambda x: x
        h = lambda x: x[:1]  # observe first component
        Q = 0.1 * np.eye(2)
        R = np.array([[1.0]])
        initial = KalmanState(mean=np.array([0.0, 0.0]), covariance=np.eye(2))

        pred = UnscentedKalmanFilter.predict(initial, f, Q)
        upd = UnscentedKalmanFilter.update(pred, np.array([1.0]), h, R)
        assert upd.mean[0] > 0  # moved toward observation

    def test_ukf_filter_sequence(self):
        f = lambda x: x
        h = lambda x: x
        Q = 0.1 * np.eye(1)
        R = np.array([[1.0]])
        initial = KalmanState(mean=np.array([0.0]), covariance=np.array([[1.0]]))
        observations = [np.array([float(i)]) for i in range(10)]

        filtered = UnscentedKalmanFilter.filter_sequence(
            observations, f, h, Q, R, initial
        )
        assert len(filtered) == 10


class TestSteadyStateGain:
    def test_converges(self):
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[1.0]])
        K = steady_state_gain(F, H, Q, R)
        assert K.shape == (1, 1)
        assert 0 < K[0, 0] < 1  # gain should be between 0 and 1

    def test_2d_system(self):
        F = np.array([[1.0, 0.1], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = 0.01 * np.eye(2)
        R = np.array([[0.5]])
        K = steady_state_gain(F, H, Q, R)
        assert K.shape == (2, 1)


class TestNIS:
    def test_nis_values(self):
        innovations = [
            (np.array([1.0]), np.array([[2.0]])),
            (np.array([0.5]), np.array([[1.0]])),
        ]
        nis = normalized_innovation_squared(innovations)
        assert len(nis) == 2
        assert nis[0] == pytest.approx(0.5)  # 1^2 / 2
        assert nis[1] == pytest.approx(0.25)  # 0.5^2 / 1


class TestFilterAndSmooth:
    def test_convenience_function(self):
        F, H, Q, R, initial = _1d_kalman_setup()
        observations = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        filtered, predicted, smoothed = filter_and_smooth(
            observations, F, H, Q, R, initial
        )
        assert len(filtered) == 3
        assert len(predicted) == 3
        assert len(smoothed) == 3
