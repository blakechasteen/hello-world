"""
Unified Thompson Sampler factory and interface.

Single entry point for creating any Thompson Sampling model:
- Discrete Bernoulli (Beta-Bernoulli MAB)
- Bayesian Linear (Gaussian contextual bandit)
- Neural Bandit (deep contextual bandit)
- GP-TS (Gaussian Process continuous optimization)
- Deep Kernel GP-TS (neural embedding + GP)

Example:
    >>> from hololoom.ts_core import create_thompson_sampler
    >>>
    >>> # Discrete MAB
    >>> sampler = create_thompson_sampler("discrete", n_arms=5)
    >>>
    >>> # Bayesian Linear
    >>> sampler = create_thompson_sampler("linear", context_dim=50, n_actions=10)
    >>>
    >>> # Neural Bandit
    >>> sampler = create_thompson_sampler("neural", context_dim=384, n_actions=5)
    >>>
    >>> # GP-TS
    >>> sampler = create_thompson_sampler("gp", param_dim=10)
"""

from typing import Literal, Any
from hololoom.ts_core.models.discrete_bernoulli import DiscreteBernoulliTS, create_discrete_ts
from hololoom.ts_core.models.bayes_linear import BayesianLinearTS, create_bayesian_linear_ts
from hololoom.ts_core.models.gp_ts import GaussianProcessTS, create_gp_ts


def create_thompson_sampler(
    sampler_type: Literal["discrete", "linear", "neural", "gp", "deep_kernel_gp"],
    **kwargs: Any,
) -> Any:
    """
    Unified factory for Thompson Sampling models.

    Creates the appropriate TS model based on type and forwards kwargs.

    Args:
        sampler_type: Type of Thompson Sampler:
            - "discrete": Discrete Bernoulli (Beta-Bernoulli MAB)
            - "linear": Bayesian Linear (Gaussian contextual bandit)
            - "neural": Neural Bandit (deep contextual bandit)
            - "gp": Gaussian Process TS
            - "deep_kernel_gp": Deep Kernel GP-TS (future)
        **kwargs: Type-specific arguments

    Returns:
        Initialized Thompson Sampler

    Raises:
        ValueError: If sampler_type unknown

    Examples:
        >>> # Discrete MAB (A/B testing, agent routing)
        >>> sampler = create_thompson_sampler(
        ...     "discrete",
        ...     n_arms=5,
        ...     alpha_prior=1.0,
        ...     beta_prior=1.0
        ... )
        >>> arm = sampler.select()
        >>> sampler.update(arm, reward=1.0)
        >>>
        >>> # Bayesian Linear (news recommendation)
        >>> sampler = create_thompson_sampler(
        ...     "linear",
        ...     context_dim=50,
        ...     n_actions=10,
        ...     lambda_prior=1.0
        ... )
        >>> action = sampler.select(context)
        >>> sampler.update(context, action, reward)
        >>>
        >>> # Neural Bandit (tool selection with embeddings)
        >>> sampler = create_thompson_sampler(
        ...     "neural",
        ...     context_dim=384,
        ...     n_actions=5,
        ...     hidden_dims=[256, 128],
        ...     backend="bootstrap"
        ... )
        >>> # See HoloLoom.bandits for full neural usage
        >>>
        >>> # GP-TS (hyperparameter tuning)
        >>> sampler = create_thompson_sampler(
        ...     "gp",
        ...     param_dim=10,
        ...     kernel="rbf",
        ...     bounds=(0.0, 1.0)
        ... )
        >>> params = sampler.select()
        >>> sampler.update(params, reward)
    """
    if sampler_type == "discrete":
        # Extract discrete-specific args
        n_arms = kwargs.pop("n_arms")
        prior = kwargs.pop("prior", "uniform")
        seed = kwargs.pop("seed", None)

        # Handle both factory and direct constructor
        if isinstance(prior, str):
            return create_discrete_ts(n_arms=n_arms, prior=prior, seed=seed)
        else:
            # Direct construction with alpha/beta
            alpha_prior = kwargs.pop("alpha_prior", 1.0)
            beta_prior = kwargs.pop("beta_prior", 1.0)
            return DiscreteBernoulliTS(
                n_arms=n_arms,
                alpha_prior=alpha_prior,
                beta_prior=beta_prior,
                seed=seed,
            )

    elif sampler_type == "linear":
        # Extract linear-specific args
        context_dim = kwargs.pop("context_dim")
        n_actions = kwargs.pop("n_actions")
        prior = kwargs.pop("prior", "default")
        seed = kwargs.pop("seed", None)

        # Handle both factory and direct constructor
        if isinstance(prior, str):
            return create_bayesian_linear_ts(
                context_dim=context_dim,
                n_actions=n_actions,
                prior=prior,
                seed=seed,
            )
        else:
            lambda_prior = kwargs.pop("lambda_prior", 1.0)
            sigma_noise = kwargs.pop("sigma_noise", 1.0)
            return BayesianLinearTS(
                context_dim=context_dim,
                n_actions=n_actions,
                lambda_prior=lambda_prior,
                sigma_noise=sigma_noise,
                seed=seed,
            )

    elif sampler_type == "neural":
        # Neural bandit uses existing HoloLoom.bandits
        from hololoom.bandits.config import BanditConfig, create_neural_ts_policy

        config = BanditConfig(**kwargs)
        return create_neural_ts_policy(config)

    elif sampler_type == "gp":
        # Extract GP-specific args
        param_dim = kwargs.pop("param_dim")
        kernel = kwargs.pop("kernel", "rbf")
        bounds = kwargs.pop("bounds", (0.0, 1.0))
        seed = kwargs.pop("seed", None)

        return create_gp_ts(
            param_dim=param_dim,
            kernel=kernel,
            bounds=bounds,
            seed=seed,
        )

    elif sampler_type == "deep_kernel_gp":
        raise NotImplementedError(
            "Deep Kernel GP-TS not yet implemented. "
            "Use 'neural' for deep contextual bandits or 'gp' for GP-TS."
        )

    else:
        raise ValueError(
            f"Unknown sampler_type: {sampler_type}. "
            f"Must be one of: discrete, linear, neural, gp, deep_kernel_gp"
        )


# ============================================================================
# Convenience functions
# ============================================================================


def create_discrete_mab(n_arms: int, **kwargs) -> DiscreteBernoulliTS:
    """Convenience: Create discrete multi-armed bandit."""
    return create_thompson_sampler("discrete", n_arms=n_arms, **kwargs)


def create_contextual_bandit(
    context_dim: int,
    n_actions: int,
    model: Literal["linear", "neural"] = "linear",
    **kwargs,
) -> BayesianLinearTS | Any:
    """
    Convenience: Create contextual bandit.

    Args:
        context_dim: Context dimension
        n_actions: Number of actions
        model: "linear" (Bayesian) or "neural" (deep)
        **kwargs: Model-specific args

    Returns:
        BayesianLinearTS or NeuralThompsonPolicy

    Example:
        >>> # Linear contextual bandit
        >>> sampler = create_contextual_bandit(
        ...     context_dim=50,
        ...     n_actions=10,
        ...     model="linear"
        ... )
        >>>
        >>> # Neural contextual bandit
        >>> sampler = create_contextual_bandit(
        ...     context_dim=384,
        ...     n_actions=5,
        ...     model="neural",
        ...     hidden_dims=[256, 128]
        ... )
    """
    return create_thompson_sampler(
        model,
        context_dim=context_dim,
        n_actions=n_actions,
        **kwargs,
    )


def create_continuous_optimizer(
    param_dim: int,
    **kwargs,
) -> GaussianProcessTS:
    """
    Convenience: Create continuous parameter optimizer (GP-TS).

    Args:
        param_dim: Parameter space dimension
        **kwargs: GP-specific args (kernel, bounds, etc.)

    Returns:
        GaussianProcessTS

    Example:
        >>> optimizer = create_continuous_optimizer(
        ...     param_dim=10,
        ...     kernel="rbf",
        ...     bounds=(0.0, 1.0)
        ... )
    """
    return create_thompson_sampler("gp", param_dim=param_dim, **kwargs)
