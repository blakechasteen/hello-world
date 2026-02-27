"""Thompson Sampling models."""

from HoloLoom.bandits.models.discrete_bernoulli import DiscreteBernoulliTS
from HoloLoom.bandits.models.bayes_linear import BayesianLinearTS

__all__ = [
    "DiscreteBernoulliTS",
    "BayesianLinearTS",
]
