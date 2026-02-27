"""Thompson Sampling models."""

from hololoom.bandits.models.discrete_bernoulli import DiscreteBernoulliTS
from hololoom.bandits.models.bayes_linear import BayesianLinearTS

__all__ = [
    "DiscreteBernoulliTS",
    "BayesianLinearTS",
]
