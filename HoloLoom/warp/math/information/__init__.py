"""Information Theory module."""
from .information_theory import (
    entropy, cross_entropy, kl_divergence, mutual_information,
    jensen_shannon, conditional_entropy, information_gain,
    InformationMetrics, DistributionPair
)

__all__ = [
    'entropy', 'cross_entropy', 'kl_divergence', 'mutual_information',
    'jensen_shannon', 'conditional_entropy', 'information_gain',
    'InformationMetrics', 'DistributionPair'
]
