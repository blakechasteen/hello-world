"""
Decision & Information Module - Information Theory, Game Theory, Operations Research
===================================================================================

Complete framework for decision-making, optimization, and strategic interaction.

Modules:
    information_theory: Entropy, mutual information, coding theory
    game_theory: Nash equilibria, mechanism design, auctions
    operations_research: Linear programming, network flows, scheduling

Sprint 5: Decision & Information
"""

# Information Theory
from .information_theory import (
    Entropy,
    MutualInformation,
    DivergenceMetrics,
    ChannelCapacity,
    SourceCoding,
    ErrorCorrection,
    RateDistortion,
)

# Game Theory
from .game_theory import (
    Strategy,
    NormalFormGame,
    NashEquilibrium,
    MechanismDesign,
    AuctionTheory,
    CooperativeGame,
    EvolutionaryGame,
)

# Operations Research
from .operations_research import (
    LinearProgramming,
    NetworkFlows,
    IntegerProgramming,
    Scheduling,
    DynamicProgramming,
    InventoryTheory,
)

__all__ = [
    # Information Theory
    "Entropy",
    "MutualInformation",
    "DivergenceMetrics",
    "ChannelCapacity",
    "SourceCoding",
    "ErrorCorrection",
    "RateDistortion",

    # Game Theory
    "Strategy",
    "NormalFormGame",
    "NashEquilibrium",
    "MechanismDesign",
    "AuctionTheory",
    "CooperativeGame",
    "EvolutionaryGame",

    # Operations Research
    "LinearProgramming",
    "NetworkFlows",
    "IntegerProgramming",
    "Scheduling",
    "DynamicProgramming",
    "InventoryTheory",
]
