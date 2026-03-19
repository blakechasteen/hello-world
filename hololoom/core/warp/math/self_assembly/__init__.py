"""
Self-Assembly — Evolutionary, Swarm, Reaction-Diffusion, and Collective Behavior
==================================================================================

Bio-inspired methods that emerge from simple rules applied to populations.

Modules:
    genetic_algorithms: GA, Differential Evolution
    swarm: Particle Swarm Optimization, Ant Colony Optimization
    reaction_diffusion: Turing patterns, Gray-Scott, FitzHugh-Nagumo
    flocking: Boids, consensus protocols, Byzantine fault tolerance
"""

# Advanced Flocking Metrics
from .boids_advanced import (
    FlockingMetrics,
)
from .flocking import (
    Boids,
    ByzantineFaultTolerance,
    ConsensusProtocol,
    ConsensusResult,
    FlockState,
)
from .genetic_algorithms import (
    DifferentialEvolution,
    GeneticAlgorithm,
    Individual,
)

# Genetic Programming
from .genetic_programming import (
    NEAT,
    GeneticProgramming,
    GPNode,
    GrammarGP,
    NEATGenome,
    NeuralProgramInduction,
)
from .reaction_diffusion import (
    FitzHughNagumo,
    GrayScott,
    RDResult,
    ReactionDiffusion,
)

# Self-Modifying Systems
from .self_modifying import (
    EchoStateProperty,
    LiquidStateMachine,
    OpenEndedEvolution,
    Quine,
    ReflectiveTower,
)
from .swarm import (
    AntColonyOptimization,
    ParticleSwarmOptimization,
    stigmergy_update,
)

__all__ = [
    # Genetic Algorithms
    "Individual",
    "GeneticAlgorithm",
    "DifferentialEvolution",
    # Swarm Intelligence
    "ParticleSwarmOptimization",
    "AntColonyOptimization",
    "stigmergy_update",
    # Reaction-Diffusion
    "RDResult",
    "ReactionDiffusion",
    "GrayScott",
    "FitzHughNagumo",
    # Flocking & Consensus
    "FlockState",
    "ConsensusResult",
    "Boids",
    "ConsensusProtocol",
    "ByzantineFaultTolerance",

    # Genetic Programming
    "GPNode",
    "GeneticProgramming",
    "NEATGenome",
    "NEAT",
    "GrammarGP",
    "NeuralProgramInduction",

    # Advanced Flocking Metrics
    "FlockingMetrics",

    # Self-Modifying Systems
    "LiquidStateMachine",
    "EchoStateProperty",
    "Quine",
    "ReflectiveTower",
    "OpenEndedEvolution",
]
