"""
Decision & Information Module - Information Theory, Game Theory, Operations Research
===================================================================================

Complete framework for decision-making, optimization, and strategic interaction.

Modules:
    information_theory: Entropy, mutual information, coding theory
    game_theory: Nash equilibria, mechanism design, auctions
    operations_research: Linear programming, network flows, scheduling
    multi_agent_rl: Multi-agent coordination (independent, CTDE, mean-field, communication)
    inverse_reinforcement_learning: Learn rewards from demonstrations (MaxEnt, Bayesian, apprenticeship)
    hierarchical_rl: Options framework, MAXQ decomposition, feudal hierarchy
    risk_sensitive_rl: CVaR, entropic risk, distributional RL, mean-variance
    optimal_control: Pontryagin maximum principle, LQR, MPC
    value_based_rl: Q-Learning, Double Q-Learning, SARSA, N-step returns
    model_based_rl: World model, Dyna, MCTS with learned model, MB policy optimization
    offline_rl: Conservative Q-Learning (CQL), Batch-Constrained Q (BCQ), Implicit Q-Learning (IQL)
    reward_machines: Finite-state reward specification, RM Q-learning, LTL conversion

Sprint 5: Decision & Information
"""

# Information Theory
# Constrained MDPs
from .constrained_mdp import (
    CMDPPolicy,
    CMDPSpec,
    ConstrainedMDP,
    SafeRL,
)

# Counterfactual Regret Minimization
from .counterfactual_regret import (
    CFRPlus,
    CFRResult,
    CounterfactualRegretMinimization,
    GameNode,
    InfoSet,
    LinearCFR,
)
from .curriculum_learning import (
    AutomaticCurriculum,
    CurriculumScheduler,
    CurriculumState,
    SelfPacedLearning,
)

# Curriculum Learning
from .curriculum_learning import (
    Task as CurriculumTask,
)

# Evolution Strategies
from .evolution_strategies import (
    CMAES,
    CMAESState,
    ESResult,
    NaturalEvolutionStrategy,
    SimpleES,
)

# Game Theory
from .game_theory import (
    AuctionTheory,
    CooperativeGame,
    EvolutionaryGame,
    MechanismDesign,
    NashEquilibrium,
    NormalFormGame,
    Strategy,
)

# Hierarchical RL
from .hierarchical_rl import (
    MAXQ,
    FeudalRL,
    HRLResult,
    Option,
    OptionsFramework,
)
from .information_theory import (
    ChannelCapacity,
    DivergenceMetrics,
    Entropy,
    ErrorCorrection,
    MutualInformation,
    RateDistortion,
    SourceCoding,
)

# Inverse Reinforcement Learning
from .inverse_reinforcement_learning import (
    ApprenticeshipLearning,
    BayesianIRL,
    Demonstration,
    IRLResult,
    MaxEntIRL,
)

# Meta-Learning
from .meta_learning import (
    MAML,
    MAMLState,
    MetaLearnResult,
    Reptile,
    Task,
    TaskDistribution,
)

# Model-Based RL
from .model_based_rl import (
    MCTS_RL,
    Dyna,
    MBPolicyOptimization,
    Transition,
    WorldModel,
)

# Multi-Agent RL
from .multi_agent_rl import (
    CentralizedCritic,
    CommProtocol,
    IndependentLearners,
    MARLAgent,
    MARLResult,
    MeanFieldMARL,
)

# Offline RL
from .offline_rl import (
    BatchConstrainedQ,
    ConservativeQ,
    ImplicitQLearning,
    OfflineDataset,
)

# Operations Research
from .operations_research import (
    DynamicProgramming,
    IntegerProgramming,
    InventoryTheory,
    LinearProgramming,
    NetworkFlows,
    Scheduling,
)

# Optimal Control
from .optimal_control import (
    LQR,
    ControlResult,
    LQRResult,
    ModelPredictiveControl,
    MPCResult,
    PontryaginMaximum,
)

# Policy Gradient Methods
from .policy_gradient import (
    REINFORCE,
    SAC,
    TRPO,
    ActorCritic,
    ActorCriticResult,
    PolicyGradientResult,
    SoftmaxPolicy,
)

# Reward Machines
from .reward_machines import (
    LTLFormula,
    RewardMachine,
    RMQLearning,
)

# Risk-Sensitive RL
from .risk_sensitive_rl import (
    CVaRMDP,
    DistributionalRL,
    EntropicRisk,
    MeanVarianceRL,
    RiskMetrics,
    compute_risk_metrics,
)

# Value-Based RL
from .value_based_rl import (
    SARSA,
    DoubleQLearning,
    NStepReturn,
    QLearning,
    RLResult,
)

# Correlated Equilibria
try:
    from .correlated_equilibria import (
        CorrelatedEquilibrium,
        RegretMatchingCE,
    )
except ImportError:
    from .correlated_equilibria import CorrelatedEquilibrium
    RegretMatchingCE = None

# MDP Foundations
# Deep RL
from .deep_rl import (
    DQN,
    DoubleDQN,
    DQNResult,
    DuelingDQN,
    PrioritizedReplay,
    RegretMatching,
    SafetyShield,
)
from .mdp_foundations import (
    MDP,
    MDPResult,
    TemporalDifference,
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

    # Constrained MDPs
    "CMDPSpec",
    "CMDPPolicy",
    "ConstrainedMDP",
    "SafeRL",

    # Counterfactual Regret Minimization
    "InfoSet",
    "CFRResult",
    "GameNode",
    "CounterfactualRegretMinimization",
    "CFRPlus",
    "LinearCFR",

    # Multi-Agent RL
    "MARLAgent",
    "MARLResult",
    "IndependentLearners",
    "CentralizedCritic",
    "MeanFieldMARL",
    "CommProtocol",

    # Inverse Reinforcement Learning
    "Demonstration",
    "IRLResult",
    "MaxEntIRL",
    "BayesianIRL",
    "ApprenticeshipLearning",

    # Hierarchical RL
    "Option",
    "HRLResult",
    "OptionsFramework",
    "MAXQ",
    "FeudalRL",

    # Risk-Sensitive RL
    "RiskMetrics",
    "CVaRMDP",
    "EntropicRisk",
    "DistributionalRL",
    "MeanVarianceRL",
    "compute_risk_metrics",

    # Evolution Strategies
    "CMAESState",
    "ESResult",
    "CMAES",
    "NaturalEvolutionStrategy",
    "SimpleES",

    # Meta-Learning
    "Task",
    "MAMLState",
    "MetaLearnResult",
    "MAML",
    "Reptile",
    "TaskDistribution",

    # Optimal Control
    "ControlResult",
    "LQRResult",
    "MPCResult",
    "PontryaginMaximum",
    "LQR",
    "ModelPredictiveControl",

    # Value-Based RL
    "RLResult",
    "QLearning",
    "DoubleQLearning",
    "SARSA",
    "NStepReturn",

    # Model-Based RL
    "Transition",
    "WorldModel",
    "Dyna",
    "MCTS_RL",
    "MBPolicyOptimization",

    # Offline RL
    "OfflineDataset",
    "ConservativeQ",
    "BatchConstrainedQ",
    "ImplicitQLearning",

    # Reward Machines
    "RewardMachine",
    "RMQLearning",
    "LTLFormula",

    # Policy Gradient Methods
    "PolicyGradientResult",
    "ActorCriticResult",
    "SoftmaxPolicy",
    "REINFORCE",
    "ActorCritic",
    "TRPO",
    "SAC",

    # Curriculum Learning
    "CurriculumTask",
    "CurriculumState",
    "CurriculumScheduler",
    "SelfPacedLearning",
    "AutomaticCurriculum",

    # Correlated Equilibria
    "CorrelatedEquilibrium",
    "RegretMatchingCE",

    # MDP Foundations
    "MDPResult",
    "MDP",
    "TemporalDifference",

    # Deep RL
    "DQNResult",
    "DQN",
    "DoubleDQN",
    "DuelingDQN",
    "PrioritizedReplay",
    "SafetyShield",
    "RegretMatching",
]
