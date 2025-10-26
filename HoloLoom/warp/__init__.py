"""
Warp - Tensioned Tensor Field
==============================
The computational manifold where threads are under active tension.

Exports:
- WarpSpace: Tensioned subset of Yarn Graph (core)
- Advanced operations: Differential geometry, tensor decomposition, quantum-inspired
- Optimized: GPU acceleration, sparse tensors, lazy evaluation
- Topology: Persistent homology, Mapper, TDA features
- Combinatorics: Chain complexes, discrete Morse theory, sheaf theory
- Category Theory: Functors, natural transformations, limits, Yoneda
- Representation Theory: Group representations, characters, equivariant maps
- Math/Analysis: Complete 7-module suite (real, complex, functional, measure, Fourier, stochastic, advanced)
"""

from .space import WarpSpace

# Optional advanced operations
try:
    from .advanced import (
        RiemannianManifold,
        TensorDecomposer,
        QuantumWarpOperations,
        FisherInformationGeometry
    )
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False

# Optional optimizations
try:
    from .optimized import (
        GPUWarpSpace,
        SparseTensorField,
        LazyWarpOperation,
        TensorMemoryPool,
        BatchedWarpProcessor
    )
    HAS_OPTIMIZED = True
except ImportError:
    HAS_OPTIMIZED = False

# Optional topology
try:
    from .topology import (
        PersistentHomology,
        PersistenceDiagram,
        PersistenceInterval,
        VietorisRipsComplex,
        MapperAlgorithm,
        TopologicalFeatureExtractor
    )
    HAS_TOPOLOGY = True
except ImportError:
    HAS_TOPOLOGY = False

# Optional combinatorics
try:
    from .combinatorics import (
        ChainComplex,
        DiscreteMorseFunction,
        Sheaf
    )
    HAS_COMBINATORICS = True
except ImportError:
    HAS_COMBINATORICS = False

# Optional category theory
try:
    from .category import (
        Category,
        Morphism,
        Functor,
        NaturalTransformation,
        Limit,
        Colimit,
        YonedaEmbedding,
        MonoidalCategory
    )
    HAS_CATEGORY = True
except ImportError:
    HAS_CATEGORY = False

# Optional representation theory
try:
    from .representation import (
        Group,
        Representation,
        CharacterTable,
        EquivariantMap,
        cyclic_group,
        symmetric_group,
        trivial_representation,
        regular_representation
    )
    HAS_REPRESENTATION = True
except ImportError:
    HAS_REPRESENTATION = False

__all__ = ["WarpSpace"]

if HAS_ADVANCED:
    __all__.extend([
        "RiemannianManifold",
        "TensorDecomposer",
        "QuantumWarpOperations",
        "FisherInformationGeometry"
    ])

if HAS_OPTIMIZED:
    __all__.extend([
        "GPUWarpSpace",
        "SparseTensorField",
        "LazyWarpOperation",
        "TensorMemoryPool",
        "BatchedWarpProcessor"
    ])

if HAS_TOPOLOGY:
    __all__.extend([
        "PersistentHomology",
        "PersistenceDiagram",
        "PersistenceInterval",
        "VietorisRipsComplex",
        "MapperAlgorithm",
        "TopologicalFeatureExtractor"
    ])

if HAS_COMBINATORICS:
    __all__.extend([
        "ChainComplex",
        "DiscreteMorseFunction",
        "Sheaf"
    ])

if HAS_CATEGORY:
    __all__.extend([
        "Category",
        "Morphism",
        "Functor",
        "NaturalTransformation",
        "Limit",
        "Colimit",
        "YonedaEmbedding",
        "MonoidalCategory"
    ])

if HAS_REPRESENTATION:
    __all__.extend([
        "Group",
        "Representation",
        "CharacterTable",
        "EquivariantMap",
        "cyclic_group",
        "symmetric_group",
        "trivial_representation",
        "regular_representation"
    ])
