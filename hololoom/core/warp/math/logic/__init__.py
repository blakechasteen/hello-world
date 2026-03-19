"""
Logic & Foundations Module - Mathematical Logic, Computability Theory
====================================================================

Complete framework for formal foundations of mathematics and computation.

Modules:
    mathematical_logic: Propositional/first-order logic, model theory, Gödel
    computability_theory: Turing machines, decidability, complexity classes

Sprint 6: Logic & Foundations
"""

# Mathematical Logic
# Computability Theory
from .computability_theory import (
    ChurchTuringThesis,
    ComplexityClasses,
    Decidability,
    Direction,
    HaltingProblem,
    NPCompleteness,
    TuringMachine,
    TuringState,
)
from .mathematical_logic import (
    FirstOrderLogic,
    GodelTheorems,
    LogicOperator,
    ModelTheory,
    ProofTheory,
    Proposition,
    PropositionalLogic,
    SetTheory,
    TypeTheory,
)

__all__ = [
    # Mathematical Logic
    "LogicOperator",
    "Proposition",
    "PropositionalLogic",
    "FirstOrderLogic",
    "ModelTheory",
    "ProofTheory",
    "GodelTheorems",
    "SetTheory",
    "TypeTheory",

    # Computability Theory
    "Direction",
    "TuringState",
    "TuringMachine",
    "ChurchTuringThesis",
    "Decidability",
    "HaltingProblem",
    "ComplexityClasses",
    "NPCompleteness",
]
