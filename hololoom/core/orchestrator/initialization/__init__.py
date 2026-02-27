#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Orchestrator Initialization Module
============================================

Initialization logic for the WeavingOrchestrator.

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: lines 401-1003 (~600 lines)

This module provides clean, single-responsibility initialization functions
for setting up the orchestrator's components.

Public API:
    initialize_config_and_memory: Configure memory sources and protocols
    initialize_reflection_and_caching: Set up learning and caching systems
    initialize_recursive_learning: Configure recursive learning components
    initialize_components: Create all weaving architecture components
    initialize_production_hardening: Set up production monitoring/safety
    initialize_semantic_cache: Configure 244D semantic projection cache
    initialize_linguistic_gate: Set up Phase 5 linguistic filtering

Created: 2025-11-22 (Elegance Pass Refactoring - Phase 2)
"""

from hololoom.core.orchestrator.initialization.config_init import initialize_config_and_memory
from hololoom.core.orchestrator.initialization.reflection_init import initialize_reflection_and_caching
from hololoom.core.orchestrator.initialization.recursive_init import initialize_recursive_learning
from hololoom.core.orchestrator.initialization.component_init import initialize_components
from hololoom.core.orchestrator.initialization.production_init import initialize_production_hardening
from hololoom.core.orchestrator.initialization.cache_init import initialize_semantic_cache
from hololoom.core.orchestrator.initialization.linguistic_init import initialize_linguistic_gate

__all__ = [
    'initialize_config_and_memory',
    'initialize_reflection_and_caching',
    'initialize_recursive_learning',
    'initialize_components',
    'initialize_production_hardening',
    'initialize_semantic_cache',
    'initialize_linguistic_gate',
]
