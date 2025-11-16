"""
SOAR Pre-built Playbooks

5 production-ready incident response playbooks:
1. SQL Injection Response (CRITICAL)
2. Brute Force Response (WARNING)
3. DDoS Mitigation (CRITICAL)
4. Data Breach Containment (CRITICAL)
5. Anomaly Investigation (WARNING)

**Implemented: 2025-11-16**
"""

from typing import List, Dict, Any
from ..core import SOAREngine, PlaybookMetadata

# Import all playbook modules
from . import sql_injection_response
from . import brute_force_response
from . import ddos_mitigation
from . import data_breach_containment
from . import anomaly_investigation


# List of all playbook modules
ALL_PLAYBOOKS = [
    sql_injection_response,
    brute_force_response,
    ddos_mitigation,
    data_breach_containment,
    anomaly_investigation,
]


def register_all_playbooks(soar_engine: SOAREngine) -> List[str]:
    """
    Register all pre-built playbooks with SOAR engine

    Args:
        soar_engine: SOAR engine instance

    Returns:
        List of registered playbook names
    """
    registered = []

    for module in ALL_PLAYBOOKS:
        metadata = module.METADATA
        execute_func = module.execute

        # Register using decorator pattern
        soar_engine.registry.register(metadata, execute_func)
        registered.append(metadata.name)

    return registered


def get_playbook_catalog() -> List[Dict[str, Any]]:
    """
    Get catalog of all available playbooks

    Returns:
        List of playbook metadata dictionaries
    """
    catalog = []

    for module in ALL_PLAYBOOKS:
        metadata: PlaybookMetadata = module.METADATA

        catalog.append({
            "name": metadata.name,
            "description": metadata.description,
            "severity": metadata.severity.value,
            "triggers": metadata.triggers,
            "version": metadata.version,
            "author": metadata.author,
            "requires_approval": metadata.requires_approval,
            "timeout_seconds": metadata.timeout_seconds,
            "tags": metadata.tags,
        })

    return catalog


__all__ = [
    "sql_injection_response",
    "brute_force_response",
    "ddos_mitigation",
    "data_breach_containment",
    "anomaly_investigation",
    "register_all_playbooks",
    "get_playbook_catalog",
    "ALL_PLAYBOOKS",
]
