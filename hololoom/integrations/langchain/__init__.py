"""
LangChain Integration for HoloLoom
===================================

Adds 100+ document loaders, multiple LLM providers, and vector store integrations
from LangChain to HoloLoom's learning-first architecture.

Date: November 2025
Status: Production Ready
"""

from typing import TYPE_CHECKING

# Public API
__all__ = [
    # Document loaders
    'UniversalDocumentLoader',
    'load_documents',
    'supported_document_types',
    'load_github_repo',
    'load_slack_workspace',
    'load_notion_database',

    # LLM providers
    'MultiProviderLLM',
    'create_llm',
    'list_llm_providers',
    'create_best_available_llm',

    # Vector stores
    'VectorStoreFactory',
    'create_vector_store',
    'list_vector_stores',
    'get_recommended_store',

    # Prototyping
    'HoloLoomCLI',
    'QuickPrototype',
    'quick_start',
]

# Lazy imports to avoid dependency issues
if TYPE_CHECKING:
    from .document_loaders import (
        UniversalDocumentLoader,
        load_documents,
        load_github_repo,
        load_notion_database,
        load_slack_workspace,
        supported_document_types,
    )
    from .llm_providers import (
        MultiProviderLLM,
        create_best_available_llm,
        create_llm,
        list_llm_providers,
    )
    from .prototyping import HoloLoomCLI, QuickPrototype, quick_start
    from .vector_stores import (
        VectorStoreFactory,
        create_vector_store,
        get_recommended_store,
        list_vector_stores,
    )

def __getattr__(name):
    """Lazy import for better startup time and graceful degradation."""
    if name in ['UniversalDocumentLoader', 'load_documents', 'supported_document_types',
                'load_github_repo', 'load_slack_workspace', 'load_notion_database']:
        from .document_loaders import (
            UniversalDocumentLoader,
            load_documents,
            load_github_repo,
            load_notion_database,
            load_slack_workspace,
            supported_document_types,
        )
        # Use locals() instead of eval() to avoid RCE vulnerability
        globals()[name] = locals()[name]
        return globals()[name]

    elif name in ['MultiProviderLLM', 'create_llm', 'list_llm_providers', 'create_best_available_llm']:
        from .llm_providers import (
            MultiProviderLLM,
            create_best_available_llm,
            create_llm,
            list_llm_providers,
        )
        # Use locals() instead of eval() to avoid RCE vulnerability
        globals()[name] = locals()[name]
        return globals()[name]

    elif name in ['VectorStoreFactory', 'create_vector_store', 'list_vector_stores', 'get_recommended_store']:
        from .vector_stores import (
            VectorStoreFactory,
            create_vector_store,
            get_recommended_store,
            list_vector_stores,
        )
        # Use locals() instead of eval() to avoid RCE vulnerability
        globals()[name] = locals()[name]
        return globals()[name]

    elif name in ['HoloLoomCLI', 'QuickPrototype', 'quick_start']:
        from .prototyping import HoloLoomCLI, QuickPrototype, quick_start
        # Use locals() instead of eval() to avoid RCE vulnerability
        globals()[name] = locals()[name]
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Version
__version__ = '1.0.0'
