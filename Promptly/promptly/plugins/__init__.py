#!/usr/bin/env python3
"""
Promptly Plugin System

Provides extensibility through:
- Custom evaluators
- Custom storage backends
- Custom chain step processors
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Type, Optional, Any

from .base import (
    EvaluatorPlugin,
    StorageBackend,
    ChainStepProcessor,
    BaseEvaluator,
    BaseStorageBackend,
    BaseChainStepProcessor
)


class PluginRegistry:
    """Central registry for all plugins"""

    def __init__(self):
        self.evaluators: Dict[str, Type] = {}
        self.storage_backends: Dict[str, Type] = {}
        self.chain_processors: Dict[str, Type] = {}

    def register_evaluator(self, evaluator_class: Type) -> None:
        """Register an evaluator plugin"""
        # Check if it implements the protocol or inherits from BaseEvaluator
        if not (isinstance(evaluator_class, type) and
                (issubclass(evaluator_class, BaseEvaluator) or
                 hasattr(evaluator_class, 'evaluate'))):
            raise ValueError(f"{evaluator_class} does not implement EvaluatorPlugin protocol")

        # Create instance to get name
        instance = evaluator_class()
        name = instance.name
        self.evaluators[name] = evaluator_class

    def register_storage_backend(self, backend_class: Type) -> None:
        """Register a storage backend plugin"""
        if not (isinstance(backend_class, type) and
                (issubclass(backend_class, BaseStorageBackend) or
                 hasattr(backend_class, 'save_prompt'))):
            raise ValueError(f"{backend_class} does not implement StorageBackend protocol")

        instance = backend_class()
        name = instance.name
        self.storage_backends[name] = backend_class

    def register_chain_processor(self, processor_class: Type) -> None:
        """Register a chain step processor plugin"""
        if not (isinstance(processor_class, type) and
                (issubclass(processor_class, BaseChainStepProcessor) or
                 hasattr(processor_class, 'process'))):
            raise ValueError(f"{processor_class} does not implement ChainStepProcessor protocol")

        instance = processor_class()
        name = instance.name
        self.chain_processors[name] = processor_class

    def get_evaluator(self, name: str, **kwargs) -> Any:
        """Get an evaluator instance by name"""
        if name not in self.evaluators:
            raise ValueError(f"Evaluator '{name}' not found. Available: {list(self.evaluators.keys())}")
        return self.evaluators[name](**kwargs)

    def get_storage_backend(self, name: str, **kwargs) -> Any:
        """Get a storage backend instance by name"""
        if name not in self.storage_backends:
            raise ValueError(f"Storage backend '{name}' not found. Available: {list(self.storage_backends.keys())}")
        return self.storage_backends[name](**kwargs)

    def get_chain_processor(self, name: str, **kwargs) -> Any:
        """Get a chain processor instance by name"""
        if name not in self.chain_processors:
            raise ValueError(f"Chain processor '{name}' not found. Available: {list(self.chain_processors.keys())}")
        return self.chain_processors[name](**kwargs)

    def list_evaluators(self) -> List[Dict[str, str]]:
        """List all registered evaluators"""
        result = []
        for name, cls in self.evaluators.items():
            instance = cls()
            result.append({
                'name': name,
                'description': instance.description
            })
        return result

    def list_storage_backends(self) -> List[Dict[str, str]]:
        """List all registered storage backends"""
        result = []
        for name, cls in self.storage_backends.items():
            instance = cls()
            result.append({
                'name': name,
                'description': instance.description
            })
        return result

    def list_chain_processors(self) -> List[Dict[str, str]]:
        """List all registered chain processors"""
        result = []
        for name, cls in self.chain_processors.items():
            instance = cls()
            result.append({
                'name': name,
                'description': instance.description
            })
        return result


class PluginLoader:
    """Loads plugins from various sources"""

    def __init__(self, registry: PluginRegistry):
        self.registry = registry

    def load_builtin_plugins(self) -> None:
        """Load built-in plugins"""
        # Load evaluators
        try:
            from .evaluators.keyword import KeywordEvaluator
            from .evaluators.semantic import SemanticSimilarityEvaluator, ExactMatchEvaluator

            self.registry.register_evaluator(KeywordEvaluator)
            self.registry.register_evaluator(SemanticSimilarityEvaluator)
            self.registry.register_evaluator(ExactMatchEvaluator)
        except ImportError as e:
            print(f"Warning: Could not load evaluator plugins: {e}")

        # Load storage backends
        try:
            from .storage.sqlite import SQLiteStorage
            from .storage.json_file import JSONStorage

            self.registry.register_storage_backend(SQLiteStorage)
            self.registry.register_storage_backend(JSONStorage)
        except ImportError as e:
            print(f"Warning: Could not load storage backend plugins: {e}")

    def load_from_directory(self, plugin_dir: Path) -> None:
        """
        Load plugins from a directory

        Expected structure:
        plugin_dir/
            my_evaluator.py  (contains class implementing EvaluatorPlugin)
            my_storage.py    (contains class implementing StorageBackend)
            my_processor.py  (contains class implementing ChainStepProcessor)
        """
        if not plugin_dir.exists():
            return

        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue

            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(
                    plugin_file.stem,
                    plugin_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find and register plugins
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if name.startswith("_"):
                        continue

                    # Try to register as evaluator
                    if (issubclass(obj, BaseEvaluator) or
                        (hasattr(obj, 'evaluate') and hasattr(obj, 'name'))):
                        try:
                            self.registry.register_evaluator(obj)
                        except Exception:
                            pass

                    # Try to register as storage backend
                    if (issubclass(obj, BaseStorageBackend) or
                        (hasattr(obj, 'save_prompt') and hasattr(obj, 'name'))):
                        try:
                            self.registry.register_storage_backend(obj)
                        except Exception:
                            pass

                    # Try to register as chain processor
                    if (issubclass(obj, BaseChainStepProcessor) or
                        (hasattr(obj, 'process') and hasattr(obj, 'name'))):
                        try:
                            self.registry.register_chain_processor(obj)
                        except Exception:
                            pass

            except Exception as e:
                print(f"Warning: Could not load plugin from {plugin_file}: {e}")

    def load_plugin_module(self, module_path: str) -> None:
        """Load a specific plugin module by import path"""
        try:
            module = importlib.import_module(module_path)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if name.startswith("_"):
                    continue

                # Try registering as different plugin types
                try:
                    if hasattr(obj, 'evaluate'):
                        self.registry.register_evaluator(obj)
                except Exception:
                    pass

                try:
                    if hasattr(obj, 'save_prompt'):
                        self.registry.register_storage_backend(obj)
                except Exception:
                    pass

                try:
                    if hasattr(obj, 'process'):
                        self.registry.register_chain_processor(obj)
                except Exception:
                    pass

        except ImportError as e:
            print(f"Warning: Could not load plugin module '{module_path}': {e}")


# Global plugin registry
_global_registry = PluginRegistry()
_global_loader = PluginLoader(_global_registry)

# Load built-in plugins automatically
_global_loader.load_builtin_plugins()


def get_registry() -> PluginRegistry:
    """Get the global plugin registry"""
    return _global_registry


def get_loader() -> PluginLoader:
    """Get the global plugin loader"""
    return _global_loader


# Convenience functions
def get_evaluator(name: str, **kwargs):
    """Get an evaluator plugin instance"""
    return _global_registry.get_evaluator(name, **kwargs)


def get_storage_backend(name: str, **kwargs):
    """Get a storage backend plugin instance"""
    return _global_registry.get_storage_backend(name, **kwargs)


def get_chain_processor(name: str, **kwargs):
    """Get a chain processor plugin instance"""
    return _global_registry.get_chain_processor(name, **kwargs)


def list_plugins() -> Dict[str, List[Dict[str, str]]]:
    """List all available plugins"""
    return {
        'evaluators': _global_registry.list_evaluators(),
        'storage_backends': _global_registry.list_storage_backends(),
        'chain_processors': _global_registry.list_chain_processors()
    }


__all__ = [
    'PluginRegistry',
    'PluginLoader',
    'get_registry',
    'get_loader',
    'get_evaluator',
    'get_storage_backend',
    'get_chain_processor',
    'list_plugins',
    'EvaluatorPlugin',
    'StorageBackend',
    'ChainStepProcessor',
    'BaseEvaluator',
    'BaseStorageBackend',
    'BaseChainStepProcessor'
]
