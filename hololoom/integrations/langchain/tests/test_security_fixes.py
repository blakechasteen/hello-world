"""
Test security fixes for LangChain integration lazy imports.

Tests the fix that replaced eval() with locals() to prevent RCE vulnerability.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock
import importlib


class TestLazyImportSecurity:
    """Test the lazy import security fix that replaced eval() with locals()."""

    def setup_method(self):
        """Clean up any cached imports before each test."""
        # Remove the module from sys.modules to ensure clean import
        if 'hololoom.integrations.langchain' in sys.modules:
            del sys.modules['hololoom.integrations.langchain']

        # Also remove any submodules
        modules_to_remove = [
            key for key in sys.modules.keys()
            if key.startswith('hololoom.integrations.langchain')
        ]
        for module_name in modules_to_remove:
            del sys.modules[module_name]

    def test_lazy_import_document_loaders(self):
        """Test that document loader lazy imports work correctly."""
        from hololoom.integrations import langchain

        # Before accessing, the attribute shouldn't be in globals
        assert 'UniversalDocumentLoader' not in langchain.__dict__

        # Mock the document_loaders module to avoid actual import
        with patch('hololoom.integrations.langchain.document_loaders') as mock_loaders:
            mock_loader_class = MagicMock()
            mock_loaders.UniversalDocumentLoader = mock_loader_class

            # Trigger lazy import via __getattr__
            loader = langchain.UniversalDocumentLoader

            # Verify it got imported and cached
            assert loader is mock_loader_class
            assert langchain.__dict__['UniversalDocumentLoader'] is mock_loader_class

    def test_lazy_import_llm_providers(self):
        """Test that LLM provider lazy imports work correctly."""
        from hololoom.integrations import langchain

        # Before accessing, the attribute shouldn't be in globals
        assert 'MultiProviderLLM' not in langchain.__dict__

        # Mock the llm_providers module
        with patch('hololoom.integrations.langchain.llm_providers') as mock_providers:
            mock_llm_class = MagicMock()
            mock_providers.MultiProviderLLM = mock_llm_class

            # Trigger lazy import
            llm = langchain.MultiProviderLLM

            # Verify it got imported and cached
            assert llm is mock_llm_class
            assert langchain.__dict__['MultiProviderLLM'] is mock_llm_class

    def test_lazy_import_vector_stores(self):
        """Test that vector store lazy imports work correctly."""
        from hololoom.integrations import langchain

        assert 'VectorStoreFactory' not in langchain.__dict__

        with patch('hololoom.integrations.langchain.vector_stores') as mock_stores:
            mock_store_class = MagicMock()
            mock_stores.VectorStoreFactory = mock_store_class

            store = langchain.VectorStoreFactory

            assert store is mock_store_class
            assert langchain.__dict__['VectorStoreFactory'] is mock_store_class

    def test_lazy_import_prototyping(self):
        """Test that prototyping lazy imports work correctly."""
        from hololoom.integrations import langchain

        assert 'QuickPrototype' not in langchain.__dict__

        with patch('hololoom.integrations.langchain.prototyping') as mock_proto:
            mock_proto_class = MagicMock()
            mock_proto.QuickPrototype = mock_proto_class

            proto = langchain.QuickPrototype

            assert proto is mock_proto_class
            assert langchain.__dict__['QuickPrototype'] is mock_proto_class

    def test_lazy_import_caching(self):
        """Test that lazy imports are cached correctly."""
        from hololoom.integrations import langchain

        with patch('hololoom.integrations.langchain.document_loaders') as mock_loaders:
            mock_loader = MagicMock()
            mock_loaders.load_documents = mock_loader

            # First access - triggers import
            first = langchain.load_documents

            # Second access - should use cached value
            second = langchain.load_documents

            assert first is second
            assert first is mock_loader

    def test_invalid_attribute_raises_error(self):
        """Test that accessing invalid attributes raises AttributeError."""
        from hololoom.integrations import langchain

        with pytest.raises(AttributeError) as exc_info:
            _ = langchain.nonexistent_attribute

        assert "has no attribute 'nonexistent_attribute'" in str(exc_info.value)

    def test_no_eval_vulnerability(self):
        """Test that the code doesn't use eval() which could be exploited."""
        # Read the actual source code
        import inspect
        from hololoom.integrations import langchain

        # Get the __getattr__ function source
        source = inspect.getsource(langchain.__getattr__)

        # Ensure no eval() is used
        assert 'eval(' not in source
        assert 'exec(' not in source
        assert '__import__(' not in source

        # Verify locals() is used instead
        assert 'locals()' in source

    def test_locals_returns_correct_imports(self):
        """Test that locals() correctly captures imported names."""
        from hololoom.integrations import langchain

        # Test multiple imports work correctly
        with patch('hololoom.integrations.langchain.document_loaders') as mock_loaders:
            mock_loader1 = MagicMock(name='load_github_repo')
            mock_loader2 = MagicMock(name='load_slack_workspace')
            mock_loaders.load_github_repo = mock_loader1
            mock_loaders.load_slack_workspace = mock_loader2

            # Access both
            gh = langchain.load_github_repo
            slack = langchain.load_slack_workspace

            assert gh is mock_loader1
            assert slack is mock_loader2
            assert gh is not slack

    def test_security_no_code_injection(self):
        """Test that malicious attribute names can't cause code injection."""
        from hololoom.integrations import langchain

        # Try various injection attempts that would work with eval()
        malicious_names = [
            "__import__('os').system('echo hacked')",
            "'; import os; os.system('echo hacked'); '",
            "1 + 1",
            "lambda x: x",
            "__dict__",
            "__class__.__bases__[0].__subclasses__()",
        ]

        for name in malicious_names:
            with pytest.raises(AttributeError) as exc_info:
                getattr(langchain, name)

            # Should just treat it as a missing attribute, not execute it
            assert f"has no attribute" in str(exc_info.value)

    def test_all_public_api_accessible(self):
        """Test that all items in __all__ are accessible via lazy import."""
        from hololoom.integrations import langchain

        # Mock all the submodules
        with patch('hololoom.integrations.langchain.document_loaders') as mock_loaders, \
             patch('hololoom.integrations.langchain.llm_providers') as mock_providers, \
             patch('hololoom.integrations.langchain.vector_stores') as mock_stores, \
             patch('hololoom.integrations.langchain.prototyping') as mock_proto:

            # Set up mock attributes
            for attr in ['UniversalDocumentLoader', 'load_documents', 'supported_document_types',
                        'load_github_repo', 'load_slack_workspace', 'load_notion_database']:
                setattr(mock_loaders, attr, MagicMock(name=attr))

            for attr in ['MultiProviderLLM', 'create_llm', 'list_llm_providers', 'create_best_available_llm']:
                setattr(mock_providers, attr, MagicMock(name=attr))

            for attr in ['VectorStoreFactory', 'create_vector_store', 'list_vector_stores', 'get_recommended_store']:
                setattr(mock_stores, attr, MagicMock(name=attr))

            for attr in ['HoloLoomCLI', 'QuickPrototype', 'quick_start']:
                setattr(mock_proto, attr, MagicMock(name=attr))

            # Verify all __all__ items are accessible
            for name in langchain.__all__:
                attr = getattr(langchain, name)
                assert attr is not None

    def test_performance_lazy_import(self):
        """Test that lazy import doesn't import modules until needed."""
        # Start fresh
        self.setup_method()

        # Import main module
        from hololoom.integrations import langchain

        # Check that submodules aren't imported yet
        assert 'hololoom.integrations.langchain.document_loaders' not in sys.modules
        assert 'hololoom.integrations.langchain.llm_providers' not in sys.modules
        assert 'hololoom.integrations.langchain.vector_stores' not in sys.modules
        assert 'hololoom.integrations.langchain.prototyping' not in sys.modules

    def test_thread_safety_lazy_import(self):
        """Test that lazy imports are thread-safe."""
        from hololoom.integrations import langchain
        import threading
        import time

        results = []

        def access_attribute():
            """Access an attribute in a thread."""
            with patch('hololoom.integrations.langchain.document_loaders') as mock_loaders:
                mock_loaders.UniversalDocumentLoader = MagicMock()
                try:
                    loader = langchain.UniversalDocumentLoader
                    results.append(('success', loader))
                except Exception as e:
                    results.append(('error', str(e)))

        # Start multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=access_attribute)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join(timeout=5.0)

        # All should succeed
        assert len(results) == 10
        for status, _ in results:
            assert status == 'success'